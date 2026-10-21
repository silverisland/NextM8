# Covariate integration and non-weight-update adaptation for time-series foundation models (focus: TimesFM)

Scope: how TimesFM 1.x/2.x/2.5/3 and competing TSFMs consume covariates, and which adaptation routes other than weight fine-tuning exist, with evidence on which help most for solar/energy. Status as of 2026-10-15.
Labels used below:
- **[CODE]** = read in the installed package `timesfm 3.0.2` (the wheel contains both `timesfm/` (2.5 + XReg) and `timesfm3/` (3.0)) at `.../scratchpad/venv314/lib/python3.14/site-packages/` from another session.
- **[PAPER]** / **[BLOG]** = what the authors say.
- **[3RD]** = third-party write-up.

---

## Q1. How does TimesFM handle covariates (1.x/2.x XReg -> 2.5 -> 3.0)? What does fine-tuning do to the covariate pathway?

### Takeaway
TimesFM 1.x/2.x/2.5 never put covariates inside the transformer. They use "XReg": one in-context ridge regression fitted per call, with a single coefficient vector shared by the whole batch, either before TimesFM (TimesFM then forecasts the residuals) or after it (fitted on TimesFM's residuals). TimesFM-3 (Aug 2026) is the first version with **native** covariates. Each covariate becomes its own variate row of patch tokens, with its own causal RevIN. Past-future covariates (our NWP) are unmasked in the horizon, and every one of their tokens also carries a 64-step lookahead of itself. Variate attention mixes them into the target at every patch position. So in our LoRA setup the NWP pathway is the variate-attention plus input-ResBlock pathway, and which LoRA targets we pick decides whether that pathway gets adapted at all.

### Cited Findings
**Version timeline (repo README)**
- 1.0/2.0 now live in the repo's `v1` subdirectory and need `pip install timesfm==1.3.0`. 2.5 was released 2025-09-15: 200M params, 16k context, optional 30M quantile head, no frequency indicator. On 2025-10-29 the changelog says "Added back the covariate support through XReg for TimesFM 2.5". 3.0 launched Aug 2026 "with native multivariate forecasting and covariate support", weights at `google/timesfm-3.0-pytorch`, and BigQuery ML rollout finished Sept 2026. — [timesfm GitHub README](https://github.com/google-research/timesfm)
- **License:** per the README, the 3.0 weights "are restricted to non-commercial, non-production use". Code and weights up to 2.5 are Apache-2.0. This matters for production at a utility; verify the HF model card. — [timesfm GitHub README](https://github.com/google-research/timesfm); echoed by [3RD pasqualepillitteri.it](https://pasqualepillitteri.it/en/news/13739/google-timesfm-3-forecasting-model-non-commercial)
- A LoRA/PEFT fine-tuning example (HF Transformers + PEFT) was added on 2026-04-09 under `timesfm-forecasting/examples/finetuning/` and is listed under 2.5. — [timesfm GitHub README](https://github.com/google-research/timesfm)

**XReg (TimesFM 2.5 code; 1.x has the same design, but the archived v1 code was not re-read here)**
- [CODE] `TimesFM_2p5.forecast_with_covariates(inputs, dynamic_numerical_covariates, dynamic_categorical_covariates, static_numerical_covariates, static_categorical_covariates, xreg_mode="xreg + timesfm", normalize_xreg_target_per_input=True, ridge=0.0, max_rows_per_col=0, force_on_cpu=False)`. It requires `return_backcast=True` in the compiled forecast config. The forecast horizon is inferred from the length of the dynamic covariates.
- [CODE] Docstring and branch logic: **"xreg + timesfm" first fits XReg on the targets, then uses TimesFM to forecast the residuals.** "timesfm + xreg" first runs TimesFM, then fits XReg on its in-sample residuals; that mode drops the first patch, so train_len = input_len - p. Some third-party docs state the opposite order and are wrong against the code (e.g. a SKILL doc surfaced in search said the default "TimesFM forecasts first").
- [CODE] `xreg_lib.BatchedInContextXRegLinear.fit` concatenates every series in the call into one design matrix and solves a single `beta_hat = pinv(XᵀX + ridge·I) Xᵀ y`, a JAX ridge via pseudo-inverse. Forecasts are then split back per series. **The coefficients are shared across the batch.** Per-series behaviour comes only from per-input target normalisation (`normalize_xreg_target_per_input`, mean/std per series) and from static or categorical one-hots. Categorical one-hot drops the "first" level when ridge = 0. The XReg output is added to point and quantile forecasts as a constant shift, so the quantile spread does not depend on the covariates.
- XReg needs the dynamic covariates to cover both context and horizon. A 2026 user paper flags this as a limitation; for us it is a non-issue because NWP is available for the horizon. — [arXiv 2605.15752 via search snippet](https://arxiv.org/pdf/2605.15752)
- [3RD] One tutorial claimed 2.5 had dropped covariates and ported them back itself. That predates the official 2025-10-29 restoration. — [AI Horizon Forecast](https://aihorizonforecast.substack.com/p/timesfm-25-external-covariates-a)

**TimesFM-3 native covariates**
- [BLOG] Covariates come in three kinds: multiple targets, past covariates, and past-future covariates ("planned promotions or weather forecasts"). Other blog details:
  - patches of 32 steps, per-series normalisation similar to 2.5;
  - target and past-covariate tokens are built from a single patch;
  - **a past-future covariate token "concatenates the current patch with future patches (lookahead)"**;
  - alternating causal temporal attention within each series and full variate attention across series at each time step;
  - non-autoregressive decoding with Contiguous Patch Masking and 9 output quantiles;
  - 330M params, pretrained on more than 1T time points of real plus synthetic data.
  — [Google Research blog, 2026-08-31](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- [CODE] Default `TimesFM3Torch` config:
  - input patch 32, output patch 64 (rolls = 2), 20 layers, d = 1280, FFN 1280, 16 heads;
  - RMS norms and qk_norm, RoPE on the sequence axis, **no RoPE on the variate axis** (`use_rope_var=False`), `max_variates=32`.
  - My count: about 16.4M params per layer (seq-attn 6.6M + var-attn 6.6M + FFN 3.3M) times 20, which matches the blog's 330M. **Our notes call TimesFM-3 "~200M", but that is the 2.5 size.**
- [CODE] `decode()` packs the inputs as variates:
  - order is `[targets | past_only | past_future]`, with shape (b, u + v_po + w, L);
  - `patch_is_target = True` for targets **and for past-only covariates**: they are treated like targets, their horizon is masked and predicted, and the prediction is thrown away;
  - past-future covariates have `patch_is_target = False`, and their horizon values `past_future_covariates[..., context:context+horizon]` are kept unmasked.
- [CODE] `_preprocess()` builds each token as follows:
  - per-variate, per-patch **causal running mean/std (RevIN)** via `get_running_stats`, which is cumulative over unmasked points;
  - then `values_fcov = get_output_patch_via_roll(values, rolls)`, i.e. the next 64 steps of the same variate;
  - this is masked out for target/past-only variates (`masks_fcov |= patch_is_target`);
  - the ResBlock input is `[patch(32) | lookahead(64) | masks(96)]`, so 192 numbers go into `pre_transformer_resblock` and come out at d = 1280.
  - The lookahead slot is the only input-level feature that tells "known-future covariate" apart from target. Variates carry no learned identity embedding.
- [CODE] `MixingTransformer` layer, per layer:
  1. sequence attention over n patches per variate (causal by default);
  2. variate attention over v at each patch index (non-causal, no positional encoding, so permutation-invariant across covariates);
  3. FFN.
  Forecasts are read from the target variate's tokens at `num_context_patches-1+k`, taking 2 x 32 = 64 steps, then stitched (`use_stitching=True`). **Each horizon target token sees the NWP of the same time slot only through variate attention**, plus the 64-step NWP lookahead embedded in the covariate token.
- [CODE] Linear detrending (`use_linear_detrending=True`) is decided per variate, covariates included. A variate is detrended if the std after removing a linear fit is below 0.5 times the original std. The trend of a past-future covariate is extrapolated into its horizon values. `use_frozen_running_stats=False` by default, so a past-future covariate's running stats keep updating over its horizon patches with future NWP values. `TimesFM3Torch` is documented as "inference only": `decode` is `@torch.no_grad`, so training has to go through `forward()` with hand-built `values/masks/patch_is_target`, as our M5/M6 code does.
- [CODE] `TimesFM3Forecaster` truncates the context to `context_length` and keeps the covariate window aligned with the target (regression test `test_truncated_context_keeps_covariate_window_aligned`). In a mixed batch, a query without covariates gets zero placeholders that are fully masked.
- [CODE, our repo] `forecast_backbones/m5/config/finetune.yaml` LoRA targets `[query_proj, key_proj, value_proj, out_proj]`. `lora.inject_lora` matches modules by name suffix. In `timesfm3/torch/transformer.py` **both** `seq_attn` and `var_attn` are `MultiHeadAttention` with these exact names, so our LoRA adapts temporal and variate (covariate-mixing) attention. It leaves untouched the `pre_transformer_resblock` (which embeds the NWP lookahead) and the FFNs, and `output_head` only if `unfreeze_output_head` is set.

**What fine-tuning does to the covariate pathway (evidence)**
- No paper found that studies LoRA on TimesFM-3 covariate attention specifically. Related evidence:
  - Sequential fine-tuning of TimesFM-2.0 and Chronos-2 on energy data shows a stability–plasticity trade-off, i.e. forgetting. — [arXiv 2510.00809](https://arxiv.org/html/2510.00809v2)
  - In a day-ahead electricity price study, fine-tuning Moirai improved calibration while "gains from including exogenous variables are limited". — [arXiv 2609.00089 via search](https://arxiv.org/pdf/2609.00089)

### Inferences
- **Per-variate RevIN makes TimesFM-3 see NWP-GHI only as a shape relative to its own 7-day context.** Its absolute level (clear vs cloudy week) is normalised away, and the target's level comes only from the target's own history. Day-ahead PV level errors on regime changes, e.g. a sunny week followed by a rainy day, therefore depend on the model learning "covariate drops below its running mean, so target drops" through variate attention. A level-preserving channel adds information RevIN removes. Candidates: an XReg-style ridge on top ("timesfm + xreg"), or feeding a covariate already in target units such as NWP-implied clear-sky power or PV-model power.
- LoRA on attention q/k/v/o adapts the variate-mixing weights, which is plausibly why covariate LoRA helps. The NWP lookahead embedding in `pre_transformer_resblock` stays frozen, though. An ablation adding `hidden_layer/output_layer/residual_layer` (ResBlock) to the LoRA targets is cheap and directly targets "how NWP lookahead is read".
- Past-only covariates are internally predicted as extra targets. Passing measured irradiance or other only-observed signals as past-only is therefore costly: they compete for capacity and are force-forecast. Anything known for the future should go in `past_future_covariates`.
- XReg's shared-coefficient ridge across a batch is a poor fit for 28 heterogeneous stations unless each station is called separately or a station one-hot is added. TimesFM-3 has no XReg path at all [CODE: no xreg/ridge in `timesfm3_forecaster.py`], but `timesfm.utils.xreg_lib` from the same wheel can be reused as a post-hoc residual corrector.

### Gaps
- No public TimesFM-3 paper or tech report found with covariate ablations (with vs without covariates) or energy/solar numbers. The blog's charts are images without values.
- Whether the HF checkpoint config overrides the code defaults (e.g. `use_frozen_running_stats`, `input_transform`) was not checked here.
- The 1.x XReg code (v1 dir) was not re-read. The claim that 1.x matches 2.5 rests on the 2.5 port and the changelog wording.

---

## Q2. Covariate adapters and covariate-native TSFMs (ChronosX, Chronos-2, Moirai, UniCA, COSMIC, ApolloPFN, TimeXer, benchmarks)

### Takeaway
The field has converged on two designs. One is the **"covariates as extra variates plus cross-variate attention"** used by TimesFM-3, Chronos-2 (group attention) and Moirai (any-variate attention). The other is **frozen-backbone adapters**: ChronosX's injection blocks and UniCA's pre/post-backbone fusion. Independent energy evidence says channel-mixing / group-attention designs exploit weather covariates most consistently, while weak cross-channel designs can get *worse* with covariates. In PV specifically, covariate-aware TSFMs beat non-covariate baselines by large margins, but a well-featured LightGBM is still as good or better unless the TSFM is fine-tuned.

### Cited Findings
**Chronos-2** (Amazon, arXiv 2510.15821)
- Design:
  - every block alternates time attention (within a series) and **group attention** (across all series sharing a group ID at the same patch index);
  - target(s), past-only and known covariates of a task share one group ID;
  - known covariates' future values sit in the future input while targets are masked;
  - covariate predictions are discarded;
  - categorical covariates use target encoding (univariate target) or ordinal encoding.
  — [Chronos-2 paper](https://arxiv.org/html/2510.15821v1)
- Training data: "For multivariate and covariate-informed tasks, we relied entirely on synthetic data". Multivariatizers impose dependencies on AR/ETS/KernelSynth series. — [Chronos-2 paper](https://arxiv.org/html/2510.15821v1)
- Results: fev-bench SQL over the whole benchmark: Chronos-2 90.7% win rate / 47.3% skill, TimesFM-2.5 75.9% / 42.3%, TabPFN-TS 59.3% / 39.6%. On the covariate subset Chronos-2 "outperforms baselines by a large margin" with TabPFN-TS second; numbers are only in Fig. 4b. In the EPF-DE energy case, ICL with covariates is "significantly more accurate" than univariate mode (Fig. 6, no text numbers). TimesFM-2.5 is listed as lacking covariate support, so XReg is not compared. — [Chronos-2 paper](https://arxiv.org/html/2510.15821v1)

**ChronosX** (AISTATS 2025, arXiv 2503.12107)
- Modular blocks "inject past and future covariate information, without necessarily modifying the pretrained model". Evaluated on 32 synthetic covariate datasets plus real data. — [arXiv 2503.12107](https://arxiv.org/abs/2503.12107); [PMLR 258](https://proceedings.mlr.press/v258/arango25a.html)
- [3RD] Described as Input Injection (adjusting token embeddings) plus Output Injection blocks. — [Moonlight review](https://themoonlight.io/review/chronosx-adapting-pretrained-time-series-models-with-exogenous-variables)
- In an independent demand study ChronosX (on chronos-t5-small) used covariates inconsistently and only supports horizons up to 64 without retraining. — [arXiv 2602.05390](https://arxiv.org/html/2602.05390v1)

**UniCA** (ICLR 2026, arXiv 2506.22039)
- Covariate homogenisation turns heterogeneous covariates (categorical, image, text) into a temporal representation. Attention fusion happens pre- and post-backbone, and the TSFM parameters stay frozen. Code at github.com/hanlu-nju/UniCA. The paper claims TSFMs with covariates "substantially outperform specialized models". — [arXiv 2506.22039](https://arxiv.org/html/2506.22039v1); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10010335)

**COSMIC** (AWS/JKU, arXiv 2506.03128)
- Zero-shot covariate use via in-context learning. Trained without any covariate datasets thanks to "Informative Covariate Augmentation". Claims SOTA zero-shot with and without covariates, e.g. Electricity at H = 24: MASE 0.729 vs DeepAR 0.960. — [arXiv 2506.03128](https://arxiv.org/html/2506.03128v1)
- Related: Dange et al. (ICML'24 workshop) argue that a shifted causal attention lets covariates plus history implement in-context linear regression in one layer. — [ICML 2024 TF2M](https://icml.cc/virtual/2024/37853)
- Related: Baguan-TS (arXiv 2603.17439) is a sequence-native ICL model with covariates and reports solar_with_weather_15T/1H benchmarks. — [arXiv 2603.17439](https://arxiv.org/pdf/2603.17439)

**Moirai** any-variate attention
- Flattens multivariate input and applies attention with variate-aware biases. Pretrained on energy among other domains. — [Salesforce blog](https://www.salesforce.com/blog/moirai/)
- Independent demand study: Moirai zero-shot with all features "generally got worse, and the losses grew with horizon". Only fine-tuned Moirai benefited, at short horizons. — [arXiv 2602.05390](https://arxiv.org/html/2602.05390v1)

**ApolloPFN**
- A time-aware PFN with native exogenous covariates; claims SOTA on M5 and electricity price forecasting. It is proprietary, so no independent replication was found. — [via search, arXiv 2603.15802](https://arxiv.org/html/2603.15802)

**Independent energy evidence on covariate exploitation**
- Singapore/ACT electricity demand study: "TTM's channel-mixing and Chronos-2's grouped attention consistently leverag[e] exogenous features, while other approaches show inconsistent benefits". Even so, Chronos-2 hourly-Singapore MAPE got slightly worse with all ~30 features (about -0.3% to -3.7%), while daily-Singapore improved by up to 26.4%. A RevIN-LSTM often beat TSFMs at short horizons. — [arXiv 2602.05390](https://arxiv.org/html/2602.05390v1)
- FETS benchmark (54 energy datasets, 9 categories):
  - covariate-informed zero-shot TSFMs were the best group;
  - Chronos-2 median NRMSE 0.472, TiRex-2 0.474, XGBoost 0.611, RF 0.696 (the tree models trained task-specifically);
  - NWP from ECMWF-IFS runs issued at least 9 h before t0 (no leakage);
  - accuracy saturates at about 2,000 context steps (about 3 weeks at 15 min).
  — [FETS arXiv 2604.22328](https://arxiv.org/abs/2604.22328); [HTML v2](https://arxiv.org/html/2604.22328v2)
- Cold-start PV (440 sites): covariate-aware TSFMs beat baselines by about 1.7–2x. TabPFN-TS was best under real feedback (MAE 0.514 kWh/kWp/day) and Chronos-2 most robust under self-forecast feedback. A search snippet gives DKASC WAPE of about 8.99% for Chronos-2 vs about 9.37% for TimesFM 2.5 vs about 13.89% for Prophet. — [arXiv 2606.07457](https://arxiv.org/abs/2606.07457)
- ERCOT (ARPA-E PERFORM) power-system benchmark: zero-shot TSFMs were poor on solar (nMAE > 9%). Day-ahead solar CRPS: Chronos-Bolt 2.25, TimesFM 2.53, LSTM 3.32. — [arXiv 2604.22077 via search](https://arxiv.org/pdf/2604.22077)
- TimeXer (Wang et al., NeurIPS 2024): endogenous series as patch tokens plus each exogenous series as one variate token, linked through a learnable global endogenous token with cross-attention. This is the canonical task-specific exogenous design. — [arXiv 2402.19072](https://arxiv.org/abs/2402.19072) (not re-fetched this session; architecture summary from prior knowledge)

### Inferences
- TimesFM-3's design is closest to Chronos-2: variate attention, covariates as variates, synthetic-style covariate pretraining is likely. The Chronos-2 evidence that grouped attention exploits covariates most consistently probably transfers. Adapter methods such as ChronosX and UniCA mainly matter for backbones *without* native covariates (TimesFM ≤ 2.5, Chronos-Bolt), so they are lower priority for us.
- Unrelated or weak covariates can hurt cross-variate-attention models (the Chronos-2 hourly demand case). Feed a small set of physically strong covariates: forecast GHI, clear-sky GHI/power, maybe temperature. Do not dump all NWP fields.

### Gaps
- No "GIFT-Eval covariate track" found. GIFT-Eval appears univariate-focused, and fev-bench is the covariate-heavy benchmark. No reported TimesFM-3 numbers on solar-with-NWP tasks found.
- "Apollo-forecast" as named in the brief was not found. The closest match is ApolloPFN.
- Exact Chronos-2 covariate-subset numbers are only in figures.

---

## Q3. In-context fine-tuning (TimesFM-ICF): mechanism, gains, availability; could we put same-station previous days or neighbouring stations in context?

### Takeaway
ICF continues pretraining TimesFM so that its context can hold up to about 50 example series separated by a learnable separator token. It matches or slightly beats per-dataset full fine-tuning, about 3% better on Monash, with zero gradient steps at deployment. No public ICF checkpoint was found, and it was built on a TimesFM-1-era model, not 2.5/3. TimesFM-3's native variate axis offers a different, available route: pass neighbouring stations as extra *targets* or *covariates* in the same call.

### Cited Findings
- Mechanism:
  - training contexts pack up to 50 examples, each up to 640 points (512 history + 128 horizon), drawn from the same series or other series of the same dataset;
  - a learnable 1280-d separator token follows each example and is never masked;
  - causal attention spans all earlier examples;
  - the base checkpoint was recreated with NoPE (no positional encoding) for length generalisation;
  - base model: TimesFM 200M, 20 layers, 16 heads, d = 1280, input patch 32, output 128, about 400B pretraining points.
  — [ICF paper, arXiv 2410.24087](https://arxiv.org/html/2410.24087)
- Results:
  - Monash scaled MAE (geo-mean): ICF 0.643, FT-TimesFM full 0.663, FT linear-probe 0.676, base 0.694, N-BEATS 0.700;
  - ETT avg MAE at H = 96/192: ICF 0.207/0.265 vs PatchTST 0.335/0.368 vs base 0.348/0.387;
  - fine-tuning all datasets took 115 min (LP) vs about 4 min of ICF inference.
  — [arXiv 2410.24087](https://arxiv.org/html/2410.24087)
- Ablations:
  - ETT MAE falls from 0.411 (1 example) to 0.345 (50 examples);
  - long-history control: TimesFM with 2,048-step history scores 0.685, vs ICF 0.675 with 4 examples and 0.643 with 50, i.e. *related examples beat simply more own history*;
  - same-series vs other-series examples were not ablated.
  — [arXiv 2410.24087](https://arxiv.org/html/2410.24087)
- Google blog: ICF is 6.8% more accurate than base on 23 unseen datasets (GM MASE vs seasonal naive) and matches TimesFM-FT. More examples mean more accuracy but slower inference. Examples come from the series' own full history and from other series in the dataset. — [Google Research blog](https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/)
- Published at ICML 2025 (Faw, Sen, Zhou, Das), PMLR 267:16355–16374. — [mlanthology](https://mlanthology.org/icml/2025/faw2025icml-incontext)
- Availability: neither the paper, the blog nor the timesfm README mentions released ICF weights or code. — [timesfm README](https://github.com/google-research/timesfm); [blog](https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/)

### Inferences
- With TimesFM-3 we cannot use ICF separators, because the checkpoint was not trained with them. Feeding concatenated "previous days" in the sequence without a separator just reproduces the long-history baseline, and 672 steps already cover 7 days.
- Viable substitutes within the TimesFM-3 API:
  1. **Neighbouring stations as extra target variates** (multivariate mode) or as past-only covariates, so variate attention can share cloud-regime information. Limits: `max_variates=32` per call, and the variate axis is permutation-invariant with no identity.
  2. **Analog days chosen by NWP similarity, placed as past-future covariates**, e.g. a covariate series equal to "power of the most NWP-similar historical day", shifted into the horizon. This is a retrieval-as-covariate hack. It respects the model's training distribution better than inventing separators.
  3. Extending the context toward FETS's about 2,000-step saturation point (about 3 weeks). This is cheap to test.

### Gaps
- No public evidence on ICF for solar, or on ICF combined with covariates. No ICF port for TimesFM 2.5/3 found.

---

## Q4. Test-time adaptation (TTA), online calibration, and retrieval-augmented forecasting

### Takeaway
Gradient-free or light online correctors on top of a frozen TSFM consistently help in the literature:
- ELF improved MASE in every TSFM × dataset × horizon combination reported, at about 0.4 s per CPU update;
- TAFAS cut Chronos error by up to 45% under distribution shift.

Retrieval (RAF, TimeRAF, TS-RAG, PARA-PV) also helps, with bigger gains for bigger models. For PV, the cheapest high-value variant is a per-station online linear or gated output corrector, which is close to ELF/COSA in spirit and to the user's existing "refit mode".

### Cited Findings
**Test-time adaptation**
- **TAFAS** (AAAI-25): periodicity-aware adaptation scheduling plus a gated calibration module. It uses partially observed ground truth as it arrives and preserves the source model's semantics. It improves Chronos's error on unseen streams "by up to 45%". Code: github.com/kimanki/TAFAS. — [AAAI-25](https://ojs.aaai.org/index.php/AAAI/article/view/33965); [arXiv 2501.04970](https://arxiv.org/html/2501.04970v1)
- **PETSA** (ICML'25 TTA workshop) is a parameter-efficient TAFAS variant: input/output calibration with low-rank adapters and dynamic gating. **COSA** (ICLR 2026) does direct output-space correction with minimal overhead. A principled-TTA paper also exists. — [PETSA arXiv 2506.23424](https://arxiv.org/pdf/2506.23424); [COSA slides](https://iclr.cc/media/iclr-2026/Slides/10010061_jx4tDbU.pdf); [arXiv 2605.17250](https://arxiv.org/pdf/2605.17250)
- **ELF** (formerly AdapTS; ICML 2025):
  - the FM stays frozen;
  - ELF-Forecaster is a Fourier-domain linear model on recent data (keeps low-frequency components, α = 0.9), with closed-form ridge (λ = 20) and Woodbury updates, refit every M = 200 steps with L = 520;
  - ELF-Weighter mixes fast and slow exponential weights between the FM and ELF-Forecaster;
  - tested on Chronos-tiny, TTM, TimesFM, Moirai-small and VisionTS over ETT, Weather, Solar, ECL and Traffic;
  - MASE dropped in every reported case, with VisionTS averaging more than 10%;
  - about 0.38 s per update on 2 CPUs, about 10–113x faster than TAFAS/OneNet, and about 2,506x faster than continual fine-tuning of TTM (with +5.89% accuracy on ETT).
  — [arXiv 2502.12920v3](https://arxiv.org/html/2502.12920v3); [PMLR 267](https://proceedings.mlr.press/v267/lee25ag.html)
- **ORCA** (2026): a black-box online residual adapter conditioned on the base model's input *and* output, built for API-only TSFMs and tested on 5 TSFMs × 8 datasets. — [via search, emergentmind 2606.14222](https://www.emergentmind.com/papers/2606.14222)

**Retrieval-augmented forecasting**
- **RAF** (Tire et al., AISTATS 2026): appends top-matching historical patterns to the query and relies on TSFM in-context learning, with no weight change, as "a resource-efficient alternative to fine-tuning". Gains are larger for larger models. Chronos-Mini failed to use even the oracle future as the retrieved example, while Small/Base succeeded, so this is an emergent ability. — [arXiv 2411.08249](https://www.arxiv.org/pdf/2411.08249); [PMLR v300](https://proceedings.mlr.press/v300/tire26a.html)
- **TimeRAF**: a Channel Prompting module extracts information from retrieved series and reportedly beats some full-shot methods zero-shot. — [arXiv 2412.20810](https://arxiv.org/pdf/2412.20810)
- **TS-RAG**: retrieval-augmented TSFMs as stronger zero-shot forecasters (method details not read). — [arXiv 2503.07649](https://arxiv.org/pdf/2503.07649)
- **PARA-PV** (arXiv 2607.08079), PV-specific:
  - retrieves historical patches and analog trajectories matched on shape, power level, PV operating state and intraday period (abstract does not mention NWP);
  - a **frozen Chronos prior** is calibrated by a lightweight residual adapter;
  - a physics-aware gated mean-shift/scale correction uses power, weather, timestamp and day/night;
  - the loss is reweighted over peak, ramp, night and regular regimes.
  The authors argue a standalone TSFM "is not suitable" for PV because of capacity limits, night zeros and weather regimes. Numbers were not on the abstract page. — [arXiv 2607.08079](https://arxiv.org/abs/2607.08079)

### Inferences
- For day-ahead PV, the analogue of TAFAS's "partially observed ground truth" is yesterday's realised generation versus yesterday's forecast. A per-station daily-refit output corrector is cheap, gradient-free and orthogonal to LoRA. Examples: a ridge on [TSFM quantiles, NWP-GHI, clear-sky power, hour-of-day], or a gated scale/shift per clear-sky-index regime.
- Retrieval keyed on NWP-forecast similarity, which PARA-PV does not do, is more relevant day-ahead than shape-based retrieval: tomorrow's history is unknown, but tomorrow's NWP is known. With TimesFM-3 the retrieved analog would enter as a past-future covariate (see Q3).

### Gaps
- No TTA paper found that evaluates TimesFM-3 or 15-min day-ahead PV with NWP. ELF's "Solar" is the classic Solar-Energy (Alabama) dataset with no NWP.
- PARA-PV's quantitative results and its Chronos version were not retrieved.

---

## Q5. Hybrid / stacking: TSFM + LightGBM residual correction, TSFM quantiles as features, "foundation model as prior"

### Takeaway
In the most directly relevant PV study (15-min, GFS NWP), a full-feature LightGBM (4.50% nMAE) beat fine-tuned Chronos-2 (4.59%), and both crushed zero-shot Chronos-2 (7.70%). A fusion ensemble with LightGBM as one expert was best (4.32%), but its gain over LightGBM was below the authors' own practical-significance threshold. Hybrids help modestly. The robust lesson is that **NWP-driven tree models stay a strong baseline for PV, and TSFMs add most as a prior or as one expert, not alone**. No controlled study of "TSFM quantiles as LightGBM features" was found for PV.

### Cited Findings
- Horizon-specific expert fusion study (PVDAQ 2107, 15-min, GFS NWP; train 2019–22, test Jan–Oct 2024):
  - daylight horizon-averaged nMAE: fusion ensemble 4.315%, LightGBM 4.500%, fine-tuned Chronos-2 (3,000 full-param updates) 4.592%, zero-shot Chronos-2 7.701%;
  - the ensemble gain over LightGBM was +0.185 pp (95% block CI +0.118 to +0.247), below the predeclared 0.2 pp practicality threshold;
  - by horizon (15/60/240 min), LightGBM 2.160/3.815/5.422 vs Chronos-2 FT 2.504/3.917/5.663;
  - Chronos-2 was **not** an ensemble member; LightGBM was;
  - intraday only (≤ 4 h), no day-ahead.
  — [arXiv 2609.15035](https://arxiv.org/html/2609.15035)
- PARA-PV uses a frozen Chronos as a "temporal prior" plus a residual adapter and physics correction ("foundation model as prior"). — [arXiv 2607.08079](https://arxiv.org/abs/2607.08079)
- ELF's FM + linear-forecaster weighting is itself a hybrid and improved every TSFM tested. — [arXiv 2502.12920v3](https://arxiv.org/html/2502.12920v3)
- FETS: covariate-informed zero-shot TSFMs (median NRMSE about 0.47) beat task-specific XGBoost (0.611) across 54 energy datasets. The tree baselines' feature set is not described in the abstract. — [arXiv 2604.22328](https://arxiv.org/abs/2604.22328)
- Electricity price forecasting: "only the TabPFN models consistently and significantly outperform the benchmarks". TSFMs cannot universally replace market-specific models. — [arXiv 2609.00089 via search](https://arxiv.org/pdf/2609.00089)
- Demand: a simple baseline often beat all TSFMs in Singapore at short horizons. — [arXiv 2602.05390](https://arxiv.org/html/2602.05390v1)

### Inferences
- The FETS result (TSFM > XGBoost) and the PV fusion result (LightGBM > fine-tuned Chronos-2) differ mainly in how strong the tree baseline's features are. With rich NWP plus solar-geometry features, LightGBM is hard to beat. This supports the user's planned "LightGBM baseline" lever (C/D deferred in memory).
- Two candidate stackings to test:
  - (a) LightGBM on [NWP, clear-sky, TimesFM-3 P10/P50/P90, hour] predicting power. TimesFM features carry level/shape priors from history that NWP lacks.
  - (b) LightGBM on the residual of TimesFM-3. This tends to over-fit less when TimesFM is already good.
  Use out-of-fold TimesFM predictions on the train period to avoid leakage. With LoRA-fine-tuned TimesFM, in-sample predictions are optimistic.

### Gaps
- No day-ahead (24–48 h) PV study found that pits TimesFM (any version) with NWP covariates against LightGBM-with-NWP, or that stacks them.

---

## Q6. Input/target transformations for PV with TSFMs (clear-sky index, capacity normalisation, clear-sky power as covariate)

### Takeaway
No study was found that tests clear-sky-index *targets* with TimesFM or Chronos. Classic solar work favours clear-sky normalisation for linear and NN models. For TSFMs with per-instance RevIN, capacity normalisation is mathematically irrelevant to the backbone. What matters is (i) giving the model the deterministic diurnal envelope, most safely as a past-future covariate such as clear-sky power or POA, and (ii) avoiding transforms that break RevIN statistics, like ratios that blow up at dawn and dusk.

### Cited Findings
- Clear-sky normalisation removes the diurnal and seasonal trend and yields a near-stationary series for AR/linear models. Its drawback is that clear-sky models need atmospheric parameters that may be unknown at the site. — [Lauret et al., Solar 2022 (HAL)](https://hal.univ-reunion.fr/hal-03964785v1)
- Lauret et al. found clear-sky-index forecasts better than clearness-index forecasts across persistence, AR and NN. A nonlinear NN on the clearness index can still beat linear models on the clear-sky index. — [HAL 03964785](https://hal.univ-reunion.fr/hal-03964785/document)
- [CODE] TimesFM-3 applies causal per-variate running mean/std normalisation (RevIN) and optional linear detrending, then denormalises outputs. So any per-station constant scaling, such as dividing by capacity, cancels inside the backbone. It still changes the loss weighting across stations during fine-tuning and the shared-coefficient XReg fit. — `timesfm3/torch/model.py` `_preprocess`, `decode`; `timesfm/utils/xreg_lib.py`
- PARA-PV (frozen Chronos prior) argues PV needs explicit physics: capacity caps, night zeros and regime responses. It adds gated physics correction rather than transforming the input. — [arXiv 2607.08079](https://arxiv.org/abs/2607.08079)
- Cold-start PV: a synthetic production history built from metadata plus weather (i.e. a physics-model power curve), given as context, enabled covariate-aware TSFMs to beat baselines by about 1.7–2x. Accuracy depended more on having plausible temporal context than on which generator was used. — [arXiv 2606.07457](https://arxiv.org/abs/2606.07457)
- SPIRIT (zero-shot irradiance with foundation models) uses clear-sky GHI as a baseline to estimate cloud impact (details not verified). — [arXiv 2502.10307](https://arxiv.org/pdf/2502.10307)

### Inferences
- Clear-sky power (or a PV-model power from NWP-GHI, e.g. the user's h-series clear-sky ratio and envelope work) belongs in `past_future_covariates`. It is exactly known in the horizon, shares the target's shape and units, and gives variate attention a clean template. This is more TSFM-friendly than a clear-sky-index target, which:
  - is undefined or noisy near sunrise and sunset;
  - breaks the zero-night structure that RevIN statistics, linear detrending and the TSFM's pretraining distribution assume;
  - requires multiplying back, which amplifies errors at high elevation angles.
- Because RevIN removes level per variate, a covariate in **target units** (expected power) is more informative than raw GHI only if combined with an absolute-level corrector (XReg-style ridge or LightGBM on top). Inside TimesFM-3 both are reduced to shapes.
- Capacity normalisation should be kept for training-loss balance across 28 stations and for metrics. The memory notes "cap = station_meta cap_mw", which is consistent. It will not change the zero-shot backbone output.

### Gaps
- No controlled TSFM experiment found on any of these: clear-sky-index target vs raw-power target, clear-sky power as covariate vs not, or night-masking/zeroing effects on RevIN statistics. These are worth running in-house, since there is no external evidence.
