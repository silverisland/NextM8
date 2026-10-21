# SFT and PEFT of Time-Series Foundation Models (TSFMs), small-data focus — research notes for TimesFM-3 PV fine-tuning

Scope: fine-tuning strategies for TSFMs: full FT, partial FT, linear probing, LoRA and other PEFT, prompt tuning, and multi-entity adapters. Our baseline for comparison is **LoRA r8 on attention q/k/v/o + unfrozen output head, capacity-normalized pinball loss, lr 1e-4** on TimesFM-3 (~200M), 28 PV stations, 1–2 years of 15-min data each, ctx 672 / horizon 192, with NWP GHI and temperature as covariates.

Evidence labels used below:
- **[TSFM]**: measured on a time-series foundation model.
- **[LLM/vision-only]**: evidence from language or vision models only.
- **[pre-2024]**: older evidence.
- **[snippet]**: comes from a search-result summary; I did not read the full text.

Searched and read 2026-10-15.

---

## Q1. Full fine-tuning vs partial (last-N / head-only / linear probing / LP-FT) vs LoRA for TSFMs: numbers, data sizes, overfitting and forgetting

### Takeaway
Full FT of a ~100–200M TSFM is the riskiest option at small data. Several 2024–2026 TSFM studies find it worse than LoRA. Some find it worse than zero-shot, and one sets out catastrophic forgetting as a possible explanation. Full FT wins only with lots of data per entity and a very small lr (1e-6). LoRA or linear probing (input/output blocks only) is the safer default, and "is fine-tuning better than zero-shot at all?" depends more on the dataset than on the method.

### Cited Findings
- **TimesFM (1.0 base, 2024), per-dataset FT on Monash [TSFM]:**
  - Two variants were tried: full FT, and "LP", which here means freezing all transformer layers and tuning only the **input and output residual (MLP) blocks**. Settings: batch 16, ≤10k iterations, original decoder-only loss.
  - FT-TimesFM (Full) improved geometric-mean scaled MAE by **~4% over zero-shot base**. TimesFM-ICF, a zero-shot in-context model with no gradient updates, was a further **~3% better than full FT**.
  - The authors suggest that "in many of the smaller datasets in Monash, fine-tuning the weights of a foundation model can actually lead to catastrophic forgetting". ICF beat FT on small datasets (tourism yearly, bitcoin, us births). FT won on larger ones such as Australian electricity demand.
  - LP across all datasets took 115 min, against 4 min of ICF inference.
  - Source: [In-Context Fine-Tuning for TSFMs, arXiv 2410.24087 (§6.3, App. A.5)](https://arxiv.org/abs/2410.24087)
- **Chronos-2 and TTM-R3 across dataset sizes (2026) [TSFM]** ([arXiv 2607.23146](https://arxiv.org/html/2607.23146v1)):
  - Setup: one fixed hyperparameter set, 4,000 steps. FFT lr 1e-4 for Chronos-2. LoRA r8/α16 at lr 1e-5.
  - Small datasets, Chronos-2 (MWQLₙ / MAPEₙ): zero-shot **0.423 / 0.544**, FFT **0.577 / 0.732**, LoRA-full **0.477 / 0.647**, LoRA-20%-data 0.471 / 0.612. Every fine-tuning variant was worse than zero-shot, and FFT was worst.
  - Medium and large datasets: LoRA-full was best for Chronos-2, but only by about 2–3% over zero-shot (0.357 vs 0.364 medium; 0.301 vs 0.310 large). FFT was worse than zero-shot in every size bin for Chronos-2.
  - For the small TTM-R3 (1–36M), FFT ≈ zero-shot and LoRA was slightly worse.
  - Zero-shot Chronos-2 won 5 of 15 MWQL configurations, and zero-shot Chronos-2 had the best MWQL on the Energy domain (ETT, Electricity, Solar) at all horizons.
  - Caveat: the abstract's claim that fine-tuning "consistently improves" accuracy is not supported by its own tables. Chronos-2's FFT lr of 1e-4 is ~100× the Chronos-2 default of 1e-6 (see Q4), which may exaggerate FFT's overfitting.
- **Chronos-2 for grid load forecasting (2026) [TSFM]** ([arXiv 2609.06656](https://arxiv.org/html/2609.06656v1)):
  - Setup: daily data, ISO-NE 8 zones × 3,653 training days, ENTSO-E 6 countries × 1,461 days. Weather and calendar covariates. AdamW, **full FT lr 1e-6, LoRA lr 1e-5**, 20 epochs, best validation checkpoint, 5 seeds.
  - With covariates, MAPE % at H = 30 / 60 / 90 days:

    | Dataset | Zero-shot | LoRA | Full FT |
    |---|---|---|---|
    | ISO-NE | 6.71 / 10.61 / 15.78 | 4.33 / 5.51 / 5.81 | **4.06 / 5.19 / 5.36** |
    | ENTSO-E | 5.63 / 6.57 / 7.67 | 4.54 / 4.91 / 5.32 | **4.31 / 4.66 / 4.86** |

  - Without covariates, LoRA and full FT were within noise of each other (ISO-NE 10.02 vs 10.07 at H30).
  - Takeaway from this study: full FT beat LoRA consistently, especially when covariates were present. This held with long histories per entity and a 1e-6 lr. LoRA rank and target modules were not reported.
- **Chronos (T5) PEFT on ICU vitals (2024) [TSFM]** ([Beyond LoRA, arXiv 2409.11302](https://arxiv.org/html/2409.11302v1)):
  - Data: about 3.2k training windows (4,020 samples, 1,442 patients, 8:1:1 split).
  - **Full FT made heart-rate forecasts worse than zero-shot at every model size**:

    | Chronos size | Zero-shot HR MSE | Full FT HR MSE | LoRA HR MSE |
    |---|---|---|---|
    | Tiny | 7.37 | 8.80 | 7.22 |
    | Small | 7.19 | 10.04 | — |
    | Base | 7.33 | 10.15 | — |
    | Large | 7.21 | 9.56 | — |

  - On mean BP, full FT and PEFT both beat zero-shot (~25.6 → ~20 MSE).
  - The authors conclude that PEFT beats both zero-shot and full FT "in nearly all configurations".
- **Lag-Llama, MOIRAI and Chronos LoRA on sepsis vitals (2024) [TSFM, secondary]:** a vendor blog summarizing [arXiv 2405.10216](https://arxiv.org/pdf/2405.10216) reports that LoRA beat full FT in 4 of 6 model-metric combinations, attributed to overfitting on small clinical data. Source: [TSFM.ai blog](https://tsfm.ai/blog/lora-peft-time-series-foundation-models) [snippet; not peer reviewed].
- **Process-model forecasting (2025) [TSFM]:** with Chronos-Bolt, MOIRAI-1.1 and Chronos-2, LoRA and full FT both beat zero-shot, but gains were dataset-dependent, sometimes small, and could vanish on smaller or more complex datasets. Sources: [arXiv 2512.07624](https://arxiv.org/html/2512.07624v1), as summarized in [Post-training survey, arXiv 2607.20002](https://arxiv.org/html/2607.20002v3).
- **Head-only / linear probing vs LoRA, MOMENT (TRACE, 2025) [TSFM]** ([arXiv 2503.16991](https://arxiv.org/html/2503.16991)):
  - Linear probing (head only) ≈ LoRA. MSE, LP vs LoRA: ETTh1-192 **0.417 vs 0.420**; Weather-192 0.204 vs 0.207; Exchange-96 0.122 vs 0.119.
  - Long-horizon flattening heads are large (9.4M params at H=192, 35M at H=720) and "lead to overfitting". Shrinking the head by up to 98% matched or slightly improved results.
- **Survey position (2026) [TSFM]:** PEFT suits small-to-moderate target data. Full FT needs moderate-to-large volumes and is "vulnerable to overfitting" and forgetting with scarce or noisy data. The survey gives no controlled experiment for this. Wasserstein-loss FT improves point accuracy but may hurt weighted quantile loss. Source: [arXiv 2607.20002](https://arxiv.org/html/2607.20002v3)
- **LP-FT (linear probe first, then full FT) [LLM/vision-only, pre-2024]:**
  - Fine-tuning can distort pretrained features. Probing first and then fine-tuning improves out-of-distribution accuracy (~10% OOD in Snorkel's image/NLP experiments). Sources: [Snorkel summary](https://snorkel.ai/blog/boost-foundation-model-results-with-linear-probing-fine-tuning/); Kumar et al. 2022, [arXiv 2202.10054](https://arxiv.org/abs/2202.10054) (known literature, not re-fetched here).
  - I found **no TSFM study of LP-FT**.
- **"LoRA learns less and forgets less" [LLM-only, 2024]:** in LLMs, LoRA underperforms full FT on the target domain but forgets less of the source domain. Source: Biderman et al., [arXiv 2405.09673](https://arxiv.org/abs/2405.09673) (known literature, not re-fetched).

### Inferences
- Our data is about 28 stations × 35k–70k steps at 15-min resolution. That is far more per entity than the "small" regimes above (ICU, Monash yearly), but there are fewer than 100 series and the data is extremely periodic. The Chronos-2 grid-load study is our closest analogue: long single-entity histories, known-future weather covariates, day-scale horizon. There, **full FT at lr 1e-6 beat LoRA by ~5–10% relative MAPE with covariates**. Two conditions made this work: a tiny lr and best-validation checkpointing over 5 seeds.
- A careful full-FT or last-N-layers FT run at lr ≤1e-5 (not our 1e-4), with early stopping, is therefore a legitimate candidate to beat LoRA-r8. The evidence that full FT overfits mostly comes from runs at lr 1e-4 or from tiny datasets.
- The main reason full FT helped "especially with covariates" is plausibly that covariate interaction is a new capability rather than a re-weighting of existing ones. This matters for TimesFM-3's covariate path. If TimesFM-3's covariate inputs feed through embedding or tokenizer layers that LoRA on q/k/v/o does not touch, those layers are prime candidates to unfreeze. **I have not verified TimesFM-3's internal covariate pathway; check the 3.0.x code.**
- Linear probing (head + input tokenizer only) is a cheap, strong ablation that we have not run. TimesFM's own paper used exactly this variant ("LP" = input + output residual blocks). It should be run as a control against LoRA-r8 + head, so we can see what the attention LoRA actually adds.
- Because our output head is unfrozen, the head may already carry most of the fine-tuning gain, and also most of the overfitting risk. TRACE shows that large long-horizon heads overfit. Ablate head frozen vs unfrozen.

### Gaps
- I found no published fine-tuning numbers for TimesFM 2.5 or TimesFM 3 from Google itself, and no TSFM paper doing a controlled sweep of last-N-layer unfreezing.
- I found no TSFM LP-FT study.
- I found no directly measured forgetting metric, such as zero-shot performance on held-out domains after FT, for any TSFM.
- I did not find Moirai, Moirai-MoE, Timer, Sundial or Toto FT-vs-LoRA comparisons with numbers in this session.

---

## Q2. Other PEFT methods (DoRA, IA3, BitFit, LayerNorm, adapters, prefix/prompt, VeRA, AdaLoRA, FourierFT): TSFM evidence, which layers, rank sensitivity

### Takeaway
TSFM-specific evidence is thin and mostly comes from one Chronos study on ICU data. There, ultra-light methods (BitFit, LayerNorm-only, VeRA, FourierFT with 0.2k–50k params) matched or beat LoRA, and the differences were small. The strongest 2026 energy evidence (FETS) says rank, lr, steps and target modules barely change the outcome; the dataset decides. **Swapping LoRA for DoRA, IA3 or VeRA is unlikely to be a big lever.** The choice of what to target (all-linear vs attention-only; tokenizer, covariate path, head) and the regularization regime matter more.

### Cited Findings
- **Chronos-T5, eICU vitals (2024) [TSFM]** ([arXiv 2409.11302](https://arxiv.org/html/2409.11302v1)):
  - Chronos-Tiny mean-BP MSE (×1e-4) by method, with trainable params:

    | Method | MSE | Trainable params |
    |---|---|---|
    | Zero-shot | 25.60 | — |
    | Full FT | 19.90 | 8.3M |
    | LoRA | 19.79 | 49k |
    | BitFit | 20.68 | 0.2k |
    | LayerNorm | 20.25 | 5k |
    | VeRA | 20.16 | 13k |
    | **FourierFT** | **19.51** | **2.4k** |

  - On Chronos-Base, BitFit gave the best MSE (19.77 with ~0.7k params).
  - On Chronos-Large (708M), BitFit, LN and VeRA gave no gain over zero-shot (25.09 vs 25.54). The table shows identical values for all three, which may be a reporting error.
  - Setup: VeRA r16; FourierFT n=50, α=300; targets were attention q/k/v/o; lr swept from 1e-2 to 1e-5.
- **Rank, lr, steps and modules barely matter: FETS benchmark, Chronos-2, energy (2026) [TSFM]** ([arXiv 2604.22328, App. E, Table E.3](https://arxiv.org/abs/2604.22328)):
  - Default config: LoRA r8, α16, targets {q,k,v,o, output_patch_embedding}, lr 1e-6, 1,000 steps, batch 8.
  - They varied one factor at a time: lr 1e-8 to 1e-4, steps 500 or 1,500, rank 4 or 16, modules {q} or {q,k,v,o}. The median NRMSE change versus zero-shot stayed between −18% and −28% across all variants.
  - Within each dataset the result is "essentially invariant to the configuration": "whether fine-tuning helps is governed by the dataset, not the configuration."
- **Per-module LoRA selection (TRACE, MOMENT) [TSFM]** ([arXiv 2503.16991](https://arxiv.org/html/2503.16991)):
  - Gated importance-based LoRA module selection beat plain LoRA and AdaLoRA slightly (ETTh1-192 MSE 0.412 vs LoRA 0.420). The paper shows module-importance maps across 12 layers but does not describe the pattern in words.
  - AdaLoRA's gradient-based importance correlated *negatively* (−22.1%) with Shapley importance, against 63.3% for TRACE's gate.
- **AdaLoRA and LoRA on MOIRAI q/k/v with a trainable head (MSFT, 2025) [TSFM]:** used as baselines within a multi-scale adapter method. Source: [arXiv 2506.14087](https://arxiv.org/pdf/2506.14087) [snippet].
- **DoRA [TSFM, limited]:**
  - PR #104 in google-research/timesfm added LoRA *and DoRA* to a fine-tuning pipeline (full FT, linear probing, LoRA, DoRA) for the TimesFM 1.0/2.0 era, merged by rajatsen91 ([PR #104](https://github.com/google-research/timesfm/pull/104)). I found no reported DoRA-vs-LoRA numbers.
  - LoRA and DoRA adapters on Moirai have been trained with a decision-focused (dispatch-cost) loss, giving −9.45% daily cost vs prediction-focused FT ([survey 2607.20002](https://arxiv.org/html/2607.20002v3); [ScienceDirect paper](https://www.sciencedirect.com/science/article/pii/S2666546825000655)). This is a loss-function result, not a DoRA-vs-LoRA comparison.
- **Which layers, from the LLM literature [LLM-only]:**
  - The original LoRA paper found adapting attention W_q and W_v sufficient at fixed budget: Hu et al. 2021, [arXiv 2106.09685](https://arxiv.org/abs/2106.09685) [pre-2024].
  - QLoRA found that applying LoRA to *all linear layers*, including MLP, was needed to match full FT: Dettmers et al. 2023, [arXiv 2305.14314](https://arxiv.org/abs/2305.14314) [pre-2024].
  - DoRA ([arXiv 2402.09353](https://arxiv.org/abs/2402.09353)), IA3 ([arXiv 2205.05638](https://arxiv.org/abs/2205.05638), pre-2024), VeRA ([arXiv 2310.11454](https://arxiv.org/abs/2310.11454), pre-2024) and AdaLoRA ([arXiv 2303.10512](https://arxiv.org/abs/2303.10512), pre-2024) report gains over LoRA on NLP benchmarks, typically ≤1–2 points. These are known literature, not re-fetched here.
- **Google's own TimesFM LoRA example targets `all-linear`, not attention-only [TSFM, official]:** the default is r=4, α=8, dropout 0.05, about 1.4M of ~232M params (0.6%). Source: [google-research/timesfm examples/finetuning README](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/finetuning)
- **Adapters (Houlsby or parallel) on TSFMs:** I found no head-to-head TSFM evidence against LoRA. Covariate adapters exist: ChronosX, UniCA, CoRA-Cov, and WindPrompt (wind power). Grouped adapters exist too (MixFT). Source: [survey 2607.20002](https://arxiv.org/html/2607.20002v3)

### Inferences
- Our LoRA covers q/k/v/o only, while Google's official example uses all-linear (attention + MLP). In LLMs, MLP layers matter for absorbing new "knowledge". The PV diurnal shape under NWP-driven modulation is plausibly a "new pattern" in that sense. **Switching to all-linear at r4–r8 is the cheapest change with a plausible upside.** FETS suggests the effect will be small either way.
- DoRA, VeRA, IA3, FourierFT and AdaLoRA are low-value experiments for us. Their TSFM evidence comes from one ICU study whose effect sizes are within noise of LoRA, and FETS shows insensitivity to the configuration. They are only worth running if we need to store per-station adapters cheaply (FourierFT or VeRA at ~kB per station).
- BitFit and LayerNorm-only tuning are worth one run each as a *strong-regularization control*. They train a few thousand params and can hardly overfit. If BitFit or LN ≈ LoRA-r8 on our metric, the gain is coming from re-calibration (scale, bias, quantile spread), not from new pattern learning.
- Rank sensitivity: no TSFM evidence that rank matters between 4 and 16. FETS rank 4 vs 16 gave the same median.

### Gaps
- I found no TSFM paper that compares DoRA vs LoRA, IA3, or Houlsby/parallel adapters with numbers.
- I found no study of LoRA on TimesFM's input residual tokenizer or on the covariate projection.
- I found no rank-sweep study on TimesFM.

---

## Q3. Does "prompt learning" (soft prompts, prefix patches, reprogramming) make sense for a patch-token TSFM? Does it beat LoRA?

### Takeaway
Yes, it is meaningful: learnable prefix tokens prepended to patch embeddings work on frozen TSFMs, including TimesFM-2.5. The best 2026 evidence (PaCTS) shows instance-adaptive soft prompts slightly **beating both LoRA and full FT on GIFT-Eval**, and being the *only* method that beats zero-shot on an unseen-task benchmark. The gains are small (~1.5% MASE, ~3% CRPS). LLM-reprogramming approaches (Time-LLM, TEMPO, Time-Prompt) are a different paradigm with weak evidence for TSFM users.

### Cited Findings
- **PaCTS, instance-adaptive soft prompts as context (2026) [TSFM]** ([arXiv 2609.34786](https://arxiv.org/html/2609.34786)):
  - Method: frozen Chronos-2, PatchTST-FM, or **TimesFM-2.5**. Ten latent prompt tokens are prepended to the patch embeddings (position ID 0). The tokens combine a shared learned part with an instance-adaptive part generated from 10 scale-invariant statistics (mean, trend, autocorrelation…) through a rank-4 basis, plus cross-attention over up to 16 recent segments. One module is trained jointly across datasets.
  - Trainable params: 153k for Chronos-2 and **245k (0.106%) for TimesFM-2.5**.
  - Chronos-2 on GIFT-Eval (97 configurations), MASE / CRPS:

    | Method (context) | MASE | CRPS |
    |---|---|---|
    | Zero-shot (4096) | 0.708 | 0.496 |
    | Zero-shot (8192) | 0.704 | 0.491 |
    | Full FT (4096) | 0.703 | 0.487 |
    | LoRA (4096) | 0.703 | 0.493 |
    | **PaCTS (4096)** | **0.697** | **0.480** |

  - On the unseen TIME benchmark (98 tasks), PaCTS was the only method beating zero-shot on both metrics. **All five weight-space baselines (FT, LoRA, etc.) were worse than zero-shot.**
  - LoRA and full FT gain more on shorter histories.
- **Generalized Prompt Tuning, MOMENT/GPT4TS on healthcare (2024) [TSFM]** ([arXiv 2411.12824](https://arxiv.org/html/2411.12824)):
  - MOMENT MSE: LoRA 3.109 vs prompt tuning 2.918. One version of the paper also reports **linear probing at 2.622, better than both**.
  - GPT4TS: LoRA 3.247 ± 0.337 vs prompt 3.105 ± 0.429, which is within noise.
  - Prompt tuning was reported to train slower than LoRA in their implementation.
- **Time-Prompt, LLM-based TS (2025) [LLM-backbone, not TSFM]:** soft prompt + LoRA ablation on ETTh1 gave full model 0.430, without soft prompt 0.434, without LoRA 0.435. The effects are tiny and complementary. Source: [arXiv 2506.17631](https://arxiv.org/pdf/2506.17631)
- **Time-LLM-style reprogramming applied to distributed PV forecasting** exists ([arXiv 2503.06216](https://arxiv.org/pdf/2503.06216)) [snippet; I did not read its numbers]. It uses a frozen LLM, not a TSFM.
- **The survey lists "generalized prompt tuning" and WindPrompt under covariate/multivariate adapters** but reports no prompt-vs-LoRA numbers. Source: [arXiv 2607.20002](https://arxiv.org/html/2607.20002v3)
- **TimesFM-ICF uses a learnt separator token,** a 1280-dimensional embedding, during continued pretraining to delimit in-context examples. This is a soft-token mechanism inside TimesFM itself. Source: [arXiv 2410.24087, App. A](https://arxiv.org/abs/2410.24087)

### Inferences
- For our 28 stations, a **per-station or station-conditioned learnable prefix** (k ≈ 4–16 prefix patch embeddings, backbone frozen) is a sensible, cheap experiment. It is a natural "station embedding" for a decoder-only patch model, and it carries minimal forgetting risk. PaCTS's instance-adaptive variant can generate a prompt from context statistics, so it extends to unseen stations without training.
- A practical caveat: in a decoder-only model with causal attention and patch-wise output, prefix tokens shift positions. They must be placed so that the output head reads only real patches. TimesFM-3's positional scheme needs checking.
- Do not expect prompt tuning to beat LoRA by much on in-distribution stations; the measured differences are ≤1–2%. Its main argument is robustness and generalization to unseen stations, plus composability with LoRA.
- Time-LLM-type reprogramming (an LLM backbone) is not worth trying: it is a different backbone class, and there is no evidence it beats native TSFM fine-tuning.

### Gaps
- PaCTS's per-model breakdown for TimesFM-2.5 and any energy-specific results were not visible in the extract.
- I found no study of per-entity soft prompts as station embeddings on TSFMs.
- I found no prompt-tuning results on TimesFM-3 with covariates.

---

## Q4. Official fine-tuning guidance: TimesFM (Google), Chronos/Chronos-2 (Amazon), Moirai (Salesforce)

### Takeaway
Google's current official TimesFM recipe is LoRA via HF PEFT on **all linear layers, r4/α8, lr 1e-4, 10 epochs, random windows, raw (un-normalized) inputs**, demonstrated on TimesFM-2.5. Amazon treats Chronos-2 fine-tuning as experimental. It recommends at least ~100 series and a history over 3× the horizon, uses lr **1e-6** by default, and AutoGluon uses LoRA by default with a zero-shot + FT ensemble preset. I could not retrieve Moirai's official recipe in this session.

### Cited Findings
- **TimesFM official LoRA example (added 9 Apr 2026)** ([README](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/finetuning); [repo README](https://github.com/google-research/timesfm)):
  - Files: `timesfm-forecasting/examples/finetuning/finetune_lora.py` + README. Model `google/timesfm-2.5-200m-transformers`, HF Transformers + PEFT.
  - Defaults: `target_modules="all-linear"`, r=4, α=8, dropout 0.05, lr 1e-4, 10 epochs, batch 32, 5,000 random (context, horizon) windows. The suggested custom run is **20 epochs, batch 64, lr 5e-5, r8, α16**.
  - Guidance: "Do not normalise your data externally — feed raw values and let the model handle it." Use random windows rather than one fixed window per series; the README credits Chronos-2 for this.
  - The loss is computed internally when `future_values` are passed.
  - The README does **not** mention covariates or TimesFM-3.
- **Older TimesFM fine-tuning:**
  - `notebooks/finetuning_torch.ipynb`, and PR #104, which offered full FT, linear probing, LoRA and DoRA ([PR #104](https://github.com/google-research/timesfm/pull/104)).
  - Issue #242: the PyPI package once lacked the fine-tuning module, so install from source ([issue #242](https://github.com/google-research/timesfm/issues/242)).
  - Issue #257: users hit errors combining fine-tuning with covariates (XReg) ([issue #257](https://github.com/google-research/timesfm/issues/257)) [snippet; may be outdated].
  - The repo notes covariate support was restored for 2.5 via XReg, and TimesFM 3.0 adds past-only and past+future covariates ([repo](https://github.com/google-research/timesfm)) [snippet].
- **TimesFM original FT protocol (2024):** batch 16, ≤10k iterations, pretraining loss, full or LP (input + output residual blocks). Source: [arXiv 2410.24087 App. A.5](https://arxiv.org/abs/2410.24087)
- **Chronos-2 defaults from Amazon's technical documentation,** as reproduced by FETS: 1,000 steps, **lr 1e-6**, batch 8, 75% window overlap, LoRA r8/α16/dropout 0, targets {q,k,v,o,output_patch_embedding}. Amazon "treats fine-tuning as an experimental capability recommended only for collections of roughly a hundred or more time series". Source: [FETS, arXiv 2604.22328, Table A.3 and §6.1](https://arxiv.org/abs/2604.22328)
- **AutoGluon Chronos-2:**
  - `fine_tune=True` uses LoRA by default; `fine_tune_mode="full"` is optional. An example shows lr 1e-4, 2000 steps, batch 32.
  - Recommendation: >100 series with median history >3× prediction_length. Limited data can cause overfitting.
  - The `chronos2_ensemble` preset combines zero-shot and fine-tuned models as a more robust option.
  - In the docs' load example, FT improved test MASE (−0.688 vs −0.696) but *worsened* validation (−0.832 vs −0.817).
  - Source: [AutoGluon docs](https://auto.gluon.ai/dev/tutorials/timeseries/forecasting-chronos.html)
  - Amazon's own repo defaults to full FT, while AutoGluon defaults to LoRA ([Darts issue #2964](https://github.com/unit8co/darts/issues/2964)) [snippet].
- **Chronos-2 grid-load study's working recipe:** full FT at lr 1e-6, LoRA at lr 1e-5, AdamW, ReduceLROnPlateau (halve after 3 epochs without validation improvement), 20 epochs, best checkpoint, 5 seeds. Source: [arXiv 2609.06656](https://arxiv.org/html/2609.06656v1)
- **MixFT recipe on Chronos-Bolt and Moirai-1.1-R:** LoRA r2, α16, dropout 0.1, OLoRA init, AdamW lr 5e-5, batch 256, **MixUp + experience replay with pretraining data**. Source: [arXiv 2603.02840](https://arxiv.org/html/2603.02840v1)

### Inferences
- Our setup (lr 1e-4, r8, attention-only, unfrozen head) is roughly Google's default lr with a narrower target set. Amazon's evidence points to much lower lr (1e-5 for LoRA, 1e-6 for full FT) being safer. **An lr sweep {1e-5, 3e-5, 1e-4} with EMA/early stopping is cheaper and better-evidenced than switching PEFT method.** This agrees with the project's 2026-10-13 review (lr 1e-4 vs smaller model, EMA, 5 seeds).
- Google's "don't normalize externally" advice conflicts with our capacity-normalized loss only if we also normalize the *inputs*. Normalizing the loss by capacity is fine. Normalizing inputs could interact with TimesFM's internal instance normalization (RevIN-like). This needs checking in our pipeline.
- Experience replay with generic pretraining-like data (from MixFT) is a cheap anti-forgetting regularizer we are not using.
- With only 28 series, we are below Amazon's ~100-series threshold. However, each series is very long (≫3× horizon), so in windows the dataset is large. The risk is less "too few samples" and more "too little diversity", that is, overfitting to station-specific regimes.

### Gaps
- I did not retrieve Moirai (uni2ts) official fine-tuning config values (lr, freezing, steps); they need a separate lookup of the uni2ts `cli/conf/finetune` YAMLs.
- I did not verify whether the TimesFM LoRA example works with TimesFM-3.0.x or with the covariate inputs.
- I did not read Google's TimesFM-3 model card for FT recommendations.

---

## Q5. Multi-station: per-station vs pooled fine-tuning, station embeddings or per-station LoRA (mixture of LoRAs), unseen-station generalization

### Takeaway
Pooled or grouped adaptation beats per-entity adaptation in the TSFM evidence. Per-series FT often hurts (FETS: Chronos-2 fine-tuned on one target series improved only 16 of 54 series). Per-dataset LoRAs can be worse than no fine-tuning. **A small number of cluster-level LoRAs routed by a mixture model (MixFT, K=2) beat both one shared LoRA and per-dataset LoRAs, including on unseen series.** Weight-space FT tends to hurt on unseen tasks, while soft prompts generalize better.

### Cited Findings
- **MixFT, Chronos-Bolt-small and Moirai-1.1-R-small (2026) [TSFM]** ([arXiv 2603.02840](https://arxiv.org/html/2603.02840v1)):
  - Method: a Bayesian GMM in TSFM embedding space splits the fine-tuning data into sub-domains, with one LoRA per sub-domain. At inference each unseen context is routed (hard argmax) to the most likely LoRA. Routing entropy was ≈0.01.
  - Evaluation was on *unseen* datasets.
  - Average rank, Chronos-Bolt: MixFT 2.0, Poly 2.8, **Base (zero-shot) 2.9, Shared single LoRA 3.0**, per-dataset μ-Datasets 6.3.
  - Average rank, Moirai: MixFT 2.1, Shared 2.9, Base 4.3, per-dataset 6.3.
  - "Some per-dataset methods perform worse than not fine-tuning at all". For Chronos-Bolt, every non-MixFT method lost to Base by ≥0.01 MASE on at least three datasets.
  - Design ablations: K=2 was best on validation; hard routing (rank 1.6) beat soft weighting (4.4).
- **Per-series FT in energy (FETS, 2026) [TSFM]** ([arXiv 2604.22328 §6.1, App. E](https://arxiv.org/abs/2604.22328)):
  - Chronos-2 was fine-tuned on one target series per dataset (default LoRA recipe) and improved **16 of 54 series**.
  - The aggregate median regressed: overall NRMSE 0.599 FT vs 0.472 zero-shot with covariates. FT nonetheless had the highest single-experiment win rate (18%).
  - Gains concentrated in the hardest, least forecastable series, with Spearman r = 0.39 to zero-shot difficulty. There was no correlation with dataset length or covariate count (|r| < 0.1).
- **Shared adapter, not per-entity (Chronos-2 grid-load) [TSFM]:** all zones were modeled jointly. The authors propose, but do not test, per-zone LoRA on a frozen backbone. Source: [arXiv 2609.06656](https://arxiv.org/html/2609.06656v1)
- **Unseen-task generalization after FT (PaCTS) [TSFM]:** on TIME, a benchmark with no overlap with the training data, all five weight-space baselines (full FT, LoRA, …) trained on GIFT-Eval were *worse than zero-shot* Chronos-2. The soft-prompt module was the only method above zero-shot. Source: [arXiv 2609.34786](https://arxiv.org/html/2609.34786)
- **Cold-start PV:** covariate-aware zero-shot TSFMs, including TimesFM 2.5, reached ~1.7–2× lower error than Prophet on unseen PV sites without fine-tuning. Source: [arXiv 2606.07457](https://arxiv.org/html/2606.07457v1) [snippet]
- **Federated or multi-client full-FT variants** exist (FedChronos and others). Source: [survey 2607.20002](https://arxiv.org/html/2607.20002v3). One federated Chronos-family study on 30 short price series found **unregularized LoRA worse than zero-shot**, with validation loss bottoming out early. Source: [FedChronos, arXiv 2608.01290](https://arxiv.org/pdf/2608.01290) [snippet].

### Inferences
- **Recommended structure for 28 stations:** one pooled LoRA (as now) as the baseline, then compare:
  - (a) K=2–4 cluster-level LoRAs, routed by an embedding GMM or by simple station metadata such as climate zone, tracker vs fixed tilt, or curtailment regime.
  - (b) a pooled LoRA plus a per-station soft prefix (station embedding).
- Per-station LoRAs trained independently (28 × 1–2 years) are the configuration the evidence warns against. Use a pooled backbone adapter first.
- **Unseen-station protocol:** the evidence (MixFT, PaCTS) says weight-space FT can lose to zero-shot on unseen entities. Always report seen vs unseen-station splits; the project already tracks "seen" scores. Keep a zero-shot fallback or ensemble, like AutoGluon's `chronos2_ensemble`.

### Gaps
- I found no TSFM paper testing explicit learned station-ID embeddings vs per-entity LoRA.
- I found no study of a "mixture of LoRAs" across PV sites specifically.
- I found no quantitative leave-station-out evaluation of fine-tuned TimesFM.

---

## Q6. Typical fine-tuning gains over zero-shot on energy, solar and load data, and when fine-tuning hurts

### Takeaway
Gains range from large (load with covariates and long history: ISO-NE MAPE 6.7→4.1% at H30, 15.8→5.4% at H90) to small (2–4% on GIFT-Eval or Monash) to negative. Fine-tuning hurts on easy, well-structured series where zero-shot is already near-optimal. That case explicitly includes **PV Hong Kong (rooftop PV, 15-min, day-ahead)**, where Chronos-2 LoRA FT made NRMSE ~3× worse. On that dataset the degradation did not change with lr, which hints at a pipeline effect.

### Cited Findings
- **Load, daily, with covariates (Chronos-2) [TSFM]:** the large gains above, MAPE −40% to −66% relative. ENTSO-E 5.63→4.31% at H30. Without covariates the gains were small: ENTSO-E 5.78→5.16 at H30, and 7.53→7.36 at H90. Source: [arXiv 2609.06656](https://arxiv.org/html/2609.06656v1)
- **FETS energy benchmark, Chronos-2 LoRA, per-series (2026) [TSFM]** ([arXiv 2604.22328, Tables E.1/E.3](https://arxiv.org/abs/2604.22328)):
  - NRMSE, zero-shot → fine-tuned (higher is worse):

    | Series | Zero-shot | Fine-tuned | Change |
    |---|---|---|---|
    | DE load | 0.027 | 0.160 | worse (interval coverage collapsed 0.94→0.27) |
    | **PV Hong Kong** (15-min rooftop, ctx 2000 / hor 192) | **0.198** | **0.688** | worse |
    | **DK Solar** (country level) | 0.648 | 0.555 | +14% better |
    | NetzeBW feeder | 0.535 | 0.483 | +10% better |

  - For PV Hong Kong the degradation was −304% at the default lr 1e-6 and stayed −303% to −305% across lr 1e-8 to 1e-5. Ranks 4 and 16 and modules {q} or {qkvo} gave −307% to −320%.
  - Train and validation loss kept decreasing while test accuracy got worse.
  - The authors offer catastrophic forgetting or overfitting to the fine-tuning window as possible explanations.
- **Energy domain in GIFT-type data (2026) [TSFM]:** zero-shot Chronos-2 had the best MWQLₙ on Energy (ETT, Electricity, Solar) at all horizons (0.520 / 0.255 / 0.225). LoRA-few-shot was best on MAPEₙ. Source: [arXiv 2607.23146](https://arxiv.org/html/2607.23146v1)
- **ERCOT power-system benchmark (2026) [TSFM]:** zero-shot TimesFM solar nMAE 9.25%, nRMSE 15.1%. The authors call zero-shot accuracy too low for grid operations. TimesFM wind MAE fell from 8.51 to 3.84 as the training share went from 20% to 100%; I could not retrieve the solar FT numbers. Source: [arXiv 2604.22077](https://arxiv.org/html/2604.22077v1) [snippet]
- **Building energy (TimesFM 1.0, Chronos):** fine-tuned TSFMs beat zero-shot TSFMs and deep baselines. Source: [arXiv 2506.00630](https://arxiv.org/pdf/2506.00630) [snippet]
- **Monash with TimesFM 1.0:** full FT gave ~4% over zero-shot on aggregate, losing on small datasets. The Monash benchmark includes solar weekly. Source: [arXiv 2410.24087](https://arxiv.org/abs/2410.24087)
- **AutoGluon load example:** a marginal test gain with a validation loss (see Q4). Source: [AutoGluon docs](https://auto.gluon.ai/dev/tutorials/timeseries/forecasting-chronos.html)
- **The survey notes that point-loss FT can improve point accuracy while hurting probabilistic quality** (Wasserstein FT example). Source: [arXiv 2607.20002](https://arxiv.org/html/2607.20002v3)

### Inferences
- The FETS PV Hong Kong result is a warning but probably not a law of nature. A degradation that is identical at lr 1e-8 (effectively no weight change) and at 1e-5 cannot come from learned weights. It points to something that changes between the zero-shot and fine-tuned inference paths in their pipeline: context length, covariate handling, or which prediction method is used. **Lesson for us: always evaluate a "fine-tuned for 0 steps / lr≈0" checkpoint through the same FT inference path, to isolate pipeline effects from learning effects.** This connects to past project bugs (cap convention, night zeroing, clear-sky ceiling) that changed scores without any change in the model.
- FT tends to help where zero-shot is poor and to hurt where zero-shot is already strong. So per-station gains will be heterogeneous, and the per-station zero-shot score is a reasonable prior for where FT pays off. Report per-station deltas, not just the mean.
- Our 南网 score is RMSE-like at the daily level, and we train with pinball loss. The survey's point/probabilistic tension suggests checking whether the quantile spread collapses after FT, as FETS saw coverage collapse. The decision quantile (p*) used for the score should be re-selected after each FT.

### Gaps
- I found no published TimesFM (any version) fine-tuned vs zero-shot comparison on PV *with NWP covariates*.
- I found no Chinese-grid (南网/国网) TSFM fine-tuning study.
- I could not extract the solar fine-tuning table of the ERCOT paper.

---

## Q7. Synthesis for our setup: which strategies are likely to beat LoRA-r8-on-attention + unfrozen head, and which are not worth trying

### Takeaway
The evidence says the *regime* (lr, regularization, target set, pooling, evaluation hygiene) matters more than the PEFT *family*. The best-supported candidates are:
1. lower lr + EMA/early stopping + multiple seeds;
2. all-linear LoRA (Google's default) and/or unfreezing the covariate/tokenizer path;
3. carefully regularized full or last-N FT at lr ~1e-6 to 1e-5, the only TSFM evidence of beating LoRA with covariates;
4. cluster-level LoRAs (MixFT-style) or a soft prefix for station conditioning.

Low priority: DoRA, VeRA, IA3, AdaLoRA, FourierFT, Houlsby adapters, and Time-LLM-style reprogramming.

### Cited Findings
- Full FT > LoRA when covariates are present, with long per-entity history and lr 1e-6 ([arXiv 2609.06656](https://arxiv.org/html/2609.06656v1)). Full FT < LoRA < zero-shot on small data at lr 1e-4 ([arXiv 2607.23146](https://arxiv.org/html/2607.23146v1)).
- The configuration (rank, lr, modules, steps) matters far less than the dataset ([FETS App. E](https://arxiv.org/abs/2604.22328)).
- Google's official TimesFM LoRA uses all-linear, r4, lr 1e-4 ([TimesFM finetuning README](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/finetuning)).
- Linear probing ≈ LoRA on MOMENT ([TRACE](https://arxiv.org/html/2503.16991)). LP beat both LoRA and prompt tuning on MOMENT healthcare data ([arXiv 2411.12824](https://arxiv.org/html/2411.12824)).
- Grouped LoRAs > shared LoRA > per-dataset LoRA, including on unseen series ([MixFT](https://arxiv.org/html/2603.02840v1)).
- Soft prompts ≥ LoRA/FT in distribution, and better out of distribution ([PaCTS](https://arxiv.org/html/2609.34786)).
- Ultra-light PEFT ≈ LoRA on Chronos ([arXiv 2409.11302](https://arxiv.org/html/2409.11302v1)).

### Inferences (prioritized experiment list; each line says what it would show)
1. **Hygiene first:** add a "FT path, 0 steps" control (FETS lesson). Use 3–5 seeds, EMA, and best-validation checkpointing. Report seen and unseen stations separately, and use a night-zeroed metric. Expected effect: removes the noise that currently makes variants look ≈ equal.
2. **lr sweep for LoRA:** {1e-5, 3e-5, 1e-4} with cosine decay or ReduceLROnPlateau. Amazon and the grid-load study use 1e-5 for LoRA; Google uses 1e-4 at r4.
3. **Head ablation:** (a) frozen head + LoRA; (b) LP only (head + input tokenizer, backbone frozen); (c) current setup. This tells us whether the gain is from the head (re-calibration) or from attention adaptation.
4. **Target set:** all-linear LoRA r4–r8 (Google default). Separately, add LoRA or unfreezing on the covariate-input and patch-embedding layers of TimesFM-3. This is plausibly where NWP→PV mapping is learned; it needs code inspection.
5. **Regularized full or last-N-layer FT** at lr 1e-6 to 1e-5, with early stopping, optionally with LP-FT warm start (LLM/vision evidence only) and experience replay. This is the only family with TSFM evidence of beating LoRA when covariates are present. Fp32 on a V100 for ~200M params is feasible memory-wise with a small batch (not verified for our sequence length).
6. **Station conditioning:** K = 2–4 cluster LoRAs (MixFT), or pooled LoRA + per-station soft prefix (PaCTS-style). Test on held-out stations.
7. **Ensemble zero-shot + FT** (AutoGluon `chronos2_ensemble` idea) as a robustness fallback, especially for unseen stations.
8. **Not worth it (low expected value):** DoRA, VeRA, IA3, AdaLoRA, FourierFT swaps (≤ noise in TSFM evidence; FETS shows configuration insensitivity); rank sweeps beyond 4–16; Houlsby/parallel adapters (no TSFM evidence); Time-LLM/TEMPO reprogramming (LLM backbones); independent per-station LoRAs or full FT per station (evidence of harm).

### Gaps
- None of the cited studies uses TimesFM-3. Every transfer of conclusions from Chronos-2, Chronos-T5, MOMENT or TimesFM 1.0/2.5 to TimesFM-3 is an inference.
- None uses a capacity-normalized daily RMSE score like 南网's, so effect sizes may not carry over.
