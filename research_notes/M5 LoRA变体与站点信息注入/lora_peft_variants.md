# LoRA variants and the wider PEFT family for adapting TimesFM-3 to day-ahead PV (M5)

Researched 2026-10-16 (about 16 web search/fetch calls, plus in-repo reading). This note builds on and does not repeat `research_notes/M5 基础模型微调方法调研/sft_peft.md` (cited below as **[sft_peft]**) and `research_notes/M5 适配方法分类选型/weight_finetuning.md` (**[wft]**). Those notes already cover FETS, MixFT, PaCTS, the Beyond-LoRA Chronos/ICU study, TRACE, WiSE-FT, L2-SP and the TimesFM-3 module map. This note adds the LoRA-optimisation literature from 2024–2026, the mixture/multi-LoRA family, a closely matched 2026 Chinese-market Chronos-2 Gated-LoRA paper, and one in-repo finding about the rank sweep.

Labels: **[TSFM]** = measured on a time-series foundation model. **[TS-non-FM]** = measured on a time-series model that is not a foundation model. **[LLM-only]** = evidence from language models only (or language plus vision classification). **[abstract]** = I read only the abstract. **[repo]** = read in this repo. **[est]** = my own arithmetic.

Scale used to judge effect sizes (from `research_notes/M5 适配方法分类选型/hybrid_ensemble.md`): at our score level, a 1% relative error reduction ≈ 0.24 南网 points, and seed noise is about ±0.6 points. LLM papers report "+1–2 accuracy points" on benchmarks such as GSM8K. That unit is not comparable to ours and usually amounts to an lr retune (see Q3).

**Verdict summary (details in each section)**

| Candidate | Verdict | One-line reason |
|---|---|---|
| **rsLoRA-consistent rank sweep** (change only the `alpha` values in `s5_rank*.yaml`) | **ADD (zero code)** | The current sweep uses α = 2r, so the scale is a constant 2. rsLoRA theory then gives an effective update size that grows as √r, which confounds rank with lr (×0.71 / ×1.41 / ×2 for r4/16/32). |
| LoRA+ (lr_B = λ·lr_A) | ADD as 1 cheap variant, low expectation | About 15 lines in `_M5Trainer.create_optimizer`. Its edge over tuned LoRA is ≈0.08 pt in the unified benchmark, and width 1280 is small for the theory. |
| Gated-LoRA (scalar gate on the LoRA delta, from a regime/metadata vector) | TRY LATER (stage 5) | Same setting as ours (Chronos-2, Chinese provincial 15-min data, ctx 672, LoRA r8 α16 q/k/v/o + output layer). Result: −3.05% MAE vs vanilla LoRA, p≈0.074, 6 trainable params. |
| Shared LoRA + small per-cluster delta (HydraLoRA-style shared A / per-cluster B, hard routing) | TRY LATER (after per-station deltas are analysed) | Grouped beats shared in MixFT (K=2). Separate per-cluster LoRAs for 3–5 clusters give each one ~6–9 correlated stations, which is likely noise-level. |
| Separate LoRA per cluster (K=3–5) | REJECT in favour of the row above | Splits the data. MixFT found K=2 best, and per-entity models performed worst. |
| MoLE / MoLoRA / MixLoRA / LoRAHub (token routers, LoRA libraries) | REJECT | LLM multi-task instruction tuning only. They add capacity, and M8 already showed that capacity does not help. LoRAHub needs a library of LoRAs we do not have. |
| Virtual-variate / prefix soft prompts | TRY LATER (already in [wft] §4) | PaCTS gives ≈1% MASE on TimesFM-2.5. No study covers covariates. Note that the variate axis has RoPE (see Q2). |
| Time-LLM / TEMPO / TEST reprogramming | REJECT | These target LLM backbones and do not apply to a numeric TSFM. |
| PiSSA / MiLoRA / OLoRA / LoRA-GA / LoRA-Pro | REJECT | Their gains shrink to 1–2% (LLM) once lr is tuned. LoRA-GA's large wins appear only at lr 1e-6. They break the "zero-init = exact zero-shot" invariant that WiSE-FT relies on. |
| DoRA (re-check) | REJECT (confirmed) | Within 1–2% of LoRA after lr tuning. No TSFM numbers exist. |
| VeRA, AdaLoRA, FourierFT | REJECT (confirmed) | Only useful for per-entity storage. AdaLoRA's importance scores anti-correlate with Shapley importance on a TSFM. |
| LoRA-FA, GaLore, QLoRA, Delta-LoRA | REJECT | These are memory savers, and we have no memory pressure. Delta-LoRA also modifies the base. |
| (IA)³ / RMSNorm-gain tuning | TRY LATER as a regularised *control* only | BitFit is not applicable (no biases). Run only if head-only ≈ LoRA. |
| OFT / BOFT | REJECT | Evidence only on vision, diffusion and LLMs. Higher compute. No benefit we can argue for. |
| Houlsby / parallel adapters, AdapterFusion | REJECT (generic) / TRY LATER (nonlinear adapter on the resblock, only if resblock variants win) | No TSFM head-to-head exists. Adapters cannot be merged. AdapterFusion assumes many tasks. |

---

## Q1. Mixture / clustered / multi-task LoRA (MoLE, MoLoRA, LoRAHub, HydraLoRA, MixLoRA, metadata-routed LoRA, per-cluster LoRA): evidence for forecasting / multi-entity regression, and is 3–5 station clusters data-feasible?

### Takeaway
The mixture-of-LoRA literature is almost entirely LLM multi-task instruction tuning. Forecasting evidence comes from three papers:
- **MixFT** [TSFM]: K=2 hard-routed cluster LoRAs beat one shared LoRA, and per-dataset LoRAs were the worst option.
- **Gated-LoRA** (Aug 2026) [TSFM]: a near-identical setting to ours. One LoRA was scaled by a 6-parameter gate on a market-state vector, giving −3.05% MAE vs vanilla LoRA (p≈0.074). A state-independent global gate gave no gain.
- **MoLA** [TS-non-FM]: horizon-segment LoRA experts on iTransformer gave tiny gains (≈0.005 MSE).

For 28 stations, separate LoRAs for 3–5 clusters are data-unwise. A shared LoRA plus a small cluster- or regime-conditioned component, trained jointly, is the defensible form.

### Cited Findings
**Forecasting / TSFM evidence**
- **Gated-LoRA on Chronos-2 for Chinese day-ahead electricity prices** [TSFM] (arXiv 2608.11359v1, 11 Aug 2026):
  - Setup: frozen Chronos-2 (120.7M). LoRA **rank 8, α=16, on q/k/v/o + the output-patch layer**, ~1.21M params, 1500 steps, batch 8, lr 1e-4. Data: 4 provincial markets (Guangdong, Liaoning, Shandong, Shanxi), **15-min resolution, 672-step context**, 96-step horizon, leave-one-market-out.
  - Gate: γ = clip(1 + β·tanh(wᵀz + b), 1, 3) with β=2, and W' = W + γ(α/r)BA. z is a 5-dim market-state vector (net load, renewable share, reserve adequacy, capacity margin, volatility), standardised on source markets. **Only the 6 gate parameters are trained**, for 300 steps, with the LoRA frozen.
  - Four-market average: Source-LoRA MAE 76.98 / RMSE 142.90 / CRPS 62.66 vs Gated 74.63 / 137.87 / 61.23. That is **−3.05% MAE, −3.52% RMSE, −2.3% CRPS**. Pooled DM p ≈ 0.074 for MAE.
  - Most of the gain is on one market (Liaoning MAE 147.73 → 139.24). Shanxi is essentially unchanged. PICP@80 drops to 70.23%, so calibration worsens.
  - Ablations: a **learned global scalar gate (no state dependence) gives no gain (MAE 77.00)**. Shuffled states give 74.87, which the authors note means "causal evidence is limited". No full-FT baseline is reported.
  - Source: [arXiv 2608.11359](https://arxiv.org/html/2608.11359)
- **MixFT** [TSFM] (Chronos-Bolt-small, Moirai-1.1-R-small): K sub-domains from a Bayesian GMM on embeddings, one LoRA (r2) per sub-domain, hard routing. Average rank: MixFT 2.0, shared single LoRA 3.0, zero-shot 2.9, per-dataset LoRAs 6.3. K=2 was best on validation, and hard routing (rank 1.6) ≫ soft routing (4.4) — [arXiv 2603.02840](https://arxiv.org/html/2603.02840v1) (numbers from [sft_peft]/[wft]).
- **MoLA, Mixture of LoRA for multi-step forecasting** [TS-non-FM]:
  - Design: the backbone is iTransformer pretrained for one-step prediction on the same benchmark, **not an external TSFM**. The horizon is split into K segments, and P shared LoRA experts are mixed by a learnable per-segment weight vector (not an input-dependent router).
  - vs the base model (MSE): ETTh2 0.377 vs 0.390, Weather 0.266 vs 0.281, PEMS08 0.138 vs 0.159.
  - vs plain step-specific LoRA: ETTh2 0.377 vs 0.381, Weather 0.266 vs 0.272. These are single-run point estimates.
  - Source: [arXiv 2505.17872](https://arxiv.org/html/2505.17872)
- **Federated "unilateral" LoRA for PV**: aggregates generalizable LoRA modules across plants and keeps task-specific ones local. The authors claim noise reduction and faster convergence; I saw no numbers. The backbone is an LLM, not a TSFM — [IEEE 11359056](https://ieeexplore.ieee.org/abstract/document/11359056/) [abstract].
- I found **no paper** that trains separate LoRAs per climate/geography cluster of PV, wind or load sites (search 2026-10-16). The nearest designs are the Gated-LoRA and federated papers above — [arXiv 2608.11359](https://arxiv.org/html/2608.11359); [IEEE 11359056](https://ieeexplore.ieee.org/abstract/document/11359056/).

**Mixture-of-LoRA mechanics (LLM-only evidence)**
- **HydraLoRA** (NeurIPS 2024): one shared A matrix and K B matrices combined by a trainable MoE router (y = W₀x + Σᵢ G(x)ᵢ BᵢAx). Motivation: some LoRA parameters learn what all data has in common and others learn component-specific aspects. Code: github.com/Clin0212/HydraLoRA. Secondary summaries claim a 1.96× training speedup; I could not verify this — [arXiv 2404.19245](https://arxiv.org/html/2404.19245v2); [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2024/file/123fd8a56501194823c8e0dca00733df-Paper-Conference.pdf); [GitHub](https://github.com/clin0212/hydralora).
- A 2025 critique (MASA) argues that the single shared A matrix is a representational bottleneck — [arXiv 2510.06005](https://arxiv.org/html/2510.06005v2). A 2025 single-rank MoE-LoRA paper (SMoRA) treats each rank as an expert for multi-task learning — [arXiv 2501.15103](https://arxiv.org/pdf/2501.15103).
- A unified 2026 LoRA-variant study treats "integration with Mixture-of-Experts" as one of four taxonomy axes. It concludes that with proper hyperparameters, vanilla LoRA "consistently matches or surpasses the performance of most of its variants". Its tasks are NLG, NLU and image classification — [arXiv 2601.22708](https://arxiv.org/abs/2601.22708) [abstract].
- MoLE (learned per-layer gating over pre-trained LoRAs, [arXiv 2404.13628](https://arxiv.org/abs/2404.13628)), MoLoRA (MoE of LoRA/IA³ experts with soft token routing, [arXiv 2309.05444](https://arxiv.org/abs/2309.05444)), MixLoRA (top-k router over LoRA experts in the FFN, [arXiv 2404.15159](https://arxiv.org/abs/2404.15159)) and LoRAHub (gradient-free weighted composition of LoRAs pre-trained on other tasks, [arXiv 2307.13269](https://arxiv.org/abs/2307.13269)) are all evaluated on LLM instruction/multi-task benchmarks [LLM-only; IDs from my knowledge, not re-fetched this session].

**Repo facts that bear on clustering**
- M5 trains one LoRA on all 28 stations with a global batch of 16 for 4000 steps. Early stopping is on `nanwang_after_pstar` [repo, `forecast_backbones/m5/config/finetune.yaml`].
- The earlier verdict was that per-station LoRA is rejected and K=2 clustered LoRA is "TRY LATER (stage 5)", to be run only if per-station deltas from the pooled LoRA are bimodal [wft §5].

### Inferences
- **Data arithmetic for 3–5 clusters [est].** 28 stations × ~1–2 years ≈ 10–20k station-days. With 3–5 clusters, each separate LoRA sees ≈ 2–7k station-days from 6–9 stations. Stations in the same geographic cluster share weather days, so the effective number of independent weather situations per cluster is much smaller than the station-day count. Evaluating per cluster on 6–9 stations also inflates seed noise beyond ±0.6. MixFT's own optimum was **K=2**, with far more heterogeneous datasets than our 28 southern-China PV sites. **Separate per-cluster LoRAs are not reasonable data-wise. K ≥ 3 is very likely noise or overfitting.**
- **The defensible form is shared trunk + small conditioned component, trained jointly:**
  - (i) HydraLoRA-style: shared A and the shared B as today, plus a per-cluster B_k (or a rank-2 per-cluster residual), with hard routing by station cluster. All 28 stations train the shared part.
  - (ii) Gated-LoRA-style: a scalar or per-layer gate γ(z) on the frozen pooled LoRA. Here z would be **per-issue weather regime from NWP** (for example D+1 forecast daily clear-sky index, intra-day NWP-GHI variability, temperature), optionally plus station metadata (cap_mw, tracker flag if known). A regime-conditioned z also serves unseen stations, which a cluster id does not. One station has no coordinates, so geography-only routing fails there.
- **Gated-LoRA's global-scalar ablation (no gain) is informative for us.** It is the training-time analogue of our post-hoc WiSE-FT α. It suggests that a single scalar strength is not where the gain is; any gain needs **state dependence**. Expect a small effect: 3% MAE in their unseen-market transfer is roughly 0.7 南网 points if it transferred one-to-one [est]. That is near the noise band, and their own significance was marginal.
- **Mechanism risk specific to PV.** Station heterogeneity in PV is mostly capacity factor / DC-AC ratio, curtailment, tracker vs fixed and shading. These are per-station *level/shape* effects, which the output head plus RevIN largely absorbs. A LoRA mixture adds attention capacity, and the M8 round-2 lesson was that added capacity ≈ noise (project memory, M8 status). So token-level routers (MoLoRA/MixLoRA) are the least promising members of the family here.
- **Implementation cost with our minimal LoRA [repo, `lora.py`]:**
  - Gated-LoRA: add a `gate` multiplier to `LoRALinear.forward`, set from the batch before the forward call. Then add a 6–20-param gate module trained for ~300 steps with the LoRA frozen. About 40 lines plus collate support for z.
  - Hydra/shared-plus-cluster: `LoRALinear` holds K B-matrices plus a per-sample cluster index. Because cluster is per sample, it needs a batched gather. About 60–80 lines plus save/load and WiSE handling.
  - Separate per-cluster LoRA: zero code (filter the station list and train K runs), which is the only reason it is tempting.

### Gaps
- There is no PV, wind or load paper with per-cluster LoRAs, and no TSFM paper testing HydraLoRA/MoLE/MixLoRA.
- Gated-LoRA does not ablate individual gate inputs and has no full-FT baseline. Its evaluation is a 4-market leave-one-out on prices, not PV power.
- HydraLoRA's quantitative results were not verified this session (only secondary summaries were read).

---

## Q2. Soft prompts / prefix tuning / P-tuning / reprogramming for TSFMs: published use on TimesFM/Chronos/Moirai, could TimesFM-3 take learned prompt tokens as extra variates or patches, and evidence vs LoRA

### Takeaway
Published prompt tuning on TSFMs covers:
- PaCTS (2026): TimesFM-2.5 and Chronos-2, ≈1–1.5% MASE gain, and the only method beating zero-shot on an unseen benchmark.
- Gen-P-Tuning (MOMENT/GPT4TS): LoRA 3.109 vs prompt 2.918 MSE, but linear probing reached 2.622.
- CPFM (MOMENT, black-box domain adaptation).

None uses past-future covariates, and none uses TimesFM-3. On TimesFM-3, learned prompt rows on the variate axis are the cleanest insertion point. Unlike [wft] §4 assumed, the released forecaster **uses variate RoPE** (`use_rope_var=True`), so prompt rows must be appended *after* the existing variates to leave target-covariate relative positions unchanged. Time-LLM/TEMPO/TEST are LLM reprogramming methods and do not apply.

### Cited Findings
- **PaCTS** [TSFM], TimesFM-2.5 on GIFT-Eval: MASE/CRPS 0.717/0.514 → 0.706/0.507 at 4k context. Prompts are prepended to patch embeddings with position id 0. On the unseen TIME benchmark (Chronos-2), PaCTS gives 0.661/0.555 vs zero-shot 0.664/0.558, while all five weight-space baselines were *worse* than zero-shot. Static (non-adaptive) prompts "improve CRPS but provide little improvement in MASE" — [arXiv 2609.34786](https://arxiv.org/html/2609.34786) (from [wft] §4).
- **Gen-P-Tuning** (late 2024): a learned cross-channel prefix per layer on frozen univariate TSFMs (MOMENT, GPT4TS, with TimesFM as a special case of the general form) — [arXiv 2411.12824](https://arxiv.org/html/2411.12824); [PMLR v259](https://proceedings.mlr.press/v259/liu25a.html). On MOMENT healthcare data: LoRA 3.109 vs prompt tuning 2.918 vs linear probing 2.622 MSE ([sft_peft] Q3).
- **CPFM** (Oct 2025): prompts injected into the MHSA of a frozen MOMENT model. This is equivalent to concatenating the same prompt parameter to q/k/v. Its target is black-box domain adaptation — [arXiv 2510.00487](https://arxiv.org/pdf/2510.00487).
- A 2026-10-16 web search found no 2025–2026 soft-prompt paper on Chronos-2/Moirai-2/TimesFM-2.5/3 other than PaCTS. Newer work on those models relies on LoRA or in-context fine-tuning — [ICF, arXiv 2410.24087](https://arxiv.org/pdf/2410.24087); [process-model forecasting, arXiv 2512.07624](https://arxiv.org/html/2512.07624v1).
- The original TimesFM paper notes that prompt-tuning techniques are less clear for TSFMs — [arXiv 2310.10688](https://arxiv.org/pdf/2310.10688).
- **TimesFM-3 variate axis [repo / local code, from `research_notes/M5 适配方法分类选型/in_context.md` finding 1]:**
  - `timesfm3_forecaster.py:305-306` builds the transformer with `use_rope_seq=True, use_rope_var=True`, and `TimesFM3Evaluator` (used by M5) subclasses it.
  - No `var_segment_pos` is passed, so variate positions = `arange(v)` in packing order `[targets | past_only | past_future]`.
  - This corrects [wft] §4, which read the `TimesFM3Torch` default `use_rope_var=False` and concluded that var_attn is permutation-invariant.
- Covariate/multivariate prompt-like adapters (WindPrompt, generalized prompt tuning) are listed in the 2026 post-training survey without prompt-vs-LoRA numbers — [arXiv 2607.20002](https://arxiv.org/html/2607.20002v3).
- Time-LLM (reprogramming patches into an LLM's word-embedding space), TEMPO (decomposition plus a prompt pool on GPT-2) and TEST (contrastive embedding alignment to an LLM) are designed to make *language* models forecast — [Time-LLM arXiv 2310.01728](https://arxiv.org/abs/2310.01728); [TEMPO arXiv 2310.04948](https://arxiv.org/abs/2310.04948); [TEST arXiv 2308.08241](https://arxiv.org/abs/2308.08241) [IDs from my knowledge, not re-fetched].

### Inferences
- **Virtual-variate prompts on TimesFM-3, corrected design:**
  - Append k learned rows *after* the last real variate, at positions v..v+k−1. RoPE is relative, so target↔covariate geometry is unchanged, and each prompt row gets a fixed offset from the target, which is what makes it learnable.
  - Inserting prompt rows before the covariates would shift every covariate's relative index and partly invalidate pretrained var_attn behaviour.
  - Rows are excluded from the output head and RevIN. Params: k×1280 shared (2.6–5.1k), or a small generator from context statistics for unseen stations (see [wft] §4).
  - Cost: ≈ 1.7–2.3× tokens per step and a fork of `TimesFM3Torch.forward` (~50–80 lines) [est, from [wft] §4].
- **Time-axis prefix patches** collide with the causal seq_attn and the `arange(n)+next_index` positions. They need `segment_pos` plumbing that `forward` does not pass [wft] §4. This is more invasive and has no extra upside.
- **Expected value:** ≈1% MASE-equivalent on seen stations, i.e. ≪ 0.6 南网 points. The argument for prompts is unseen-station robustness and composability with LoRA, not beating LoRA. Static shared prompts mostly move CRPS (PaCTS ablation). Our metric is point-like (p* on quantiles), so a static prompt is the weakest variant.
- **Verdict:** prompt rows = TRY LATER (unchanged), with the RoPE correction recorded. Reprogramming methods = REJECT (they need an LLM backbone).

### Gaps
- No prompt-tuning result exists on any TSFM with native past-future covariates. No study measures how variate-RoPE offsets interact with learned prompt variates.

---

## Q3. LoRA optimisation variants (LoRA+, rsLoRA, PiSSA/MiLoRA/OLoRA, LoRA-GA, LoRA-Pro, DoRA, VeRA, AdaLoRA, LoRA-FA, Delta-LoRA, GaLore, (IA)³, BitFit, LN tuning, OFT/BOFT): robust evidence beyond LLM instruction tuning, and which are near-free

### Takeaway
The 2026 evidence is consistent across studies. **Once lr is tuned per method, vanilla LoRA matches nearly all variants within 1–2% (LLM).** Lee et al. compared nine variants. The unified benchmark (NLG/NLU/image classification) found the best variant at lr 1e-4, LoRA+, only +0.08 pt ahead. Most published variant gains came from comparisons at one shared lr. LoRA-GA, for example, wins by 11 points only at lr 1e-6.

None of these variants has TSFM evidence except OLoRA-init used in MixFT's recipe (no ablation), and DoRA/VeRA/AdaLoRA/BitFit/LN in small TSFM studies within noise.

Two items are near-free and worth doing for *protocol* reasons rather than expected gains:
- **rsLoRA-consistent α in the rank sweep**: our current α = 2r confounds rank with effective step size.
- **LoRA+**: one ratio hyperparameter.

### Cited Findings
**Meta-evidence (2026)**
- **"Learning Rate Matters: Vanilla LoRA May Suffice"** (Lee, Ko, Chen, Yeh; v1 4 Feb 2026, v2 19 May 2026) [LLM-only]:
  - Nine variants vs vanilla LoRA, with searches over lr, batch size, rank and duration. "Different LoRA methods favor distinct learning rate ranges". After tuning, all methods peak "within 1–2%".
  - The differing optimal lrs are explained by the largest Hessian eigenvalue — [arXiv 2602.04998](https://arxiv.org/abs/2602.04998) [abstract].
  - Search-result snippets of the same paper say that PiSSA needs a lower lr than LoRA in every model-task combination, that PiSSA trails LoRA at low rank and overtakes it at high rank (MiLoRA the opposite), and that gaps are 0.43–1.81% — [arXiv 2602.04998 PDF](https://arxiv.org/pdf/2602.04998) [snippet, not verified in full text].
- **"A Unified Study of LoRA Variants"** (He et al., 30 Jan 2026, TPAMI submission):
  - Tasks: NLG, NLU and **image classification** (the only non-LLM evidence in this family). LoRA and its variants are "much more sensitive to learning rate than to other hyperparameters". With proper hyperparameters LoRA "consistently matches or surpasses the performance of most of its variants". Codebase: LoRAFactory — [arXiv 2601.22708](https://arxiv.org/abs/2601.22708) [abstract].
  - Search snippets add that LoRA-GA leads LoRA by >11 points on GSM8K at lr 1e-6, while the best variant (LoRA+) is only +0.08 at lr 1e-4 — [arXiv 2601.22708 PDF](https://arxiv.org/pdf/2601.22708) [snippet].
- **"Beware of the Batch Size"** (2026): the optimal lr depends jointly on batch size. PiSSA vs MiLoRA show a crossover with batch size, and tuned vanilla LoRA remains competitive — [arXiv 2602.09492](https://arxiv.org/html/2602.09492v2) [snippet].

**Individual methods**
- **rsLoRA** (Kalajdzievski, 2023):
  - Claim: the α/r factor causes gradient collapse at larger r, so larger ranks perform no differently from small ones. Adapters are rank-stabilized only if the scale γ_r ∈ Θ(1/√r). With α/√r, higher ranks do improve performance. Inference cost is unchanged — [arXiv 2312.03732](https://arxiv.org/pdf/2312.03732) [LLM-only].
  - A practitioner replication found the rsLoRA-vs-LoRA difference "minimal" once the effective lr is higher — [determined.ai blog](https://determined.ai/blog/lora-parameters).
- **LoRA+** (Hayou, Ghosh, Yu; Feb 2024, v2 Jul 2024): a width-scaling argument that equal lrs for A and B are suboptimal, so it sets lr_B > lr_A by a fixed ratio. Reported gain: "1–2%" performance and up to ~2× faster fine-tuning at equal compute — [arXiv 2402.12354](https://arxiv.org/abs/2402.12354) [abstract; the recommended ratio is not stated in the abstract].
- **PiSSA** (NeurIPS 2024 spotlight):
  - Mechanism: initialise A and B from the top-r singular components of W and freeze the residual.
  - Results: Mistral-7B GSM8K 72.86% vs LoRA 67.7%. "Consistently outperformed" across 12 models (184M–70B) and 13 tasks (5 NLG, 8 NLU) **under identical setups** — [arXiv 2404.02948](https://arxiv.org/abs/2404.02948).
  - Lee et al. show that these identical-setup gains largely vanish with per-method lr — [arXiv 2602.04998](https://arxiv.org/abs/2602.04998).
- **LoRA-GA** (Jul 2024): initialise so that the first-step gradient of BA matches the full-FT gradient. Results: T5-Base GLUE +5.69%, Llama-2-7B GSM8K +11.52%, HumanEval +5.05%, convergence 2–4× faster — [arXiv 2407.05000](https://arxiv.org/abs/2407.05000).
  - Its comparisons use one lr for the whole LoRA family — [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/62c4718cc334f6a0a62fb81c4a2095a1-Paper-Conference.pdf) [snippet]. Its advantage disappears at lr 1e-4 ([arXiv 2601.22708](https://arxiv.org/pdf/2601.22708) [snippet]).
- **LoRA-Pro**: adjusts the A/B gradients so that the equivalent low-rank gradient approximates the full-FT gradient — [arXiv 2407.18242](https://arxiv.org/pdf/2407.18242) [LLM-only; title/abstract level].
- **"LoRA vs Full Fine-tuning: An Illusion of Equivalence"** (v1 Oct 2024, v3 Oct 2025): LoRA creates "intruder dimensions", which are new high-ranking singular vectors. LoRA forgets less than full FT, and the forgetting is localised to the intruders. Scaling the intruders down recovers pre-training modelling with a minor drop in downstream performance — [arXiv 2410.21228](https://arxiv.org/abs/2410.21228) [LLM-only, abstract].
- **TSFM evidence for other PEFT** (from [sft_peft] Q2):
  - Chronos on ICU data: BitFit/LN/VeRA/FourierFT matched or beat LoRA within small margins — [arXiv 2409.11302](https://arxiv.org/html/2409.11302).
  - TRACE on MOMENT: AdaLoRA's gradient importance correlated −22.1% with Shapley importance — [arXiv 2503.16991](https://arxiv.org/pdf/2503.16991).
  - TimesFM PR #104 added DoRA without reported numbers — [PR #104](https://github.com/google-research/timesfm/pull/104).
  - MixFT used OLoRA init without ablating it — [arXiv 2603.02840](https://arxiv.org/html/2603.02840v1).
- **DoRA** (magnitude/direction decomposition): [arXiv 2402.09353](https://arxiv.org/abs/2402.09353). It is one of the variants that tie vanilla LoRA after lr tuning — [arXiv 2602.04998](https://arxiv.org/pdf/2602.04998) [snippet].
- **TimesFM-3 has no biases outside `output_head`**, so BitFit is not applicable. RMSNorm-gain tuning has ≈163k params — [wft] §0.

**Repo facts (our minimal LoRA)**
- `LoRALinear`: A is Kaiming-initialised, B is zero, scaling = α/r, so outputs are bit-identical to zero-shot at mount. `inject_lora` matches by name tail or by `re:` full path [repo, `forecast_backbones/m5/lora.py`].
- **Rank sweep configs use α = 2·rank, so scaling = 2 for r = 4, 8, 16, 32** [repo, `config/variants/s5_rank{4,16,32}.yaml`, comment "与 l1 的缩放 alpha/rank = 2 相同"].
- Training uses HF Trainer AdamW (wd 0.01, lr 1e-4, cosine, grad clip 1.0) through a `_M5Trainer(Trainer)` subclass [repo, `train.py:296,508`; `transformer_trainer.py:135`]. That subclass is where a custom `create_optimizer` (LoRA+) would go.
- WiSE-FT is implemented as "LoRA delta ×α, head interpolated to pretrained, α=0 bit-identical to zero-shot" [repo, commit 735c166 message]. That property depends on zero-init B.

### Inferences
- **rsLoRA, ADD, zero code (a protocol fix, not a method bet).**
  - With Adam and a fixed scale s, rsLoRA's analysis implies the size of the learned update BA·x grows ~√r. Relative to a rank-stable s ∝ 1/√r, our α = 2r sweep therefore gives an effective step size of **×0.71 (r4), ×1 (r8), ×1.41 (r16), ×2 (r32)** [est, from the rsLoRA theorem].
  - At lr 1e-4, already the top of our lr grid, a worse r32 could be an "lr too high" effect rather than a capacity effect. A better r4 could be a "more conservative" effect.
  - Fix: keep s = 2 at r8 and use the rsLoRA scale s = 2·√(8/r). That gives **α = s·r = 11.31 (r4), 22.63 (r16), 32.0 (r32)**, and only the yaml numbers change.
  - Alternatively, keep α = 2r and read the sweep as rank×lr confounded, or co-sweep lr. Given Lee et al., expect the de-confounded rank curve to be flat. Its value is to stop a false "rank matters" or "rank hurts" conclusion.
- **LoRA+, ADD as one variant, low expectation.**
  - Implementation: `create_optimizer` override with two param groups, B at λ·lr. The head goes in its own group at the base lr. About 15 lines. The ratio is not in the abstract; λ = 16 is the commonly used value, which I did not verify this session.
  - Its theoretical benefit grows with width. Our d = 1280 is 3–6× smaller than the LLM widths it was derived on, and the unified benchmark shows +0.08 at lr 1e-4.
  - Run only lr_A = 3e-5 with λ = 16 (B at 4.8e-4) as a single probe, after the s2 lr sweep reports. If the best lr sits at the top edge (1e-4), that hints B is under-stepped, and LoRA+ is the principled way to raise only B's step.
- **PiSSA / MiLoRA / OLoRA / LoRA-GA, REJECT.**
  - Gains vanish under per-method lr tuning (LLM).
  - LoRA-GA's large wins are at lr ≤ 1e-6, which we do not use.
  - All of them change the frozen base (W_res = W − B₀A₀) or start from nonzero BA. That breaks the "mount = exact zero-shot" invariant, the WiSE-FT α = 0 semantics and the save format (the residual base or B₀A₀ would also have to be stored, with SVD sign ambiguity).
  - PiSSA's one structural argument, a better fit at high rank, is irrelevant because the rank sweep is expected flat.
- **LoRA-Pro, REJECT:** the optimizer complexity is not justified. It has LLM-only evidence and falls into the same lr-tuning caveat.
- **DoRA re-check, REJECT confirmed:** no new TSFM numbers, and LLM ties after lr tuning.
- **VeRA / AdaLoRA / FourierFT, REJECT confirmed:** VeRA and FourierFT are only valuable for kB-sized per-entity adapters, and per-station LoRA is rejected. AdaLoRA's importance criterion is anti-correlated with Shapley on a TSFM.
- **LoRA-FA (freeze A) / GaLore / Delta-LoRA, REJECT.** LoRA-FA and GaLore are memory savers, and full FT fits in ≈8 GB fp32 ([wft] §7). Delta-LoRA also writes ΔBA into W, which destroys base-freezing and WiSE semantics, and its evidence is LLM-only. (IDs: LoRA-FA [arXiv 2308.03303](https://arxiv.org/abs/2308.03303), Delta-LoRA [arXiv 2309.02411](https://arxiv.org/abs/2309.02411), GaLore [arXiv 2403.03507](https://arxiv.org/abs/2403.03507); from my knowledge, not re-fetched.)
- **(IA)³ / RMSNorm-gain, TRY LATER as controls only.**
  - (IA)³ is a learned rescaling of k, v and the FFN hidden activations ([arXiv 2205.05638](https://arxiv.org/abs/2205.05638)). On TimesFM-3 that is ≈ 20 × (4×1280 + 1280) ≈ 128k params [est]: k and v for each of the two attentions plus the ff hidden.
  - RMSNorm-gain tuning is 163k params. Both are "re-calibration only" controls. Run them only if `s2_head_only` ≈ LoRA r8, to decide whether the LoRA gain is pattern learning or re-scaling. BitFit is N/A.
- **OFT / BOFT, REJECT.** Multiplicative orthogonal updates preserve the angles between neurons. Evidence is on diffusion, ViT and LLMs, with no regression/TSFM evidence ([OFT arXiv 2306.07280](https://arxiv.org/abs/2306.07280); [BOFT arXiv 2311.06243](https://arxiv.org/abs/2311.06243); IDs from my knowledge). It costs extra compute (Cayley or butterfly products). Its "keeps pretrained geometry" argument is already covered more cheaply by WiSE-FT and EMA.
- **The intruder-dimension result** gives an LLM-side mechanism for why WiSE-FT α < 1 can help unseen stations: it shrinks the LoRA delta, intruders included. This supports keeping WiSE in the evaluation grid rather than adding a new method.

### Gaps
- There is no TSFM study of LoRA+, rsLoRA, PiSSA, LoRA-GA or LoRA-Pro.
- I did not read the full text of Lee et al. or the unified study. The per-method numbers above marked [snippet] come from search summaries.
- The exact LoRA+ ratio recommendation was not verified this session.

---

## Q4. QLoRA: confirm or overturn the rejection

### Takeaway
**Confirm REJECT.** QLoRA's only benefit is memory. At 330.7M params the fp32 base is ≈1.32 GB, and full FT with AdamW fits in ≈8 GB ([wft] §7). Quantising the frozen base to 4-bit NF4 adds approximation error to the zero-shot and LoRA paths, with no accuracy upside. No TSFM QLoRA-vs-LoRA comparison exists.

### Cited Findings
- QLoRA = frozen 4-bit NF4 base + LoRA. Its accuracy claims are "matches 16-bit fine-tuning" on LLMs; it is a memory method — [arXiv 2305.14314](https://arxiv.org/abs/2305.14314) [LLM-only].
- QPiSSA vs QLoRA (LLaMA-3-70B GSM8K 86.05% vs 81.73%) shows that quantisation error matters enough that initialisation choices recover it. That result is for a 70B model under memory pressure — [arXiv 2404.02948](https://arxiv.org/abs/2404.02948).
- An LLM-based PV paper uses "DQ-LoRA" (dynamic quantized LoRA) on an LLM backbone, not a TSFM — [ScienceDirect S037877962501048X](https://www.sciencedirect.com/science/article/abs/pii/S037877962501048X) [abstract].
- Memory arithmetic and the absence of any TSFM QLoRA benchmark — [wft] §7.

### Inferences
- Nothing new overturns the rejection. On V100, 4-bit dequant kernels would also slow each step, because there is no bf16 and compute stays in fp32 or fp16.
- The only scenario where QLoRA would matter is a ≥1B-param future TimesFM on 16 GB V100s, which does not apply.

### Gaps
- None material.

---

## Q5. Adapters (Houlsby, parallel adapters, AdapterFusion): evidence on TSFMs

### Takeaway
There is **no head-to-head TSFM study** of bottleneck or parallel adapters vs LoRA. Adapter-like modules on TSFMs exist only as covariate-injection adapters (ChronosX, UniCA, CoRA-Cov, WindPrompt) and grouped adapters (MixFT). In NLP, adapters and LoRA are close, and LoRA can be merged at inference. AdapterFusion assumes many task adapters to fuse and does not fit one task. The only adapter idea with a code-level rationale for us is a small **nonlinear adapter on the covariate input pathway** (the resblock). It is worth considering only if the already-planned resblock variants (`s2_lora_resblock`, `s2_probe_resblock`, `s2_lora_var_resblock`) show that the input pathway is the lever.

### Cited Findings
- The 2026 TSFM post-training survey lists covariate adapters (ChronosX, UniCA, CoRA-Cov, WindPrompt) and grouped adapters (MixFT) without adapter-vs-LoRA numbers — [arXiv 2607.20002](https://arxiv.org/html/2607.20002v3) (via [sft_peft] Q2).
- A web search on 2026-10-16 found no TSFM results for adapters or prompt tuning head-to-head against LoRA. The LoRA paper's own (LLM) comparison has LoRA slightly ahead of adapters, prefix and prompt tuning — [Raschka summary](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html) [secondary].
- EnergyGPT uses frequency/temporal/spatial adapters on an LLM for multi-energy load (organised by signal dimension) — [ScienceDirect S0960148125009759](https://www.sciencedirect.com/science/article/abs/pii/S0960148125009759) [abstract; LLM backbone].
- A 2026 load-forecasting paper on a GPT-style model injected LoRA into attention *and FFN* and reports beating frozen and partially-unfrozen variants on AEP hourly — [Electronics Letters 2026](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ell2.70612?af=R) [abstract].
- TimesFM-3 resblock input = [32-step patch | 64-step lookahead | 96 mask bits] = 192 dims → 1280, with no bias — [wft] §0.
- Houlsby adapters ([arXiv 1902.00751](https://arxiv.org/abs/1902.00751)), the parallel-adapter "unified view" ([arXiv 2110.04366](https://arxiv.org/abs/2110.04366)) and AdapterFusion ([arXiv 2005.00247](https://arxiv.org/abs/2005.00247)) are NLP methods [LLM/NLP-only; IDs from my knowledge, not re-fetched].

### Inferences
- **Generic Houlsby/parallel adapters in every layer, REJECT.** They have no TSFM evidence. Unlike LoRA they cannot be merged, so inference gets slower. They cannot do anything a LoRA on q/k/v/o/ff cannot already do in the linear regime. The extra nonlinearity is capacity, which the M8 lessons say is not the bottleneck.
- **AdapterFusion, REJECT:** a single task with no library of adapters.
- **Nonlinear covariate adapter, TRY LATER, conditional.** Example: a parallel 192→64→1280 GELU bottleneck added to the resblock output, B zero-initialised so that mount = zero-shot, ≈ 94k params [est]. Run it only if the resblock LoRA/probe variants beat plain attention LoRA, because that would localise the gain to how NWP enters the model. Cost: ~30 lines, a wrapper around `pre_transformer_resblock` registered in `unfreeze_modules`-style save/load.

### Gaps
- There are no TSFM adapter-vs-LoRA numbers, and no numbers for ChronosX/UniCA-style covariate adapters on PV with NWP were collected this session.

---

## Q6. Published fine-tuning benchmarks for TSFMs comparing LoRA vs full FT vs linear probing vs adapters / prompts

### Takeaway
There is no single benchmark that compares all PEFT families on TSFMs. The pieces agree:
- LoRA ≈ full FT at small data. Full FT needs very small lr and overfits.
- Linear probing or head-only is sometimes as good or better.
- Configuration (rank, targets, steps) matters less than the dataset (FETS).
- Weight-space FT can lose to zero-shot on unseen entities (PaCTS, MixFT).
- A matched 2026 Chinese-market paper uses exactly our LoRA recipe and gets most of its gain from LoRA itself, with the gate adding ~3%.
- Official TimesFM added a PEFT-LoRA example in April 2026, for TimesFM 2.5-era code.

### Cited Findings
- **Beyond LoRA** (Gupta et al., Sep 2024) [TSFM]: BitFit, LN-tuning, VeRA and FourierFT vs LoRA on Chronos sizes for ICU vital-sign forecasting. Some PEFTs beat LoRA in parameter efficiency and domain adaptation — [arXiv 2409.11302](https://arxiv.org/html/2409.11302).
- **Low-Rank Adaptation of TSFMs for out-of-domain modality** (2024) [TSFM]: zero-shot, full FT and LoRA on Moirai/Chronos and others. Mixed: LoRA does not uniformly beat full FT — [arXiv 2405.10216](https://arxiv.org/pdf/2405.10216) [snippet].
- **MSFT** (2025) [TSFM]: baselines are full FT, linear probing, LoRA and AdaLoRA on Moirai/Moment q/k/v with a trainable head. Full FT needs a small lr to be stable — [arXiv 2506.14087](https://arxiv.org/pdf/2506.14087) [snippet].
- **TRACE** (2025) [TSFM]: importance-gated LoRA module selection. The fine-tuned prediction head tends to overfit, and reconstructing the head beats linear probing. Retained LoRA modules vary by task and horizon — [arXiv 2503.16991](https://arxiv.org/pdf/2503.16991).
- **TSFMs for process-model forecasting** (Dec 2025) [TSFM]: Chronos-2, Moirai-2.0 and TimesFM-2.5 improve strongly over zero-shot with both LoRA and full FT. Limited task data raises the overfitting risk of full FT — [arXiv 2512.07624](https://arxiv.org/html/2512.07624v1).
- **Building-energy probabilistic forecasting with TSFMs** (2025): uses LoRA as the PEFT choice — [arXiv 2506.00630](https://arxiv.org/pdf/2506.00630) [snippet].
- **FETS** (2026) [TSFM, energy]: rank, lr, steps and target modules barely change the outcome; the dataset decides — [arXiv 2604.22328](https://arxiv.org/html/2604.22328v2) (via [sft_peft]).
- **PaCTS** and **MixFT** [TSFM]: weight-space baselines lost to zero-shot on unseen data. Grouped LoRA > shared > per-dataset — [arXiv 2609.34786](https://arxiv.org/html/2609.34786); [arXiv 2603.02840](https://arxiv.org/html/2603.02840v1).
- **Gated-LoRA** [TSFM]: Source-LoRA on Chronos-2 vs zero-shot. The gated version beat MSMI zero-shot by −6.24% MAE and −7.99% RMSE, so most of that is plain LoRA (−3%) plus the gate (−3%) — [arXiv 2608.11359](https://arxiv.org/html/2608.11359).
- **Official TimesFM repo:** the README states that a fine-tuning example using HF Transformers + PEFT (LoRA) was added on 9 Apr 2026 under `timesfm-forecasting/examples/finetuning/`. I could not open the directory to confirm the target version or config — [github.com/google-research/timesfm](https://github.com/google-research/timesfm). Earlier PR #104 offered full FT, linear probing, LoRA and DoRA — [PR #104](https://github.com/google-research/timesfm/pull/104).
- **Chronos-2** has native LoRA via AutoGluon and Moirai 2.0 has research-only LoRA, according to a secondary blog — [tsfm.ai blog](https://tsfm.ai/blog/lora-peft-time-series-foundation-models) [secondary, unverified].
- **LLM context:** "LoRA learns less and forgets less" — [arXiv 2405.09673](https://arxiv.org/abs/2405.09673) (via [sft_peft]). Intruder dimensions — [arXiv 2410.21228](https://arxiv.org/abs/2410.21228).

### Inferences
- Our planned grid already spans what the TSFM literature compares: head-only, frozen head, resblock probe, LoRA targets, full FT at 1e-6/1e-5, partial ends, EMA and WiSE. **The PEFT-variant literature adds no new axis that has evidence of beating tuned LoRA.**
- The two changes this note recommends adding are both protocol fixes:
  - (1) rsLoRA-consistent α for the rank sweep.
  - (2) Compare variants at their own best lr. In practice: run any new variant (LoRA+, prompts, gate) at ≥2 lrs, not just 1e-4, following Lee et al. and "Beware of batch size".
- Our batch size is fixed at a global 16, so "Beware of the batch size" mainly warns against importing LLM-tuned lrs.
- If something structural is tried beyond the current grid, the best-matched evidence points to **state-conditioned scaling of the pooled LoRA** (Gated-LoRA), with z = NWP-derived weather regime. It is evaluated seen vs unseen against WiSE-FT α as the state-independent control. It ranks above any clustered or mixture LoRA.

### Gaps
- There is no TSFM benchmark that includes adapters, LoRA+, rsLoRA or SVD-init variants. There is no published PEFT comparison on TimesFM-3 or on PV with NWP covariates.
- The official TimesFM PEFT example's config and target version were not verified.
