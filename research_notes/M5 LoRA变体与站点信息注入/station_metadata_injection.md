# Injecting static station information and solar-geometry priors into TimesFM-3 (+LoRA) for PV day-ahead forecasting

Scope: TimesFM-3 (timesfm 3.0.2, ~330.7M, 20 layers, seq_attn + var_attn with variate RoPE, per-variate RevIN, covariates fed as extra variates), LoRA r8 q/k/v/o + unfrozen head, inputs GHI + temperature (NWP + history), 28 stations, 15-min, ctx 672 / horizon 192, metric 1 − sqrt(mean(((P−P̂)/max(P,0.2·cap))²)).
Evidence labels: [paper] = read in primary source; [abstract] = abstract/summary only; [snippet] = search-result summary, not opened; [TK] = my training knowledge, not re-verified this session; [inference] = my reasoning.

## Q1. Derived time-varying physics channels as extra variates: which survive per-variate RevIN, and what is the evidence?

### Takeaway
Geometry channels (solar elevation, clear-sky power) mostly survive RevIN only as *shape and timing*: their amplitude/offset is normalised away, and clear-sky power and sin(elevation) become close to redundant with each other. The channels that should carry the most new information after RevIN are *ratio* or *data-derived envelope* channels: the clear-sky index of the NWP GHI, and a history envelope (per time-of-day q0.9 of past power). These are not affine transforms of anything TimesFM already sees. The literature supports clear-sky / geometry priors for deep PV models. I found no study that isolates this effect inside a TSFM.

### Cited findings
- TimesFM 3.0 natively supports "past-only and past-and-future dynamic covariates" through `past_only_covariates` / `past_future_covariates` in `predict_batch`. The README mentions no static covariates for 3.0 [paper] — [TimesFM README](https://github.com/google-research/timesfm)
- Normalising by a clear-sky model beat normalising by extraterrestrial irradiance (clearness index) for every model type tested: persistence, linear AR and a neural net. A non-linear NN on the clearness index could still beat linear models on the clear-sky index [abstract] — [Lauret et al. 2022, *Solar*](https://hal.univ-reunion.fr/hal-03964785v1)
- NWPsolarNet (CIGRE 2026) scales PV by a data-derived factor (10th percentile of the clear-sky-irradiance/production ratio) instead of rated power. It reports −4.0% MAE vs capacity normalisation and says the scaling is robust to noisy metadata and degradation [abstract only] — [CIGRE D2-10112-2026](https://www.e-cigre.org/publications/detail/d2-10112-2026-nwpsolarnet-a-scalable-deep-learning-framework-for-medium-term-solar-forecasting-across-europe.html)
- Day-ahead PV with NAM NWP: in feature screening, solar zenith angle and solar radiation had the lowest error scores (<10%), versus >60% for irrelevant features. This is a feature ranking, not an accuracy ablation [snippet] — [OSTI 1810250](https://www.osti.gov/servlets/purl/1810250)
- A 2026 benchmark of 12 ML/DL models for NWP-based day-ahead PV found that "simple ML with hybridization and feature engineering rivals complex DL". Transformers were competitive but not superior, and the best RNN reached a 47.7% skill score [snippet] — [ScienceDirect S266654682600159X](https://www.sciencedirect.com/science/article/pii/S266654682600159X)
- SPIRIT (irradiance, vision-encoder based, not a pure TSFM) uses clear-sky values and tilted irradiance as features. It reports up to 70% better zero-shot transfer than SOTA, with further gains from fine-tuning [snippet] — [arXiv 2502.10307](https://arxiv.org/pdf/2502.10307)
- PV-specific TSFM usage: Chronos-2 and TimesFM-2.5 with covariates beat Prophet by about 1.7–2× under real feedback. Daily resolution, cold start, 440 sites. MAE (kWh/kWp/d): TabPFN-TS 0.514, Chronos-2 0.537, TimesFM-2.5 0.599, Prophet 1.007. The univariate-only TSFMs were far worse: TiRex 1.061, Moirai-2.0 1.109. In this paper, physics (pvlib geometry, transposition, thermal) enters only through a synthetic *history*, not through clear-sky covariates [paper] — [arXiv 2606.07457](https://arxiv.org/html/2606.07457v1)
- Internal survey note: RevIN degraded Solar-Energy in the CycleNet appendix (0.250 vs 0.208) but helped day-length windows in PV-Client. RevIN is unstable when the window spans night zeros, and the survey recommends capacity normalisation plus clear-sky/elevation inputs plus a daytime mask [internal, cites arXiv 2409.18479 and 2406.03808] — `docs/survey_covariate_pv_models.md` line 86; [CycleNet](https://arxiv.org/abs/2409.18479), [PV-Client](https://arxiv.org/abs/2406.03808)
- Chronos-2 diagnostic: feeding the target itself as a covariate cuts solar error sharply (1.3238 → 0.8333, 24-24 setting). The authors argue that benchmark wins "offer limited insight into actual covariate usage" [snippet] — [TSALM 2026 covariate paper](https://helios2.mi.parisdescartes.fr/~themisp/publications/tsalm26-covariateforecasting.pdf)

### Inferences
- [inference] RevIN algebra: each variate is mapped x → (x − μ_ctx)/σ_ctx (TimesFM-3 normalises per variate on the context, iteratively). The consequences:
  - A **constant** channel (lat, lon, alt, cap repeated in time) becomes 0 or NaN-guarded, so the model cannot see it.
  - Any **affine map** of a channel is invisible: clear-sky power = cap·f(geometry) looks identical whatever the cap.
  - Clear-sky power and sin(elevation)⁺ differ only by the air-mass nonlinearity and the night clip, so feeding both is close to redundant. Keep one, or keep both only if the variate budget is cheap.
  - What does survive is the information in the *shape*: exact sunrise, sunset and solar-noon timing on the 15-min grid (longitude shifts noon against the timestamp; latitude and season set day length), and the change in day length from context to horizon. This is how lat/lon "leak in" usefully even though RevIN removes constants: **geometry channels are the RevIN-compatible way to inject coordinates.**
- [inference] The M5 tiny-test backbone config sets `use_linear_detrending=True` (forecast_backbones/m5/backbone.py:83). If the production config also detrends per variate, then any linear trend across the 7-day window (the seasonal drift in daily peak) is removed from every covariate as well. Verify this in the real timesfm3 config.
- [inference] Highest-value new channels, ranked:
  1. **NWP clear-sky index** k* = GHI_nwp / GHI_cs. Clip it to [0, 1.3] and set it to 0 or hold it constant at night. This is a ratio, not an affine transform of GHI, so it gives cloudiness separated from geometry. It is also the direct fix for the "horizontal-GHI clear-sky ceiling" issue noted for station 1027.
  2. **History envelope** (per time-of-day q0.9 over the past 7 days, rolled into the horizon). It is coordinate-free, it encodes the real panel orientation, shading and clipping, and it worked in M8 for the station without coordinates.
  3. **sin(elevation)⁺ or clear-sky power** (already implemented as variant g1).
  4. **Day-of-year / time-of-day encodings** add little: they are smooth, and RevIN keeps only their within-window shape, which the NWP GHI already carries.
  5. **Extraterrestrial irradiance** is near-collinear with clear-sky power after RevIN, so skip it.
- [inference] Variate RoPE makes order matter. **Append new channels after the existing (target, GHI, temp) positions** so that the pretrained relative positions of the channels you already use are unchanged.

### Gaps
- No paper found that ablates clear-sky / geometry channels inside a fine-tuned TSFM (TimesFM, Chronos or Moirai) for PV. Direct evidence is limited to classical NN/irradiance work and abstracts.
- Not verified this session: TimesFM-3's exact RevIN implementation and whether linear detrending is active in the released checkpoint config (the timesfm3 package was not installed locally).

## Q2. Clear-sky-normalised target (predict clear-sky index, multiply back): fit with TimesFM's RevIN

### Takeaway
This is plausible but risky for a pretrained TSFM. It removes the diurnal envelope that TimesFM's pretrained prior handles well, and it introduces divide-by-near-zero noise at dawn and dusk. The metric's max(P, 0.2·cap) denominator makes dawn/dusk errors cheap, which helps. A safer variant is to normalise the target by the **history envelope** or a **capacity-scaled clear-sky power with a floor**, and to compute the loss in power space after multiplying back.

### Cited findings
- Clear-sky-index forecasting is better than clearness-index forecasting for classical and NN models (irradiance) [abstract] — [Lauret et al. 2022](https://hal.univ-reunion.fr/hal-03964785v1)
- Data-derived clear-sky scaling beats capacity scaling by about 4% MAE for PV [abstract] — [NWPsolarNet](https://www.e-cigre.org/publications/detail/d2-10112-2026-nwpsolarnet-a-scalable-deep-learning-framework-for-medium-term-solar-forecasting-across-europe.html)
- Large PV metadata (capacity, tilt, coordinates) is "frequently missing, erroneous, or set to default values" [snippet] — [Two-stage PV forecasting, arXiv 2603.04132](https://arxiv.org/pdf/2603.04132)
- Fine-tuning TSFMs on solar is fragile. In the FETS LoRA benchmark, fine-tuning helped 16 of 54 energy series: DK Solar NRMSE 0.648 → 0.555, but PV Hong Kong 0.198 → 0.688 (worse) [snippet, Table E.1] — [FETS, arXiv 2604.22328](https://arxiv.org/pdf/2604.22328)

### Inferences
- [inference] RevIN interaction: dividing P by a clear-sky envelope is *not* affine. It changes the shape of the series that RevIN sees, from a diurnal hump to a noisy step-like ratio near 1 on clear days. That is out of distribution for TimesFM's pretraining, so LoRA r8 must re-learn the dynamics. Expect a zero-shot drop and a need for more steps.
- [inference] A safer form: target = P / max(E, ε·cap), where E is the history envelope or capacity-scaled clear-sky power. Mask night (E small) and multiply back before the pinball loss. Leave the input history in raw power or provide both: keep the raw-power target and add the ratio series as an extra past-only variate. That variant keeps the pretrained prior and adds the information.
- [inference] With a wrong or missing clear-sky model (one station has no coordinates, another has wrong borrowed coordinates), clear-sky-index targets inherit the error multiplicatively. Envelope-based normalisation is immune to coordinate errors. That is a strong argument for E over clear-sky power at this site set.

### Gaps
- I found no paper that fine-tunes a TSFM on clear-sky-index targets vs power targets for PV.

## Q3. Static metadata via conditioning: FiLM/adaLN, hyper/conditional LoRA, station embedding/prefix token, constant variate

### Takeaway
Because of per-variate RevIN, static metadata must enter **after normalisation, in hidden space**. Four mechanisms are viable, each with a published precedent:
- (a) an additive zero-init embedding on the target's patch tokens (ChronosX IIB / UniCA pre-fusion pattern);
- (b) a site token that attends with the series tokens (Tyan-WP prepends a site token; MetaTST uses metadata tokens);
- (c) FiLM/adaLN-zero modulation of per-layer norms;
- (d) a residual output-head correction (ChronosX OIB).

Hypernetwork-generated LoRA is the most expressive and the least suited to 28 stations. Feeding metadata as a constant extra variate does not work under RevIN.

### Cited findings
- **ChronosX** (AISTATS 2025): the Input Injection Block updates pretrained token embeddings, f = h_emb(z) + FFN(ReLU(h_emb·W_emb ⊕ x·W_cov)). The Output Injection Block adds FFN(ReLU(h_out·W_out ⊕ x_future·W_cov)) to the logits; for point/patch models, ẑ + FFN(…⊕ ẑ·W_p). The new parameters are trained from scratch, either as adapter-only or with full fine-tuning (FF). It extends to TimesFM ("TimesFMX") by patching covariates to the same patch length. Results:
  - ChronosX beats Chronos-Small zero-shot by 22% in WQL and MASE on synthetic data.
  - On 18 real covariate datasets (mostly energy, including 15-min), the adapter-only variants beat their zero-shot counterparts.
  - The full-FF variants underperformed several baselines, attributed to data scarcity.
  - For TimesFMX, FF slightly *hurt* on complex synthetic data.
  [paper, read in full text] — [arXiv 2503.12107](https://arxiv.org/abs/2503.12107)
- **UniCA**: the backbone is frozen. Static covariates S are embedded by a new layer ρ and used as the *context of conditional attention pooling* (GRN + GLU). Pre-fusion (past covariates) is added to target tokens before the encoder; post-fusion (future covariates) is concatenated with the encoder output, followed by self-attention, then the predictor. Backbones: Chronos-Bolt, TimesFM, Time-MoE, Moirai. A linear homogeniser (d_het = 4) is enough. The frozen backbone beat full FT of Chronos-Bolt (Fig. 4b). Reported gains: −6.5% error for TimesFM and −5.9% for Chronos-Bolt on the image-covariate MMSP benchmark [paper] — [arXiv 2506.22039v2](https://arxiv.org/html/2506.22039v2)
- **ContextFormer**: categorical and continuous metadata pass through FC encoders and a small transformer, then are injected into a pretrained forecaster (PatchTST, iTransformer) by **cross-attention, with history hidden states as queries**. The new weights are **zero-initialised** so the model "matches the base model at the start". Training is two-stage: base frozen except the final layer. Results: ECL (user ID + time) −42.1% MSE / −28.1% MAE on average; average MSE gains about 13.9% (PatchTST) and 15.5% (iTransformer). No solar data [paper] — [arXiv 2410.12672v5](https://arxiv.org/html/2410.12672v5)
- **Tyan-WP** (wind-power FM, 2026): the site embedding is e_site = LN(W_s[W_g·(sin φ, cos φ, sin λ, cos λ); Emb_terr; Emb_eco] + b). It is **prepended as a site token**, "so that site priors broadcast to all patch tokens through self-attention". Ablation on zero-shot unseen sites: removing the site embedding raises MAE by 1.8%, RMSE 1.2%, CRPS 1.6% (127 WTK sites); on Kelmarsh (6 turbines, same farm) MAE +0.9%, RMSE +2.1%. The gain is small and scales with geographic diversity [paper, read in full text] — [arXiv 2606.08630](https://arxiv.org/abs/2606.08630)
- **MetaTST** turns metadata into templated text, encodes it with an LLM into metadata tokens, and concatenates them with series tokens in a transformer encoder [snippet] — [arXiv 2410.03806](https://arxiv.org/abs/2410.03806)
- A 2026 preprint states that published pretrained time-series transformers have no metadata-conditioning channel ("fine-tuning sharpens that prior but does not introduce a metadata-conditioning channel"). It proposes a learned metadata head per backbone [snippet] — [RAID, arXiv 2606.16925](https://arxiv.org/pdf/2606.16925)
- **adaLN-Zero** (DiT) predicts per-block scale/shift/gate from the condition, (1+γ)·LN(h)+β with a gated sublayer output. The projection is zero-initialised so each block starts as identity. Some later works deliberately use default init instead [snippet] — search summary; original [DiT, arXiv 2212.09748](https://arxiv.org/abs/2212.09748) [TK]; FiLM [Perez et al. 2018, arXiv 1709.07871](https://arxiv.org/abs/1709.07871) [TK]
- **Hypernetwork LoRA**:
  - HyperLoRA (EMNLP Findings 2024) generates LoRA parameters from task information for cross-task generalisation [snippet] — [ACL Anthology 2024.findings-emnlp.956](https://preview.aclanthology.org/watermark/2024.findings-emnlp.956)
  - "Task-Agnostic Low-Rank Adapters for Unseen English Dialects" (a hypernetwork conditioned on dialect feature vectors, aimed at *unseen* dialects) is the closest analogue to coords-to-LoRA for unseen stations [title from search; content TK] — [arXiv 2311.00915](https://arxiv.org/pdf/2311.00915)
  - HypeLoRA: a shared hypernetwork produces A and B from per-layer embeddings, which cuts parameters [snippet] — [arXiv 2603.19278](https://arxiv.org/pdf/2603.19278)
  - Text-to-LoRA: zero-shot LoRA generated from a text description [snippet] — [ICML 2025 slides](https://icml.cc/media/icml-2025/Slides/43471.pdf)
- **TimesFM XReg** (2.x): an in-context linear regression on covariates, with two modes: "xreg + timesfm" (default; TimesFM first, then XReg on residuals) and "timesfm + xreg". Third-party skill docs list static_numerical and static_categorical types, but I found no verified call example for static numerics [snippet; third-party] — [AI Horizon Substack](https://aihorizonforecast.substack.com/p/timesfm-25-external-covariates-a), [skill doc](https://cdn.jsdelivr.net/npm/@saibolla/ada@0.1.3/skills/timesfm-forecasting/references/data_preparation.md). The v1 approach was ridge on covariates, then TimesFM on residuals [snippet] — [arXiv 2605.15752](https://arxiv.org/pdf/2605.15752). A preprint claims TimesFM-2.5's default covariate interface is "memoryless" [snippet, unverified] — [alphaXiv 2610.08118](https://www.alphaxiv.org/abs/2610.08118)

### Inferences
- [inference] **A constant extra variate fails.** RevIN maps it to 0 (σ=0). Even a hand-made non-constant encoding (e.g. lat·sin(hour)) loses its amplitude to RevIN. The only "variate-route" ways to carry static information are (i) the geometry channels (Q1), whose shape encodes coordinates, and (ii) a **pseudo-variate token injected after RevIN at the embedding level**, i.e. a site token with no normalisation.
- [inference] **Where to put it in TimesFM-3:**
  - (a) **Additive token bias**: h_tok ← h_tok + W_up·MLP(s), applied at the output of `pre_transformer_resblock` on the target variate (optionally all variates), with W_up zero-initialised. Cheapest: about 0.1–1M parameters, and the model starts identical to the current checkpoint. This is ChronosX IIB without covariates, or UniCA pre-fusion with only static context.
  - (b) **Site pseudo-variate in var_attn**: append one extra "variate" whose token at every patch is e_site (the Tyan-WP site-token idea, adapted to TimesFM's axial attention). var_attn then lets the target attend to it at each time patch. Caveat: a zero vector still takes softmax mass, so zero-init does **not** give exact identity. Gate its value path, h + g·Attn_site with g = 0 at init, or mask it at step 0. It costs one extra variate (about +1/(C+1) var_attn FLOPs) and takes a RoPE slot: place it last.
  - (c) **FiLM/adaLN-zero per layer**: γ_l, β_l = W_l·MLP(s) with W_l = 0, applied to the RMSNorm outputs before seq_attn/var_attn/FFN. 20 layers × 2·d parameters per norm. More expressive than (a), but with 28 stations it is easy to overfit; use weight decay plus metadata dropout.
  - (d) **OIB-style head correction**: q̂ = q̂_base + FFN([h_out, s, future geometry]), zero-init last layer. This works in normalised space and is simple. It is good for station-specific bias such as clipping and inverter limits.
- [inference] **Hyper/conditional LoRA**: full hypernetwork LoRA (s → A, B) with 28 training stations is badly under-determined. A lighter **conditional-gate LoRA**, ΔW = B·diag(1 + tanh(G·MLP(s)))·A with G zero-init, adds about r parameters per layer per condition and degrades gracefully to the plain shared LoRA. Prefer it over full HyperLoRA if (a)–(d) show signal.
- [inference] **Feature vector s**: (sin/cos lat, sin/cos lon or lon offset relative to the timezone meridian, alt/1000, log cap, has_coords flag, history-derived stats such as envelope peak/cap and daily energy/cap). The history-derived stats are coordinate-free and available for every station, including the no-coords one. Use a metadata dropout of about 0.2–0.3 (replace s by a learned "unknown" vector) so that a valid no-metadata mode exists.

### Gaps
- No published experiment found that adds FiLM/adaLN or a site token to TimesFM specifically, and none comparing additive embedding vs site token vs FiLM in a TSFM. The UniCA appendix ablation on static covariates (G.4) was not retrievable in my fetch.
- Gains from static site embeddings in the best domain analogue (Tyan-WP) are modest (about 1–2%). ContextFormer's large gains come from ECL, where the metadata is a user ID on a large panel, not from geographic coordinates.

## Q4. Station-cluster LoRA vs single LoRA

### Takeaway
There is weak but relevant evidence that sub-domain LoRAs can beat a single LoRA (MixFT). With 28 stations, cluster LoRAs (2–4 clusters by climate or envelope shape) split already-small data. This is a low-priority experiment, and a conditional-gate LoRA (Q3) is the smoother alternative.

### Cited findings
- MixFT (2026) argues that per-dataset LoRAs are suboptimal. It re-partitions fine-tuning data into latent sub-domains with a Bayesian mixture on TSFM embeddings, trains one LoRA per sub-domain, and at inference selects the LoRA from the context's inferred sub-domain [abstract/snippet] — [arXiv 2603.02840](https://arxiv.org/abs/2603.02840)
- MoLA uses a mixture of LoRA experts with partial parameter sharing, but across forecast steps, not domains [snippet] — [arXiv 2505.17872](https://arxiv.org/abs/2505.17872)
- LoRA on TSFMs (Lag-Llama, Moirai, Chronos) for out-of-domain modalities [snippet] — [arXiv 2405.10216](https://arxiv.org/html/2405.10216v1)
- In ChronosX, lower-capacity adapters beat full FT in data-sparse regimes [paper] — [arXiv 2503.12107](https://arxiv.org/abs/2503.12107)

### Inferences
- [inference] Cluster assignment should be data-based (envelope shape, latitude band, NWP bias regime), and inference-time routing must work for unseen stations. MixFT's embedding-based routing does; ID-based routing does not.
- [inference] Given the M8 lesson that variants are close to seed noise, cluster LoRA needs at least 3–5 seeds and a leave-stations-out split to show a real effect.

### Gaps
- No PV-specific evidence on cluster-wise adapters for TSFMs.

## Q5. Spatial / neighbour information for day-ahead

### Takeaway
For day-ahead, neighbours' target-day actuals are unavailable. What remains is neighbours' *history* (already summarised by the station's own history), NWP fields at neighbouring grid points, and neighbour-informed metadata (e.g. borrowing an envelope for a station with no history). Expected value is low relative to cost.

### Cited findings
- Irradiance forecasting at sites without on-site data selects training sites in a learned metric space built from exogenous variables including lat/lon. "Geographic location does not necessarily coincide with the location in the new metric space" [snippet] — [Renewable Energy 2020, S0960148120301142](https://www.sciencedirect.com/science/article/abs/pii/S0960148120301142)
- Chronos-2's group attention shares information within groups across the batch axis, so related series or covariates can be grouped. With covariates, fev-bench skill rises 40.9% → 47.0% for Chronos-2 and 34.1% → 42.5% for TabPFN-TS (a poster gives 39.9% as the baseline) [snippet] — [Chronos-2, arXiv 2510.15821](https://arxiv.org/html/2510.15821v1)

### Inferences
- [inference] A practical neighbour use is **metadata repair**, not a model input: for the station with wrong borrowed coordinates, estimate effective longitude from the solar-noon timing of its clear days and latitude from day length. Patents describe solar noon (midpoint of the production day) → longitude and day length → latitude via the sunrise equation [snippet] — [US10365309](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10365309), [US10564315](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10564315). pvanalytics fits a quadratic to each day to find the daily-peak time, but uses it for orientation, not location [doc] — [pvanalytics infer_orientation_daily_peak](https://pvanalytics.readthedocs.io/en/stable/generated/pvanalytics.system.infer_orientation_daily_peak.html)
- [inference] Neighbour NWP (a k×k grid patch around the site) as extra variates fits TimesFM's variate axis, but it multiplies variates and var_attn cost, and it is a separate covariate study.

### Gaps
- No day-ahead PV TSFM paper found that uses spatial graphs.

## Q6. Evidence on TSFMs with covariates and static features (ChronosX, Moirai, XReg, TTM, Chronos-2, PV-specific)

### Takeaway
Covariate adapters on frozen or LoRA'd TSFMs reliably beat zero-shot when the covariates are informative: ChronosX, UniCA, TimesFMX. Full fine-tuning often does not beat adapters on small data. Native static support is rare: TabPFN-TS takes static inputs, TTM handles exogenous inputs at fine-tune time, and Moirai's any-variate attention is order-invariant with no static channel found. PV-specific evidence for TSFMs with covariates exists only for zero-shot and cold-start settings.

### Cited findings
- **ChronosX / TimesFMX**: adapter-only beats zero-shot on 18 real covariate datasets; FF underperforms on sparse data [paper] — [arXiv 2503.12107](https://arxiv.org/abs/2503.12107)
- **UniCA**: frozen backbone better than full FT for Chronos-Bolt; static covariates condition the attention pooling [paper] — [arXiv 2506.22039v2](https://arxiv.org/html/2506.22039v2)
- **TTM**: pre-trained channel-independent, then fine-tuned with channel mixing over targets and exogenous channels. An optional exogenous mixer handles future-known variables. The backbone is frozen while the decoder and head are trained. I found no explicit static-categorical handling [snippet] — [TTM, arXiv 2401.03955](https://www.arxiv.org/abs/2401.03955), [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/874a4d89f2d04b4bcf9a2c19545cf040-Paper-Conference.pdf)
- **Moirai**: flattens variates into one sequence with binary learnable attention biases (within- vs cross-variate), so it is invariant to variate order, unlike TimesFM-3's variate RoPE. No static-covariate mechanism found. Moirai fine-tuning used LoRA on q/k/v plus the head and soft prompts, with very small learning rates (5e-6 / 5e-7) [snippet] — [Moirai, arXiv 2402.02592](https://arxiv.org/abs/2402.02592), [Salesforce blog](https://www.salesforce.com/blog/moirai/)
- **PV cold start**: only TabPFN-TS injects static plant metadata (peak power, tilt, azimuth, technology), and it was best under real feedback (MAE 0.514, WAPE 15.46%). Chronos-2 was close behind without static inputs (0.537, 16.14%); TimesFM-2.5 + XReg scored 0.599 / 18.11%. The authors' conclusion: "accuracy is driven more by the availability of plausible temporal context than by the specific generator" [paper] — [arXiv 2606.07457](https://arxiv.org/html/2606.07457v1)
- **SolarFM** (Applied Energy 2026): a spatio-temporal irradiance forecaster built on a TSFM, with 2.6–15.1% gains over baselines [abstract] — [ScienceDirect S0306261926001844](https://www.sciencedirect.com/science/article/abs/pii/S0306261926001844)
- A TFT on Open Climate Fix Quartz PV uses capacity, lat, lon, tilt and orientation as static features (GSoC report, not peer-reviewed) [snippet] — [gist](https://gist.github.com/roshnaeem/f3f66bc7e76a4eff79b07568b99819f4)

### Inferences
- [inference] The cold-start result supports the M8 finding that a **history-based prior** (envelope) matters more than raw metadata. A static-aware model (TabPFN-TS) won only narrowly over a static-free one (Chronos-2) that had good context.
- [inference] ChronosX's finding that FF hurt TimesFMX is consistent with keeping the trunk frozen or LoRA'd, and adding small zero-init injection modules.

### Gaps
- No paper found on TimesFM-3 (3.0.x) fine-tuning with covariates, PV or otherwise. The TimesFM README added a HF+PEFT LoRA example in April 2026 but reports no covariate results.

## Q7. 28 stations: is a per-station embedding over-parameterised, and what generalises to unseen stations?

### Takeaway
An ID embedding is fine for seen stations if it is tiny (8–16 dims, projected, with dropout). It gives nothing for unseen stations and will memorise. Coordinate- and history-derived conditioning generalises; ID does not. Use coords + history stats as the main conditioner, ID only as an optional residual with ID dropout, and always run leave-stations-out validation.

### Cited findings
- Tyan-WP builds the site embedding only from generalisable metadata (sin/cos coords, terrain and ecoregion categories), and evaluates zero-shot on unseen sites with a +1–2% error penalty when it is removed [paper] — [arXiv 2606.08630](https://arxiv.org/abs/2606.08630)
- ContextFormer's ECL gain relies on user ID over a large panel (hundreds of series) [paper] — [arXiv 2410.12672v5](https://arxiv.org/html/2410.12672v5)
- Metadata is often wrong or missing in PV fleets [snippet] — [arXiv 2603.04132](https://arxiv.org/pdf/2603.04132)

### Inferences
- [inference] With 28 stations, an MLP on about 8 metadata features has far fewer effective degrees of freedom than 28 free d-dim vectors. Still, 28 points in coordinate space is a sparse sample, so expect the conditioner to learn a low-dimensional "regional bias", not a physical mapping.
- [inference] Handling the three special cases:
  - **No coordinates**: has_coords = 0, plus a learned unknown vector (trained via metadata dropout), plus the history envelope.
  - **Wrong coordinates**: repair them from data (Q5), or set has_coords = 0.
  - **Unseen station at test time**: coords and history stats only; the ID path is disabled.

### Gaps
- No PV-specific study found that quantifies ID-embedding vs coordinate-embedding generalisation on held-out sites.

## Q8. Practical recommendation: ranked experiments for TimesFM-3 + LoRA (cheapest/safest first)

### Takeaway
Run physics/envelope *channels* first, because they are RevIN-compatible and already supported. Then add a zero-init post-normalisation static injector. Hyper/cluster LoRA comes last. Each experiment should be judged with 3–5 seeds, on seen vs held-out stations, with night-zeroed scoring (same as production).

### Cited findings (design precedents)
- Zero-init so that training starts at the base model: [ContextFormer](https://arxiv.org/html/2410.12672v5); adaLN-Zero ([DiT](https://arxiv.org/abs/2212.09748) [TK])
- Adapter-only beats full FT on sparse data: [ChronosX](https://arxiv.org/abs/2503.12107), [UniCA](https://arxiv.org/html/2506.22039v2)
- Site token prepended and broadcast by attention: [Tyan-WP](https://arxiv.org/abs/2606.08630)

### Inferences (concrete plan)
1. **E1: geometry channels (already implemented as g1) + NWP clear-sky index channel.**
   - Design: append [sin(elev)⁺ or clear-sky power, k*_nwp = clip(GHI_nwp / GHI_cs, 0, 1.3), night → 0] as past_future variates *after* GHI and temp.
   - Zero-shot is not identical, because the new variates enter var_attn. Mitigate by keeping the original variate order.
   - Confirm: gain on clear and partly cloudy days, and at stations where NWP GHI over- or under-shoots (e.g. 1027).
   - Refute: no gain over g1 alone, meaning RevIN redundancy dominates.
   - Cost: retrain only.
2. **E2: history-envelope channel** (per-ToD q0.9 of the past 7 days, rolled forward 2 days as a past_future variate).
   - It needs no coordinates, so it covers the no-coords and wrong-coords stations.
   - Confirm: the biggest gain at the no-coords station, and gains not reduced by having E1 present.
   - Refute: no gain on top of E1 means geometry already covers it.
   - Cost: feature code only (port it from the M8 h4 history_envelope).
3. **E3: zero-init additive static embedding (ChronosX-IIB / UniCA style).**
   - Design: MLP(s) → W_up (zero-init) → added to the target-variate tokens after `pre_transformer_resblock`.
   - s = (sin/cos lat, lon offset, alt, log cap, has_coords, envelope peak/cap, mean daily energy/cap).
   - Training: metadata dropout 0.25, LoRA as now.
   - Starts identical to the current model.
   - Confirm: gain on seen stations and no loss on held-out stations, in a leave-k-stations-out split.
   - Refute: gain on seen stations only, which is memorisation.
   - Cost: about 1 day of code and less than 1M parameters.
4. **E4: FiLM/adaLN-zero on per-layer norms, or a gated site pseudo-variate in var_attn.**
   - Use the same s, zero-init γ/β or a zero-init gate g on the site-token attention output.
   - Try it only if E3 shows a signal and E3's gain saturates.
   - Confirm/refute: as for E3, and it must beat E3 by more than seed noise.
5. **E5: conditional-gate LoRA (ΔW = B·diag(1 + tanh(G·MLP(s)))·A, G = 0 init) or 2–4 cluster LoRAs.**
   - Most ambitious and highest overfitting risk with 28 stations.
   - Confirm: a held-out-station gain over E3.
- Not recommended:
  - static values as constant variates (zeroed by RevIN);
  - full HyperLoRA with 28 stations;
  - clear-sky-index *target* replacement before E1/E2 are measured (high risk, Q2);
  - spatial graphs for day-ahead.

### Gaps
- All expected effect sizes are extrapolated from M8 and analogous domains (wind FM: 1–2%; covariate adapters: 5–22%). There is no direct TimesFM-3 PV measurement.
