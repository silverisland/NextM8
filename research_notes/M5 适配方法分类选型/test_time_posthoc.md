# Test-time adaptation and post-hoc (output-level) correction for M5 (TimesFM-3, day-ahead PV, 南网 score)

Scope reminder: M5 = TimesFM-3 (~330M), 9 quantiles × 192 × 15-min steps, issued daily ~10:00 for D+1 (D+1 = steps 55–151 in the old replay code), 28 stations, NWP = GHI + temperature. Score = (1 − sqrt(mean(((P_real − P_pred)/max(P_real, 0.2·cap))²)))×100 per day, **all points including nights** (`pipelines/nanwang_metric.py`, per `reports/M5 基础模型微调方法调研.md` line 29). Noise band ±0.6. Existing decision layer: `forecast_backbones/m5/decision.py` (physics_clip, enforce_monotone, p_star / p_star_l1 / median / global_shrink / quantile:τ, by_step rules).

Feedback timing in our setting (derived from the old replay code, `git show 966aec2^:forecast_backbones/m5/conformal.py`): at issue time D 10:00, the newest *fully scored* forecast is the one issued D−2 (for D−1), so `lag_days ≥ 2` was enforced. The forecast issued D−1 has only D 00:00–10:00 observed (mostly night + morning ramp). One labelled forecast per station per day ⇒ ~365 labelled issues/station/year, ~10k across 28 stations.

## Q1. Test-time adaptation for TSFMs (ELF, TAFAS, 2025–2026 methods) and behaviour with ~1–2-day feedback delay

### Takeaway
Published TSFM TTA methods (ELF, TAFAS, PROCEED, δ-Adapter, BC-ACI) all report gains, but on stride-1 streaming benchmarks without future covariates; with one labelled day-ahead issue per station per day and a 2-issue delay, their fast-adaptation machinery mostly reduces to "rolling per-station output correction with a 2-day lag", which is cheap to build ourselves inside the decision layer. ELF's *online expert weighting* (fast/slow exponential weights between FM and a second forecaster) is the one transferable component; gradient-based TTA through the 330M backbone (TAFAS-style) is not worth it.

### Cited Findings
- **ELF (Lee et al., ICML 2025)**: frozen FM + a Fourier-domain linear forecaster (keeps lowest α=0.9 of frequencies, complex weight matrix W) solved in closed form by ridge (λ=20), inverse updated with Woodbury so past data is not stored; final forecast = weighted average of FM and ELF-forecaster, weights from a slow exponential weighter (full loss history) + fast weighter (last B=5 updates) + merge weighter, η=0.5; first 5 updates use the FM only — [arXiv 2502.12920](https://arxiv.org/html/2502.12920v3)
- ELF updates every M=200 steps using losses of the most recent forecasts, **only after their true values are observed** (adaptation lags by up to one window); more frequent updates help at more compute; ~0.38 s/update on 2 CPUs (ETTh1, H=96), ~10× faster than TTM+TAFAS (3.76 s), ~2506× faster than continual fine-tuning of TTM — [arXiv 2502.12920](https://arxiv.org/html/2502.12920v3)
- ELF on the **Solar** dataset with TimesFM: MASE 1.097→1.039 (−5.3%, H=30), 1.201→1.104 (−8.1%, H=96), 1.248→1.148 (−8.0%, H=336); Moirai gains larger (−0.12 to −0.16 MASE); Chronos/TTM/VisionTS −0.02 to −0.09. Uniform 0.5 averaging performs poorly — the weighting is essential. Context fixed at 520, hyperparameters not tuned per dataset; univariate (no covariates) — [arXiv 2502.12920](https://arxiv.org/html/2502.12920v3)
- **TAFAS (Kim et al., AAAI 2025)**: frozen forecaster with an input and an output Gated Calibration Module (L×L or H×H linear map + bias, tanh gate per variable, zero-initialised), trained by gradient on "partially observed ground truth" (POGT) of length p = dominant period (6–96 steps across datasets) plus full-GT loss once available; adapting the backbone/normalisation instead of GCMs gives smaller gains or hurts. The paper states POGT "does not eliminate the adaptation delay"; no runtime reported in the paper — [arXiv 2501.04970](https://arxiv.org/html/2501.04970v1)
- TAFAS gains: +4.95–6.30% MSE at H=336/720 on iTransformer/DLinear; Chronos at H=96 reductions ~21–48% (e.g. Chronos-base ETTh2 1.357→0.708); "up to 45%" headline — on ETT/Exchange-type benchmarks with strong distribution shift — [arXiv 2501.04970](https://arxiv.org/html/2501.04970v1); [AAAI-25](https://ojs.aaai.org/index.php/AAAI/article/view/33965)
- **PROCEED (Zhao & Shen, KDD 2025)** explicitly argues that online-learning methods wrongly assume prompt ground truth; labels arrive only after the horizon, which itself creates a train–test drift gap; it estimates drift between recent training samples and the test sample and generates parameter adjustments with a generator trained on synthetic drifts — [arXiv 2412.08435](https://arxiv.org/abs/2412.08435v2)
- **δ-Adapter / "The Forecast After the Forecast" (Liang et al., ICLR 2026)**: post-hoc, backbone-agnostic input nudging + bounded output residual correction, with a Quantile Calibrator and Conformal Corrector; claims O(δ) drift bounds and gains "with little extra compute"; I only saw the abstract, no tables — [arXiv 2601.20280](https://arxiv.org/abs/2601.20280); [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6f7d90b1198fec96defd80b5ebd5bc81-Abstract-Conference.html)
- The 2026 TSFM post-training survey (Xie et al.) lists TTA/online methods ELF, TAFAS, CANDI, AdaNODEs, DynaTTA, RG-TTA and output refinement (δ-Adapter, RefineBridge, TFMAdapter); it states TTA "can be unstable when the test stream contains anomalies, delayed labels, abrupt regime changes, or weak self-supervised signals", that recent work moves to *controlled* adaptation (gates, curated samples, regime detectors), and that a calibrator fit on a fixed model becomes stale if the model is adapted online — calibration should be re-fit as a dependency of TTA — [arXiv 2607.20002](https://arxiv.org/pdf/2607.20002)
- Same survey: forecast refinement is "most valuable when TSFM errors contain stable, target-specific structure such as horizon-dependent bias, scale mismatch, or recurring residual patterns"; main challenge is estimating the residual map from limited data and detecting when to leave the raw forecast unchanged — [arXiv 2607.20002](https://arxiv.org/pdf/2607.20002)
- Earlier internal survey already positioned ELF/TAFAS-style correction as "stage 6, per-station daily corrector" (ridge on [TSFM quantiles, NWP-GHI, envelope power, hour] or clear-sky-index-gated scale/offset) — `reports/M5 基础模型微调方法调研.md` (lines 84, 128, 165)

### Inferences
- ELF's Fourier ridge maps a 520-step context to H outputs (hundreds × 192 complex weights). Under daily issuance we get ~1 labelled pair per station per day, not one per 15-min step; ridge with λ=20 will be dominated by the prior for months. ELF also ignores NWP, which is the main source of day-ahead skill. So ELF's forecaster is mismatched; its *weighting* is not — a fast/slow exponentially weighted mixture of M5 with M8 (or LightGBM) per station, updated with a 2-day lag, is a direct transplant and degrades gracefully (weight → FM if the second expert is bad).
- The ELF Solar gains (5–8% MASE) are vs. a *zero-shot univariate* TimesFM on a dataset with no covariates. For a LoRA-fine-tuned M5 that already sees NWP and the station's own history in context, most systematic bias is already absorbed; residual drift (soiling, curtailment, inverter outages, seasonal NWP bias) is what remains, so expect a fraction of that.
- TAFAS's POGT in our setting is the D 00:00–10:00 segment of the D−1 forecast — mostly night and the sunrise ramp, ~2–3 daylight hours, which is weak evidence about D+1. TAFAS would also need backprop through the 330M model for 28 stations daily. Its input-side GCM duplicates what TimesFM's in-context RevIN already does for level. Not worth it.
- PROCEED is the only method designed around delayed labels, but needs a trained adaptation generator per backbone; too heavy for a +0.x point target.
- Because TimesFM conditions on the last context window every day, it is already a "test-time adapting" model for level/shape; the remaining lever is *output-side* bias/scale/dispersion correction, which is what Q2–Q4 cover.

### Gaps
- No paper evaluates ELF/TAFAS/PROCEED on day-ahead PV with NWP covariates and daily issuance; no measured TTA gain under a 2-day label delay at 1 sample/day.
- Could not read δ-Adapter tables or the TCP (Aich 2026) / DynaTTA / RG-TTA numbers.

## Q2. Online / rolling bias correction for day-ahead PV (per station, per hour, per clear-sky-index regime; Kalman / RLS / MOS)

### Takeaway
Power-side MOS is the evidence-backed place to correct (Mayer & Yang: up to 8.5% RMSE reduction on power vs ≤5.3% on GHI, and GHI correction adds ~nothing once power is corrected), but those gains are measured against *physical model chains*, not a model already trained on power. Short-window Kalman/rolling corrections remove bias fast but over-correct clear-sky days unless conditioned on regime. For M5 the realistic gain is from per-station multiplicative scale conditioned on clear-sky/NWP-cloudiness regime, with shrinkage to a pooled value.

### Cited Findings
- Mayer & Yang (Applied Energy 371, 2024): 3 utility-scale plants in Hungary, 2017–2020, MOS (Lorenz) and KCDE applied at 4 workflow positions; best = quarterly-grouped KCDE, 3 years training, **power-side only**: nRMSE 43.4% → 39.7% (−8.5%); GHI-only post-processing gives up to −5.3%; "if the PV power forecasts are post-processed, only marginal, if any, further error reduction can be achieved by also correcting the GHI forecasts"; model chains re-introduce bias even into unbiased GHI — [Applied Energy 2024 (OA PDF)](https://real.mtak.hu/205425/1/Optimalplacetoapplypost-processinginthedeterministicphotovoltaicpowerforecastingworkflow.pdf)
- Same paper on training length: with 1 year of data a single model for the whole year is best, quarterly close behind, **monthly fitting substantially worse**; only with 3 years does monthly fitting win; KCDE (non-parametric) beats MOS in all cases — [Mayer & Yang 2024](https://real.mtak.hu/205425/1/Optimalplacetoapplypost-processinginthedeterministicphotovoltaicpowerforecastingworkflow.pdf)
- Kalman filter on day-ahead WRF-Solar GHI (Manila): a 3-day training window already cuts MBE by 70–94% but RMSE only 8–12%; optimal window 14 days (JJA) / 42 days (JFM) gives ≥17% RMSE reduction in cloudy periods; KF **underestimates clear-sky GHI** because the bias of dominant cloudy periods is carried into clear days — [Ateneo / Manila Observatory](https://archium.ateneo.edu/manila-observatory/14/)
- NRCan (2011): KF bias removal + spatial averaging gave ~43% RMSE reduction vs persistence and ~15% vs raw NWP; bias removal mattered mostly for the regional aggregate, little for single systems (RMSE 6.4–9.2% of rated power) — [NRCan report](https://www.nrcan.gc.ca/sites/www.nrcan.gc.ca/files/canmetenergy/files/pubs/2011-062.pdf)
- Decaying-average bias correction of WRF-Solar irradiance improved downstream day-ahead solar power forecasts by 15% (Taiwan, transformer model) — reported via search summary of [ETASR article](https://etasr.com/index.php/ETASR/article/download/13866/6057/76379) (not read in full)
- NOAA operational KF bias-correction code computes coefficients from data at the **same time of day** (alternative mode: recent rolling window) — [NCEP kf_luca.f90](https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/aqm.v7.0.11/sorc/AQM-utils/lib/lib.analog/kf_luca.f90)
- Internal: M8's k_nwp and nwp_resid channels are already an implicit power-side MOS; M8 survey §7.5 concluded clear-sky should be a soft prior with per-station envelope calibration — `docs/survey_m8_architecture_review.md` §7.5

### Inferences
- The 8.5% (Mayer & Yang) is an *upper-ish* reference: their raw forecast is a physical chain never fitted to the plant's power. A fine-tuned M5 has already been fitted on these stations' power, so static per-station bias is mostly gone; what remains is drift between training period and test period. A per-station rolling correction is worth testing, but a realistic prior is 1–3% relative RMSE ⇒ roughly +0.2 to +0.7 points at a ~25-point error level (score ≈ 100 − daily relative-RMSE%). Mostly below or at the ±0.6 band on average, but potentially well above it at a few drifting stations (curtailment, capacity change, soiling).
- Kalman's clear-sky underestimation is exactly the failure that would hurt the 南网 metric on high-output days. Condition the correction on regime: bins of NWP clear-sky index k = GHI_nwp/GHI_clear (e.g. clear k>0.7, mixed, overcast) and coarse time-of-day (morning ramp / midday / evening ramp), and make the correction **multiplicative** on power (bias scales with output; additive corrections inject power at night/dawn).
- With ~1 year of validation, Mayer & Yang's training-length result says: do not fit monthly cells; fit per-station × regime (≤3 bins) × ≤3 time-of-day bins, with shrinkage to the pooled-over-stations value (the old conformal used w = D/(D+k), k=15 — reusable).
- RLS/KF vs rolling window: equivalent for a scalar per cell; an exponentially weighted ratio estimator (half-life ~14–30 days, lag 2 days) is the simplest robust form, and the old replay harness design (`window_days=45`, `lag_days=2`, station/pool/shrink controls) already encodes the right protocol.

### Gaps
- No found study reports before/after Kalman or rolling-MOS gains on *day-ahead PV power from a model already trained on power* (as opposed to physical chains or GHI).
- No study found on regime-conditioned (clear-sky-index-binned) rolling correction specifically for TSFM outputs.

## Q3. Quantile recalibration (CQR, ACI, multi-step conformal, isotonic, QRA) and whether calibration improves p* / the score

### Takeaway
Conformal/recalibration reliably improves interval scores (14–17% WIS/Winkler; ~15% CRPS for nonlinear QR post-processing), but the 南网 score is a point metric: calibration helps only through the point extracted from the distribution. Classic ACI only changes width, not centre; the point-relevant pieces are (a) re-centring (bias, as BC-ACI does) and (b) dispersion of the *lower* tail, because p*'s weight 1/max(y,0.2C)² emphasises low outcomes. The old M5/M6 Mondrian one-sided conformal module was built, then deleted without a recorded score; I found no evidence it was ever measured to help.

### Cited Findings
- Renkema, Visser, AlSkaif (Solar Energy Advances 4, 2024): day-ahead PV power (Netherlands), CP on top of linear/RF point models; best = CP with kNN + Mondrian binning after RF, **WIS −14%** vs multiple linear quantile regression; no point-accuracy effect reported — [WUR record](https://research.wur.nl/en/publications/enhancing-the-reliability-of-probabilistic-pv-power-forecasts-usi/)
- BC-ACI (arXiv 2604.13253, preprint): standard ACI "can only adjust the interval width, but not shift the interval center"; adds an EWMA online bias estimate to re-centre scores, with a dead-zone to skip corrections within noise; 688 runs: Winkler score −13–17% under mean/compound shift (Wilcoxon p<0.001), **≈no change on stationary data (ratio 1.002)** — [arXiv 2604.13253](https://arxiv.org/pdf/2604.13253); also summarised in [arXiv 2607.20002](https://arxiv.org/pdf/2607.20002) §7.2
- Multi-step conformal options: Bonferroni per-horizon (Stankeviciute 2021), Bellman conformal inference (Yang, Candès, Lei 2024), AcMCP using serial correlation of multi-step errors (Wang & Hyndman) — [arXiv 2410.13115](https://arxiv.org/pdf/2410.13115)
- Zaffran et al.: ACI suited to time series with general dependence; efficiency depends on the learning rate — via search summary of [arXiv 2202.07282](https://arxiv.org/pdf/2202.07282v1.pdf)
- Mayer, Baran, Lerch, Horat, Yang, Baran (Solar Energy 307, 2026): 7 Hungarian PV plants, 7 post-processing methods on PV-power ensembles; **every** statistical post-processing beats the raw ensemble, non-parametric > parametric, advanced nonlinear QR best (≈14.7% CRPS reduction per search snippet); all methods improve reliability "at the cost of decreased resolution (sharpness)" — [arXiv 2508.15508](https://arxiv.org/abs/2508.15508)
- Isotonic distributional regression is a competitive generic CRPS benchmark but can be beaten by tuned parametric post-processing; per-quantile recalibration can produce quantile crossing that needs rearrangement — [ECMWF IDR slides](https://www.ecmwf.int/sites/default/files/elibrary/2019/18962-isotonic-distributional-regression-idr-powerful-nonparametric-calibration-technique.pdf); DNI calibration abstract (CRPS −20%, crossing caveat) [ICEM 2019](https://icem2019-abstract-submission.p.wemc.currinda.com/days/2019-06-27/abstract/738)
- Achour et al. 2025 (zero-shot TSFM + CP: more data can be reserved for calibration since no training is needed); Temporal Conformal Prediction (Aich 2026: rolling split-conformal + Robbins–Monro) — [arXiv 2607.20002](https://arxiv.org/pdf/2607.20002) §7.2
- Internal history: M6 added "Mondrian one-sided conformal quantile calibration (fit/apply/hierarchical fallback)" (commit 6257b72); M5 added an offline rolling conformal replay (station / class-pool / shrinkage w = D/(D+k), window 45 d, lag 2 d, cells elev×variability×lead; commit 6a44866); M6 then switched to "LoRA only, no conformal" (4231911) and M5 deleted conformal in the 2026-10-14 refactor (966aec2, design doc `docs/superpowers/specs/2026-10-14-m5-lora-finetune-design.md`). No commit message or doc records a conformal score delta — `git log -i --grep=conformal`
- Internal: FETS fine-tuning saw interval coverage collapse 0.94→0.27 (DE load); risk that LoRA narrows M5's spread — `research_notes/M5 基础模型微调方法调研/sft_peft.md` (line 270, 288)

### Inferences
- Mechanism of calibration → score: p* = Σ y·w / Σ w over the quantile function with w = 1/max(y,0.2C)². Lower quantiles get larger weight, so p* sits below the mean; if the lower tail is too narrow (over-confident in high output) p* is too high on cloudy-surprise days, and vice versa. Therefore what matters for the score is (i) the centre/bias and (ii) the lower-tail dispersion, not nominal coverage at 0.1/0.9. Upper-tail calibration is nearly irrelevant (capped at cap and down-weighted).
- Full conformal is the wrong tool shape: it targets coverage, needs per-level D_min samples (the old code needed ⌈3/min(τ,1−τ)⌉ station-days per cell), and its one-sided shifts add a constant δ·p_clear that is undefined at night/for stations without coordinates. A 2–3 parameter **dispersion/scale recalibration selected directly on the 南网 score** captures the point-relevant part: q̃ = a·(m + s·(q − m)), m = median, s = spread factor (lower half only, optionally), a = scale. global_shrink is the special case s=0 on the median.
- If LoRA collapses spread (FETS precedent), p* degenerates toward the median and its +1 point may vanish; a spread factor s>1 restores it. This makes "spread scale" the most valuable recalibration knob after fine-tuning, and it should be re-selected per checkpoint (as p* already is).
- ACI-style online width updates add value only for interval products (not scored); BC-ACI's re-centring is equivalent to the rolling bias correction in Q2 — implement once, not twice.

### Gaps
- No published result shows calibrating quantiles improves a downstream RMSE-type point score; the link is theoretical (Bayes rule under the predictive distribution) — [Gneiting, Making and Evaluating Point Forecasts](https://ar5iv.arxiv.org/html/0912.0902).
- The deleted M5/M6 conformal results (if any were run on the server) are not in the repo; ask the user whether a replay table exists.
- Could not access full CQR-for-PV numbers (MDPI 2026 MTSA-QRN post-calibration values truncated) — [MDPI Sustainability 18(2):739](https://www.mdpi.com/2071-1050/18/2/739).

## Q4. Point-forecast selection under the 南网 metric (per-station / per-horizon / per-regime p* or shrink), overfitting risk, day-block bootstrap

### Takeaway
Theory and solar evidence both say: extract the functional the operator's score elicits (p* already does this) — elicited points beat "best-guess" points under every scoring function tested. The remaining knobs (per-station shrink, per-regime spread) have few parameters each, so they are safe if selected with day-block bootstrap and shrinkage; per-step/per-horizon rules have many and are the main overfitting risk with ~1 year of validation.

### Cited Findings
- Gneiting: if the scoring function is specified ex ante, the forecaster should issue the Bayes rule (the functional for which the score is strictly consistent); expectations, ratios of expectations and quantiles are elicitable — [Gneiting 2011 (arXiv 0912.0902)](https://ar5iv.arxiv.org/html/0912.0902)
- Yang & Kleissl (IJF 39(4), 2023): grid operators specify the scoring function ex ante; points elicited from ECMWF-EPS solar ensembles were **consistently more accurate than best-guess forecasts whatever scoring function was used**; but quality→economic value is non-linear and non-monotone, and optimally elicited forecasts can give lower value under some penalty triggers — [IDEAS record](https://ideas.repec.org/a/eee/intfor/v39y2023i4p1640-1654.html)
- Wang, Yang, Kleissl et al. (RSER 199, 2024): respecting consistency/elicitability when summarising NWP ensembles reduced reserves and plant penalties in an IEEE-30-bus day-ahead scheduling study — [IDEAS record](https://ideas.repec.org/a/eee/rensus/v199y2024ics1364032124001710.html)
- Internal: M8 d2 p* = +1.05 points, the only item clearly above noise (others: scheduling +0.26, 3-seed ensemble +0.3–0.5) — `docs/survey_m8_architecture_review.md` line 61
- Internal: M6 already used per-station routing by **paired day bootstrap**: choose LoRA only if the α-quantile of the LoRA−zero-shot mean difference > min_gain_pct and paired days ≥ min_days — commit 0fb4bc4 message
- Mayer & Yang: with 1 year of data, finer temporal partitioning of post-processing (monthly) hurt vs one yearly model — [Mayer & Yang 2024](https://real.mtak.hu/205425/1/Optimalplacetoapplypost-processinginthedeterministicphotovoltaicpowerforecastingworkflow.pdf)

### Inferences
- Because 南网 score uses max(P_real, 0.2·cap) and a daily sqrt, p* (squared relative error) is the right per-step functional; the daily sqrt and 1/RMSE_d-type day weighting mean a slight further shrink toward lower values can help on bad days — this is what a tuned global/per-station shrink picks up.
- Parameter budget vs data: 28 stations × ~365 days. A per-station scalar (shrink or spread) has ~365 correlated daily scores to estimate one number — fine with shrinkage to the global value. Per-station × per-horizon-step rules (by_step with 192 steps) or per-station × regime × hour cells are where selection noise exceeds the ±0.6 band.
- Recommended protocol: (1) candidate set small and pre-declared (median, p*, p*+global shrink, p*+per-station shrink with James–Stein/D/(D+k) pooling, p*+spread s∈{0.8,1,1.2,1.5}); (2) score per station-day; (3) moving/day-block bootstrap over days (blocks of 7 days to respect weather autocorrelation), paired across candidates; (4) adopt a per-station deviation from the global choice only if the paired lower α-quantile of gain > 0 (M6 routing rule); (5) confirm on the held-out split (seen/unseen) — never select on the same days you report.
- by_step should stay coarse (e.g. D+1 block vs. first-16-step ultra-short block, as M8 already does), not per-step.

### Gaps
- No literature quantifies overfitting of per-station point-functional selection with 1 year of data in PV; the bootstrap protocol is standard practice, not PV-validated.

## Q5. Night handling and physics clipping for TSFM outputs

### Takeaway
Under this metric every night step with P_real=0 is scored with denominator 0.2·cap, so any positive night output is pure loss; zeroing nights is free and production already does it. Do it with a station-specific, data-derived daylight mask (historical envelope), not a geometric clear-sky ceiling. Keep capacity clipping with the fixed per-station cap_mw; do not use clear-sky ceilings as hard caps.

### Cited Findings
- Metric includes all points including nights (denominator max(P_real, 0.2·cap)), lower bound 0 — `reports/M5 基础模型微调方法调研.md` line 29 / `pipelines/nanwang_metric.py`
- M8 lessons: clear-sky ceiling hurt tracking/bifacial plants; the ceiling silently zeroed nights until commit 86a9c8c (evaluations not aligned); production zeroes nights for all models; fixed per-station cap from `station_meta` cap_mw is the scoring convention — user memory / `docs/survey_m8_architecture_review.md` §7 (prescription: estimate sunrise/sunset and per-time-slot envelope from station history; geometry cannot be a hard ceiling)
- `decision.physics_clip` already supports `clear_sky_cap_ratio=None` (M5 default: capacity-only upper bound, no implicit night zeroing), `night_ceiling`, `clip_negative`, and an explicit `night` mask — `forecast_backbones/m5/decision.py`
- Kalman/regime-blind corrections bias clear-sky days low — [Manila KF study](https://archium.ateneo.edu/manila-observatory/14/)

### Inferences
- Night mask design: night[t] = True if the station's observed power at the same 15-min slot was ≤ ε·cap on (nearly) all of the last N=14–30 days (lagged by 2 days) — a per-station historical envelope. This handles missing/incorrect coordinates (1026-type stations) and tracking plants, and moves naturally with season. Optionally OR with "solar elevation < −2°" when coordinates are trusted. Twilight steps should not be zeroed if any recent day had output there.
- Order of operations in the decision layer: enforce_monotone → capacity clip → (optional correction a·, s·) → clip again to [0, cap] → night zero → point selection. Corrections applied after clipping must be re-clipped; night zeroing must be last so multiplicative/additive corrections cannot resurrect night power.
- Evaluate all variants with `diagnose --night-zero` (production convention) so night zeroing is not counted as a model gain.

### Gaps
- No external literature specifically on night handling for TSFM PV outputs; recommendations are derived from metric structure and internal M8 findings.

## Q6. Final selection: adopt / try later / reject, with implementation sketch and expected gain vs ±0.6 band

### Takeaway
Adopt only cheap, decision-layer, few-parameter items that are re-selectable per checkpoint (p* everywhere, data-derived night mask, spread-scale recalibration, per-station shrink with pooling). Try a per-station regime-conditioned rolling multiplicative correction and online M5/M8 expert weighting next. Reject gradient-based TTA, ELF's Fourier forecaster, PROCEED, and coverage-targeted conformal as score levers.

### Cited Findings
- Evidence anchors: p* +1.05 (M8, internal); power-side post-processing up to −8.5% RMSE vs physical chain ([Mayer & Yang 2024](https://real.mtak.hu/205425/1/Optimalplacetoapplypost-processinginthedeterministicphotovoltaicpowerforecastingworkflow.pdf)); ELF −5–8% MASE on Solar for zero-shot univariate TimesFM ([arXiv 2502.12920](https://arxiv.org/html/2502.12920v3)); BC-ACI ≈0 on stationary data, −13–17% Winkler under shift ([arXiv 2604.13253](https://arxiv.org/pdf/2604.13253)); conformal WIS −14% for day-ahead PV ([Renkema et al. 2024](https://research.wur.nl/en/publications/enhancing-the-reliability-of-probabilistic-pv-power-forecasts-usi/))

### Inferences (decision table; gains are my estimates in 南网 points, not measured)

| # | Method | Verdict | Implementation sketch (our decision layer) | Expected gain |
|---|---|---|---|---|
| 1 | p* as default point on every M5 checkpoint (z0 and LoRA) | **Adopt** | already exists: `select_point(default="p_star")`, keep `by_step` coarse (ultra-short block → `p_star_l1`) | +0.5–1.0 (M8 got +1.05); above band |
| 2 | Data-derived night mask + capacity-only clip | **Adopt** | build `night` per station from lagged 14–30-day observed envelope; `physics_clip(clear_sky_cap_ratio=None, capacity_cap=True, night=mask)`; night zero last | 0 to +0.3 vs no night-zero (larger if TSFM leaks dawn/dusk power); ~0 vs production night-zero; removes geometry failures |
| 3 | Spread-scale recalibration before p* | **Adopt (cheap test)** | new `recalibrate(q, s_lo, s_hi)`: q̃ = m + s_lo·(q−m) below median, s_hi above; grid s_lo∈{0.8,1,1.25,1.5}, global, select by day-block bootstrap; re-select per checkpoint | 0 to +0.4 zero-shot; potentially more after LoRA if spread collapses; mostly within band |
| 4 | Per-station shrink factor (pooled) | **Adopt if bootstrap-positive** | `global_shrink_factor` per station on validation, shrunk: f_s = w·f_station + (1−w)·f_global, w = D/(D+k); apply as `a_s·p*` | +0.1–0.3 avg; >0.6 at drifting stations |
| 5 | Rolling per-station × regime multiplicative correction (online MOS / EW ratio, lag 2 d) | **Try later (next round)** | cells = station × NWP clear-sky-index bin (3) × time-of-day block (3); ratio r = Σ y·p/d² / Σ p²/d² over EW window (half-life 14–30 d), pooled via D/(D+k) to class; apply r to all quantiles; dead-zone if |r−1| < noise (BC-ACI idea); replay harness = resurrect `conformal_replay.py` protocol (lag_days=2) | +0.2–0.7 avg (my estimate), concentrated in a few stations; must beat #4 to be kept |
| 6 | Online expert weighting M5 vs M8 (ELF-style fast/slow exponential weights, lag 2 d) | **Try later** (after M8/LightGBM aligned) | per-station weights from exponentiated recent 南网 daily losses, fast (5 d) + slow (all) merged; combine quantile functions (Vincentization) then p* | +0.3–0.6 (ensemble range seen in M8: +0.3–0.5) |
| 7 | Per-horizon / per-step p* or shrink (`by_step` fine-grained) | **Reject** beyond 2–3 blocks | — | gains likely < selection noise |
| 8 | Conformal (CQR, Mondrian, ACI) for coverage | **Reject as score lever**; keep only if intervals become a product | if needed, BC-ACI-style EWMA re-centring + width on top of #5 | ~0 on score (coverage not scored) |
| 9 | Isotonic / per-quantile recalibration | **Reject** (dominated by #3, crossing risk, needs many samples) | — | ~0 |
| 10 | ELF Fourier-ridge forecaster | **Reject** (univariate, ~1 labelled sample/day, no NWP) | take only its weighting (#6) | — |
| 11 | TAFAS (gradient GCMs through backbone) | **Reject** | POGT = D-morning night/ramp only; backprop 330M daily | — |
| 12 | PROCEED / learned drift generator | **Reject** | heavy training for +0.x | — |
| 13 | Hard clear-sky ceiling (1.05·p_clear) | **Reject** (keep only as diagnostic) | `clear_sky_cap_ratio=None` | negative on tracking/bifacial |

- Stacking order to evaluate (each step must clear paired day-block bootstrap with lower α-quantile > 0): 1 → 2 → 3 → 4 → 5 → 6. Expect cumulative +0.5–1.5 over median-without-night-zero, of which only p* is individually reliably above ±0.6.
- Reuse: the deleted `conformal_replay.py` (predict only the ≤10:00 row per station-day, then replay with `lag_days ≥ 2`, station/pool/shrink controls) is the right harness for #5/#6; restore from `git show 966aec2^:forecast_backbones/m5/conformal_replay.py` and swap the conformal step for the ratio correction.

### Gaps
- All expected gains in the table are inferences from mechanism + cited magnitudes; none are measured on M5. The only measured anchor in our setting is M8's p* +1.05.
- Unknown whether M5 (zero-shot or LoRA) is over- or under-dispersed on our 28 stations; a PIT/coverage table on the validation split would decide whether #3 is worth more than noise.
