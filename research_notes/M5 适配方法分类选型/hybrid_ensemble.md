# Hybrid, ensemble, stacking and distillation methods that combine a TSFM with other models, for day-ahead PV (M5 / M8 / M6)

Scope note: researched 2026-10-15. I used 17 web search/fetch calls and read these in-repo sources: `docs/survey_m8_architecture_review.md` (§2.2, §8–§12), `reports/M5 基础模型微调方法调研.md`, `research_notes/M5 基础模型微调方法调研/solar_tsfm_evidence.md`, `experiments/nwp_covariate/rank_lgbm.py` (untracked, read only), `forecast_backbones/m5|m8/evaluate.py`, `forecast_backbones/m5/decision.py` and `forecast_backbones/m6/README.md`. In-repo facts are cited by file path. Web facts are cited by URL and come from the page text as the fetch tool returned it. Several pages (the IJF PDF, the CQRA PDF, HAL) could not be read in full, and the Gaps sections say so.

**Rule of thumb for converting effect sizes.** The 南网 daily score is 100·(1 − relative RMSE with a 0.2·cap floor). At M8's level (about 76, so relative error about 0.24), a 1% relative error reduction is worth roughly 0.24 points. The literature effects below therefore map to: 4% ≈ 1 point, 10% ≈ 2.4 points. This is only approximate, because the literature reports nMAE, pinball or CRPS, not floored relative RMSE, and the daily score averages a per-day sqrt. The in-repo noise band is ±0.6 points (seed noise, `docs/survey_m8_architecture_review.md` §2.2).

---

## Q1. LightGBM / GBDT for day-ahead PV with NWP: features, global vs per-station, quantile GBDT, competition evidence, accuracy vs deep models

### Takeaway
Every competition and benchmark found puts NWP-driven GBDT (LightGBM or CatBoost, one model per quantile or a multi-quantile loss) at the bar or above it. The best TSFM result on 15-min PV (fine-tuned Chronos-2) is slightly worse than a full-feature LightGBM (nMAE 4.59 vs 4.50). The repo still has no GBDT baseline, and `rank_lgbm.py` already contains about 70% of the sample-building code one would need.

### Cited Findings
- **HEFTCom2024, 3.6 GW GB wind+solar portfolio, day-ahead, 3 months live.** The organisers conclude the competition "reaffirms the competitiveness of popular gradient boosted tree algorithms" and that gradient boosting trees were common among top teams. Top teams chose features through validation or feature importance. — [Browell et al., HEFTCom2024, arXiv 2507.01579v2](https://arxiv.org/html/2507.01579v2)
- **HEFTCom2024 winner (SVK), pinball 22.18 vs 2nd 23.18 vs benchmark 53.58.** SVK trained one CatBoost per NWP source (DWD, GFS, MEPS), separately for wind and solar, with a MultiQuantile loss over 9 quantiles. "The only hyper-parameter tuned was the number of boosting iterations." The 3rd-place team (Rnt) was neural, using embeddings from in-house AI weather models. — [arXiv 2507.01579v2](https://arxiv.org/html/2507.01579v2)
- **SVK, value of a third NWP source:** adding MEPS to GFS+DWD gave "an 8% improvement in Pinball Score" on 2023 validation. — [arXiv 2507.01579v2](https://arxiv.org/html/2507.01579v2)
- **HEFTCom2024 4th place (GEB), solar model:**
  - LightGBM, one model per quantile over 101 levels; each quantile model trains in under 30 s.
  - 13 features built from max/mean/min of downward solar radiation and cloud cover over 20 grid points, plus lagged, leading and current steps and hour-of-day. No clear-sky or solar-geometry features are mentioned.
  - Optuna tuning (3-fold CV) was run on the median quantile only and reused for all quantiles. — [GEB, arXiv 2505.10367v2](https://arxiv.org/html/2505.10367v2)
- **GEB, online solar post-processor.** A cubic polynomial correction fit by LASSO-penalised quantile regression and retrained daily on recent samples cut mean pinball loss from 15.22 to 13.72 (−9.86%) against the offline model, and by −3.52% against offline post-processing. — [arXiv 2505.10367v2](https://arxiv.org/html/2505.10367v2)
- **PVDAQ station 2107, 15-min, archived GFS, daylight nMAE:**
  - Full LightGBM 4.500, fine-tuned Chronos-2 4.592 (3,000 full-parameter steps at lr 1e-6), N-HiTS 4.763, TimeMixer 4.936, Cross-UNet + retrieval 5.248, zero-shot Chronos-2 7.701, persistence 19.5.
  - The LightGBM uses one regressor per horizon with 311 columns: 170 shared, 137 horizon-specific and 4 solar-derived. Inputs are recent lags, rolling summaries, availability masks, solar features, weather descriptors, and a "normalised recent power / clear-sky irradiance" index. — [Horizon-specific expert fusion, arXiv 2609.15035](https://arxiv.org/html/2609.15035)
- **GEFCom2014 solar (hourly, ECMWF, 3 farms).** Huang & Perry used gradient-boosted regression trees with a separate model for each non-zero-radiation hour, plus k-NN for the probabilistic intervals. — [IJF 32(3) 1081–1086 (IDEAS)](https://ideas.repec.org/a/eee/intfor/v32y2016i3p1081-1086.html). On GEFCom2014 solar, the full LightGBM (2.1428 nMAE) ties an 8-expert fusion (2.1405, difference not significant). — [arXiv 2609.15035](https://arxiv.org/html/2609.15035)
- **Earlier in-repo synthesis (M8 review §9):**
  - LightGBM 6.07 vs linear regression 7.27 RMSE on 100 New York sites.
  - LightGBM 4.54 < iTransformer 5.03 on PVDAQ.
  - SolarBoost beats CNN-LSTM by 3–14% on 5 Chinese cities with operational NWP.
  - The one large counterexample is NVIDIA Cross-UNet: 30–40% lower MAE than XGBoost/GBDT on 4 Hebei PVOD stations with WRF NWP, where the tree feature set is not described.
  - Multi-site joint training: −15% for LightGBM and −40% for LSTM (PVOD 10 stations, GitHub page). — `docs/survey_m8_architecture_review.md` §9, §10 (refs [64][65][67][69][70][71], W24)
- **Clear-sky model choice barely matters.** Yang compared Ineichen-Perez, McClear and REST2: forecast RMSEs were similar, and no clear-sky model yields a stationary clear-sky index. Accessibility should drive the choice. — [AIP Scilight on Yang 2020](https://www.aip.org/scilights/when-it-comes-to-forecasting-solar-irradiance-a-complex-model-is-not-necessarily-the-best)
- **Repo fact, `experiments/nwp_covariate/rank_lgbm.py` (untracked, user's work):**
  - It already builds (issue row, lead) samples using the D+1 metric's own convention: the last row ≤ issue time per station-day, leads 55–150. It keeps daylight samples only (clear-sky GHI ≥ min_clear).
  - Background features: lead, target-time clear-sky GHI, last observed power ratio, same-time-yesterday power ratio.
  - Target is either the residual vs the C0 baseline divided by cap, or y/cap with `--no_baseline`.
  - It uses a chronological date split with early stopping. The run reports permutation importance, TreeSHAP and gain, with a Gaussian noise reference column. — `experiments/nwp_covariate/rank_lgbm.py` lines 1–15, 61–107

### Inferences
- **Recommended LightGBM design for our setting** (synthesis of SVK, GEB, PVDAQ and rank_lgbm):
  - **Model scope:** one global model across the 28 stations, with target y/cap_mw (the station_meta convention) and station id as a categorical.
  - **Features:**
    - NWP GHI and T2m at the target step, ±1–4 steps, and day-level max/mean/sum.
    - k_nwp = GHI_nwp / GHI_clear and p_clear/cap.
    - Solar elevation/zenith and tod/doy sin-cos.
    - The per-station 7–14 day per-tod max of observed power, i.e. the history envelope already planned as M8 h4. This is the key feature for stations without coordinates.
    - Last observed power ratio, yesterday same-time ratio, and recent-days NWP residual bias.
    - A has_coords flag and static fields (lat/lon/alt/cap).
  - **Objective:** 9 quantile models (or one with lead as a feature and α looped). Then reuse `m5.decision.p_star` unchanged.
  - **Point-model variant:** a second LightGBM trained with sample weight w = 1/max(y, 0.2·cap)² under L2 gives a direct estimate of the metric-optimal point. This mirrors what p* computes from quantiles (`reports/M5 基础模型微调方法调研.md` §1).
- **Nights:** GBDT should be trained on daylight samples only, with nights zeroed by the decision layer. Production zeroes nights for all models anyway (MEMORY note), and rank_lgbm already filters by clear-sky GHI.
- **Cost:** half a day of engineering if built on rank_lgbm's `build_samples`. Training is seconds to minutes on CPU.
- **Expected standalone result vs M8:** within ±1 point, direction unknown. With only GHI+T2m from one NWP source, GBDT's usual edge from multi-source features (the SVK +8%) is unavailable. Its value is as the reference point and as a diverse ensemble member, not as a replacement.

### Gaps
- I found no head-to-head of TSFM vs LightGBM vs a from-scratch small transformer on multi-station 15-min PV with operational NWP restricted to 2 variables. The closest are single-station PVDAQ and a HEFTCom portfolio aggregate.
- No Chinese PV competition write-up (e.g., a State Grid / 南网 forecasting contest) with method details was found in this pass.
- The global vs per-station GBDT evidence is limited to one GitHub page (PVOD 10 stations, −15%) cited secondhand in the M8 review.
- HEFTCom2024 gives no solar-only scores; all numbers are wind+solar portfolio pinball.

---

## Q2. Stacking with TSFM outputs: TSFM quantiles/points as GBDT features, residual vs direct, out-of-fold construction, evidence of gains

### Takeaway
Feeding a frozen TSFM's forecast into a learner that holds the NWP information, and learning the residual, has single-digit-% evidence in PV: PARA-PV reports about −7.5% MAE, and SVK used quantile meta-regression. Stacking two similar GBDTs gives almost nothing (GEB +0.15–0.29%). For us the key structural fact is that **zero-shot M5 predictions are leak-free on every date, because TimesFM-3 never saw our data**. A GBDT can use them as features over the whole training period without out-of-fold (OOF) gymnastics. M8 and LoRA-M5 outputs need OOF or validation-period-only training.

### Cited Findings
- **PARA-PV (2 State Grid stations, 15-min).** A frozen Chronos median serves as an additive base, and an adapter learns the residual. Removing Chronos raises MAE from 0.1352 to 0.1461 and RMSE from 0.3052 to 0.3314 (about −7.5% / −7.9% attributable to the TSFM). — `docs/survey_m8_architecture_review.md` §8, ref [58] (arXiv 2607.08079)
- **TimesFM XReg covariate example (electricity price):** MAE 6.73 → 5.34 (−21%). AutoGluon ships a built-in option where a tabular model handles covariates and the TSFM forecasts the residual (`covariate_regressor`). For drought, residual refinement cut TimesFM MSE by 18.9%. — `docs/survey_m8_architecture_review.md` §8, refs [59][60][61]
- **HEFTCom2024 SVK stacking.** Per target quantile, a linear quantile-regression meta-model takes all 27 quantiles from its three CatBoost models as covariates. Missing NWP inputs were handled by filling with predictions from the remaining models, which gave robustness to missing inputs. The paper does not separately quantify the stacking gain. — [arXiv 2507.01579v2](https://arxiv.org/html/2507.01579v2)
- **GEB stacking of DWD- and GFS-based LightGBMs (solar).** Mean pinball loss went 13.35 → 13.33 (0.15%) and 13.72 → 13.68 (0.29%). The authors attribute the small gain to the two NWPs having identical solar coverage, so stacking mostly combined model differences. — [arXiv 2505.10367v2](https://arxiv.org/html/2505.10367v2)
- **PVDAQ fusion weights** were fit on chronological validation segments, not OOF predictions: level 1 on Jul–Sep 2023 and level 2 on Oct–Dec 2023. For the retrieval gate the authors admit "training-bank residuals are in-sample residuals," which may hurt gate generalisation. — [arXiv 2609.15035](https://arxiv.org/html/2609.15035)
- **Other domains:** a TFT + LightGBM residual corrector cut RMSE 0.128 → 0.113 (−11.7%) vs TFT alone. This is a secondary summary from a search result and not verified in full text. — [Frontiers in Energy Research 2026](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2026.1715418/full)
- **Zero-shot TSFMs vs GBDT outside PV:** ChronosBolt did not beat LightGBM/XGBoost on monthly US renewable consumption. — [MDPI Sustainability 18(13) 6730](https://www.mdpi.com/2071-1050/18/13/6730)
- **Earlier repo guidance.** The previous M5 survey requires stacking features from LoRA-M5 to be OOF, because in-sample LoRA predictions are optimistic. It lists "LightGBM stacking with M5 quantiles as features" as stage 6. — `reports/M5 基础模型微调方法调研.md` §3.1, §4
- **Earlier repo guidance on where residual learning pays.** The residual learner must hold information the TSFM lacks (NWP, geometry). Gains are largest where history dominates, data are scarce, or the station is unseen. — `docs/survey_m8_architecture_review.md` §8

### Inferences
- **Two stacking designs are worth distinguishing for us.**
  1. **GBDT-on-M5zs (direct with TSFM features):** the LightGBM from Q1 plus features M5-zs p10/p50/p90, M5-zs p50/cap minus history envelope, and M5-zs spread (p90−p10). There is no leakage at all, because M5-zs is a fixed function of the input history. It can therefore be trained on the full train split.
     - This is the cheapest TSFM hybrid and the most defensible one.
     - It is also the GBDT analogue of M8's d3/d4 hint channel.
     - Expected gain over plain LightGBM: largest at no-coordinate and low-data stations (国能和煦, 白马湖). Fleet gain is probably +0.2–0.8 points; PARA-PV's −7.5% is an upper-end analogue.
  2. **Residual stacker on M8 (or ensemble) output:** target = (y − M8 p*)/cap; features = NWP + M5-zs quantiles + M8 quantile spread + k_nwp + lead. M8 trained on the train split, so its predictions on train dates are in-sample and overfit (M8 peaks at 0.4–0.6 epoch, `docs/survey_m8_architecture_review.md` §2.2). There are two leak-free options:
     - (a) Blocked time-series CV refits of M8, e.g. 4 monthly-block folds × 3 seeds × about 20 min ≈ 4 GPU-h, to produce OOF M8 forecasts over the train period.
     - (b) Train the stacker only on the validation year with nested blocked CV (by week or month) and evaluate on test.

     Option (b) is cheaper but has about 1 year × 28 stations of data. It is enough for a shallow, heavily regularised GBDT (num_leaves ≤ 15, min_data_in_leaf large), not for a deep one.
- **The `nanwang_online` fallback pattern** (SVK filling missing NWP from other models) is a separate operational benefit of keeping multiple families. If NWP is missing, M5-zs can still issue a history-only forecast.
- **Use quantiles, not just the median, as stacking features.** The spread carries regime information (clear vs variable days). This is the cheapest available proxy for the "weather-dependent gating" that no PV paper has implemented (see Q3).

### Gaps
- No peer-reviewed paper feeds TimesFM/Chronos quantiles into a GBDT for PV and reports the gain vs the GBDT alone. PARA-PV uses a neural adapter, not a GBDT. The extended search found none.
- No quantified OOF vs in-sample stacking comparison was found for energy forecasting. The PVDAQ authors flag the issue but do not measure it.

---

## Q3. Ensembling M5 + M8 (+ zero-shot): quantile averaging vs Vincentization vs probability averaging; static vs per-station/horizon/regime weights; meta-learners; forecast-combination puzzle

### Takeaway
Use **quantile averaging (Vincentization) in real units, followed by one pass of the existing p* decision layer**. Both M5 and M8 `evaluate.py` already do this within a family. Use a **small number of weights**: one global weight, optionally split by a structural flag (has_coords) or by lead block, fitted with a constrained pinball/南网 objective and shrunk toward equal weights.

Do **not** use equal weights blindly when one member is much weaker. On PVDAQ an equal-weight 8-expert average (5.23 nMAE) was worse than LightGBM alone (4.50), while learned convex weights beat it (4.32).

### Cited Findings
- **Definitions and properties.**
  - Probability averaging (vertical, linear pool) gives a mixture.
  - Quantile averaging (horizontal, Vincentization) averages quantile functions.
  - Both give the same mean.
  - Quantile averaging is sharper and unimodal; the linear pool is wider, because disagreement on the mean adds spread. Averaging calibrated distributions by probability yields an over-dispersed, uncalibrated result.
  - Quantile averaging is vulnerable when members are underdispersed.
  - Differences between schemes grow when members are biased. — summarised from [Wang, Hyndman et al., "Forecast combinations: an over 50-year review", arXiv 2205.04216](https://arxiv.org/pdf/2205.04216) and [Quantile aggregation of density forecasts](https://metatoc.com/papers/101958-quantile-aggregation-of-density-forecasts) (via search-result summaries; full texts not read)
- **Energy evidence on the choice is unsettled.** Electricity-price work notes the literature disagrees. Horizontal averaging is described as more robust and sharper; vertical averaging can increase variance and multimodality. — [Combining predictive distributions of electricity prices, arXiv 2308.15443](https://arxiv.org/pdf/2308.15443) (search summary)
- **Constrained quantile regression averaging (CQRA) for probabilistic load forecasts.** Weights are non-negative, sum to one, and are fit by a linear program minimising pinball. The ensemble beats the best individual member by 4.39% pinball on average (ISO-NE and Irish smart-meter data) and beats nine other ensemble methods. — [Wang et al., "Combining Probabilistic Load Forecasts", arXiv 1803.06730](https://arxiv.org/abs/1803.06730)
- **Solar CQRA.** Yang et al. post-processed ECMWF ensemble irradiance with 5 QR and 5 EMOS variants and combined them by linear programming. The combination improved on the calibrated forecasts at every location. — [HIT scholar page, Renewable Energy 2023](https://scholar.hit.edu.cn/en/publications/combining-quantiles-of-calibrated-solar-forecasts-from-ensemble-n/)
- **PVDAQ horizon-specific expert fusion.**
  - Non-negative weights per horizon in two levels (32 LPs in total), fixed across forecast origins, with no weather-dependent gating.
  - Results: learned fusion 4.315 vs LightGBM 4.500 (−4.11%) vs equal-weight 5.234.
  - Learned fusion beat equal weighting by 0.92 pp on PVDAQ and 0.27 pp on GEFCom2014, but was not significantly better than LightGBM on GEFCom2014.
  - The authors conclude "the value of a larger ensemble depends on its measurable complementarity."
  - Only removing N-HiTS stayed significantly positive after multiple-comparison adjustment; removing climatology or an analog expert changed nothing. — [arXiv 2609.15035](https://arxiv.org/html/2609.15035)
- **Forecast combination puzzle.** Equal weights often beat estimated weights (Stock & Watson 2004; Clemen 1989). The main explanation is weight-estimation error (Timmermann 2006; Smith & Wallis 2009; Claeskens et al. 2014). Elliott (2011) finds the best possible gain of optimal over equal weights is often small enough to be swamped by estimation noise. Shrinkage-modified combinations can beat both. — [arXiv 2205.04216](https://arxiv.org/pdf/2205.04216); [Qian et al., "On the forecast combination puzzle", Econometrics 7(3):39](https://arxiv.org/pdf/1505.00475)
- **Qian et al.** distinguish "combining for adaptation" (one member dominates, so the combination should find it) from "combining for improvement" (members are complementary, so average them). Applying the wrong one creates the puzzle. — [arXiv 1505.00475](https://arxiv.org/pdf/1505.00475)
- **Weather-dependent weights.** I found no day-ahead PV paper that sets combination weights by sky class. Related approaches:
  - Online CRPS-minimising weights for French PV up to 6 days. This adapts implicitly and does not test by sky type.
  - Weights by inverse error SD.
  - A weighted quantile combination of 3 probabilistic PV models (Bracale et al. 2019). — [search summary listing hal-01909827 and others](https://hal-enpc.archives-ouvertes.fr/hal-01909827) (abstract not readable, blocked)
- **AutoGluon-TimeSeries (29 benchmark datasets)** ensembles diverse statistical, ML and TSFM models. The abstract says the ensemble sometimes beats the best-in-hindsight combination of prior methods but gives no figures. — [arXiv 2308.05566](https://arxiv.org/abs/2308.05566)
- **In-repo facts:**
  - **Existing ensemble code:** `forecast_backbones/m5/evaluate.py` and `forecast_backbones/m8/evaluate.py` treat multiple `--runs` as an ensemble. They average each member's raw quantiles in real units (`np.mean(qs, axis=0)`), then apply the decision layer once. `reports/members.csv` scores each member separately. So within-family Vincentization is already implemented.
  - **Seed ensembles:** M8's 3-seed ensemble gave +0.3–0.5 points.
  - **Head-to-head:** M8 beats zero-shot M5 on 15/17 test stations, but M5 wins by 12.2 at 国能和煦 (no coordinates).
  - **Decision layer:** p* is worth +1.05.
  - Sources: `docs/survey_m8_architecture_review.md` §2.2; `reports/M5 基础模型微调方法调研.md` §1.

### Inferences
- **Combination method.**
  - Vincentization fits our pipeline natively: p* integrates over a quantile grid (`decision.quantile_grid` / `p_star`), so averaged quantiles feed straight in.
  - A linear pool would need CDF inversion. It would also widen the distribution, which shifts p* (p* weights by 1/max(y, 0.2cap)², so a wider lower tail pulls p* down).
  - Recommendation: Vincentize, then monotonise, then clip, then p*. Test a linear pool once only if M5 and M8 quantiles are both underdispersed on validation.
- **Weighting scheme for M5zs + M8 (2 members), in increasing complexity:**
  - (i) a global w;
  - (ii) w split by has_coords, a structural rule with no per-station fitting;
  - (iii) w per lead block (e.g. 4 blocks over D+1);
  - (iv) w per k_nwp tercile (clear / mixed / cloudy).

  With 2 members and ≤8 weights, estimation error is small relative to about 1 year × 28 stations, so the puzzle is less threatening than with 8 members. Still, shrink toward 0.5 (or toward the global w) and confirm on test before adopting.
- **Equal weights would likely hurt.** M5zs is much weaker than M8 on most stations, so this is the PVDAQ failure mode (zero-shot Chronos-2 at 7.7 in an equal pool). The has_coords split is the "combining for adaptation" case. The other stations are "combining for improvement", where a small M5 weight (0.1–0.3) may still help through diversity.
- **Expected gains:**
  - Global-w M5zs+M8 Vincentized: +0.2–0.6 fleet (borderline vs noise).
  - With the has_coords split: 国能和煦 alone could recover several of its 12.2 points, about +0.3–0.7 on a 17-station fleet mean.
  - Adding a LightGBM member (3 families, CQRA weights): +0.5–1.5 if LightGBM lands within about 1 point of M8. The PVDAQ −4.1% maps to about +1 point.
  - Within-family seed ensembles stay at +0.3–0.5.
- **M6** (TimesFM-3 + calibration + decision ladder, zero-shot route A per `forecast_backbones/m6/README.md`) shares M5's backbone. As an extra member it adds little diversity over M5zs, so treat M5/M6 as one family.

### Gaps
- No direct energy benchmark of Vincentization vs linear pool vs CQRA on the same PV data was found.
- No PV evidence that sky-class or clear-sky-index weights beat static weights. The PVDAQ authors explicitly leave weather-dependent gating as future work.
- AutoGluon's ensemble-vs-best-single numbers were not readable from the abstract.

---

## Q4. Per-station model selection vs combination; overfitting risk with ~1 year of validation

### Takeaway
Hard per-station argmax selection is the high-variance end of combination ("combining for adaptation" with estimated weights at the 0/1 extremes). It is justified only where the validation gap is large and stable (e.g. 国能和煦's 12.2 points), not where differences sit inside the ±0.6 noise. Prefer a structural rule (has_coords, data length) or shrunk per-station weights over argmax.

### Cited Findings
- **Estimated weights can be far from equal weights even when the losses are very similar,** and weight-estimation variance is the main reason simple averages win. — [arXiv 2205.04216](https://arxiv.org/pdf/2205.04216) (search summary); [Claeskens et al. via Qian et al.](https://arxiv.org/pdf/1505.00475)
- **Screening candidate models before combining** (i.e. selection) is listed as one cause of the combination puzzle. — [Qian et al., arXiv 1505.00475](https://arxiv.org/pdf/1505.00475)
- **PVDAQ:** "the value of a larger ensemble depends on its measurable complementarity". The fusion's advantage over the single best model (LightGBM) was negligible on GEFCom2014. — [arXiv 2609.15035](https://arxiv.org/html/2609.15035)
- **In-repo:** between-variant differences of up to ±0.6 points are seed noise on the fleet mean. Per-station scores are noisier still: M8's validation score oscillates between 73.2 and 75.8 after its peak. The two laggard stations lose 6–12 points from prior mismatch, not from the backbone. — `docs/survey_m8_architecture_review.md` §2.2
- **In-repo:** fine-tuned TSFMs can degrade on unseen stations (MixFT, PaCTS), so keep a zero-shot fallback or a zero-shot + LoRA ensemble for new stations. — `reports/M5 基础模型微调方法调研.md` §6

### Inferences
- **Selection rule, if any:** switch a station's model only if the validation gap exceeds about 2× a block-bootstrap SE (resampling weeks, not days, to respect autocorrelation), and confirm the sign on the held-out test period.
  - With about 365 daily scores per station and daily-score SDs that are plausibly 5–10 points (not measured here), the SE of a station mean is about 0.3–0.5. A 12-point gap passes easily; a 1-point gap does not.
- **Prefer structural gating.** It transfers to the stations not yet in the test set (28 total vs 17 tested) and to new stations, which per-station argmax cannot do.
- **The has_coords split is the cleanest candidate.** M8's failure there is explained mechanistically (clock-time geometry), and the h4 history-envelope fix may close the gap anyway. Re-check whether gating is still needed after h4.

### Gaps
- Per-station daily-score SD and the autocorrelation of daily scores in our data were not measured. They are needed to set the selection threshold properly; this can be computed from existing `reports/members.csv` and station_report outputs.
- No PV-specific study of per-site model selection vs combination was found.

---

## Q5. Distillation (TimesFM-3 into a small model or into M8); does it improve accuracy; licence angle

### Takeaway
TSFM distillation papers show students that are **comparable** to their teachers at 1/150 of the parameters. None shows that a student distilled from a TSFM beats the same student trained on ground truth, and none uses covariates or PV.

For us distillation has no accuracy case: M8 is already small and already beats M5zs on most stations, and the d3/d4 hint channel is the stronger form of "TSFM as teacher signal". It also has no licence case. The TimesFM-3 licence explicitly lists distilling other models for commercial use as non-permitted, and bars using outputs for production purposes. **Distillation does not launder the restriction, and neither does using TimesFM-3 outputs as stacking features or ensemble members in production.** (Flag only; not legal advice.)

### Cited Findings
- **DistilTS (ICASSP 2026).**
  - Teachers: TimeMoE, MOIRAI, Chronos, TimesFM, Moment. Students: DLinear and iTransformer.
  - Data: ETTh1/h2/m1/m2 and Weather, horizons 96/192.
  - Claim: "comparable forecasting performance" with up to 1/150 of the parameters and up to 6000× faster inference.
  - It reports **no student trained from scratch on ground truth**. The undefined "Baseline" column in its ablation is 0.414 vs 0.395 MSE on ETTh1.
  - No exogenous covariates. — [DistilTS, arXiv 2601.12785](https://arxiv.org/html/2601.12785v1)
- **Battery capacity.** A fine-tuned Timer TSFM was distilled into compact experts for edge deployment, which keep "a substantial portion" of the cross-capacity gains at lower cost. — [arXiv 2505.08151](https://arxiv.org/abs/2505.08151v3) (search summary)
- **Earlier repo assessment:** distillation / self-training is feasible and "reduces variance and serving cost, does not align with the metric"; marked optional. — `reports/M5 基础模型微调方法调研.md` §3.1
- **TimesFM licensing.** The README says 3.0 weights are under `timesfm-non-commercial-license-v1.0`, "restricted to non-commercial, non-production use"; "Commercial or production use of downloaded / self-hosted weights is not permitted"; commercial use is only via authorised Google Cloud services. Code and weights up to 2.5 are Apache-2.0. — [google-research/timesfm README](https://github.com/google-research/timesfm)
- **TimesFM-3 LICENSE (Hugging Face), verbatim fragments as returned by the fetch tool:**
  - Non-commercial purpose: "means use for testing, evaluation, or research not tied to commercial gain, production deployment, or revenue generation"; this "includes internal benchmarking, academic research, and experimentation on private or public datasets, provided the results are not used in commercial decision-making, client deliverables, or paid products/services."
  - Not non-commercial: use "(b) in direct or indirect interactions with end users or production systems", and "(c) to train, fine-tune, or distill other models for commercial use".
  - Derivatives: "Derivative" includes "(ii) work that incorporates, utilizes, or is otherwise based on or derived from the TimesFM Model, its logic, or parameters".
  - Outputs: "Outputs are not considered Derivatives under this License", **but** the restriction also covers "any Outputs and data produced by the TimesFM Model … for any commercial or production purposes". — [HF LICENSE](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
- **Repo context:** M5 zero-shot already runs in `nanwang_online`. The previous survey flagged this as a compliance question and named TimesFM 2.5 (Apache-2.0, covariates via XReg only) as the fallback. — `reports/M5 基础模型微调方法调研.md` §6

### Inferences
- **Accuracy:** distilling M5 into M8 via soft targets is functionally close to d3/d4 (hint channel plus residual base), but weaker: soft targets transfer only the teacher's errors, while a hint channel lets M8 decide when to trust the teacher. With no evidence that distillation beats ground-truth training, the expected gain is ≈0 ± noise.
- **Serving cost:** TimesFM-3 inference is not a bottleneck (M8 review §9: TSFM inference about 200 ms P95 on V100 for TimesFM-2.0). This removes the main motive for distillation.
- **Licence (flag, not legal advice):** every hybrid in this note that uses TimesFM-3 at inference in production (ensemble member, GBDT feature, hint channel) or as a training teacher for a production model sits inside the restricted uses as literally worded. Research and evaluation on our data appears to fall under the permitted "internal benchmarking … experimentation", provided the results are not used in commercial decision-making.
- **Practical mitigation:** design every hybrid so the TSFM slot is backbone-agnostic. Then TimesFM 2.5 (Apache-2.0), another permissively licensed TSFM, or a Google Cloud-hosted TimesFM can be swapped in. Have legal review the HF licence before any production rollout. I did not verify Chronos-2's licence in this pass.

### Gaps
- No TSFM distillation study with covariates, quantile outputs or PV data.
- No legal interpretation was sought. Whether a power-grid forecasting deployment counts as "commercial" or "production" for an internal utility tool is for counsel. The literal text covers "production systems" regardless of revenue.

---

## Q6. Final shortlist: adopt / try later / reject, with implementation sketch and expected gain vs ±0.6 noise

### Takeaway
Adopt three cheap, low-variance items:
1. A global quantile LightGBM baseline built on rank_lgbm's sample builder.
2. A cross-family Vincentized M5zs + M8 ensemble with at most a has_coords-split weight.
3. GBDT-with-M5zs-features (leak-free by construction).

Try later: a 3-family CQRA ensemble, the residual stacker on M8 with OOF, lead-block weights, and sky-regime gating. Reject: per-station argmax, equal-weight pooling of weak members, linear pooling as the default, and distillation.

### Cited Findings
- Evidence base for the effect sizes in the table below:
  - PVDAQ fusion −4.1% vs LightGBM, equal weighting worse than LightGBM. — [arXiv 2609.15035](https://arxiv.org/html/2609.15035)
  - CQRA −4.39% vs the best member. — [arXiv 1803.06730](https://arxiv.org/abs/1803.06730)
  - GEB stacking of similar models +0.15–0.29%; online post-processor −9.86%. — [arXiv 2505.10367v2](https://arxiv.org/html/2505.10367v2)
  - SVK CatBoost + QR meta-model, extra NWP +8%. — [arXiv 2507.01579v2](https://arxiv.org/html/2507.01579v2)
  - PARA-PV TSFM residual −7.5%. — `docs/survey_m8_architecture_review.md` §8
  - In-repo seed ensemble +0.3–0.5, p* +1.05, M5 +12.2 at the no-coordinate station. — `docs/survey_m8_architecture_review.md` §2.2

### Inferences

| # | Method | Verdict | Implementation sketch (repo-specific) | Cost | Expected gain (南网 points) vs ±0.6 noise |
|---|---|---|---|---|---|
| 1 | **Global quantile LightGBM baseline** | **Adopt** | Extend `rank_lgbm.build_samples` (D+1 leads 55–150, issue ≤10:00) with target y/cap_mw. Features as in Q1 (NWP ±k steps, k_nwp, p_clear/cap, elevation, tod/doy, history envelope, last/yday ratios, station id, has_coords). 9 `objective=quantile` models, then `m5.decision.p_star`. Plus one L2 model with weights 1/max(y, 0.2cap)². Score with `eval_shell` on the m8_v1 split, nights zeroed. | 0.5–1 day, CPU | Standalone: within ±1 of M8, direction unknown. Its main value is positioning M5/M8 and serving as an ensemble member. |
| 2 | **Vincentized M5zs + M8 ensemble, weight split by has_coords** | **Adopt** | Generalise the existing within-family quantile average in `evaluate.py` to members from different families (same row schema, real units). Grid-search w ∈ {0, 0.1, …, 0.5} on validation, separately for has_coords = 0/1, shrunk toward the global w. Then one decision layer pass (p*). No retraining. | Hours | +0.2–0.6 fleet with global w (borderline). With the split, likely +0.3–0.7, concentrated at 国能和煦. Re-check after M8 h4. |
| 3 | **LightGBM with M5-zs quantiles as features** (direct, leak-free) | **Adopt** (right after #1) | Add M5zs p10/p50/p90 and spread as features to #1. Zero-shot outputs are leak-free on all dates, so no OOF is needed. Cache M5zs forecasts once per issue row. | +0.5 day plus M5zs batch inference | +0.2–0.8 over plain LightGBM, mostly at low-data or no-coordinate stations. |
| 4 | **3-family CQRA ensemble** (M8 + LightGBM(+M5zs) + M5zs or M5-LoRA) | **Try later** (after #1–#3 have scores) | LP or grid over non-negative weights summing to 1, minimising validation pinball or 南网 score after p*. ≤3 weights global, or per 4 lead blocks at most. Validate on test. | Hours | +0.5–1.5 if LightGBM is within about 1 point of M8. ≈0 if the members are not complementary (GEFCom2014 pattern). |
| 5 | **Residual stacker on M8 output** (shallow GBDT, target (y − M8 p*)/cap, features NWP + M5zs quantiles + M8 spread) | **Try later** | Either (a) blocked-CV refits of M8 for OOF (about 4 GPU-h) or (b) stacker trained on the validation year with nested blocked CV. Keep it shallow (num_leaves ≤ 15). | 1 round | +0.3–1.0. Leakage risk if in-sample M8 outputs are used. Gains overlap with #4 and M8 d3/d4. |
| 6 | **Lead-block or sky-regime (k_nwp tercile) weights** | **Try later** (only if #4 shows complementarity) | Same as #4 but weights per block. Shrink toward global weights. | Hours | ≤ +0.3 beyond #4. No PV evidence for regime gating, and estimation-error risk. |
| 7 | **Online daily post-processor** (GEB-style LASSO-QR polynomial correction on the ensemble output, retrained daily) | **Try later** (adjacent; overlaps M5 survey stage 6 "station-level daily corrector") | Per-station or global cubic correction of the quantiles on a recent window (e.g. 30–60 days). | 0.5 day | Literature −3.5% to −9.9% pinball under capacity drift. For us, depends on drift: +0 to +1. |
| 8 | **Per-station argmax model selection** | **Reject** (except as structural rule #2) | — | — | High variance. Small gaps are inside noise and do not transfer to untested or new stations. |
| 9 | **Equal-weight pool including weak members** (e.g. M5zs + M6 + M8 at 1/3 each) | **Reject** | — | — | Likely negative. PVDAQ: equal 5.23 vs best single 4.50. |
| 10 | **Linear pool (probability averaging) as default** | **Reject as default**, one-off check | Only if both families are underdispersed on validation. | — | ≈0 or negative for p*: wider tails bias p* down. |
| 11 | **Distil TimesFM-3 into a small model or M8** | **Reject** | — | — | ≈0 accuracy evidence. Licence explicitly covers distillation for commercial use. Serving cost is not a bottleneck. |
| 12 | **Adding M6 as a separate ensemble member** | **Reject** (merge with the M5 family) | — | — | Same backbone, low diversity. |

- **Ordering:** #1 → #2 (both independent, can run in parallel) → #3 → #4. Stop the hybrid track if #4's gain over the best single model is below about 0.6 on test. That would be the GEFCom2014 "no measurable complementarity" outcome.
- **Licence gating:** #2–#5 depend on TimesFM-3 at inference. Build them backbone-agnostic so a TimesFM 2.5 / permissive-TSFM member can replace M5zs if counsel rules out TimesFM-3 production use. #1 has no licence exposure.
- **Interaction with the M8 roadmap:** M8 d3/d4 (TimesFM hint plus residual) and #3/#5 address the same complementarity. If d3/d4 lands +1, the expected gains of #2, #3 and #5 shrink, because the M5 information is already absorbed. Run d3/d4 before investing in #5.

### Gaps
- None of the expected gains above has been measured on our data. They are literature-to-metric translations (≈0.24 points per 1% relative error) and need confirmation on the m8_v1 test split with ≥3 seeds for the neural members.
- Daily-score SD per station and the M5zs-vs-M8 error correlation (the complementarity measure that decides #4) were not computed. Both are available from existing `reports/members.csv` and station_report outputs, without retraining.
