# Empirical evidence: time-series foundation models (TSFMs) for solar PV and related forecasting, especially day-ahead with NWP

Scope note: researched 2026-10-15 with 16 search/fetch calls. The PV-specific TSFM literature is thin and mostly preprints. No paper found evaluates TimesFM-3 (or any TSFM) on 15-min day-ahead PV with NWP GHI/temperature against a tuned LightGBM with identical NWP features. Each source below has a quality flag.

Source quality key: [A] official model paper or peer-reviewed competition paper; [B] arXiv preprint with a clear protocol; [C] preprint with internal inconsistencies, secondary/vendor summary, or thesis-level work.

---

## Q1. Zero-shot vs fine-tuned TSFM vs gradient boosting vs task-specific deep models on PV (day-ahead): who wins, by how much, at what data size? Does the value come from NWP covariates or from history?

### Takeaway
On solar, zero-shot univariate TSFMs are clearly worse than models fitted to the site. In the one multi-model power-system benchmark, solar nMAE was 9–14% zero-shot versus about 3.7–4.7% after fine-tuning. Fine-tuned TSFMs only roughly tie well-trained task-specific transformers (TimeXer, TFT, PatchTST) once all the data is used. No PV study shows a TSFM beating a tuned GBDT that has the same NWP features. Every top HEFTCom2024 entry found was GBDT, and none used a TSFM. Covariate-aware TSFMs (Chronos-2 in-context mode, TabPFN-TS) are the only ones with reported wins over XGBoost-style baselines on energy data, and those figures are secondhand. For day-ahead PV, most of the skill comes from the NWP covariates, not from long history.

### Cited Findings
**ERCOT power-system benchmark (ARPA-E PERFORM, 5-min data, solar/wind/load; nMAE/nRMSE as % of series max) [C: the text contradicts its own tables in several places; no GBDT baseline]** — [Empirical Assessment of TSFMs for Power System Forecasting, arXiv 2604.22077](https://arxiv.org/html/2604.22077v1)
- Zero-shot solar MAE/RMSE (%): TimesFM 9.25/15.1; Chronos-Bolt 11.21/17.43; Moirai-L 10.91/17.5; MOMENT 11.26/18.62; TTM 14.03/22.03. TimesFM is best zero-shot on all three domains.
- Fine-tuned on 100% of the data, solar day-ahead (24 h) MAE/RMSE: TimesFM 3.71/7.02; Chronos-Bolt 3.78/7.03; Moirai-L 4.03/7.72; MOMENT 4.18/8.05; TTM 4.71/8.31. Trained from scratch: TimeXer 3.85/5.82 (best RMSE); TFT 3.91/6.02; PatchTST 4.18/7.82; LSTM 6.71/9.04; 1D-CNN 6.34/8.85.
  - So fine-tuning cuts TimesFM's solar MAE by about 60% (9.25 → 3.71). The fine-tuned TSFM beats the best from-scratch transformer by only about 4% MAE (3.71 vs 3.85) and is worse on RMSE (7.02 vs 5.82). RMSE is the relevant error for the 南网 metric.
- Low-data regime, 20% of training data, solar MAE/RMSE: TimesFM 8.36/13.82; Chronos-Bolt 7.79/13.15; TTM 7.82/11.92; PatchTST 16.25/22.13; TimeXer 14.15/18.2; LSTM 27.13/33.14. TSFMs are about 2× better than from-scratch deep models when data is scarce. The advantage disappears by 100% (about one year of 5-min data). Fine-tuning curves flatten after about 80% of the data.
- The paper claims "FMs with 20% of data beat LSTM/CNN at 100%", but its own Table 4 contradicts this (TimesFM 8.36 at 20% vs LSTM 6.71 at 100%).
- Adding weather covariates to day-ahead solar (MAE, univariate → +weather): Moirai-L 4.03 → 3.65 (−9.4%); PatchTST 4.18 → 3.74 (−10.5%); TimeXer 3.85 → 3.53 (−8.3%); TFT 3.91 → 3.66 (−6.4%); LSTM 6.71 → 6.70 (no change). Covariates were irradiance (global, direct, clear-sky), 2 m temperature, pressure and 10 m wind. The paper does not say whether these are forecasts or actuals, so the gains may be optimistic.
- Context length (solar MAE at 24 h): longer context helps every model class. The context and horizon tables appear to be conflated, so the effect size is unreliable.
- Probabilistic CRPS (% of nominal), solar: Chronos-Bolt 2.25 (best); TimesFM 2.53; TimeXer 2.84; MOMENT 2.87; TFT 3.18; Moirai-L 3.38.
- Authors' conclusion: zero-shot TSFMs "often underperform specialized models"; fine-tuning closes the gap, but the models are "not yet competitive for direct deployment".

**FETS benchmark (54 energy datasets, covariate mode) [B]** — [FETS Benchmark, arXiv 2604.22328](https://arxiv.org/html/2604.22328v2)
- Overall median NRMSE: Chronos-2 (covariates) 0.472, TiRex-2 (covariates) 0.474. Chronos-2 wins 57% of paired comparisons. TiRex-2 is "slightly ahead" on the Non-Dispatchable Generation (PV/wind) category. Per-category numbers are only in figures.
- Weather protocol: historical weather forecasts are used as context covariates. For the horizon, it uses the latest ECMWF IFS run issued at least 9 h before the forecast origin. This is a realistic, leakage-free setup.
- Accuracy saturates at about 2,000 context steps, roughly 3 weeks at 15-min resolution.
- Secondhand only: a Timescale vendor blog says covariate-fed Chronos-2 beat a heavily tuned XGBoost by about 23% overall and 32–34% on wind/solar. I could not confirm this in the paper text. [C] — [Timescale/TigerData blog](https://www.tigerdata.com/learn/foundation-models-predictive-maintenance)
- Also secondhand: on OpenSTEF's 55 grid targets, zero-shot Chronos-2 rCRPS 0.044 vs a trained XGBoost preset 0.065. [C, vendor] — [same blog](https://www.tigerdata.com/learn/foundation-models-predictive-maintenance)

**Chronos-2 official paper [A]** — [Chronos-2, arXiv 2510.15821](https://arxiv.org/html/2510.15821)
- Across 16 fev-bench energy tasks with dynamic covariates, Chronos-2 "consistently outperforms [TabPFN-TS, TiRex] by a wide margin". The numbers are only in a figure.
- fev-bench includes "Solar with Weather" tasks at 15T (horizon 96) and H (horizon 24), with 2 past-only and 7 known covariates. This is the closest public benchmark to our setup.
- EPF-DE case study: univariate mode gives "reasonable but imprecise" forecasts; in-context learning with load and solar+wind forecasts gives "significantly more accurate" ones (qualitative only).
- Aggregates: fev-bench SQL win rate 90.7% and skill 47.3% (TiRex 80.8% / 42.6%); GIFT-Eval WQL win rate 81.9% and skill 51.4%.

**Cold-start PV (440 rooftop sites, daily yield, ERA5 used as perfect-foresight covariates) [B; daily resolution, single seed, optimistic covariates]** — [Physics-Informed Synthetic Histories for Cold-Start PV, arXiv 2606.07457](https://arxiv.org/html/2606.07457v1)
- With real observed context (rolling 15-day forecasts): TabPFN-TS MAE 0.514 kWh/kWp/day (best); Chronos-2 0.537; TimesFM 2.5 0.599. Covariate-aware TSFMs are about 1.7–2× better than Prophet and persistence.
- Error roughly doubles when the real context is replaced by synthetic history. The authors conclude that "plausible context" matters more than which simulator generates it.
- LightGBM was not evaluated.

**Competition evidence (HEFTCom2024, day-ahead wind+solar, NWP-driven) [A]**
- Overall winner: stacked CatBoost with separate wind and solar models and three NWP sources — [Olauson et al., IJF 42(3) 724–735](https://ideas.repec.org/a/eee/intfor/v42y2026i3p724-735.html)
- 4th in forecasting and 3rd in trading (team GEB): LightGBM dense quantile models, each quantile model trained in under 30 s, plus an online solar post-processor for capacity changes — [arXiv 2505.10367](https://arxiv.org/html/2505.10367v2)
- No top HEFTCom2024 entry found used a TSFM. — same sources

**Not found:** a HKUST day-ahead PV thesis compares Chronos-2/TimesFM-2.5/TiRex against LightGBM/LSTM with Open-Meteo NWP, but no results are published on the repo page [C] — [lorenzospolti/solar-pv-forecasting](https://github.com/lorenzospolti/solar-pv-forecasting)

### Inferences
- For day-ahead PV with NWP, the evidence is consistent with our M8 experience: GBDT with NWP features is the bar to beat, and a fine-tuned TSFM is at best on par with a well-built task model. The ERCOT gap between a fine-tuned TSFM and a from-scratch transformer (about 4% relative MAE, worse RMSE) is in the same range as our ±0.6-point seed noise.
- The value of TSFM history is largest for cold-start or data-poor stations, or under 20% data. With about a year or more per station across 28 stations, expect only small gains from the pretraining prior.
- Zero-shot univariate TimesFM is not a reasonable day-ahead PV contender. Covariate-aware inference is the minimum: Chronos-2 ICL, TimesFM 2.5's covariate path, or TabPFN-TS.
- FETS's 2,000-step saturation implies that our 7-day context (672 steps at 15 min) may be below the point of diminishing returns for Chronos-2-class models. Testing 14–21 days of context is a cheap experiment.
- If TimesFM gains anything day-ahead, it most likely comes from how it combines NWP GHI with site-specific day shape from recent history. That matches our finding that history day-shape envelopes help.

### Gaps
- No peer-reviewed head-to-head of TSFM vs LightGBM on 15-min day-ahead PV with operational NWP.
- No public per-dataset fev-bench "Solar with Weather" numbers were retrievable as text.
- No published TimesFM-3 PV results.
- I could not confirm the Timescale "32–34% over XGBoost on solar/wind" figure in the FETS paper.

---

## Q2. Does fine-tuning a TSFM jointly on many PV stations help generalization to unseen stations? Per-station vs pooled?

### Takeaway
There is limited but consistent evidence that pooled fine-tuned TSFMs degrade much less on unseen sites than from-scratch deep models: about 27% vs about 84% MAE inflation. Cross-domain joint training (solar+wind+load) gives only small and inconsistent gains. No paper directly compares per-station and pooled TSFM fine-tuning on PV.

### Cited Findings
- Unseen sites, day-ahead solar MAE (seen → unseen): TimesFM 3.71 → 4.73 (+27%); Chronos-Bolt 3.78 → 4.91 (+30%); LSTM 6.71 → 12.35 (+84%); 1D-CNN 6.34 → 11.49 (+81%). Wind: TimesFM 3.84 → 4.55; LSTM 6.85 → 11.42. [C] — [arXiv 2604.22077](https://arxiv.org/html/2604.22077v1)
- Joint solar/wind/load training, solar MAE: Moirai-L 4.03 → 3.92; TFT 3.91 → 3.84. For wind, TimeXer got worse (4.01 → 4.08). Load benefited most (Moirai-L 2.95 → 2.73). — [arXiv 2604.22077](https://arxiv.org/html/2604.22077v1)
- Cold-start: covariate-aware zero-shot TSFMs with any plausible context (even synthetic, from PVGIS or the OPAQUE simulator) give usable forecasts, so generalization is driven by context availability. — [arXiv 2606.07457](https://arxiv.org/html/2606.07457v1)
- HEFTCom2024 GEB: rescaling forecasts by the change in capacity was unreliable because capacity growth was not spatially uniform. A learned online post-processor was needed. — [arXiv 2505.10367](https://arxiv.org/html/2505.10367v2)

### Inferences
- For 28 stations, the safer default is one pooled fine-tune with capacity-normalized targets, which helps unseen and new stations. Per-station adaptation can then be layered on through cheap heads: per-station quantile choice, hour-of-day bias correction, or a small LoRA.
- Our M8 observation that "m8_all" mixed splits leaked regime information means any seen/unseen claim must be evaluated by holding out whole stations. The ERCOT paper's unseen-site test is the right protocol to copy.

### Gaps
- No PV paper compares per-station TSFM fine-tuning with pooled fine-tuning, or tests station embeddings or IDs in a TSFM.
- No evidence on how many stations are needed before pooled fine-tuning saturates.

---

## Q3. Reported recipes: learning rate, steps, layers, loss, normalization, night handling, ensembling, post-processing

### Takeaway
Published PV recipes are sparse. The recurring choices are:
- Small learning rates (around 1e-5 for full fine-tuning of Chronos-2).
- Low-rank LoRA (r = 4–16) when data is limited.
- Clear-sky-index or capacity normalization.
- Treating impossible night values as missing rather than zero.
- Using the mean of quantiles, or a chosen quantile, as the point forecast.
- Online or learned post-processing for distribution shift.

Very few papers state their fine-tuning hyperparameters.

### Cited Findings
- Chronos-2 PV fine-tune (HKUST thesis): full fine-tuning (not LoRA), lr 1e-5, 5 epochs, one training window per day (stride 24 h), fresh model per fit. Target is the clear-sky index, using a clear-sky power template binned by sun position and fit on the first 365 days only. Impossible night readings are set to NaN; daytime gaps of up to 2 h are bridged; outages (3 h or more of zeros under high sun) are treated as missing. The point forecast is the mean of 9 quantiles. Evaluation uses day-block bootstrap with Diebold–Mariano/Holm tests. [C] — [lorenzospolti/solar-pv-forecasting](https://github.com/lorenzospolti/solar-pv-forecasting)
- Chronos LoRA vs full fine-tuning on building energy: LoRA mostly beats full fine-tuning, except for point estimates on HVAC. Low rank (r = 4 or 16) is recommended for limited data. [B] — [arXiv 2506.00630](https://arxiv.org/pdf/2506.00630)
- Official TimesFM repo added a HF Transformers + PEFT (LoRA) fine-tuning example in April 2026. Community PR #104 adds full FT, linear probing, LoRA and DoRA. — [google-research/timesfm](https://github.com/google-research/timesfm); [PR #104](https://github.com/google-research/timesfm/pull/104)
- ERCOT benchmark: Adam with early stopping and Nixtla-tuned hyperparameters. No learning rate, steps or layers are given; TFT uses a monotone quantile loss. — [arXiv 2604.22077](https://arxiv.org/html/2604.22077v1)
- Chronos-2 outputs 21 quantiles (0.01 … 0.99) rather than the 9-quantile grid. Known future covariates go into a future input matrix with a mask; past-only covariates are masked in the horizon. Pretraining context is 2,048, extended to 8,192 in post-training. A 28M "small" model is within about 1 point of the 120M model on GIFT-Eval skill. — [arXiv 2510.15821](https://arxiv.org/html/2510.15821)
- FETS NWP protocol: use the latest IFS run issued ≥ 9 h before the origin. Historical archived forecasts, not analyses, are used as context covariates. — [arXiv 2604.22328](https://arxiv.org/html/2604.22328v2)
- HEFTCom2024 GEB:
  - Dense quantile LightGBM, one model per quantile.
  - Final quantiles are taken from the inverse of the aggregated CDF.
  - An online solar post-processor handles shift caused by capacity changes.
  - [arXiv 2505.10367](https://arxiv.org/html/2505.10367v2)
- HEFTCom2024 winner: stacked CatBoost, three NWP sources, separate wind and solar models. Operational robustness (no missed submissions, handling a cable outage) is credited as part of the win. — [IJF 42(3)](https://ideas.repec.org/a/eee/intfor/v42y2026i3p724-735.html)

### Inferences
- Our recipe is in line with what little has been published: small model, lr 1e-4 for the from-scratch M8, EMA, 5 seeds, LoRA for TimesFM. For a TimesFM LoRA, start at r = 4–16 and lr around 1e-5 to 1e-4 with early stopping on held-out stations.
- A metric-aligned point decision has an independent justification. The 南网 score is RMSE-based, and the RMSE-optimal point is the conditional mean, not the median. PV predictive distributions are skewed (bounded by 0 and the clear-sky ceiling), so the median is biased. Choosing p* or the quantile mean approximates the conditional mean, which is consistent with our +1 point. Expect the best p* to vary with the hour and with sky-condition regime.
- Multiple NWP sources and ensembles were part of the HEFTCom winner's edge. If a second NWP source is available, it is likely worth more than architecture changes.

### Gaps
- No published learning rate, step count or layer choice for TimesFM fine-tuning on PV.
- No direct pinball-vs-MSE loss comparison for TSFM PV fine-tuning.
- No conformal or hour-of-day bias-correction results specific to TSFM PV forecasts.

---

## Q4. Known failure modes of TSFMs on PV and their remedies

### Takeaway
Documented failure modes:
- Drift away from the true production envelope when the model lacks horizon weather covariates.
- Regression to the climatological mean when the context is uninformative.
- Large zero-shot errors at day-ahead horizons.

Night zeros, ramps, abrupt cloud events and phase errors are widely assumed to be problems but are almost never quantified for TSFMs in the literature found. The remedies used are covariate-aware models, clear-sky-index targets, night masking, and post-processing.

### Cited Findings
- Without horizon meteorological covariates, TiRex drifts from the measured envelope when fed its own forecasts (WAPE 49.77% vs Chronos-2 30.67%). Chronos-2 and Moirai 2.0 keep the seasonal structure better. With synthetic-only context, decoders tend toward the simulator's climatological mean, with positive bias at some cohorts. — [arXiv 2606.07457](https://arxiv.org/html/2606.07457v1)
- Zero-shot solar nMAE of 9–14% is judged "too high for practical grid operations". The ERCOT paper does not analyze night, ramp or smoothing failures. — [arXiv 2604.22077](https://arxiv.org/html/2604.22077v1)
- Night handling in the HKUST thesis: night readings set to NaN, clear-sky-index target, outages masked. — [lorenzospolti/solar-pv-forecasting](https://github.com/lorenzospolti/solar-pv-forecasting)
- NWP can differ significantly from local measurements, "seriously degrading" PV forecasts. NWP correction or downscaling is a standard remedy. One study reports a 36.4% error reduction from downscaled NWP. [B/C] — search summary of [Robustness of DL PV models under NWP errors, arXiv 2607.12954](https://arxiv.org/pdf/2607.12954) and [CNN-LSTM with spatial downscaling, PMC12846104](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12846104/) (numbers not verified against the full text)

### Inferences
- The clear-sky-index normalization used in the literature carries the same risk we found. A horizontal-GHI clear-sky ceiling under-predicts tracking and bifacial plants, so these stations need a plane-of-array or history-derived envelope. Our history day-shape envelope is the safer choice.
- Smoothing toward climatology is a direct risk for TSFMs on cloudy days. Their quantile median tends to be over-smooth, which is one more reason a p* or mean decision layer helps under RMSE.
- Production zeroes nights for all models, so night errors should be removed from comparisons. Night handling inside the model mainly matters for training-signal efficiency, not for the score.

### Gaps
- No quantitative TSFM study of ramp-event skill, phase or timing errors, or cloudy-day vs clear-day breakdowns for PV.

---

## Q5. Results on GEFCom, PVOD, Chinese grid datasets, SKIPP'D, NREL, and solar tasks in GIFT-Eval or fev-bench

### Takeaway
The public TSFM energy evidence comes from five places:
- ERCOT PERFORM (NREL/ARPA-E data).
- fev-bench's "Solar with Weather" and GFC12/14/17 (GEFCom) tasks inside the Chronos-2 evaluation.
- FETS (54 energy datasets including HEFTCom and PV Hong Kong).
- Moirai's Monash "Solar" result.
- The 440-site cold-start study (Ausgrid, DKASC, UK-PV, PVDAQ).

I found no TSFM results on PVOD, SKIPP'D (sky images, not a TSFM target), or Chinese State Grid/南网 station datasets.

### Cited Findings
- fev-bench energy tasks: ENTSO-e load, EPF-BE/DE/FR/NP/PJM, GFC12/14/17, Solar with Weather (15T horizon 96; H horizon 24) and KDD Cup 2022 wind. Chronos-2 leads on energy-with-covariates. — [arXiv 2510.15821](https://arxiv.org/html/2510.15821)
- FETS: 54 energy datasets including PV Hong Kong, HEFTCom wind/PV and the Hill of Towie wind farm. Chronos-2 and TiRex-2 are the leaders, with TiRex-2 slightly ahead on non-dispatchable generation. — [arXiv 2604.22328](https://arxiv.org/html/2604.22328v2)
- Moirai-Large on Monash Solar: CRPS 0.406, MSIS 6.250, beating several full-shot neural baselines (per the search summary of the Moirai paper). — [Moirai, arXiv 2402.02592](https://arxiv.org/pdf/2402.02592)
- ERCOT PERFORM 5-min solar/wind/load: see Q1. — [arXiv 2604.22077](https://arxiv.org/html/2604.22077v1)
- Cold-start daily PV on Ausgrid (300), DKASC (10), UK-PV (100) and PVDAQ (30). — [arXiv 2606.07457](https://arxiv.org/html/2606.07457v1)
- Irradiance foundation-model work mostly does not use time-series FMs:
  - SPIRIT uses vision encoders on sky images plus clear-sky features, and reports about 70% better zero-shot transfer than its baselines — [arXiv 2502.10307](https://arxiv.org/html/2502.10307v1).
  - ZSIF uses a ViT on satellite images plus an iTransformer — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2352467725004266).
  - SolarFM (Applied Energy 2026) is a spatio-temporal irradiance model "inspired by" TSFMs; its baselines were not verifiable — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306261926001844).
- Other TSFMs (Sundial, Toto, Lag-Llama, Timer, Moirai 2.0): no PV or GHI-specific evaluations found. In non-energy benchmarks (realized volatility, air quality), their rankings are domain-dependent and they sit within narrow bands of each other — [arXiv 2607.05291](https://arxiv.org/html/2607.05291v1); [arXiv 2607.19381](https://arxiv.org/pdf/2607.19381)

### Inferences
- fev-bench "Solar with Weather" (15T, horizon 96, known weather covariates) is the closest public proxy for our task. Running TimesFM-3, Chronos-2 and LightGBM on it, or on a 28-station replica, would ground expectations better than any paper.
- Model choice among frontier TSFMs (Chronos-2, TiRex-2, TimesFM-2.5/3, Moirai-2) appears to matter less than covariate support and context.

### Gaps
- No TSFM results on PVOD or any Chinese PV station dataset.
- GIFT-Eval solar subset numbers for the individual models were not retrieved.

---

## Q6. 南网 (China Southern Grid) accuracy scoring and techniques to maximize it

### Takeaway
The 2018 Southern-region PV implementation rules score daily-forecast accuracy each day over n = 96 fifteen-minute points against rated capacity (Cap), with an 85% threshold. The penalty energy is (85% − daily accuracy) × Cap × 1 h, counted daily and assessed monthly. The full accuracy formula was not retrievable as text, but its symbols (PM_i, PP_i, Cap, n = 96) fit the user's "1 − RMSE/Cap" form. The 2022 rules (南方监能市场〔2022〕91号) and the 2025 PV annex are in force. I found no public description of competitor techniques for maximizing the score.

### Cited Findings
- 2018 南方区域光伏电站并网运行及辅助服务管理实施细则:
  - Daily and short-term forecasts are both at 15-min resolution.
  - A daily forecast accuracy below 85% is penalized: penalty energy = (85% − daily accuracy) × Cap × 1 h, where Cap is rated capacity and n = 96 points per day.
  - Accuracy is counted daily and assessed monthly.
  - The point-wise formula uses measured PM_i and forecast PP_i (exact expression truncated in the source).
  - [北极星太阳能光伏网 repost](https://mguangfu.bjx.com.cn/mnews/20180118/875022.shtml) (via search snippet)
- A secondary interpretation gives minimum requirements of 85% for daily forecasts and 90% for short-term forecasts. The 90% figure is unverified. — same search
- The 2025 南方区域光伏发电并网运行及辅助服务管理实施细则 (Annex 4) says forecast accuracy must meet national and industry standards; its formula text was not extracted. — [NEA 南方监管局 PDF](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585645833212.pdf)
- Compared with other regions, the 2022 Southern rules add a medium-term (0–240 h) accuracy assessment. The cited penalty example: a 10-point drop in ultra-short-term accuracy at a 100 MW wind farm costs 200,000 kWh. — [Tencent Cloud article comparing regional 两个细则](https://cloud.tencent.com.cn/developer/article/2200593)
- 2026 spot-market draft: stations that bid both quantity and price are temporarily exempt from the 两个细则 forecast-deviation penalties. Empty short-term or ultra-short-term submissions are scored as 0. — [search summary of 北极星 2026-05-19](https://news.bjx.com.cn/html/20260519/1496189.shtml)
- Nationally, older rules scored PV day-ahead by monthly mean absolute error and wind by single-point deviation plus daily RMSE. Newer national rules moved toward single-point deviation. This is a national comparison, not a Southern-specific one. — search snippet ([i光伏](http://www.ipvei.com/news/report/pdf.jsp?id=fbb9615f674dbd3a0167a63f948d008e))

### Inferences
- If the score is 1 − sqrt(mean over 96 points of (PM − PP)²)/Cap per day:
  - Night points with zero error stay in the denominator count, which dilutes the score. Zeroing nights is free accuracy.
  - The RMSE-optimal point forecast is the conditional mean, which justifies the p* or quantile-mean decision layer.
  - Daily averaging makes bad cloudy days dominate the monthly loss. A day-level shrinkage toward a smoother profile when NWP uncertainty is high can reduce RMSE (hedging).
  - The penalty only applies below 85%, so improvements matter most on low-accuracy days. A worthwhile diagnostic is the distribution of daily accuracies, not just their mean.
- Empty submissions score as 0, so operational robustness (fallback forecasts) is worth points. The HEFTCom winner also credited "no missed submissions".

### Gaps
- Exact current 南网 day-ahead accuracy formula (2022/2025 rules): Cap vs startup capacity, whether night or curtailed points are excluded, and any clipping.
- No public competitor write-ups on maximizing the 南网 score (point-forecast selection, day-level bias correction). Practitioner knowledge exists in vendor marketing (for example a 国能日新 broker report — [东方财富 PDF](https://pdf.dfcfw.com/pdf/H3_AP202305311587381480_1.pdf), not read) but not as quantitative evidence.
