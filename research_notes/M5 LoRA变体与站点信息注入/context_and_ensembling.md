# In-context adaptation (input-side priors, context, retrieval) and ensembling/stacking for TSFM-based PV day-ahead forecasting (M5 TimesFM-3 LoRA + M8)

Scope note: research run 2026-10-16, about 16 search/fetch calls. Every cited number comes from a fetched page or a search-result summary, and the latter is marked "(summary)" where I could not open the full text. Design guidance tied to our pipeline (C1–C3, H1–H3) is in the "Inferences" subsections; it is my reasoning, not something the sources state.

## Q1. Input-side priors for TSFMs: do physical, climatology or envelope channels help? What form should the envelope take? Covariate or target normaliser, given per-variate RevIN?

### Takeaway
Recent energy benchmarks show covariate-aware TSFMs beating univariate ones on PV, sometimes by about 2x in MAE. But no study I found ablates a single physically derived channel such as clear-sky or envelope inside a TSFM, so C1 and C2 have to be validated on our own data. The PV literature has a long-established "statistical clear-sky" from power history alone: a high-quantile, kernel-smoothed envelope (Bacher et al. 2009). It is normally used to normalise power (a clear-sky index), not as an extra input.

### Cited Findings
- **Covariate-capable TSFMs beat univariate ones on PV.** 440 PV sites, daily resolution, ERA5 covariates with perfect foresight, rolling 15-day "real feedback" (RF) setting. MAE in kWh/kWp/day: TabPFN-TS 0.514, Chronos-2 0.537 and TimesFM 2.5 0.599 (all three use covariates), against Moirai 2.0 1.109 and TiRex 1.061 (univariate). WAPE under RF: TabPFN-TS 15.46%, Chronos-2 16.14%, TimesFM 18.11%, Prophet 31.40%, naive 42.65%. Caveats: covariate use is confounded with model identity; the paper has no covariate-on/off ablation of the same model, no clear-sky input to the forecasters, a single seed and daily resolution — [arXiv 2606.07457](https://arxiv.org/html/2606.07457v1)
- **Mechanisms differ by model** (same paper). TimesFM 2.5 uses a univariate backbone plus an auxiliary linear regressor on the covariates, fitted at inference on the context window. Chronos-2 and TabPFN-TS model target and covariates jointly. TabPFN-TS also takes static plant metadata (peak power, tilt, azimuth). — [arXiv 2606.07457](https://arxiv.org/html/2606.07457v1)
- **Plausible context matters more than the generator** (same paper). When target history is missing, filling it with a physics-simulated history (OPAQUE: transposition + NOCT derating + PVWatts) or with PVGIS gives similar accuracy, and the model ranking holds. The authors conclude that accuracy depends more on having plausible temporal context than on the specific generator. Example: Chronos-2 cold-start MAE is 1.019 with OPAQUE and 0.907 with PVGIS — [arXiv 2606.07457](https://arxiv.org/html/2606.07457v1)
- **FETS energy benchmark: covariate-informed TSFMs score best.** Chronos-2 with covariates has the lowest overall median NRMSE (0.472), with TiRex-2 with covariates at 0.474. Caveat: the tree baselines were tuned on full histories, while the TSFMs saw only a limited context window — [FETS, arXiv 2604.22328](https://arxiv.org/pdf/2604.22328) (summary)
- **ERCOT zero-shot solar.** TimesFM MAE 9.25 / RMSE 15.1; Chronos-Bolt 11.21 / 17.43; Moirai-L 10.91 / 17.5. Covariate use is not stated — [arXiv 2604.22077](https://arxiv.org/html/2604.22077v1) (summary)
- **Counter-evidence on exogenous data (electricity demand).** In Singapore and Australia, ChronosX, Moirai, MOMENT, TTM and Chronos-2 vary widely in effectiveness. A simple baseline often beat all of them in Singapore's stable climate. FMs showed advantages mainly in variable climates — [arXiv 2602.05390](https://arxiv.org/abs/2602.05390) (summary)
- **ChronosX** injects past and future covariates through adapter blocks on a frozen pretrained model. It was evaluated on 32 synthetic plus real datasets with WQL and MASE; full fine-tuning improved further at higher cost. I found no solar evaluation — [Amazon Science / AISTATS 2025](https://www.amazon.science/publications/chronosx-adapting-pretrained-time-series-models-with-exogenous-variables), [arXiv 2503.12107](https://arxiv.org/abs/2503.12107)
- **Statistical clear-sky from power alone** (Bacher, Madsen, Nielsen 2009, Solar Energy 83:1772–1783). PV power is normalised by a clear-sky estimate built from observed power only, using statistical smoothing and quantile regression. Data: 21 PV systems, one year at 15-min resolution. The normalised series (a clear-sky index) is then forecast — [DTU copy of the article](https://02417.compute.dtu.dk/material/week13/article-solar-power-forecasting.pdf) (summary)
- **Weighted quantile regression clear-sky** (EMS2014). A smoothing function recursively estimates the local maximum (envelope) of a year of PV power. It does better than deterministic clear-sky models at very low solar elevation and captures local shading — [EMS2014-556 abstract](https://meetingorganizer.copernicus.org/EMS2014/EMS2014-556.pdf), [slides](https://presentations.copernicus.org/EMS2014/EMS2014-556_presentation.pdf)
- **Limitation:** the statistical (Bacher-type) clear-sky can be exceeded by true irradiance, so the index is not bounded by 1 — [arXiv 2002.09267](https://arxiv.org/pdf/2002.09267)
- **Alternative envelope estimator:** the Statistical Clear Sky Fitting algorithm (Meyers et al.) estimates a PV clear-sky baseline from measured power by generalised low-rank modelling. It needs no weather, irradiance or system metadata — [arXiv 1907.08279](https://arxiv.org/pdf/1907.08279)
- **Retrieval/context augmentation helps attention TSFMs but not non-attention models** (indirect evidence that TSFMs can use extra in-context structure). RAF gains: Chronos +15.8% WQL, TimesFM +16.7% WQL; DLinear, GBDT and ARIMA see about 0 or negative change — [arXiv 2411.08249](https://arxiv.org/html/2411.08249)

### Inferences
- **What per-variate RevIN does to the envelope channel.** RevIN removes each variate's own context mean and scale, so the channel reaches attention as a *shape*: diurnal profile, sunrise/sunset timing, day-to-day drift. Its absolute ratio to the target is lost. That is mostly fine. The useful information in C1 is the per-time-of-day maximum profile, including local shading and inverter clipping that coordinate-based clear-sky misses (cf. the EMS2014 low-elevation and shading result). It also explains why C1 helped M8 most on the station without coordinates: there it is the only source of geometry.
- **Two things RevIN does destroy.**
  - The level relationship between target and envelope over the context, i.e. "how cloudy the last week was relative to clear". The model can only recover it through cross-variate attention on normalised values.
  - The envelope's absolute scale relative to capacity.
  - Mitigation (cheap): add the context-mean clear-sky index (Σ P / Σ envelope over the last 7 days) as a constant-valued or scalar covariate. Or feed the envelope already rescaled by the target's RevIN statistics. With per-variate RevIN, a constant channel is normalised to 0/0, so check how TimesFM-3 handles zero-variance variates (epsilon). A constant channel may be useless; prefer a per-day stepwise "daily clear-sky index" series.
- **Covariate vs target normalisation (forecast P/envelope).** Bacher-style normalisation is the PV-classic route. For a TSFM it has three costs:
  - (a) nights and dawn give 0/0 and spikes, so denominators need clipping, e.g. max(env, 0.05·cap) with night masking;
  - (b) the pretrained model never saw a clear-sky-index-like series. LoRA may adapt, but zero-shot quality will drop;
  - (c) the 南网 metric weights low-power hours by 1/max(P, 0.2·cap)². Errors at dawn and dusk come from envelope timing, and dividing by a small envelope amplifies them.
  - Recommendation: keep the envelope as a *covariate* (C1) by default. Test target normalisation as a separate ablation only for M8 or LightGBM, where we control training.
- **Recommended C1 form** (no paper gives optimal hyperparameters for TSFM inputs, so this is my synthesis):
  - per-time-of-day q0.9 over the last 7 days (as in M8);
  - a ±1–2 step (15–30 min) moving max or average along time of day;
  - fallback: if fewer than 2 of the 7 days are clear (q0.9 well below the coordinate clear-sky or below a fraction of cap), widen to 14–21 days or use q0.95. This needs C3-length rows. A cloudy week makes the 7-day q0.9 underestimate the envelope.
  - Seasonal drift over 14–21 days (sunrise shifts by minutes per day) is small relative to 15-min steps except near equinoxes. This is general solar geometry, not from a fetched source.
- **C2 (envelope × clip(NWP GHI / NWP clear-sky GHI, 0, ~1.1)).** This is effectively a physics-informed point forecast on the target scale. It plays the role of TimesFM 2.5's in-context linear-regression covariate path. Expect it to help most where the NWP clear-sky ratio is informative (clear/overcast days) and to add little over raw NWP GHI on variable days. It is also a strong LightGBM feature and baseline (see Q5). Ablate C1 alone, C2 alone and C1+C2, because C2 is collinear with NWP GHI already in the variates.
- **Distraction risk.** Every extra variate adds cross-variate attention load. TimesFM-3's variate RoPE gives no channel-type identity unless the order is fixed. Keep channel order fixed across train and test, and keep the number of channels small; about 5–6 seems safe based on the M5 variants already run.

### Gaps
- No paper found ablates clear-sky, envelope or climatology channels inside TimesFM, Chronos-2 or Moirai for PV at 15-min resolution. All TSFM-covariate PV evidence is model-confounded or daily-resolution.
- The exact Bacher 2009 kernel bandwidth and quantile level (I believe τ≈0.85–0.9, but did not verify) could not be read from the full text.
- No source on how RevIN-type normalisation interacts with envelope covariates.

## Q2. Context length (672 → 2016 steps) for TSFMs on strongly daily-seasonal data

### Takeaway
On strongly daily-seasonal series, several 2025–2026 application studies find that longer TSFM context helps substantially, with diminishing and sometimes non-monotonic returns. No study covers 15-min PV specifically. Expect a modest gain from 7 → 14 days and an uncertain one from 14 → 21 days. Inference cost is small for a patch transformer.

### Cited Findings
- **Pedestrian counts** (hourly, strong daily cycle). TimesFM context 16,000 h vs 168 h cut MAE by about 16–49%. Chronos-2 at 8,192 h gave about 12–13% at the 24 h horizon. The authors call context length "a primary factor" for seasonal regimes — [arXiv 2609.16415](https://arxiv.org/pdf/2609.16415) (summary)
- **Building occupancy and energy.** Accuracy rose up to 5 workdays (480 steps), with diminishing returns beyond 5 days for two TSFMs. Contexts of 48 h or less degraded TSFMs substantially — [arXiv 2506.00630](https://arxiv.org/pdf/2506.00630) (summary)
- **Hourly energy load.** TSFMs improved across windows from 24 to 2048 h without saturation — [arXiv 2602.10848](https://arxiv.org/html/2602.10848) (summary)
- **Seasonal series in a broader benchmark.** Chronos-2 error fell from 6.70% at C=120 to 4.4% at C=624, with a similar trend for Moirai-2.0. A separate benchmark found TimesFM error went 36% (C=180) → 21% (C=1260) → 22.5% (C=2460): a mild degradation at very long context. These two results came from a search summary that did not clearly attribute them; the candidates are [arXiv 2609.25788](https://arxiv.org/pdf/2609.25788) and [arXiv 2602.12147](https://arxiv.org/pdf/2602.12147). Verify before quoting.
- **Inference scaling study (Chronos-T5).** Longer contexts consistently lower the error floor, but without input perturbations the gains saturate quickly. Diversifying inputs (perturbations) and aggregating adds further gains — [arXiv 2601.17376](https://arxiv.org/pdf/2601.17376) (summary)
- **Max context:** TimesFM 2.5 4096 steps, Chronos-2 8192 (benchmark listing); TimesFM 1.0 512 — [search summary citing benchmark listings](https://arxiv.org/pdf/2609.16415)
- **RAF:** Chronos Mini shows more variability at long contexts and Base gains more, i.e. larger models use long context better — [arXiv 2411.08249](https://arxiv.org/html/2411.08249)

### Inferences
- **For 15-min PV, 672 steps = 7 daily cycles; 2016 = 21.** By analogy with the occupancy result (gains saturate around 5 cycles), the expected gain from 7 → 21 days is probably small for the target channel alone. The main benefits are likely:
  - (i) more clear days in context, so better implicit envelope and level estimation, which overlaps with C1;
  - (ii) more day-to-day weather-to-power examples for in-context NWP→power regression across variates. That is plausibly the bigger effect for a cross-variate model fed NWP history.
  - So C1 (with a 14–21-day window) and C3 are partly substitutes. Run C3 both with and without C1.
- **Cost.** With 32-step input patches (TimesFM 1/2 convention; check TimesFM-3's patch size), 672 → 2016 steps means 21 → 63 patches per variate. With about 6 variates under cross-variate attention, the token count roughly triples, so attention FLOPs grow up to about 9x per layer and MLP cost about 3x. That is cheap at inference and modest for LoRA training. The data cost is real: rows must be rebuilt with 2016-step history, and stations with less history need padding or masking.
- **RevIN side effect.** A longer context changes the RevIN mean and scale. Over 21 days the scale is more stable (fewer outlier weeks), which is a quiet benefit for unseen stations.
- **Cheap self-ensemble:** run M5 at context 7/14/21 days and average quantiles. This is "input diversification" in the sense of arXiv 2601.17376, and a free H2-style ensemble member.

### Gaps
- No PV-specific or 15-min TSFM context-length ablation found. The TimesFM-3 context limit and patch size are not in public sources I checked.

## Q3. Retrieval-augmented forecasting and analog ensembles as extra context or variates

### Takeaway
Analog ensembles (AnEn: match today's NWP forecast to similar past NWP forecasts, take the observed power on those days) are a mature PV probabilistic method. Retrieval-augmented TSFMs (RAF, TS-RAG, TimeRAF) show 5–25% WQL gains for attention-based TSFMs, but only for univariate series in published work. For M5 the natural form is an "analog power for D+1" future-known covariate built from the station's own history, or from other stations' history for new stations.

### Cited Findings
- **Alessandrini, Delle Monache, Sperati, Cervone (2015), Applied Energy 157:95–110.** The AnEn takes past power observations recorded on days whose past NWP forecasts are most similar to the current forecast, for each lead time and location. Tested on 3 Italian solar farms against quantile regression and a persistence ensemble (last 20 measurements at the same lead time). Evaluation covered consistency, reliability, resolution and skill — [UCAR OpenSky](https://opensky.ucar.edu/islandora/object/articles%3A17670), [RePEc](https://ideas.repec.org/a/eee/appene/v157y2015icp95-110.html), [NSF PAR full text](https://par.nsf.gov/servlets/purl/10037418). I did not retrieve the skill-score numbers.
- **Follow-ups:** ANN + AnEn for short-term PV (Cervone et al. 2017); AnEn for rare events (Alessandrini et al., EMS2018). Only titles were retrieved — [UCAR dataset page](https://ckanprod.data-commons.k8s.ucar.edu/dataset/short-term-photovoltaic-power-forecasting-using-artificial-neural-networks-and-an-analog-ensemb), [EMS2018-647](https://meetingorganizer.copernicus.org/EMS2018/EMS2018-647.pdf)
- **kNN in competition winners.** The GEFCom2014 solar winners (Huang & Perry, IJF 32(3):1081–1086) blended gradient boosting with k-nearest-neighbours regression on ECMWF NWP. A nearest-neighbours quantile filter was "very close" to the best GEFCom2014 entries — [RePEc](https://ideas.repec.org/a/eee/intfor/v32y2016i3p1081-1086.html), [arXiv 1903.07390](https://arxiv.org/pdf/1903.07390)
- **RAF (Tire et al., AISTATS 2026)**
  - Method: embed series with the Chronos encoder, retrieve the top-1 by ℓ2, instance-normalise the retrieved segment and the query separately, shift the retrieved segment to match the query's first value, and prepend [retrieved context + retrieved future] to the query context. The database holds only segments ending before the forecast window.
  - Gains: Chronos-Base overall about 22.8% WQL / 9.2% MASE; Mini about 17.7% / 6.1%. Across TSFMs: Chronos 15.8% / 12.4%, Moirai 6.0% / 4.5%, TimesFM 16.7% / 4.8%, Lag-Llama 6.1% / 6.4%.
  - Non-attention baselines gain about 0 or lose. Fine-tuning with retrieval improves further, e.g. NN5 MASE 0.481 → 0.378 for Base.
  - Limits: univariate only; gains shrink on noisy series; it needs a sufficiently large database.
  - — [arXiv 2411.08249](https://arxiv.org/html/2411.08249), [PMLR v300](https://proceedings.mlr.press/v300/tire26a.html)
- **TS-RAG** reports beating RAF_Chronos across benchmarks. These are the authors' own comparisons — [arXiv 2503.07649](https://arxiv.org/pdf/2503.07649)
- **TimeRAF** (Microsoft) uses a learnable retriever plus "Channel Prompting", which injects retrieved series along the channel dimension. That is structurally close to adding variates in TimesFM-3. The abstract claims significant gains; I found no numbers — [arXiv 2412.20810](https://arxiv.org/html/2412.20810v1)
- **A different "RAF"** (NeurIPS 2025, model-agnostic) consistently improves zero-shot accuracy for Chronos, Lag-Llama, MOMENT and Toto on physiological and epidemiological data — [NeurIPS 2025](https://neurips.cc/virtual/2025/132332)

### Inferences
- **Feasible M5 design (R1, "analog covariate")**
  - For issue day D, compute the D+1 NWP GHI daily profile (and optionally temperature). Find the k=5–20 most similar past days in the station's history (the 21–60 days available in rows, or a longer store).
  - Similarity: weighted ℓ2 on the daytime NWP GHI profile normalised by clear-sky. Restrict to ±30–45 days of day-of-year to keep geometry comparable.
  - Feed the per-time-of-day mean or median (and optionally the q0.9) of the observed power on those days as a future-known covariate over the horizon. Also feed the same quantity computed for the context days (leave-one-out within history), so the cross-variate attention learns the analog→power mapping in context.
  - This is "AnEn as a channel"; TimeRAF-style channel prompting is the published analogue. Leakage rule: the analog pool contains only days with actuals available at the 10:00 issue time, i.e. up to D−1 at best and D−2 to be safe. Same rule as Q6.
- **Unseen stations** have no local analog pool. Retrieve from other stations' history, normalised by cap and by the clear-sky/envelope profile. That needs a cross-station feature store.
- **RAF-style prepending** (put the analog day's context and future in front of the target history) is also possible with C3-length context. It is riskier with per-variate RevIN, because the prepended segment shifts the normalisation statistics.
- The AnEn quantiles themselves are also a cheap, diverse ensemble member for H2/H3, CPU only.

### Gaps
- I did not retrieve the 2015 AnEn skill numbers versus QR. No published retrieval-augmented TSFM work covers multivariate or covariate PV.

## Q4. Ensembling probabilistic forecasts: quantile averaging vs linear pooling, weights, overfitting, typical gains

### Takeaway
Quantile averaging (Vincentization) and linear pooling have the same mean. Quantile averaging is sharper; linear pooling is wider and can be multimodal. Which is better is empirical, but quantile averaging often wins when components are biased differently and is the natural choice for 9-quantile outputs. Gains depend on diversity: two similar NWP-driven LightGBMs stacked gave only about 0.15–0.3% in HEFTCom2024 solar. Equal weights are a strong default given few validation days.

### Cited Findings
- **Basic relationship.** Quantile averaging = horizontal averaging of CDFs; linear pooling = vertical averaging. Both have the same mean. The quantile average is always sharper (lower variance); the linear pool adds spread from disagreement between component means. For location-scale families, quantile averaging stays in the family — [Forecast combinations: 50-year review, arXiv 2205.04216](https://arxiv.org/pdf/2205.04216)
- **Lichtendahl, Grushka-Cockayne, Winkler (2013), Management Science, "Is it better to average probabilities or quantiles?"** In Survey of Professional Forecasters data, the quantile average beat the probability average both when the latter was over-confident and when it was under-confident — [INFORMS](https://pubsonline.informs.org/doi/fpi/10.1287/mnsc.1120.1667), [Duke Scholars](https://scholars.duke.edu/publication/960692) (summary)
- **Caveats.**
  - Quantile averaging can propagate under-dispersion. Cooke et al. (2021) advise against quantile averaging based on empirical results. Others call the choice an empirical question to be decided by data.
  - "Angular" combination interpolates between the two, with the angle tuned by a proper score.
  - Monte Carlo: quantile aggregation lies between linear and log pooling. The schemes differ most when components are biased.
  - — [arXiv 2205.04216](https://arxiv.org/pdf/2205.04216), [Quantile aggregation of density forecasts](https://metatoc.com/papers/101958-quantile-aggregation-of-density-forecasts) (summary)
- **Deep ensembles.** Vincentization works as a quantile-linear aggregation; allowing unnormalised weights plus an intercept corrects systematic bias — [arXiv 2204.02291](https://arxiv.org/pdf/2204.02291)
- **Solar-specific.** Yang, Yang & Liu (2023, Renewable Energy 215) post-process ECMWF ensemble irradiance with 5 QR + 5 EMOS variants, then combine quantiles via constrained quantile regression averaging (a linear program). Combining quantiles improved the calibrated forecasts "across all locations studied" — [HIT scholar](https://scholar.hit.edu.cn/en/publications/combining-quantiles-of-calibrated-solar-forecasts-from-ensemble-n/), [RePEc](https://ideas.repec.org/a/eee/renene/v215y2023ics0960148123008996.html)
- **Weight estimation.** Regularised constrained QRA for PV combines 4 probabilistic solar methods on 6 years of rooftop data, with bootstrap uncertainty on the weights — [PNNL 2025](https://www.pnnl.gov/publications/enhancing-solar-power-forecasting-regularized-constrained-quantile-regression). An Energies 2019 PV study compared unconstrained, constrained and regularised weight estimation and noted that combining probabilistic forecasts is "much more challenging" than combining point forecasts — [RePEc](https://ideas.repec.org/a/gam/jeners/v12y2019i6p1011-d214129.html)
- **Decision-focused linear pooling** learns pool weights by minimising downstream cost, tested on solar trading — [Imperial Spiral](https://spiral.imperial.ac.uk/entities/publication/f34f203b-ea22-4115-81a9-b9bbb9e87aea). This is relevant because our score is a decision (p*) score, not pinball loss.
- **Diversity determines gain.**
  - HEFTCom2024 solar, team GEB: stacking LightGBM quantile models driven by DWD vs GFS NWP gave MPL 13.35 / 17.54 → 13.33 (0.15%) and 13.72 / 15.07 → 13.68 (0.29%). The authors call the effect minimal because both share the same inputs and area — [arXiv 2505.10367](https://arxiv.org/html/2505.10367)
  - Electricity prices: combining diverse post-processing methods (QRA, conformal, isotonic distributional regression) improved by about 7.5% over a Gaussian-error benchmark — [arXiv 2404.02270](https://ar5iv.labs.arxiv.org/html/2404.02270)
- **FM + domain hybrid.** On Belgian load, hybrids of TSFMs with regression stacking, bagging and residual modelling beat standalone TSFMs at 1–24 h — [arXiv 2508.16641](https://arxiv.org/html/2508.16641v1) (summary; no headline % retrieved)

### Inferences
- **H2 recipe**
  - Average the 9 quantiles of M5 and M8 per time step: q_ens(τ) = w·q_M5(τ) + (1−w)·q_M8(τ). Sort the result (non-crossing), clip to [0, cap], then compute p* from the ensemble quantiles.
  - Do not average the two p* values. p* weights 1/max(y, 0.2cap)² act on the distribution, and p*(avg quantiles) ≠ avg(p*). Report both as a check.
  - Start with w = 0.5. Then try a global w on a grid {0.3…0.8} chosen on validation days.
  - Per-station weights: overfitting risk is high with few validation days and they are impossible for unseen stations. If needed, condition w on observable regimes (time-of-day block, NWP clear-sky ratio bins) and shrink toward the global w.
- **Expected gain.** M5 (pretrained TSFM + LoRA) and M8 (from-scratch with priors) differ in architecture, training data and inductive biases, so they are far more diverse than the HEFTCom DWD/GFS pair. A gain well above 0.3% is plausible, but no PV-specific TSFM+domain number exists. Treat about 0.5–2 accuracy points on the 南网 metric as an unverified prior to test. The gain will shrink if M8 is uniformly worse; the combination literature's rule of thumb (not verified here) is that a weaker but diverse model still helps.
- **Linear pooling** (mixture of the two quantile sets → resample → p*) widens the distribution. Under the p* decision, more spread pulls p* toward low values (high weight on small y). Quantile averaging is therefore safer for our metric. Test both, since the literature says it is empirical.
- Add cheap members (M5 at several context lengths, M5 seeds, the AnEn of Q3) before building complex stackers.

### Gaps
- No GEFCom2014-solar-specific evidence on combination weights; no study combining a TSFM with a domain PV model under a point metric like 南网.
- Did not fetch Hong et al.'s GEFCom2014 overview (IJF 2016) for cross-team combination statistics.

## Q5. Stacking and residual models (LightGBM on TSFM outputs + NWP), leakage control

### Takeaway
Gradient boosting with NWP features has been the competition workhorse for solar since GEFCom2014 (GBM+kNN winners; GBM per-quantile in the wind track) and in HEFTCom2024 (LightGBM quantile models). Feeding TSFM quantiles in as features is a sensible extension, but I found no published PV example. Leakage control means rolling-origin, out-of-fold base predictions and respecting the actuals reporting lag (HEFTCom: data to day n predicts n+2).

### Cited Findings
- **GEFCom2014 solar winners:** gradient boosting + kNN regression on ECMWF NWP, three farms, hourly — [IJF 32(3):1081–1086](https://ideas.repec.org/a/eee/intfor/v32y2016i3p1081-1086.html). Wind track (Landry et al.): GBM multiple quantile regression, one model per quantile and zone — [RePEc](https://ideas.repec.org/a/eee/intfor/v32y2016i3p1094-1102.html) (summary)
- **HEFTCom2024 solar (team GEB, 4th forecasting / 3rd trading).**
  - Model: LightGBM, one model per quantile, 13 features (max/mean/min of solar radiation and cloud cover over 20 site coordinates, lagged / leading / current steps, hour of day), from DWD and GFS NWP.
  - Hyperparameters tuned by 3-fold CV on a set mirroring online conditions (00:00 reference, 23–47 h lead); only q50 tuned.
  - Rolling test trains to day n and predicts day n+2 because actuals lag one day.
  - Authors stress strict offline-online alignment to catch bugs.
  - — [arXiv 2505.10367](https://arxiv.org/html/2505.10367)
- **Stacking practice.** Base learners trained with a rolling time-series split feed out-of-fold predictions to a meta-learner (Queensland demand, stacked MAPE about 0.81%) — [journal2.uad.ac.id](https://journal2.uad.ac.id/index.php/biste/article/view/14410). The FETS paper notes GBDTs dominated recent forecasting competitions — [arXiv 2604.22328](https://arxiv.org/pdf/2604.22328)
- **Residual modelling on TSFM outputs** corrected systematic bias in Belgian load hybrids — [arXiv 2508.16641](https://arxiv.org/html/2508.16641v1)
- **Regime-dependent linear correction** of regional PV gave day-ahead NRMSE 5.28%, 30.6% below the best single model — [Stony Brook](https://researchconnect.stonybrook.edu/en/publications/a-machine-learning-approach-for-regional-photovoltaic-power-forec/) (summary)
- RAF found GBDT gains nothing from retrieval-augmented context, so trees use explicit features, not in-context structure — [arXiv 2411.08249](https://arxiv.org/html/2411.08249)

### Inferences
- **H1 / H3 design**
  - Target: P/cap. A residual target (P − p*_M5)/cap is an alternative.
  - Per 15-min step, features:
    - M5 zero-shot or LoRA quantiles /cap (9 values, plus spread q0.9−q0.1);
    - M8 quantiles;
    - NWP GHI and temperature (plus the ±1–2 step lags and leads that HEFTCom found useful);
    - Solargis if available;
    - C1 envelope /cap and C2 envelope × NWP clear-sky ratio;
    - time of day, solar elevation (if coordinates), day of year;
    - context-mean clear-sky index of the last 7 days.
  - Loss for the point forecast: L2 with sample weight w = 1/max(P, 0.2·cap)². Within a day, the minimiser of the expected weighted squared error is the w-weighted mean, i.e. the same p* principle. The metric's per-day sqrt and the averaging across days make this an approximation.
  - Also train quantile LightGBMs (one per τ) if the output must stay probabilistic for H2.
- **Leakage**
  - (a) Base-model predictions used as stacker features must be out-of-sample for the training day. Use M5 zero-shot, which needs no training, or LoRA/M8 predictions on days outside their training split. In-sample M8 predictions would make the stacker over-trust M8.
  - (b) Split by time (rolling origin) *and* by station: leave-station-out folds, because test includes unseen stations.
  - (c) Any history-derived feature (envelope, recent clear-sky index, bias ratio) must use only actuals available at 10:00 on D. That means D−1 is partial, so use D−2 and older. This matches HEFTCom's n → n+2.
  - (d) NWP must be the run available before 10:00 on D, not the analysis.
- **Choosing H3 complexity.** With few validation days, a low-capacity stacker is less risky than a full GBDT on residuals: e.g. linear QRA per time-of-day block, or LightGBM with strong regularisation and few leaves. The HEFTCom lesson is that stacking similar models barely helps. Value comes from inputs the base models do not see (e.g. M8's priors, Solargis) or from regime-dependent weighting.

### Gaps
- No published example of LightGBM taking TSFM quantile forecasts as features for PV. No numbers on stacking gains over a TSFM baseline in solar.

## Q6. Online / recent-error correction (lag-2-day per-station bias correction)

### Takeaway
Online, recency-weighted post-processing that retrains daily on recent data with a reporting lag is well supported in operational NWP and in HEFTCom2024 solar. There, online post-processing (trained to day n, applied to n+2) cut pinball loss by 9.9% against the raw offline model and by 3.5% against static post-processing. PV-specific evidence on window length and decay is thin.

### Cited Findings
- **HEFTCom2024 solar.** Solar capacity grew from 2,609 to 2,741 MWp mid-competition. The team fit a cubic-polynomial correction via LASSO-penalised quantile regression, retrained daily on recent samples. Rolling-test MPL / MCRPS / MWS: offline model 15.22 / 27.18 / 188.16; offline post-processing 14.22 / 25.85 / 161.00; online post-processing 13.72 / 24.92 / 157.30. That is −9.86% MPL vs offline and −3.52% vs static post-processing. Proportional capacity scaling was judged unreliable because growth was not spatially uniform — [arXiv 2505.10367](https://arxiv.org/html/2505.10367)
- **Operational NWP.** NCEP weights recent biases more heavily for SREF bias correction, trading noise reduction against regime consistency — [NWS Technical Attachment](https://www.weather.gov/media/wrh/online_publications/talite/talite0806.pdf). A decaying-average, regime-dependent bias correction works well for the first few days of lead time — [NCEP manuscript](https://www.emc.ncep.noaa.gov/gmb/ens/papers/manuscript_thorpex_bocui.pdf) (summary)
- **WRF GHI.** Kalman filter + MOS moved annual rMBE 21% → −0.7% and rRMSE 81% → 70% — search summary, source among [repositorio.conacyt.gov.py](https://repositorio.conacyt.gov.py/handle/20.500.14066/3667?show=full) (summary; verify)
- **Day-ahead PV with sliding windows.** A 250 MWp Indian park used sliding-window Lorenz-polynomial and NN bias correction against measured GHI for 10 months — [Springer s40095-023-00560-6](https://link.springer.com/article/10.1007/s40095-023-00560-6). A dual-sliding-window two-stage NWP correction (Hebei) cut day-ahead MAE by up to 39% vs raw NWP, with gains largest in transitional weather — [SciOpen PEF.2026.9650021](https://www.sciopen.com/article/10.26599/PEF.2026.9650021) (summary). Both correct NWP irradiance, not model output.
- **Timestamp caution.** A one-hour timestamp shift improved an irradiance-alignment metric but not forecast skill. Check timestamp conventions separately from bias correction — [arXiv 2608.02088](https://arxiv.org/pdf/2608.02088) (summary)

### Inferences
- **Recommended form (O1)**
  - Per station, ratio r = Σ_daytime P_obs / Σ_daytime p*_model over the last k available days (D−2 back to D−k−1).
  - Use an exponentially weighted variant (half-life about 3–5 days), shrunk toward 1 as r' = 1 + λ(r − 1) with λ≈0.3–0.5, and clipped to [0.85, 1.15].
  - Multiply the whole quantile vector (then re-clip to cap) before p*.
  - Restrict the ratio to "clear-ish" day-hours (high envelope × NWP ratio). Otherwise it partly encodes past weather error, which does not persist, rather than a station bias (soiling, curtailment, capacity or inverter changes), which does.
- **What it catches.** Likely biggest value for unseen stations, where cap or metadata errors and site-level losses are not learned, and after capacity changes or curtailment regimes. HEFTCom's case was literally a capacity drift.
- **Lag.** At a 10:00 issue, D−1 actuals are incomplete, so use D−2 (as HEFTCom used n → n+2). Optionally include D's morning actuals up to 09:45 as a same-day level indicator. Their relevance for D+1 is weak, except for detecting outages or curtailment.
- **Validation.** Evaluate O1 in a strictly rolling backtest; its value is near zero in a shuffled split. Combine with H2 (correct the ensemble, not each member) to avoid compounding shrinkage.

### Gaps
- No PV-power study found that compares window length or decay for multiplicative output-bias correction; magnitude for our metric is unknown.
- No evidence found on whether recent-error correction helps or hurts a TSFM that already conditions on the last 7 days of target history. The TSFM's in-context level adaptation may already absorb persistent bias. Testable: compare O1 gains on M5 vs M8 vs LightGBM.

### Other ideas surfaced (worth testing; evidence level noted)
- **Synthetic context for stations with short or missing history:** fill the target history with physics-simulated power (clear-sky × NWP ratio × cap, or PVWatts-style). Cold-start PV evidence says a plausible context matters more than the generator — [arXiv 2606.07457](https://arxiv.org/html/2606.07457v1)
- **Alternative or ensemble FM with native covariates:** Chronos-2 (best covariate TSFM in FETS) or TabPFN-TS (best RF MAE on 440 PV sites, uses static metadata) as a zero-shot H2 member — [arXiv 2604.22328](https://arxiv.org/pdf/2604.22328), [arXiv 2606.07457](https://arxiv.org/html/2606.07457v1)
- **Input-perturbation self-ensemble:** multiple contexts (7/14/21 d), NWP source swap, small perturbations; aggregate quantiles — [arXiv 2601.17376](https://arxiv.org/pdf/2601.17376) (summary)
- **Static metadata as input** (cap, tilt, azimuth): TabPFN-TS is the only model in the cold-start study that uses it and it was the best there. That is suggestive for a "site-info" channel in M5 but confounded — [arXiv 2606.07457](https://arxiv.org/html/2606.07457v1)
