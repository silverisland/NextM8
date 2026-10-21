# In-context (input-side) adaptation of TimesFM-3 for day-ahead PV (M5): evidence and scenario-specific shortlist

Scope: methods that change only what TimesFM-3 is given (context length, covariate rows, retrieved examples, related series), not its weights. The question is which suit M5: 28 stations in southern China, 15-min data, NWP limited to forecast GHI and temperature, scored by the 南网 daily relative RMSE. This note builds on `research_notes/M5 基础模型微调方法调研/covariates_incontext.md` (cited below as "earlier notes") and does not repeat its general literature review. Status: 2026-10-15.

Labels:
- **[CODE]** = read this session in the installed `timesfm3` package (`.../scratchpad/venv314/lib/python3.14/site-packages/timesfm3/`) or in our repo. File and line are given.
- **[PAPER]** / **[BLOG]** = what the authors claim.
- **[EST]** = my own arithmetic or inference, not measured.

**Three code findings that correct or sharpen the brief and the earlier notes** (all [CODE]):
1. **The released inference path does encode variate order.** `timesfm3/torch/timesfm3_forecaster.py:305-306` builds the transformer with `use_rope_seq=True, use_rope_var=True`, and `TimesFM3Evaluator` (which our M5 uses) subclasses that forecaster (`timesfm3/torch/evaluator.py:24-32`). The `TimesFM3Torch` constructor default is `use_rope_var=False` (`model.py:100-101`), which is what the earlier notes read. But the forecaster overrides it. `model.py` passes no `var_segment_pos`, so `MultiHeadAttention` falls back to `position = arange(v)` (`transformer.py:273-276`). The variate positions are therefore the packing order `[targets | past_only | past_future]`. There is no learned identity embedding, but RoPE does encode each variate's relative index. Our own tiny test backbone also sets `use_rope_var=True` (`forecast_backbones/m5/backbone.py:80`).
2. **Per-variate normalisation makes every covariate invariant to positive affine maps.**
   - Linear detrending is applied when `std_det < 0.5·std_orig`, a scale-free test, and the detrended values are affine-equivariant (`model.py:472-520`).
   - Every patch is then RevIN-normalised with that variate's causal running mean and std (`model.py:219, 235`). `value_clip=1e20` and `input_transform="identity"` are the forecaster defaults (`timesfm3_forecaster.py:80-83`).
   - So for any a > 0, `a·x + b` yields the same tokens as `x`, apart from the near-constant-σ guard. A per-station linear "NWP→power" mapping, capacity scaling, and `cap·GHI/1000` are all **no-ops** inside the backbone.
3. **Tomorrow's level relative to the past is not removed.** The 64-step lookahead patch is normalised with the *current* patch's running stats (`model.py:238-239`), and `use_frozen_running_stats=False` (`timesfm3_forecaster.py:74`). So the model sees "tomorrow's NWP-GHI is X σ below this week's running mean". What RevIN removes is only the absolute calibration (MW per W/m²), and the target's own history supplies that.

---

## Q1. Context length: what is optimal for 15-min solar with TSFMs, what is TimesFM-3's maximum, and what does it cost on V100?

### Takeaway
TimesFM-3's hard context cap is 15,360 steps (160 days at 15 min) [CODE]. The pretraining context length is not published. The only quantitative saturation evidence is FETS: about 2,000 steps (about 3 weeks), measured for Chronos-2 and TiRex-2 on load datasets, not on solar. M5 uses 672 steps (7 days). A zero-shot sweep over 672 / 1344 / 2016 / 2880 is cheap: cost grows roughly linearly, about 2.6x at 2016 [EST]. Production is the real obstacle: online rows carry exactly 672 history points, so a longer context needs an upstream data change.

### Cited Findings
- [CODE] `timesfm3/torch/timesfm3_forecaster.py:32` `_MAX_CONTEXT_LENGTH = 15360`. `global_context` rounds it up to a multiple of 32 (lines 362-368), and each batch's context is `min(ceil(max_ctx/32)·32, global_context)` (lines 655-661). Longer queries are left-truncated, keeping the covariate window aligned (lines 225-239). Our wrapper copies the constant: `forecast_backbones/m5/backbone.py:38` `_GLOBAL_CONTEXT_POINTS = 15360`, with the same truncate/pad logic at lines 146-157. No maximum-horizon constant exists; the horizon is padded to a multiple of `output_patch_length` = 64 (lines 480-481).
- [BLOG] The TimesFM-3 blog gives no maximum context, maximum horizon or maximum variate count. It states only patch length 32, 330M params, per-series normalisation, alternating causal temporal and full variate attention, 9 quantiles and >1T pretraining points. — [Google Research blog](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- The HF model card also gives no maximum context. It lists "Context Patch Length: 32" and "Forecast Horizon Patch Length: 64", and the licence is the "TimesFM Non-Commercial License v1.0". — [HF google/timesfm-3.0-pytorch](https://huggingface.co/google/timesfm-3.0-pytorch)
- [PAPER] FETS varied context "from 100 to 8,000 time steps" for Chronos-2 and TiRex-2. Error "decreases as the context length increases, saturating at approximately 2,000 time steps, corresponding to roughly three weeks". The four datasets were DE Power, Industrial VEA, California CAISO and Netze BW LV, none solar-specific. In-context NWP came from archived historical forecasts, and the horizon used the latest ECMWF-IFS run issued at least 9 h before the origin. — [FETS arXiv 2604.22328v2](https://arxiv.org/html/2604.22328v2)
- [PAPER] Chronos-2 was pretrained at context 2,048 and extended to 8,192 in a second stage. "Extending the context length yields gains, particularly on the GIFT-Eval benchmark", with numbers only in a figure. — [Chronos-2 arXiv 2510.15821](https://arxiv.org/html/2510.15821v1)
- [PAPER] ERCOT power-system benchmark, solar, TimesFM at its 512 maximum: context windows labelled 60 m / 6 h / 24 h gave nMAE 4.28 / 3.95 / 3.71. Longer was better, but the paper's text and table headers disagree on the window sizes. — [arXiv 2604.22077](https://arxiv.org/html/2604.22077)
- [PAPER] The ICF ablation found that related examples beat simply more own history: TimesFM with 2,048-step history scored 0.685 scaled MAE vs ICF 0.643 with 50 examples. — [ICF arXiv 2410.24087](https://arxiv.org/html/2410.24087)
- [CODE] Online rows carry a fixed 672-point history: "历史列 `X`（672 点，最后一点 = 起报时刻）". — `experiments/nwp_covariate/README.md` (field rules). The M5 config sets `dataset.context_len: 672` (`forecast_backbones/m5/config/finetune.yaml`).

### Inferences
- **Cost [EST]**, from the code structure: tokens = V × (L/32 + 192/32), where V is the variate count. `use_symmetric_averaging: true` doubles the batch (`backbone.py:136-139`).
  - Linear layers dominate. Per layer, sequence attention scores at n ≤ 132 patches cost about 2·n²·d·2 ≈ 0.09 GFLOP, vs about 4.3 GFLOP for the linear layers. So cost is roughly linear in token count.
  - With the current V = 3 (target + GHI + temp), tokens per sample (×2 for symmetric averaging) and relative cost:

    | L | tokens | relative cost |
    |---|---|---|
    | 672 | 162 | 1.0 |
    | 1344 | 288 | 1.8 |
    | 2016 | 414 | 2.6 |
    | 2880 | 576 | 3.6 |
    | 4032 | 792 | 4.9 |

  - At about 0.66 GFLOP/token forward and about 5 effective TFLOP/s fp32 on a V100 (35% of the 15.7 peak): about 45 rows/s at L = 672, about 18 rows/s at L = 2016.
  - The scored test set is about 10k rows (28 stations × ~1 year, one row per station-day). A zero-shot evaluation therefore takes minutes even at L = 2016.
  - LoRA training (backward through the frozen backbone, about 3x forward): about 1 s/step at batch 16 and L = 672, so about 1.1 h per seed for 4,000 steps; about 2.5-3 h at L = 2016. Activation memory grows linearly. These are estimates; time one step before planning.
- **Why longer context might help PV specifically.** Running RevIN stats over 7 days are dominated by that week's weather. A rainy week shrinks σ and distorts the scale the target is predicted in. 3 weeks dilutes this, and it also gives the target↔NWP relation more example days. Seasonal drift over 3 weeks is small at 22–25°N.
- **Why it might not.** M8 and the 7-day M5 already see the full diurnal cycle. FETS saturation was measured on load, where weekly seasonality needs ≥ 2 weeks. PV has no weekly cycle, so the saturation point may be shorter.
- **Cheap offline test (no upstream change).** Build an L-day context by stitching the same station's earlier rows: the row issued at t−7d holds history t−14d…t−7d. Archived NWP history columns can be stitched the same way, which keeps context NWP as *forecasts* (FETS-style).

### Gaps
- TimesFM-3's pretraining context length and context-length ablation are unpublished. Whether quality degrades beyond the pretraining length (seq-RoPE extrapolation) is unknown.
- No published context-length ablation exists for 15-min PV with NWP covariates for any TSFM.
- The V100 throughput figures are estimates, not measurements.

---

## Q2. Covariate engineering for a RevIN-per-variate TSFM: which power-unit covariates help, and how to build them without leakage

### Takeaway
Because TimesFM-3 removes each covariate's own location and scale, a covariate helps only if its **shape** (diurnal profile, cross-day relative level, nonlinearity) differs informatively from what the model already has.
- **Redundant:** a linear NWP→power map, `p_phys = cap·GHI/1000·(1+γ(Tc−25))` (shape ≈ GHI plus a few-percent midday derate), and capacity-scaled anything. They are affine or near-affine in GHI.
- **Most promising:**
  - (i) a **history envelope**: a per-slot high quantile of the station's own recent power. It carries the plant's true day-shape (tracker, bifacial, shading, clipping) and needs no coordinates;
  - (ii) an **NWP-implied power**: envelope × NWP clear-sky index, a nonlinear product that maps NWP cloudiness onto the plant's own shape.
- External evidence that physics-model power *as a covariate* helps TSFMs is **absent**. The closest work uses physics-model power as synthetic *target context* for cold start.

### Cited Findings
**The model's view of covariates [CODE]**
- Detrending is decided by a scale-free test (`std_det < linear_detrending_threshold·std_orig`, threshold 0.5) and the trend is extrapolated into past-future horizon values (`timesfm3/torch/model.py:472-520, 545-560`). RevIN uses the causal running stats (`model.py:219, 235`). The lookahead patch is normalised with the current patch's stats (`model.py:238-239`). This yields the affine invariance and the preserved relative level described in the header.
- The ResBlock input is `[patch(32) | lookahead(64) | masks(96)]` (`model.py:241-245`). Past-only covariates get `patch_is_target=True` (`model.py:577-583`), so their horizon is masked and predicted, then discarded.
- Our derived channels (`forecast_backbones/m5/transformer_models.py:428-432`):
  - `solar_elevation` (degrees) and `clear_sky_power = ghi_clear/1000·cap`;
  - for stations without coordinates, `p_clear` falls back to a clock box `cap·[ghi>0]` (`rowctx.py:116`), a rectangle that RevIN turns into a 0/1 day indicator;
  - both channels are appended after the NWP pairs, i.e. at variate positions 3 and 4.
- The repo already has a zero-shot covariate harness, `experiments/nwp_covariate/`:
  - candidates: `p_phys`, `kt_general`, `ghi_clear`, multi-source GHI fusions, grid spread, and `GHI_real` as past-only;
  - keep rule: mean gain ≥ 0.3 points, block-bootstrap 95% CI lower bound > 0, ≥ 65% of stations not worse (`experiments/nwp_covariate/config.yaml`, sections `experiments` and `evaluation`);
  - `p_phys` is exactly `cap × clip(GHI/1000, 0, 1.2) × [1 + γ(T_cell − 25)]` with Faiman cell temperature (`experiments/nwp_covariate/ops.py:336-353`).

**Physics-model power with TSFMs**
- [PAPER] Cold-start PV (440 sites): an OPAQUE generator (pvlib solar position, Hay–Davies transposition, NOCT cell temperature, PVWatts system model) or PVGIS produces a synthetic history. This history is used as **target-series context**: it "replaces the unavailable measured target history". It is **not** a covariate. TSFM covariates were daily aggregates (integrated irradiance, mean and max temperature, max wind, daylight duration).
  - Swapping generators changed little ("negligible MAE/RMSE variation and unchanged ranking"). RF MAE, OPAQUE vs PVGIS: TimesFM-2.5 0.599 vs 0.587, Chronos-2 0.537 vs 0.535, TabPFN-TS 0.514 vs 0.512.
  - The paper does not test physics power as a covariate.
  — [arXiv 2606.07457](https://arxiv.org/html/2606.07457v1)
- [PAPER] Two-stage PV: a plant model (MLP ensemble) learns weather→power from satellite weather and is fed HRRR NWP only at inference.
  - Perfect-weather MAE ≈ 2.8% of peak. With NWP, MAE rose 11% and 68% at the two sites, and the HRRR GHI bias passes straight through: the plant model is "deliberately plant-specific and not a bias corrector".
  - No single-stage comparison.
  — [arXiv 2603.04132](https://arxiv.org/html/2603.04132v1)
- [PAPER] Mayer & Yang (3 Hungarian plants, 2017-2020): post-processing "works better when applied to PV power output than to GHI". — [Applied Energy 371:123681 (OA copy)](https://real.mtak.hu/205425/1/Optimalplacetoapplypost-processinginthedeterministicphotovoltaicpowerforecastingworkflow.pdf); [IDEAS record](https://ideas.repec.org/a/eee/appene/v371y2024ics030626192401064x.html)
- [PAPER] ERCOT day-ahead solar: adding weather features improved every TSFM tested. Moirai-L nMAE 4.03 → 3.65, MOMENT 4.18 → 3.79, PatchTST 4.18 → 3.74; LSTM and CNN were unchanged. TimesFM was not in that table. — [arXiv 2604.22077](https://arxiv.org/html/2604.22077)
- [3RD] A thesis repo (day-ahead hourly rooftop PV; Chronos-2, TimesFM-2.5, TiRex) builds an empirical clear-sky power template from pvlib clear-sky GHI and applies clear-sky-index normalisation. It is a work in progress with no validated results. — [lorenzospolti/solar-pv-forecasting](https://github.com/lorenzospolti/solar-pv-forecasting)

**Envelopes without metadata**
- [PAPER] Statistical Clear Sky Fitting estimates a system's clear-sky power envelope "from measured power only… no knowledge of weather, irradiance data, or system configuration". The current Solar Data Tools version is based on "smooth, periodic quantile estimation". — [arXiv 1907.08279](https://arxiv.org/pdf/1907.08279); [Solar Data Tools demo](https://solar-data-tools.readthedocs.io/en/dev/getting_started/notebooks/clearsky_estimation_demo.html)
- M8 lessons (in-house):
  - a horizontal-GHI clear-sky ceiling is structurally low for tracking and bifacial plants (cap at 0.51·cap vs observed peaks 0.57–0.80·cap; relaxing it gave +1.7 points);
  - stations without coordinates need a history day-shape;
  - M5 beat M8 by 12 points at a no-coordinate station.
  — `docs/survey_m8_architecture_review.md` §7.2, §7.4

### Inferences
**Ranking of the candidate covariates in the brief, for TimesFM-3:**

1. **History envelope E (adopt, zero-shot first).** Definition: E(slot s) = q-quantile (q ≈ 0.9) of observed power at time-of-day slot s over the previous N days, lightly smoothed (±1 slot, max over neighbours to avoid jagged edges). In context and horizon it is a periodic or slowly varying series in the target's units.
   - What it adds that GHI lacks: the plant's own day-shape (tracker plateau, bifacial gain, morning/evening shading, inverter clipping, curtailment habits), already in power units. RevIN keeps that shape.
   - It works identically for stations without coordinates. It replaces `clear_sky_power`, whose horizontal-GHI shape is wrong for trackers and whose no-coordinate fallback is a box.
   - **Leak-free construction:**
     - (E0, zero extra data) compute E from the 672-step *target context* only (7 days ending at the issue time) and repeat it over context + horizon. Every input precedes the issue time.
     - (E1) rolling version with N = 14–30 full days strictly before each day. For context days this needs data older than the window, so precompute it in the parquet as a pair `power_env` / `power_env_predict`. For horizon days use only days < issue day.
     - Never include the horizon's observed power or the issue day's afternoon.
2. **NWP-implied power P_nwp = E(s) × k_nwp(t) (adopt, zero-shot first).** k_nwp = clip(GHI_nwp / GHI_ref, 0, 1.2) at dawn and dusk.
   - GHI_ref = pvlib clear-sky GHI where coordinates exist. Otherwise use the *NWP self-envelope*: per-slot q95 of the NWP-GHI history column over the context. That needs no coordinates and also absorbs NWP systematic bias in the clear-sky limit.
   - Because it is a product, it is not affine in GHI and gives new tokens. It is the input-side version of "correct NWP on the power side" (Mayer & Yang; M8 §7.5).
   - Mask k where GHI_ref < ~50 W/m² to avoid ratio blow-ups. Night is zero anyway.
3. **Clear-sky power, horizontal (g1, keep as a control only).** Under RevIN it is just the horizontal-GHI clear-sky shape: wrong for trackers, and meaningless (a box) for stations without coordinates. It differs from E mainly where E is better.
4. **Clear-sky index of NWP alone (k_nwp as its own row; try later, low).** It has no diurnal shape, so the model must learn the multiplication through variate attention. Ratios are noisy near sunrise and sunset, and the harness's `D_kt` experiment tests this. Prefer P_nwp, which does the multiplication explicitly.
5. **Linear NWP→power mapping, p_phys, capacity scaling (reject inside TimesFM-3).** Affine invariance [CODE] makes `a·GHI+b` identical to GHI. p_phys differs from GHI only by the temperature derate (γ = −0.004/°C, about −8% to −12% at midday with T_cell ≈ 45–55 °C) and the rarely binding 1.2 clip.
   - **Prediction:** in the harness, `D_phys` ≈ `C2_ghi` and `D_phys_ghi` ≈ `C2_ghi` within noise. If that holds, it validates the invariance argument cheaply.
   - A linear mapping still has value *outside* the backbone, as an XReg-style post-hoc ridge (earlier notes Q1/Q4).
6. **Previous-day power as a past-future covariate (reject).** At a 10:00 issue time with H = 192 (to 10:00 D+2), a lag-96 copy is unknown for every horizon slot after 10:00 D+1. A leak-free lag needs ≥ 192 steps, and that copy is already in the target context, which sequence attention reads directly.
7. **Temperature row (ablate).** The past-future pair `general_temp` costs one variate (+33% tokens) and is physically weak (derate of a few percent). Test swapping it for E at constant V = 3: `[target, ghi, E]` vs `[target, ghi, temp]`.

**Ordering matters (because of variate RoPE).** Append new derived covariates *after* the existing NWP pairs, as `transformer_models._covariates` already does. Then GHI and temperature keep variate positions 1 and 2, and an existing LoRA checkpoint sees the same relative positions for them.

### Gaps
- No published study feeds a physics-model PV power, or any power-unit envelope, as a *covariate* to TimesFM or Chronos-2 and ablates it. The rankings above rest on the code-level argument plus PV post-processing literature, not on TSFM experiments.
- Whether TimesFM-3 was pretrained with `use_rope_var=True` (as the forecaster runs it) cannot be confirmed from the blog. A cheap empirical check: swap the GHI and temperature rows and see whether zero-shot forecasts change.
- The history column `general_ghi` holds archived multi-source NWP medians (`experiments/nwp_covariate/README.md`, config `general_ghi: combine [ghi_gdfs, ghi_ec, ghi_sg] median`), not measurements, so context and horizon are consistent. This was confirmed for the online DS table only; the M5 training parquet was not re-checked.

---

## Q3. Neighbouring stations as extra variates (spatial information, up to 32 variates)

### Takeaway
**Reject for day-ahead now; revisit for ultra-short (g-series).** For D+1, neighbours' future power is unknown, so they can enter only as extra targets or past-only rows. Both are force-predicted, which costs compute, and both sit *between* the target and the NWP rows in the packing order. That shifts the NWP rows' RoPE variate positions. Published TSFM evidence shows only small gains from joint or multivariate modelling of solar, and spatial gains in the literature are for short horizons.

### Cited Findings
- [CODE] Packing order is `[targets | past_only | past_future]` (`model.py:455-468, 577-583`). Variate RoPE positions are `arange(v)` (`transformer.py:273-276`) with `use_rope_var=True` (`timesfm3_forecaster.py:306`). `max_variates: int = 32` appears only in `torch/configs.py:55`; grep found no other use, so it is a config field, not an enforced check in the decode path.
- [CODE] `decode` accepts `past_future_mask (b, w, context+horizon)` (`model.py:385, 463-467`). A neighbour could in principle be passed as a past-future row with its horizon masked, placed after the NWP rows. No pretrained behaviour is documented for that pattern.
- [PAPER] ERCOT: joint solar + wind + load output vs single-task for solar: Moirai-L nMAE 4.03 → 3.92, TFT 3.91 → 3.84, MOMENT 4.18 → 4.22, PatchTST 4.18 → 4.26. The paper calls the gains "moderate" or "relatively small". — [arXiv 2604.22077](https://arxiv.org/html/2604.22077)
- [PAPER] Chronos-2 group attention: on the multivariate subset "ICL provides only modest improvements". "The largest gains are observed on covariate-informed tasks." Group attention omits positional embeddings because series "lack a natural ordering". — [Chronos-2 arXiv 2510.15821](https://arxiv.org/html/2510.15821v1)
- [PAPER] FETS excluded multivariate mode because of "its limited improvement over the univariate setup". — [FETS arXiv 2604.22328v2](https://arxiv.org/html/2604.22328v2)
- [PAPER] Very-short-term PV (LSTM, not a TSFM): including highly correlated neighbouring plants' power "significantly improved prediction accuracy", with inverse-distance weighting used to fill missing neighbour data. — [Electricity 6(2):29](https://doi.org/10.3390/electricity6020029)
- [PAPER] SolarFM fuses TSFM zero-shot forecasts with historical observations for multi-region *irradiance*, not plant power. Abstract only seen. — [Applied Energy 2026, SolarFM](https://www.sciencedirect.com/science/article/abs/pii/S0306261926001844)

### Inferences
- What neighbours could add day-ahead: the regional cloud regime is already in the NWP, which our 28 stations share at grid scale. A neighbour's *history* adds little that the station's own history lacks, except during the station's own outages or data gaps.
- Risks specific to TimesFM-3:
  - (a) moving the NWP rows from positions 1–2 to positions 1+k changes the zero-shot target↔NWP relation the model learned, a covariate-shift in RoPE space;
  - (b) each neighbour adds 27 patches per sample, and up to 32 rows means up to about 10x tokens;
  - (c) without identity, the model cannot tell "near and correlated" from "far" neighbours except by content;
  - (d) a neighbour's own outages and curtailment leak noise into the target.
- If revisited (ultra-short, issue-time-sensitive): use one or two most-correlated neighbours as a past-future row with the horizon masked, appended last. Compare against a "same neighbour, time-shuffled" control to separate information from distribution shift.

### Gaps
- No study found that uses neighbouring PV plants as variates in any TSFM (TimesFM-3, Chronos-2, Moirai) and ablates it.

---

## Q4. Analog / similar-day retrieval (RAF, TimeRAF, PARA-PV, NWP-similarity) and how to inject it into TimesFM-3

### Takeaway
Day-ahead, the right retrieval key is **tomorrow's NWP** (known), not the recent power shape. That is the classic analog ensemble (AnEn), which beats quantile regression on rare events and at low sun elevation. For TimesFM-3, inject analogs as a **past-future covariate row**, not as a RAF-style prefix. The row holds realised power of NWP-analog days, aligned for every day in context and horizon, so the model can calibrate "analog vs actual" in context. **Try later**: it needs a per-station history store beyond the 7-day window, which is new data plumbing online.

### Cited Findings
- [PAPER] RAF (AISTATS 2026):
  - retrieves the best-matching historical segment (Chronos-Base encoder embedding, ℓ2 / approximate nearest neighbour; k = 1 in the main setup);
  - concatenates `[retrieved context, retrieved future, query context]` as the model input, after instance normalisation and level alignment to the query's first value, with no separator token;
  - naive RAF zero-shot average gains: TimesFM +16.7% WQL / +4.8% MASE, Chronos +15.8% / +12.4%, Moirai +6.0% / +4.5%;
  - Chronos-Mini "completely fails" the noiseless synthetic retrieval task, and gains were "relatively small" on Traffic and Covid Deaths;
  - no solar dataset.
  — [RAF arXiv 2411.08249](https://arxiv.org/html/2411.08249)
- [PAPER] PARA-PV (two State Grid farms, 50 MW and 35 MW, 15-min, horizons 4–96 steps):
  - retrieval keys: patch-embedding shape cosine, mean power level, a 4-state operating-state match and a circular time-of-day distance (no NWP key);
  - analogs are fused by a learned gate (output-side convex interpolation);
  - frozen Chronos-2 enters through an output residual adapter;
  - average MSE: PARA-PV 0.0962 vs w/o Chronos 0.1132 vs w/o PA-RAL (retrieval module) 0.1045 on Data 1; 0.0689 / 0.0754 / 0.0715 on Data 2;
  - whether its weather inputs are observed or forecast is not stated.
  — [PARA-PV arXiv 2607.08079](https://arxiv.org/html/2607.08079v1)
- [PAPER] Analog ensemble for solar power (Alessandrini et al., Applied Energy 2015):
  - draws on a historical archive of deterministic NWP forecasts paired with observed power, so the ensemble is built from past production on NWP-similar occasions;
  - "performs as well as QR for common events… more skill for rare events and during hours with a low solar elevation".
  — [Penn State record](https://pure.psu.edu/en/publications/an-analog-ensemble-for-short-term-probabilistic-solar-power-forec/); [IDEAS](https://ideas.repec.org/a/eee/appene/v157y2015icp95-110.html)
- [PAPER] TimeRAF (Channel Prompting module) and TS-RAG are general retrieval-augmented TSFMs, as noted in the earlier notes; neither is PV-specific. — [TimeRAF arXiv 2412.20810](https://arxiv.org/pdf/2412.20810); [TS-RAG arXiv 2503.07649](https://arxiv.org/pdf/2503.07649)
- [PAPER] A 2025 day-ahead PV study does NWP-similarity KNN after an XGBoost-reweighted feature space and averages the analogs' measured data for augmentation (search snippet only). — [Energy Informatics 2025](https://link.springer.com/article/10.1186/s42162-025-00531-7)

### Inferences
**Recommended injection design: "analog-power row" (A).** For each day d in the context and for the target day(s):
1. Score candidate days d' from a pool of the same station's days **strictly before the issue day** (excluding d itself) by NWP similarity:
   - L2 distance on the daylight NWP-GHI profile, normalised by the NWP self-envelope (so it is a k profile), plus a small weight on daily mean temperature;
   - restrict the pool to ±45 days of day-of-year within any year, to control the seasonal envelope.
2. A(d, s) = mean of realised power at slot s over the top-K analogs, with K = 5–10. AnEn practice; K = 1 is RAF's setting but noisy for PV.
3. Feed A as a past-future pair (`analog_power` 672 / `analog_power_predict` 192).

In the context the model sees, day by day, how analog power related to what actually happened (in-context calibration). In the horizon it gets an NWP-conditioned, plant-shaped prior. Because A is realised power of real days, it carries the plant's shape and the NWP's regime-specific bias, which is why AnEn helps at low elevation and on rare events. Optionally add a spread row (std over the K analogs), but each row costs +33% tokens.

**Why not RAF-style prefix (reject for TimesFM-3):**
- (a) a prefix is indistinguishable from older history under causal sequence RoPE, and TimesFM-3 has no separator token (ICF separators exist only in the unreleased ICF checkpoint);
- (b) the NWP rows would need matching prefix segments, which breaks the context/covariate alignment that `TimesFM3Forecaster` enforces;
- (c) prefixing distorts the running RevIN stats of the target.

RAF's own gains were for univariate models without a covariate pathway.

**Leakage rules.** Pool = days < issue day, so for D+1 at 10:00 D the latest usable full day is D−1. When backtesting, the pool must grow over time (no future days), and NWP for pool days must be the archived forecasts with the same lead, not analyses.

**Online cost.** A per-station rolling store of (daily NWP-GHI profile, daily power profile) for ≥ 1 year, plus a KNN per issue: trivial compute, real plumbing. Offline, a retrieval over the training parquet is enough to test.

### Gaps
- No published study combines NWP-keyed analog retrieval with any TSFM, nor injects analogs as a covariate row. Expected gain size is unknown. AnEn's advantage is concentrated in rare events and low-sun hours, which the 南网 daily RMSE with a 0.2·cap floor may weight less.

---

## Q5. In-context fine-tuning (TimesFM-ICF): released weights or reproductions as of 2026? Can related series as TimesFM-3 variates mimic it?

### Takeaway
No ICF weights or code are public as of October 2026, and ICF was built on a TimesFM-1-era 200M model, not TimesFM-3. TimesFM-3's variate axis is not a substitute for ICF's sequence-axis examples. Variates are aligned per patch index and mixed only at the same time step, so an "example series" placed as a variate is just a covariate. The analog-power row (Q4) is the closest workable approximation: a same-station, NWP-matched "example" aligned in time.

### Cited Findings
- [PAPER] ICF packs up to 50 examples, each up to 640 points, separated by a learnable 1280-d separator token, with continued pretraining on a NoPE TimesFM base (200M, 20 layers). Monash scaled MAE: ICF 0.643 vs full FT 0.663 vs base 0.694. ICML 2025. — [arXiv 2410.24087](https://arxiv.org/html/2410.24087); [Google Research blog](https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/)
- Release status: a search for ICF checkpoints or code found only secondary write-ups. None mention a GitHub release, HF model card or checkpoint. The timesfm README covers 1.0/2.0/2.5/3.0 releases and a 2.5 LoRA example, with no ICF. — [dejan.ai summary](https://dejan.ai/blog/timesfm-icf/); [MarkTechPost](https://www.marktechpost.com/2025/09/23/google-ai-research-introduce-a-novel-machine-learning-approach-that-transforms-timesfm-into-a-few-shot-learner/); [timesfm GitHub](https://github.com/google-research/timesfm)
- [CODE] TimesFM-3 has no separator token or example-boundary mechanism. The sequence attention is causal over patches of one variate. Variate attention mixes variates at the same patch index (`transformer.py:505-531`).
- [PAPER] Chronos-2's "cross-learning" across series in a batch improves univariate skill scores, "especially strong on Chronos Benchmark II", with numbers only in figures. — [Chronos-2 arXiv 2510.15821](https://arxiv.org/html/2510.15821v1)

### Inferences
- Putting another station's or another period's series as a TimesFM-3 variate is a time-aligned covariate. For the same station's past days that is pointless (they are in the context already) and for other stations it is the Q3 case. ICF's benefit came from the model *learning* to read separated examples; TimesFM-3 was not trained that way.
- What we can get: (i) the analog-power row (Q4) gives the model an NWP-matched example day per context day, aligned so variate attention can compare them; (ii) LoRA fine-tuning on our 28 stations is already the "per-dataset FT" that ICF matched.

### Gaps
- No third-party reproduction of ICF on TimesFM 2.5 or 3 was found. Whether Google ships ICF-style few-shot in TimesFM-3's BigQuery product is unknown.

---

## Q6. Interaction with LoRA fine-tuning: which input-side changes need re-fine-tuning, and which can be screened zero-shot first?

### Takeaway
All input-side changes can be screened zero-shot in minutes with the existing `experiments/nwp_covariate` harness or M5 `z0_zeroshot`. Zero-shot is a necessary *filter* for cheap ideas but not a valid *rejection* test for physically strong covariates. Models often exploit covariates only after fine-tuning. Any change that keeps the existing rows' positions (new rows appended last) can also be evaluated on the current LoRA checkpoint without retraining. Context-length changes and anything adopted for production need a LoRA retrain with multiple seeds.

### Cited Findings
- [CODE] M5 LoRA targets are `[query_proj, key_proj, value_proj, out_proj]`, with `unfreeze_output_head: true` (`forecast_backbones/m5/config/finetune.yaml`). These names match both `seq_attn` and `var_attn` (`timesfm3/torch/transformer.py:425-455`). The `pre_transformer_resblock` that embeds every covariate's patch and lookahead stays frozen (earlier notes Q1).
- [CODE] Derived covariates are appended after the `features.past_future` pairs (`transformer_models.py:419-433`). `features.derived` is part of the config, and train.py logs it into run summaries (`train.py:389`).
- [PAPER] Independent demand study: Moirai zero-shot with all features "generally got worse", and only fine-tuned Moirai benefited, at short horizons. — [arXiv 2602.05390](https://arxiv.org/html/2602.05390v1) (via earlier notes)
- [PAPER] ERCOT solar: TimesFM zero-shot nMAE 9.25 vs 3.71 fine-tuned (100% data). Zero-shot TSFMs are "not yet competitive" without adaptation. — [arXiv 2604.22077](https://arxiv.org/html/2604.22077)
- [CODE] Harness keep rule: ≥ 0.3-point mean gain, 95% block-bootstrap CI lower bound > 0, ≥ 65% of stations not worse (`experiments/nwp_covariate/config.yaml` → `evaluation.keep_rule`). Memory notes: the M8 protocol uses 5 seeds, and round-2 variants were ≈ noise.

### Inferences
**Three-stage test protocol for each input-side idea:**
1. **Zero-shot screen**, minutes on one V100 [EST]: harness or `z0_zeroshot` variant, baseline `C1_online`, keep rule as configured. Keep anything that passes. For physically strong candidates (E, P_nwp, A), also carry the ones that are neutral zero-shot to stage 2.
2. **Frozen-LoRA check**, minutes: apply the current best LoRA checkpoint with the new row appended last. NWP positions are unchanged, so the checkpoint's variate-attention adapters still match. This is valid for appended covariates only, not for context-length changes or reordering.
3. **Retrain LoRA**, about 1 h per seed at L = 672 [EST], 3–5 seeds, compared with the same seeds of the baseline.

Accept only if the retrained gain clears the noise band that rounds 1–2 established.

**Expectations by change type:**
- *New covariate row*: zero-shot may under-use it (the frozen embedding sees an unfamiliar series). LoRA on `var_attn` is exactly what learns to read it, so the main decision rests on stage 3.
- *Context length*: LoRA was trained at 672 with RevIN stats of 7-day windows. A 2016 context changes those stats and the number of seq-RoPE positions. Retrain at the new length; also train-short/test-long only as a diagnostic.
- *Removing temperature or reordering*: it shifts variate positions, so it always needs a retrain.

### Gaps
- No published study measures how LoRA on a covariate-native TSFM changes the value of individual covariates. The "zero-shot neutral, fine-tuned positive" pattern is documented for Moirai, not TimesFM-3.

---

## Q7. Final shortlist for M5: adopt / try later / reject, with implementation and cheap tests

### Takeaway
**Adopt for testing now** (zero-shot → frozen-LoRA → retrain):
- (1) history envelope row E;
- (2) NWP-implied power row P_nwp = E × k_nwp;
- (3) context-length sweep 672 → 2016.

**Try later:**
- (4) NWP-analog power row (AnEn-style);
- (5) temperature→envelope swap at constant V;
- (6) k_nwp row alone;
- (7) neighbours for ultra-short only.

**Reject:** linear NWP→power / p_phys / capacity-scaled rows inside the backbone; previous-day power row; RAF prefix and ICF-style examples; neighbours for day-ahead; measured GHI as past-only for day-ahead.

### Cited Findings
- Code facts used: `timesfm3_forecaster.py:32, 305-306`; `model.py:219-245, 472-583`; `transformer.py:273-276, 505-531`; `forecast_backbones/m5/transformer_models.py:419-433`; `rowctx.py:78, 116-119` (`envelope_used` field exists, "M5 恒 False"); `config/finetune.yaml` (`features.derived: {solar_elevation: false, clear_sky_power: false}`); `experiments/nwp_covariate/config.yaml` and `ops.py:336-353`.
- External evidence: as cited in Q1–Q6 (FETS saturation; ERCOT weather gains; cold-start synthetic history as context; Mayer & Yang power-side correction; AnEn; RAF; ICF unavailable).

### Inferences

| # | Method | Verdict | Why (scenario-specific) | Implementation in our code | Cheapest test |
|---|---|---|---|---|---|
| 1 | **History envelope row E** (per-slot q0.9 of own power, past N days) | **Adopt (test first)** | It carries the plant's true day-shape (tracker, bifacial, clipping) that horizontal clear-sky and NWP-GHI lack, and works at no-coordinate stations (the M8 §7.4 lesson). Leak-free from the context alone. | New key `features.derived.history_envelope: {enabled, q: 0.9, days: 7, smooth_slots: 1}` in `config.py` DEFAULT_CFG and `finetune.yaml`. In `transformer_models._covariates`, compute E from `tgt[:, -L:]` reshaped to (days, 96) with the slot phase taken from `ts_col`, tile it over L+H and append it **after** the NWP pairs. Set `RowCtx.envelope_used`. For N > 7, precompute `power_env`/`power_env_predict` columns offline (days strictly before each day). | Harness: add `op: envelope` to `experiments/nwp_covariate/ops.py` and experiments `E_env = C1_online + [power_env]` and `E_env_only = [general_ghi, power_env]`, then run `search_m5.py`/`run_experiments.py`. Then M5 variant `h1_envelope.yaml` (zero-shot `z0` + LoRA). |
| 2 | **NWP-implied power P_nwp = E × clip(GHI_nwp/GHI_ref, 0, 1.2)** | **Adopt (test first)** | It is nonlinear in GHI, so it is not erased by RevIN, unlike p_phys. It puts NWP cloudiness onto the plant's own shape: power-side NWP correction (Mayer & Yang) done at the input. GHI_ref = pvlib clear-sky when coordinates exist, otherwise the NWP self-envelope (q95 of the NWP-history column per slot), so it needs no coordinates. | `features.derived.nwp_power_env: {enabled, ghi_col: general_ghi, ref: auto, kmax: 1.2, min_ref_wm2: 50}`. Compute it in `_covariates` from the already-loaded GHI pair plus E. Append after E. | Harness op `env_power`, experiments `E_envpow = [env_power, general_ghi]` and `E_envpow_t = C1_online + [env_power]`. Zero-shot, then LoRA variant. |
| 3 | **Context length 672 → 1344 / 2016 (/ 2880)** | **Adopt as an experiment**; production only if the gain clears the noise band | FETS saturation is at about 2,000 steps (load, not PV). Longer context stabilises RevIN stats against a single rainy week. The cap of 15,360 is not binding. Cost ≈ 1.8x / 2.6x / 3.6x [EST]. | `dataset.context_len`. Needs parquet rows with longer `observe_power`/`general_*` history (or offline stitching of consecutive rows of the same station). The backbone already pads and truncates. Online DS rows have only 672 points, so production needs an upstream change. | Zero-shot sweep on the stitched test set (minutes). If ≥ 0.3 points, retrain LoRA at the winning length with 3 seeds. |
| 4 | **NWP-analog power row A** (top-K = 5–10 same-station days by NWP-k-profile similarity, realised power, aligned per day for context + horizon) | **Try later** | The day-ahead-appropriate retrieval key (tomorrow's NWP). AnEn helps on rare events and at low sun. As a covariate row, the model can calibrate analog vs actual in context. Needs a history store beyond the window. | Offline: precompute `analog_power`/`analog_power_predict` in the training/test parquet (pool = days < issue day, same-lead archived NWP). Add as a `features.past_future` pair, no code change. Online: a per-station rolling store of daily NWP and power profiles plus KNN. | Harness experiment with the precomputed column. Compare K = 1 vs 5 vs 10 and a "random-day" control row of the same marginal to rule out a pure envelope effect. |
| 5 | **Swap temperature for E** (keep V = 3) | Try later (cheap ablation) | Temperature is physically weak (derate of a few percent) and costs +33% tokens. E is likely more informative. But removing it shifts positions, so it needs a retrain. | Variant with `features.past_future: [[general_ghi, general_ghi_predict]]` plus `history_envelope` enabled. | Harness `C2_ghi + [power_env]` vs `C1_online`. |
| 6 | **k_nwp row alone** (NWP clear-sky index) | Try later, low | No diurnal shape, and ratio noise at dawn and dusk. P_nwp already encodes it. | Already a harness candidate (`kt_general`, `D_kt`). | Run `D_kt` in the existing harness; no new code. |
| 7 | **Horizontal clear-sky power / elevation rows (g1)** | Keep as control; do not prioritise | Under RevIN only the horizontal-GHI shape survives. It is wrong for trackers and a box at no-coordinate stations (`rowctx.py:116`). E dominates it. | Existing `g1_solar_cov.yaml`. | Compare g1 vs h1 (E) per station, especially tracker and no-coordinate stations. |
| 8 | **Linear NWP→power map, p_phys, cap-scaled rows** | **Reject (inside backbone)** | Positive-affine invariance of detrend + RevIN [CODE]. p_phys ≈ GHI apart from a midday derate of a few percent. | n/a. Keep linear MOS only as a post-hoc XReg-style ridge outside the backbone. | Confirmatory only: harness `D_phys`/`D_phys_ghi` should equal `C2_ghi` within noise. If not, revisit the invariance argument. |
| 9 | **Previous-day power row** | **Reject** | Lag-96 is unknown in the horizon at a 10:00 issue time. A leak-free lag of ≥ 192 duplicates the context that sequence attention already reads. | n/a | n/a |
| 10 | **Neighbour stations as variates** | **Reject for D+1; try later for ultra-short** | No future values day-ahead. Neighbour rows shift NWP RoPE variate positions and force-predict extra rows. TSFM multivariate gains for solar are small (ERCOT, Chronos-2, FETS). | If revisited: neighbour as a past-future row with horizon `past_future_mask=True`, appended last (custom `decode` call in `backbone.predict_quantiles`). | Ultra-short g-series only, with a time-shuffled neighbour control. |
| 11 | **RAF-style retrieval prefix / ICF examples** | **Reject** | No separator token in TimesFM-3. A prefix is indistinguishable from history, breaks covariate alignment and distorts RevIN. ICF weights are unreleased. | n/a | n/a |
| 12 | **Measured GHI (`GHI_real`) as past-only** | Reject for D+1 (low value) | Past-only rows are predicted as extra targets (compute cost). Day-ahead skill comes from the forecast NWP, not past measured irradiance. | Already harness experiment `D_obs`. | Run `D_obs` only for completeness. |

**Order of work (cheapest, highest prior first):**
1. **Free sanity checks** on the existing harness, no new code: run `D_phys`, `D_phys_ghi` and `D_kt` to confirm invariance and the ratio-row hypothesis. Run a GHI↔temperature row-swap check to confirm that variate order matters (`use_rope_var=True`).
2. **Envelope rows:** add the `envelope` and `env_power` ops to the harness (≈ 50 lines), run zero-shot on E, P_nwp and their combinations, and report the station split by tracker / no-coordinate.
3. **Context sweep:** stitch rows offline and run 672 / 1344 / 2016 zero-shot.
4. **LoRA retrain:** for survivors (and E/P_nwp regardless of the zero-shot result), port the rows into `features.derived` and train an `h1_envelope` variant with 3–5 seeds against the same seeds of the current recipe.
5. **Analogs:** only if 2–4 show that power-unit priors matter. Build the offline analog column, then decide on online plumbing.

**Expected magnitude [EST]:**
- M8 put the whole NWP pathway at about 2.6 points. Plant-shape priors matter most where geometry is wrong (trackers, no-coordinate stations): +1.7 at one station from relaxing the ceiling, and the 12-point M5-vs-M8 gap at 国能和煦.
- Realistic fleet-average gains from rows 1–2 are therefore sub-point to about 1 point, concentrated at a few stations. The decision rule should use per-station wins as well as the mean, as the harness already does.

### Gaps
- None of the adopt items has direct external TSFM evidence for PV. Their support is code-level reasoning, in-house M8 lessons and PV post-processing literature. Hence the zero-shot-first protocol.
- Whether the M5 training parquet (as opposed to the online DS table) stores more than 672 history points was not checked. This decides whether the context sweep needs stitching.
- Tracker and bifacial metadata per station is not in `station_meta.csv` as far as the M8 review shows, so the per-station breakdown in step 2 may need manual labelling.
