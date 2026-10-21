# Objective alignment (training on the 南网 metric) and an RL/preference re-check for M5 (TimesFM-3, 9 quantiles, LoRA)

Research date: 2026-10-15. This builds on `reports/M5 基础模型微调方法调研.md` and `research_notes/M5 基础模型微调方法调研/rl_preference.md` and does not repeat them. Those already cover TimeRFT, PostTime, REATS, Cazaux et al. (2603.20063), Beichter et al. DFF (2503.01936), value-oriented forecasting (2309.00803, 2405.09004), Hirsch–Ziel (2604.19580), and Gneiting 2011 consistency.

Evidence labels:
- **[DOC-VERIFIED]**: I downloaded the official PDF from nfj.nea.gov.cn and read the extracted text myself.
- **[SNIPPET]**: search-engine summary or secondary source only.
- **[MATH]**: my own derivation, checkable on paper. It is not a literature claim.

Repo facts I checked:
- `pipelines/nanwang_metric.py` short-term score: daily score = (1 − sqrt(mean(((P_real − P_pred)/max(P_real, 0.2·cap))²)))×100. It uses 96 points including night, 10:00 issue (`predict[55..150]`), and floors the daily score at 0.
- `forecast_backbones/m5/decision.py`: `p_star` = Σ y·w / Σ w on a 0.005-step quantile grid with w = 1/max(y, 0.2·cap)². The grid is built by linear interpolation of the 9 quantiles and is **flat beyond q0.1 and q0.9**.
- `forecast_backbones/m6/losses.py::nanwang_r2_loss(p, y, cap, mask)` = Σ mask·((y − p)/max(y, 0.2·cap))² / Σ mask. It is a pooled per-point squared relative error: no per-day grouping, no sqrt, no threshold.
- `forecast_backbones/m5/losses.py::pinball_loss` is the plain 9-level pinball on cap-normalized values, masked and averaged over (t, τ).

---

## Q1. What are the real 南方区域 "两个细则" rules for day-ahead PV accuracy (2022 and 2025 versions)?

### Takeaway
Both the 2022 version (effective 2023-01-01) and the 2025 revision (published 2025-02-14) define day-ahead PV accuracy exactly as our repo does: per day, 1 − sqrt of the mean over 96 points of ((PM − PP)/max(PM, 0.2·Cap))². The rules differ from our repo metric in four ways:
- **Threshold.** The threshold is 65%, much lower than the old 85%.
- **Penalty form.** The penalty is a per-day hinge: each started percentage point below 65 costs Cap × 1 × A3 hours. A3 = 1 in 2022 and 0.2 in 2025, in all five provinces.
- **Monthly cap.** The 2025 revision caps all forecast penalties at 5 Cap·h per station per month.
- **Low-power exclusion.** Points where forecast, available power and actual power are *all* below 10% of capacity "may be excluded" (可不纳入). In practice this means nights.

So the denominator question is settled: it is max(P_real, 0.2·Cap), applied per point. The loss form must reflect a per-day hinge at 65, not only the mean daily score.

### Cited Findings
**2025 revision (currently applicable) [DOC-VERIFIED]**
- The notice "南方区域'两个细则'（修订版）" was posted 2025-02-14. It has 6 attachments. Attachment 4 is 《南方区域光伏发电并网运行及辅助服务管理实施细则》修订版. The notice page states no effective date. — [nfj.nea.gov.cn notice](https://nfj.nea.gov.cn/xwzx/tzgg/202502/t20250214_276632.html)
- Scope: 广东、广西、云南、贵州、海南. Applies to centralized PV directly dispatched by prefecture-level or higher dispatch. — [Attachment 4 PDF, 第一、二条](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585645833212.pdf)
- 第二十二条（三）: the station reports a medium-term forecast "每日 9 时前" (before 09:00 daily) covering the next 240 h, which "应含短期功率预测结果". The ultra-short forecast (15 min–4 h) is reported every 15 min. Resolution is 15 min. — [same PDF](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585645833212.pdf)
- 第二十二条（五）, verbatim: "光伏电站短期功率日前预测结果按日进行统计，按月进行考核。"
  - The formula is an embedded equation, garbled in the text layer but decodable. It is a two-branch per-point expression under one (1/96)Σ and one sqrt:
    - ((P_Mi − P_Pi)/P_Mi)² when P_Mi ≥ 0.2Cap;
    - ((P_Mi − P_Pi)/(0.2Cap))² when P_Mi < 0.2Cap.
  - "限电时刻 PMi 为 i 时刻的可用功率，不限电时刻 PMi 为 i 时刻的实际功率 … Cap 为光伏电站总当月装机容量。"
  — [same PDF, p.9](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585645833212.pdf)
- Threshold and penalty, verbatim: "光伏电站短期功率预测结果中日前预测准确率应不低于 65%，每降低一个百分点（不足一个百分点的按一个百分点计）按当月装机容量×1×A3 小时的标准进行考核。" — [same PDF, p.9](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585645833212.pdf)
- 附表1 (2025): A3 (短期预测考核系数) = **0.2** in all five provinces. A4 (超短期) = 0.2. A2 (中期) = 1. — [same PDF, p.33](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585645833212.pdf)
- Medium-term day 4 (hours 73–96): the threshold is 45%, and the penalty is Cap × 0.2 × A2 h per point. It uses the same formula structure. — [same PDF, p.8–9](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585645833212.pdf)
- Ultra-short (2025): P_Pi is "i 时刻前 15 分钟至前 4 小时期间 16 次对 i 时刻的超短期功率预测平均值". **The forecasts are averaged first, then scored.** The threshold is 70%, and the penalty is Cap × 1 × A4 h per point. The 2025 ultra-short formula itself is an image I could not extract. — [same PDF, p.9–10](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585645833212.pdf)
- 第二十二条（七）, verbatim: "当预测功率、可发功率和实际功率均小于当月装机容量 10%时，可不纳入准确率统计。" — [same PDF, p.10](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585645833212.pdf)
- （八）: further exclusions are force majeure, "光伏受限时段及后一个小时" (curtailment periods plus one hour), grid-side faults, AVC tests, and approved forecast-system maintenance. — [same PDF, p.10](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585645833212.pdf)
- （十）, verbatim: "单月对单个新能源场站超短期、短期、中期功率预测上报率、准确率的总考核小时数不超过 H1 小时，暂取值 5。" — [same PDF, p.10–11](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585645833212.pdf)

**2022 version [DOC-VERIFIED]**
- Notice 南方监能市场〔2022〕91号, dated 2022-06-13. It ran 6 months of simulation, then "2023 年 1 月 1 日起正式执行". — [notice PDF](http://nfj.nea.gov.cn/file_upload/20220614/2781655174179231_908b93fb-a510-4ff6-a159-684e013d4483.pdf); [notice page](https://nfj.nea.gov.cn/xwzx/tzgg/202310/t20231009_140145.html)
- 2022 PV attachment, 第二十三条（五）: same formula and same 65% threshold, with "按当月装机容量×1×A3 小时". 附表1 (2022): A3 = **1** in all five provinces, and A4 = 1. — [2022 PV attachment PDF](http://nfj.nea.gov.cn/file_upload/20220613/93581655118302306_f33dbd04-e92e-4188-9dba-d0b1f58aa316.pdf)
- 2022 ultra-short is scored on the **4th-hour** forecast (PPi = "4 小时前对 i 时刻的预测功率"), with the same RMSE-type formula and a 70% threshold. The same 10% low-power exclusion (七) applies. **The 2022 text has no H1 monthly cap clause** (searched; none found). — [2022 PV attachment PDF](http://nfj.nea.gov.cn/file_upload/20220613/93581655118302306_f33dbd04-e92e-4188-9dba-d0b1f58aa316.pdf)

**Changes 2022 → 2025 [DOC-VERIFIED, diff of the two texts]**
- A3: 1 → 0.2.
- A4: 1 → 0.2.
- Ultra-short: from 4th-hour point to the mean of 16 rolling forecasts.
- New: H1 = 5 h monthly cap.
- Unchanged: the 65% threshold and the max(P, 0.2Cap) formula.

**Old (superseded) 2018 rule [SNIPPET]**
- "日预测准确率小于85%的 … 考核电量为(85%-日准确率)×Cap×1小时 … 按日统计，按月考核." This confirms the meaning of "按日统计，按月考核": a penalty is computed per day and summed over the month. — [北极星 2018 repost](https://mguangfu.bjx.com.cn/mnews/20180118/875022.shtml) (snippet only)
- **Contradiction note.** One search-engine summary claimed that "65% is a wind ultra-short threshold, PV day-ahead is 85%". That claim mixes the 2018 PV text with the 2025 wind text. The official 2022/2025 PV PDFs say 65% for PV day-ahead.

**Pricing and settlement [DOC-VERIFIED]**
- Main rules 第一百一十二条: the assessment unit price is the province's previous-year average on-grid price. In spot-settled regions, only *unplanned outage* periods use the nodal real-time price. — [2025 main rules (附件1) PDF](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585644795756.pdf)
- 第一百零八条: assessment fees are "全部向相关并网主体进行返还奖励", meaning they are redistributed to grid-connected entities. — [same PDF](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585644795756.pdf)
- 第一百零六条: "已通过市场机制完全实现的，不在本细则中重复考核。电力现货试点地区电力调度机构可 … 提出现货市场环境下并网运行考核条款调整建议，并报能源监管机构同意后施行。" — [same PDF](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585644795756.pdf)
- 第一百零九条: when one event triggers several clauses, the penalty is not cumulative and only the largest applies. — [same PDF](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585644795756.pdf)

**Spot market context [SNIPPET]**
- Guangdong (粤发改价格〔2025〕263号, implementing 136号文): all new-energy on-grid energy enters the market from 2025-11-01. The spot price ceiling is 1.8 元/kWh and the floor is −0.05 元/kWh. Existing projects get a mechanism price referenced to 0.453 元/kWh with contract-for-difference settlement. — [新华网 2025-10-23](https://www.news.cn/energy/20251023/a8e99f614d894d6ca664e36ef0227792/c.html)
- A Yunnan document reportedly says that in 2025 new-energy stations are not yet assessed in spot mode; its day-ahead/real-time accuracy items are marked "本次不执行". — [Yunnan PDF (snippet only)](https://nyj.yn.gov.cn/zfxxgk/fdzdgknr/zdjcygk/202506/W020260709837818963735.pdf)
- A Guangdong H1-2026 market report reportedly gives a new-energy average offer of −0.035 元/kWh and a settled average price of 0.269 元/kWh. It also says PV is mostly exposed to spot because PV mid/long-term contracts are scarce. — [secondary summary in search](https://www.jtcopper.com/wp-content/uploads/2026/03/2025广东电力行业市场年度报告.pdf) (not verified)

### Inferences
- **The repo formula is correct.** The per-point denominator is max(P_M, 0.2·Cap). Cap is the *monthly* installed capacity; we use the fixed station_meta `cap_mw`, which is fine unless capacity changes mid-series. P_M is *available* power during curtailment, so curtailed hours plus one hour are excluded. Training labels should mask curtailment plus 1 h if curtailment flags exist.
- **Night inclusion is the main protocol difference.**
  - Under (七), points with forecast, available and actual power all below 10% Cap "may" be excluded. With production night-zeroing, every night point satisfies this, so dispatch most likely scores only n_d ≈ daytime points.
  - The formula text still writes 1/96. Whether n is 96 or n_d after exclusion is not stated (see Gaps).
  - If n = n_d ≈ 40–52, the official daily RMSE term is about sqrt(96/n_d) ≈ 1.4× ours. Our "76" could then be about 66 on the official scale, i.e. near the 65 line. **This changes which days are penalized.** It needs a quick re-score with a night-excluded variant (exclude t where max(PP, PM) < 0.1·Cap).
- **The economic objective is a per-day hinge, not the mean score.**
  - Monthly cost (2025) = min(5·Cap, Σ_d ceil(max(0, 65 − ACC_d)) × 0.2·Cap + other forecast items) MWh, priced at last year's average on-grid price (Guangdong coal benchmark ≈ 0.453 元/kWh for scale).
  - Example: a 100 MW station loses 20 MWh (≈ 9k 元) per point-day below 65. The month is capped at 500 MWh (≈ 225k 元), and fees are partly returned.
  - So penalties come only from bad days (overcast, convective, rain onset). Days at 70–90 cost nothing.
- **The ultra-short repo metric differs from the 2025 rule.** Ours averages |error| over 16 leads (MAE-type). The rule averages the 16 forecasts and then scores, and the formula type is unverified. This is out of M5's day-ahead scope, but worth flagging.
- **Issue time.** The rule says "每日 9 时前上报". The repo uses a 10:00 issue for D+1. Data available at 10:00 (e.g., an NWP run) could make our evaluation slightly optimistic.

### Gaps
- Whether excluded low-power points reduce n (n_d) or stay in the 1/96 sum with zero error is not stated in the text. Dispatch technical-support-system (两个细则技术支持系统) implementation documents were not found.
- The 2025 revision's exact effective date was not found. The notice page gives none, and 第六十三条 of the PV attachment says "自发布之日起施行，有效期 5 年". Earlier search snippets claimed "自 2025 年 1 月 1 日起实施", but that was not verified in an official text.
- Province-level spot-market adjustments to forecast assessment (Guangdong after 2025-11-01) were not found in primary text.
- The 2025 ultra-short formula image could not be extracted.

---

## Q2. How should a differentiable surrogate match the daily score and the hinge penalty while keeping pinball calibration? What evidence is there that metric-aligned training beats pinball plus a post-hoc decision?

### Takeaway
There are three nested targets:
- (a) the pooled per-point squared relative error, which `nanwang_r2_loss` implements and for which our p* is already Bayes-optimal;
- (b) the true mean daily score, which needs the per-day sqrt and so upweights *good* days by 1/R_d;
- (c) the real penalty, a per-day hinge at 65 that weights *bad* days only.

(b) and (c) pull in opposite directions, and p* only solves (a). Both (b) and (c) depend on the joint within-day error, which 9 marginal quantiles cannot represent. That is the only real argument for putting a metric term into training rather than only into the decision layer.

The literature consistently supports a **mixed objective (decision/metric loss + proper accuracy score)** over either alone. However, no study tests it on a 两个细则-type PV score, and the expected gain over a correct Bayes decision layer is small.

### Cited Findings
- **Reporting optimization under 两个细则 (closest Chinese analog).**
  - Wu, Ke, Liu, Fang, Zheng (Wuhan University; 电工技术学报; received 2022-09-30) chose the *reported* curve by minimizing expected assessed energy over 100 error scenarios. The scenarios used Cornish-Fisher, LHS and Cholesky time correlation. The rules were 华中 (day-ahead threshold 80%).
  - On a 98.5 MW wind farm over 30 test days, reporting the raw forecast gave day-ahead accuracy 82.8% and 64.72 MWh of penalty. Day-ahead reporting optimization alone gave 85.15% and cut total penalty from 123.48 to 74.67 MWh. The full two-stage strategy with storage gave 29.37 MWh, a −76% reduction.
  - The authors state "直接上报预测值难以取得期望最优效益" because the error distribution is skewed. They list "构建以考核电量为导向的风功率预测模型" (training the forecaster on the penalty) as **future work**.
  — [电工技术学报 HTML](https://dgjsxb.ces-transaction.com/fileup/HTML/2024-1-135.htm)
- **Mixed objectives.**
  - Stratigakos, Pineda, Morales, "Decision-focused linear pooling for probabilistic forecast combination", *IJF* 41(3) 2025, 1112–1125. Weighting for forecast quality alone does not always give better decisions, and "optimizing for a mix of downstream cost and an accuracy-oriented scoring rule consistently produced better decisions while also improving forecast quality". — [IDEAS/RePEc listing](https://ideas.repec.org/a/eee/intfor/v41y2025i3p1112-1125.html); [Imperial Spiral](https://spiral.imperial.ac.uk/entities/publication/f34f203b-ea22-4115-81a9-b9bbb9e87aea) (abstract-level)
  - A 2026 *Energies* paper (19(17):4053), "Decision-Focused Wind Power Forecasting for Economic Dispatch Under Asymmetric Imbalance Costs", trains a Mamba-style forecaster with a blended objective: accuracy plus normalized recourse cost through a differentiable dispatch layer. Because the recourse has a closed-form optimal quantile, the authors can check whether training reaches it. — [MDPI (403 on fetch; search snippet only)](https://www.mdpi.com/1996-1073/19/17/4053)
- **Value-oriented forecasting.**
  - The value-oriented model has higher RMSE than the MSE model but lower operating cost (iterative learning approach). — [arXiv 2309.00803](https://arxiv.org/html/2309.00803v2)
  - The loss for linear operation problems is shown to be piecewise linear, giving lower cost than MSE-trained models. — [arXiv 2310.00571](https://arxiv.org/pdf/2310.00571)
- **Decision-focused retraining in energy.** Pinball-trained base forecasters at τ = 0.70/0.75 were best, and decision-focused retraining reduced imbalance cost for every base loss tried (ACM e-Energy 2024). — [ACM DL (403 on fetch; snippet)](https://dl.acm.org/doi/fullHtml/10.1145/3632775.3661952)
- **Weighting-by-outcome is improper.**
  - Multiplying a score by a weight of the *observed outcome* is not proper: the expected score is minimized by a *weighted version* of the true distribution.
  - Threshold-weighted CRPS (weight w(z) on thresholds) and quantile-weighted CRPS (weight v(α) on levels) retain propriety (Gneiting & Ranjan 2011, *JBES* 29(3):411–422).
  — [scoringrules docs on weighted scores](https://scoringrules.readthedocs.io/en/latest/weighted_scores.html); [RePEc](https://ideas.repec.org:443/a/taf/jnlbes/v29y2011i3p411-422.html)
- **Training with twCRPS.**
  - Training with threshold-weighted CRPS improves the targeted region but shows a body–tail trade-off (EMOS). — [arXiv 2407.15900](https://arxiv.org/html/2407.15900v2)
  - For CNNs on KNMI wind, tuned weighted CRPS improved extremes without significant bulk loss. The best weight function was model-specific. — [arXiv 2503.07374](https://arxiv.org/html/2503.07374v2)

### Inferences
**[MATH] Notation.** For day d with included points t ∈ T_d (n_d of them), let r_t = (y_t − p_t)/D_t with D_t = max(y_t, 0.2·Cap). Then R_d = sqrt(mean_t r_t²) and S_d = 1 − R_d (×100).

1. **Target (a), pooled squared relative error.**
   - L_a = Σ_d Σ_t r_t² / Σ_d n_d. This is `nanwang_r2_loss`.
   - The pointwise Bayes act is p* = E[y·D⁻²]/E[D⁻²], which is exactly `decision.p_star`.
   - So with a well-calibrated distribution, training on L_a adds *nothing in expectation* beyond p*. It can only help through finite-capacity effects: it makes the quantiles more accurate in the high-weight region, i.e. low-power points y < 0.2·Cap, where the weight is 25/Cap² versus 1/Cap² at y = Cap.
2. **Target (b), mean daily score.**
   - L_b = mean_d R_d. The gradient is ∂L_b/∂p_t = −r_t/(D_t·n_d·R_d), so day d is weighted by 1/R_d. Days that are already accurate get up to about 5× the weight of bad days (R = 0.1 vs 0.5).
   - The Bayes act is p_t = E[y_t·D_t⁻²/R_d] / E[D_t⁻²/R_d]. This is p* reweighted by the *whole-day* realized error, which needs joint scenarios. Marginal quantiles plus p* cannot reproduce it.
   - The repo's floor at 0 (S_d ≥ 0) kills gradients on catastrophic days. Do not floor in the loss.
3. **Target (c), real penalty.**
   - L_c = mean_d h(R_d − 0.35), with h a smoothed hinge.
   - L_c is **convex in p**: R_d is a weighted L2 norm, hence convex in p, and a hinge of a convex function is convex. So plain gradient descent on it is well-posed. No RL or black-box search is needed.
   - The ceil() in "不足一个百分点的按一个百分点计" adds on average 0.5 point per penalized day. Ignore it in the loss; it is a step function with zero gradient.
4. **Recommended composite loss** (all on cap-normalized values, metric terms only on t ∈ T_d of the D+1 slice):

   L = L_pin^g(q) + λ · [ L_b(p) + κ · L_c(p) ] + μ · L_cross(q)

   - **p = p*(q).** Use a torch port of `decision.p_star`: interpolate the 9 quantiles onto the 0.005 grid, set w_g = 1/max(y_g, 0.2)², and compute p = Σ y_g·w_g / Σ w_g. Detach w_g for stability; this is a first-order-equivalent "IRLS" gradient.
   - **L_b** = mean_d sqrt(mean_t r_t² + ε) with ε = 1e-4 (≈ R of 0.01, i.e. 1 score point). Use the regulation's T_d (exclude t where max(p_t, y_t) < 0.1, or keep all 96 to match the repo; run both).
   - **L_c** = mean_d softplus(β·(R_d − 0.35))/β with β ≈ 50–100 (1 point = 0.01 in R units) and κ ≈ 1–3. If the night-excluded re-score shows the official scale is lower, set the threshold from R_thr = 1 − 0.65 on the *official* n convention.
   - **L_cross** = mean relu(q_k − q_{k+1}) with μ ≈ 1. Alternatively, sort before p* as `enforce_monotone` does.
   - **L_pin^g** is pinball on the transformed scale (item 5).
5. **Proper, metric-weighted pinball via a monotone transform [MATH].**
   - Define g(y) = y/(0.2) for y < 0.2 and g(y) = 1 + ln(y/0.2) for y ≥ 0.2 (cap-normalized), so g′(y) = 1/max(y, 0.2).
   - Train ρ_τ(g(y) − g(q_τ)). Quantiles are equivariant under strictly increasing g, so the population minimizer is still the τ-quantile of y, and calibration targets are unchanged.
   - By the substitution u = g(z), CRPS on the g-scale = ∫ (F(z) − 1{y ≤ z})² g′(z) dz. This is exactly the **threshold-weighted CRPS with w(z) = 1/max(z, 0.2·Cap)**, which is proper by Gneiting–Ranjan.
   - So this is the *proper* way to put the metric's 0.2·Cap emphasis into the distributional loss. It contrasts with multiplying pinball by w(y_obs), which is improper and distorts every quantile.
   - Use the first power, not the square, to avoid over-focusing on dawn and dusk. With g(0) = 0 the transform has no singularity.
6. **Expected magnitude.**
   - p* gave +1 point on M8, and M5's p* already removes misalignment (a). What remains: the 1/R_d day weighting, the hinge, the tail truncation of p* beyond q0.1/q0.9, and low-power quantile accuracy.
   - Realistic gain on the mean score is ≤ 0.5 point, which is inside M8's ±0.6 seed noise. The gain on the **penalty metric** (Σ point-days below 65) may be relatively larger, because L_c targets it directly and nothing else does.
   - Report both: the mean daily score, and penalty point-days per station-month under the official n convention.

### Gaps
- No published study trains a PV forecaster directly on a 两个细则 accuracy formula or penalty and compares it with pinball plus a Bayes decision layer. The Wu et al. paper optimizes reporting only and lists training as future work.
- The Beichter ACM 2024 and MDPI 2026 full texts were not accessible (403), so their λ values and warm-start details are unverified.
- The quantile-equivariance and twCRPS identity is standard math. I did not find a paper that uses this exact transform for PV training.

---

## Q3. Should the metric loss act on p*(q) or on a separate point head? Risks and remedies

### Takeaway
Start with the metric loss on p*(q) at small λ: it needs no new parameters and keeps decision and training consistent. The cleaner long-term design is a **decoupled residual point head**, p = stopgrad(p*(q)) + δ_φ(h), with δ zero-initialized and trained only on L_b + κ·L_c. The quantiles then stay exactly pinball-trained (calibrated), and the head learns the day-level corrections that p* cannot express.

### Cited Findings
- Hirsch and Ziel show that quantile-based trading strategies are not proper scoring rules: distorted forecasts can earn more. Optimizing a decision metric through the quantiles can therefore reward miscalibration. — [arXiv 2604.19580](https://arxiv.org/pdf/2604.19580) (already in the earlier note; restated as the key risk)
- Decision-focused training raises forecast MSE/MAE while lowering the decision cost (Beichter DFF: MAE 0.46 vs 0.36 kW), so the target-metric loss deforms the predictive output. — [arXiv 2503.01936](https://arxiv.org/html/2503.01936v1)
- A mix of decision cost and a proper scoring rule both improves decisions and protects forecast quality. — [Stratigakos et al., IJF 2025](https://ideas.repec.org/a/eee/intfor/v41y2025i3p1112-1125.html)
- The pinball loss balances calibration and sharpness only implicitly; separate calibration/sharpness objectives have been proposed. — [Chung et al., arXiv 2011.09588](https://arxiv.org/pdf/2011.09588)

### Inferences
- **Gradient routing [MATH].** p*(q) = Σ_k a_k(q)·q_k with a_k ≥ 0 and Σ a_k = 1. Because w ∝ 1/y², a_k concentrates on the *low* quantiles, and with flat tails q0.1 gets the extra mass below τ = 0.1. A metric gradient on p therefore mainly pushes q0.1–q0.3. The risks:
  - (i) The lower quantiles drift toward the point, so lower coverage shrinks and the interval narrows on one side.
  - (ii) Crossing: q0.1 can move above q0.2.
  - (iii) "Collapse": with large λ the cheapest way to move p is to compress the quantile spread.
- **Remedies for the p*(q) route:**
  - Keep pinball always on.
  - Keep λ_eff small: normalize λ so that λ·L_metric ≈ 0.3–1× L_pin at the warm-start checkpoint.
  - Detach the w weights.
  - Add L_cross, or sort.
  - Monitor per-level coverage (|cov_τ − τ| ≤ 5 pp, as the earlier report already set), PIT/reliability, and mean interval width 0.1–0.9.
  - Abort if q0.1 coverage moves by more than 5 pp.
- **Decoupled residual head:**
  - p = p*(q).detach() + Cap·tanh(δ_φ(h_t, x_t))·s, with s = 0.1 and zero-init on the last layer.
  - Inputs: the TimesFM output embedding h_t if exposed, or simply [9 quantiles, IQR, clear-sky ratio, NWP GHI, hour-of-day, day-level features such as the predicted daily spread].
  - It is trained only on the metric terms and needs a few thousand parameters.
  - It is exactly a learned decision layer, and can be trained on a frozen checkpoint first, i.e. equal to experiment step 2 of the earlier report.
  - Day-level inputs (a predicted day-uncertainty index) are what let it approximate the 1/R_d and hinge re-weightings.
  - Production then reports p from the head, while the quantiles stay available for any probabilistic or market use.
- **TimesFM's own point/mean channel.** If TimesFM-3's head has a mean channel next to the quantiles (TimesFM 2.x heads emit mean + deciles; not verified for TimesFM-3 here), that channel can be repurposed as the metric head. Train it on L_b + κL_c and the decile channels on pinball. That avoids a new module. Verify in `forecast_backbones/m5/backbone.py` before relying on it.
- **Night zeroing.** Apply the same physics_clip/night mask inside the loss as in production. Masked steps contribute no gradient, so the model is not spending capacity on nights that production overwrites anyway.

### Gaps
- No direct comparison of "metric on quantile-derived point" versus "separate point head" for TSFMs was found. This is my design judgement, based on the gradient-routing argument and the propriety literature.

---

## Q4. Are distribution-level scores (CRPS, energy score, weighted CRPS) viable training objectives for a TSFM here?

### Takeaway
Our 9-level pinball already *is* a discretized quantile-weighted CRPS, so "switching to CRPS" changes nothing material. The one distribution-level change worth trying is the **threshold-weighted CRPS with w(z) = 1/max(z, 0.2·Cap)**, implemented as pinball on the transform g (Q2 item 5). The energy score (multivariate, for trajectories) only matters if we want joint within-day scenarios for L_b/L_c Bayes acts. A 9-quantile head cannot produce those without a copula, so defer it.

### Cited Findings
- Weighted CRPS forms: the threshold-weighted form (weight on outcome thresholds) and the quantile-weighted form (weight on levels) are both proper, and uniform weights recover CRPS. — [scoringrules docs](https://scoringrules.readthedocs.io/en/latest/weighted_scores.html); [Gneiting & Ranjan 2011 (RePEc)](https://ideas.repec.org:443/a/taf/jnlbes/v29y2011i3p411-422.html)
- Weighted-CRPS training improves the targeted region but has a body–tail trade-off for simple models; for CNNs the trade-off was not significant, and the weight function must be tuned per model. — [arXiv 2407.15900](https://arxiv.org/html/2407.15900v2); [arXiv 2503.07374](https://arxiv.org/html/2503.07374v2)
- Earlier deterministic weighted losses caused overcasting and false alarms (Hess & Boers 2022, as cited in 2407.15900). This argues for propriety-preserving weights. — [arXiv 2407.15900](https://arxiv.org/html/2407.15900v2)
- Standard TSFM probabilistic evaluation on GIFT-Eval uses CRPS. The earlier report already lists PaCTS/LoRA CRPS numbers (0.496 → 0.480), and those were pinball/quantile-loss trained. — [earlier report, PaCTS row](../../reports/M5%20基础模型微调方法调研.md)

### Inferences
- **Quantile-level weighting v(τ).** Weighting the 9 levels unequally is proper but pointless for the 南网 point: p* already uses the whole grid.
- **twCRPS via the g-transform is the right weighting.** It concentrates the loss where the metric's denominator is the 0.2·Cap floor: dawn, dusk, overcast, and rain onset. Those are exactly the bad days that drive the hinge penalty.
- **Hyperparameters for twCRPS.** Try two weights, w1 = 1/max(z, 0.2) and w½ = 1/sqrt(max(z, 0.2)), on a body–tail check: both the full-day score and the low-power-point coverage.
- **The tail problem.** The p* grid is flat beyond q0.1/q0.9. Under w ∝ 1/y², the lower tail below q0.1 carries a lot of weight, so the flat extrapolation biases p* upward on uncertain days.
  - Cheap fix in decision.py, no training: extrapolate linearly to q0.0 = max(0, q0.1 − (q0.2 − q0.1)), clipped at 0, instead of flat.
  - Training fix: add q0.02/q0.05 levels. That needs head changes, so do not do it first.
  - This is a likely source of residual misalignment that requires no metric loss at all.

### Gaps
- No published TimesFM, Chronos or Moirai fine-tune with twCRPS or an energy score on PV was found.

---

## Q5. RL re-check: any new 2026 evidence, and is any part of our scenario (threshold, monthly cap, spot market) a case for RL or bandits?

### Takeaway
There is no new evidence that changes the earlier verdict. The 2026 additions (Cast-R1, further LLM+GRPO hybrids) are LLM/tool-policy papers, not RL on a numeric quantile TSFM. The regulatory pieces that *look* non-differentiable are mostly fine:
- the hinge at 65 is convex and smoothable;
- the ceil() and the monthly H1 cap are low-dimensional and can be handled analytically or by grid search on the decision layer.

The spot market makes the objective price-weighted, but for a price-taking PV station the optimal offer is a closed-form quantile or newsvendor rule, not a sequential control problem. **RL is rejected for weights. A bandit on the decision layer is rejected for now**: the sample size per arm is about 30 days × 28 stations per month, and offline replay on history is strictly better.

### Cited Findings
- 2026 RL-for-forecasting papers:
  - Cast-R1 (arXiv 2602.13802): a Qwen3-1.7B tool-augmented forecaster with SFT, then multi-turn GRPO, G = 8.
  - TimeRFT (2605.00015): GRPO on MOIRAI-MoE, with a ground-truth reference inserted in the group.
  - REATS (2608.10149): an LLM outputs ensemble weights and GRPO optimizes MSE.
  - Cazaux et al. (2603.20063): GRPO/PPO with marginal gains.
  None tests a quantile-only TSFM, and none compares RL against supervised training on the identical target metric.
  — [Cast-R1](https://arxiv.org/pdf/2602.13802); [TimeRFT](https://arxiv.org/html/2605.00015); [REATS](https://arxiv.org/pdf/2608.10149); [2603.20063](https://arxiv.org/pdf/2603.20063)
- GRPO-TST (ESWA 2025) uses sequence-level rewards for long-horizon tokenized forecasting in continuous casting. The full text is paywalled. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417425044458)
- The scenario-based reporting optimization under 两个细则 (Wu et al.) used metaheuristics and big-M MILP on a *few-dimensional report decision*, not RL, and cut penalties by 50–76%. — [电工技术学报](https://dgjsxb.ces-transaction.com/fileup/HTML/2024-1-135.htm)
- Monthly cap (2025): ≤ 5 Cap·h per station per month across all forecast items. Non-accumulation applies to a single event. — [2025 PV attachment](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585645833212.pdf); [2025 main rules](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585644795756.pdf)
- Spot-pilot dispatch may propose adjusted assessment clauses, and items already achieved through market mechanisms are not double-assessed. — [2025 main rules 第一百零六条](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585644795756.pdf)

### Inferences
- **Monthly cap H1.** It creates a path dependency within the month: once a station hits 5 Cap·h, marginal penalties are zero for the rest of that month. In principle this is a sequential decision. In practice:
  - (i) Hitting the cap from day-ahead alone needs 25 point-days below 65 at A3 = 0.2 (e.g. 5 days at 60). That is plausible only in rainy-season months at weak stations.
  - (ii) The rational response once capped ("stop caring") gives no extra revenue, because the penalty is zero either way.
  So the cap does not justify RL. Track "station-months at cap" as a diagnostic.
- **The ceil() step.** It is irrelevant for training (Q2). In a decision layer it slightly rewards pushing a day from 64.x to ≥ 65, which is a per-day scalar shift. A grid search over a day-level shrink factor (`global_shrink_factor` already exists) on validation days captures it.
- **Spot market (Guangdong from 2025-11-01) [inference; rules not verified].** If day-ahead offers by a price-taker are settled two-price (DA quantity at the DA price, deviation at the RT price, with any deviation-recovery clause), the optimal offer is a known function of price forecasts: a newsvendor quantile τ* set by expected over/under-generation costs. The ingredients are price forecasts plus our calibrated quantiles, which is another reason to keep the quantiles proper. This is a decision-layer formula, not RL. Hirsch–Ziel warn that it should not feed back into training the quantiles.
- **When a bandit would be justified.** Only if dispatch's actual score computation is a black box that differs from our replica (unknown n convention, curtailment masks), *and* we receive monthly official scores per station. Then choose among a handful of decision rules per station, e.g. by Thompson sampling over monthly official scores. Even then, offline replay with a corrected metric replica dominates once the convention is known. **First fix the replica (Q1 gap).**

### Gaps
- No 2026 paper applies PPO/GRPO to TimesFM (any version) or to a quantile-only head.
- The Guangdong spot settlement rules for new-energy deviations were not verified in primary text.

---

## Q6. Final selection for M5: adopt / try later / reject, with formulas and hyperparameters

### Takeaway
Before any loss work, fix the scoring protocol (night-excluded variant, penalty point-days, 09:00 issue check) and the p* tail extrapolation; both are free. Then adopt, in order:
1. a decoupled metric head (or p*(q) at small λ) trained on mean-day-RMSE plus a soft 65-hinge;
2. twCRPS pinball via the g-transform.

Everything RL or preference-based stays rejected.

### Cited Findings
- Rules basis: [2025 PV attachment](https://nfj.nea.gov.cn/xwzx/tzgg/202502/P020250214585645833212.pdf), [2022 PV attachment](http://nfj.nea.gov.cn/file_upload/20220613/93581655118302306_f33dbd04-e92e-4188-9dba-d0b1f58aa316.pdf).
- Mixed-objective evidence: [Stratigakos et al. 2025](https://ideas.repec.org/a/eee/intfor/v41y2025i3p1112-1125.html); [Beichter DFF](https://arxiv.org/html/2503.01936v1) (LoRA/DoRA r8, lr 1e-4, 5 epochs).
- Report-optimization evidence: [Wu et al. 电工技术学报](https://dgjsxb.ces-transaction.com/fileup/HTML/2024-1-135.htm).
- Weighted-score propriety: [scoringrules](https://scoringrules.readthedocs.io/en/latest/weighted_scores.html), [2407.15900](https://arxiv.org/html/2407.15900v2), [2503.07374](https://arxiv.org/html/2503.07374v2).

### Inferences

| # | Method | Verdict | Concrete spec |
|---|---|---|---|
| 0a | **Official-convention re-score** | **Adopt now (no training)** | Add to `nanwang_metric`: (i) exclude t where max(PP_t, PM_t) < 0.1·Cap (option: n = n_d, or keep 1/96); (ii) mask curtailment plus 1 h if flags exist; (iii) penalty point-days = Σ_d ceil(max(0, 65 − ACC_d)) and penalty hours = 0.2 × that, capped at 5/month; (iv) issue-time check at 09:00 vs 10:00. Re-rank existing checkpoints on all of them. |
| 0b | **p* lower-tail extrapolation** | **Adopt now (no training)** | In `quantile_grid`, replace the flat tails with linear extrapolation to τ = 0/1, clipped to [0, Cap]. Score on existing checkpoints. |
| 1 | **Decoupled metric head (learned decision layer)** on a frozen checkpoint | **Adopt (first training step)** | p = p*(q).detach() + 0.1·Cap·tanh(δ_φ) with zero-init. Inputs: 9 quantiles, IQR, clear-sky ratio, NWP GHI, hour, day-level spread. Loss L_b + κL_c (ε = 1e-4, β = 50–100, κ ∈ {0, 1, 3}). AdamW lr 1e-3, a few epochs on train, select on validation. Minutes on V100. |
| 2 | **Joint LoRA fine-tune: pinball + λ·metric on p\*(q)** | **Try (after 1)** | Warm start from the best pinball LoRA. Metric phase 1–2 epochs at lr 1e-5–3e-5 (≤ ½ of the pinball lr), EMA on. λ ramped linearly over the first 10% of steps. λ_eff ∈ {0.1, 0.3, 1} × (L_pin0 / L_met0). Detach p* weights. L_cross μ = 1. Gate: coverage drift ≤ 5 pp per level, 5 seeds, gain > seed noise on *both* mean score and penalty point-days. |
| 3 | **twCRPS pinball via g-transform** | **Try (cheap, proper)** | Replace ρ_τ(y − q) with ρ_τ(g(y) − g(q)), g′ = 1/max(y, 0.2) (cap-normalized), also w½ variant. No change to outputs or decision layer. Combine with 1 or 2. |
| 4 | Plain `nanwang_r2_loss` as the metric term | **Try only as an ablation** | Bayes-equivalent to p* in expectation (Q2 item 1), so expect ≈ 0. Useful as a control to show that any gain comes from day-level terms (L_b/L_c). |
| 5 | Metric loss **multiplying** pinball by w(y_obs) | **Reject** | Improper (Gneiting–Ranjan): it distorts all quantiles toward a weighted distribution. Use #3 instead. |
| 6 | Energy score / scenario head / copula for joint-day Bayes acts | **Try later** | Only if #1 shows that day-level features matter a lot. It needs a sampling head and is a large change. |
| 7 | Scenario-based report optimization (Wu-style) | **Try later / fold into #1** | Our 65% threshold rarely binds on average days, so #1's κL_c term covers it differentiably. Revisit if the official n convention puts the mean near 65. |
| 8 | Spot-market newsvendor offer layer | **Try later (when trading)** | τ* from price forecasts applied to calibrated quantiles. Keep separate from training (Hirsch–Ziel). |
| 9 | PPO / GRPO / REINFORCE on TimesFM weights | **Reject** | No new 2026 evidence; strictly dominated by the pathwise gradient of the convex surrogate (earlier note Q4). |
| 10 | DPO / preference pairs | **Reject** | Ground truth exists; the quantile head has no likelihood. |
| 11 | Contextual bandit on decision-rule choice | **Reject now / conditional** | Only if official monthly scores come from a black box we cannot replicate. Fix the replica first (#0a). |

- **Seed and measurement protocol.** Reuse M8's lessons: 5 seeds, the same night-zero post-processing, and decision-layer re-selection for every checkpoint. Treat ±0.6 point as noise on the mean score. Penalty point-days are a lower-noise discriminator for the bad-day terms, though that is not yet measured.

### Gaps
- The size of any gain is unmeasured for our data. All effect-size priors are transferred from wind/dispatch DFL papers or from M8's p* result.
- The official n convention (96 vs non-excluded points) decides the threshold scale and must be confirmed with dispatch or station operators, or inferred from official monthly score sheets if any station has them.
