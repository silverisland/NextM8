# RL / Preference-Optimization Fine-Tuning for Time-Series Foundation Models (TSFMs) vs Direct Metric Optimization

Scope: TimesFM-3 (~200M, decoder-only, 9-quantile head, 192 x 15-min horizon, LoRA, cap-normalized pinball loss, V100 fp32). The target metric is the 南网 daily accuracy, about 1 - RMSE_daytime/Cap per day, averaged over days. The point forecast comes from a decision layer (median or a tuned p*). Research date 2026-10-15.

Evidence labels used below:
- **[LLM-TS]**: an LLM or multimodal LLM reasons about time series, and GRPO runs on its text tokens.
- **[NUM-TSFM]**: RL runs directly on a numeric forecaster or TSFM.
- **[DFL]**: decision-focused or gradient-based metric training, with no RL involved.

---

## Q1. Mechanics and literature: can a TSFM be a policy, and who has done RL fine-tuning of numeric forecasters?

### Takeaway
It is mechanically feasible. Any model with a sampleable output density can serve as the policy. The published RL fine-tuning of numeric forecasters is thin (about 3 papers). Its reported gains are small or confounded by a train/eval objective mismatch, and none of it uses a quantile-only head like TimesFM's. Most "Time-R1 / TimeMaster"-style 2025–2026 work runs GRPO on LLM text tokens. That is a different problem and is not evidence for RL on TimesFM.

### Cited Findings
**GRPO mechanics**
- GRPO (from DeepSeekMath) removes PPO's critic. It samples a group of G outputs per prompt, scores each one, and uses the group-normalized reward (r - mean)/std as the advantage, with a KL penalty to a reference policy. It needs per-sample log-probabilities πθ(o|q). — [DeepSeekMath, Shao et al. 2024](https://arxiv.org/abs/2402.03300); the same mechanics are restated for time series in [Time-R1, arXiv 2506.10630](https://arxiv.org/pdf/2506.10630)

**[LLM-TS] papers (not RL on a numeric TSFM)**
- **Time-R1** (arXiv 2506.10630, 2025) fine-tunes an LLM in two stages: SFT warm-up, then RL with "GRIP", a GRPO variant. Rewards cover logical consistency, temporal coherence and multi-horizon accuracy, and they optimize the LLM's *reasoning paths*. — [arXiv 2506.10630](https://arxiv.org/html/2506.10630v1)
- **TimeMaster** (arXiv 2506.13705) applies token-level GRPO to a multimodal LLM reasoning over *visualized* time series. The output format is reasoning + classification + extension, and the composite reward covers format, accuracy and insight. This is classification-style reasoning, not numeric forecasting. — [papers.cool summary of 2506.13705](https://papers.cool/arxiv/2506.13705)
- **REATS** (arXiv 2608.10149) applies SFT then GRPO to a 1.7B LLM whose output is a chain of thought plus *ensemble weights*, with forecast MSE as the reward. Its stated motivation is that token-level SFT loss does not directly optimize downstream MSE. — [arXiv 2608.10149](https://arxiv.org/pdf/2608.10149)
- **Conversational TSFMs** (arXiv 2512.16022) use SFT plus GRPO with a SHAP-faithfulness reward for *explanations*. — [arXiv 2512.16022](https://arxiv.org/pdf/2512.16022)
- **PostTime** (Liu, Zhou, Sen, Prakash, Das; Georgia Tech + Google Research; arXiv 2605.29401, May 2026):
  - TimesFM-2.5 produces the numeric forecast, and Gemma-3-4B acts as a text-context *reviser* trained with SFT then GRPO.
  - The reward is the improvement ratio over the TSFM prior. This cut zero-variance GRPO groups from 27.97% to 4.66%.
  - On TimesX, nMSE was ID 0.729 / OOD 0.673 for TimesFM-2.5, 0.650 / 0.623 after SFT, and 0.638 / 0.597 after SFT+RL. **Most of the gain comes from SFT. RL adds about 1.8% (ID) and 4.2% (OOD) nMSE on top.** The gain comes from *text context* that TimesFM cannot see. — [arXiv 2605.29401](https://arxiv.org/html/2605.29401v1)

**[NUM-TSFM] papers (RL directly on the forecaster)**
- **TimeRFT** (Li, Chen, Zhu, Pan, Guo, Huang, Xiong; arXiv 2605.00015, v2 Jul 2026). Setup:
  - The policy is **MOIRAI-MoE**, which has a native likelihood and supports sampling. Compatibility is also shown with ToTo-1.0 and Chronos-2. TimesFM is not tested.
  - For each context, G=8 on-policy forecasts are sampled. The importance ratio is the per-patch likelihood ratio, with a KL penalty to the pretrained model (β=0.001).
  - The reward is computed per patch: about 0.9·exp(-MSE), plus 0.1·variability (KL of softmaxed patches), plus a small FFT-frequency synergy term. A variability term is needed because accuracy-only rewards over-smooth.
  - **The ground-truth future is inserted into each group as the top-reward reference.** Rewards are shaped piecewise, and advantages are step-wise.
  — [arXiv 2605.00015](https://arxiv.org/html/2605.00015)
- TimeRFT results:
  - It reports about 10.17% MSE and 7.04% MAE average gains over TimeSFT/TimeLoRA. One example is Loop Seattle MSE, which drops from 2.421 to 2.032.
  - **The SFT baselines are trained with NLL on future values, not MSE, while evaluation is in MSE/MAE.**
  - In very-low-data settings (UCI Air Quality at 5%), both RFT methods *underperform*.
  - Training cost is 14 min for SFT versus 24 min for RFT at G=8 and 50 min at G=24 (ETT, 20% few-shot).
  - The paper has no ablation of the ground-truth-in-group trick and no discussion of collapse or reward hacking.
  — [arXiv 2605.00015 (cont.)](https://arxiv.org/html/2605.00015)
- **"Fine-tuning Timeseries Predictors Using RL"** (Cazaux et al., arXiv 2603.20063, Mar 2026). Setup and results:
  - The backbone is a supervised non-stationary iTransformer with time2vec, used as the actor with a density-valued policy. The reward is 2·exp(-MSE) - 1, bounded to [-1, 1].
  - MSE moved from backbone to GRPO as follows: Financial 0.202 to 0.198 (PPO 0.200, CMAPPO **0.324**), Industrials 0.120 to 0.118, and Technology 0.124 to 0.124.
  - GRPO was most stable with at least 50% of encoder layers frozen. CMAPPO degraded the model even with 75% frozen.
  - The authors warn of overfitting when RL fine-tuning reuses the supervised training data, and say that "fine-tuning can fail to improve an already strong model".
  — [arXiv 2603.20063](https://arxiv.org/html/2603.20063v1)
- **GRPO-TST** (Expert Systems with Applications, 2025) bins continuous sensor readings into discrete tokens so that GRPO applies, adding sequence-level rewards against long-horizon drift. Only the abstract is available. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417425044458)

### Inferences
- **Turning TimesFM into a policy.** TimesFM's head emits 9 deterministic quantiles (plus a point output), not a density, so making it a policy takes extra machinery. There are three options:
  - **Option (a): sample from the quantile function.** Draw u ~ U(0,1) and interpolate Q(u) between the 9 quantiles. The density is 1/Q'(u), which is piecewise-constant in the interior, and the tails beyond q0.1/q0.9 need an ad-hoc parametric extrapolation. Two problems follow. Log-probs then depend on the gap between adjacent quantiles, so the policy gradient will push quantiles together or apart for reward reasons, which damages calibration. Per-step independent sampling also produces white-noise paths across 192 steps with no temporal correlation.
  - **Option (b): bolt on a Gaussian or mixture head.** This creates a new head, discards the pretrained quantile head, and needs its own warm-up.
  - **Option (c): exploit autoregressive decoding.** TimesFM decodes in output patches, but the decoded values are deterministic. There is no stochastic token like Chronos's categorical bins.

  TimeRFT chose MOIRAI-MoE precisely because it has a native likelihood. The engineering cost is non-trivial for a quantile model, and the result is a policy whose "actions" are not what production uses (the decision-layer point forecast).
- **The positive [NUM-TSFM] results are weak evidence for our case.**
  - TimeRFT compares against NLL-trained SFT and evaluates on MSE, so part of its 10% gain is plausibly just *training on the evaluation metric*. A non-RL MSE fine-tune might capture that part. Its ground-truth-in-group trick also reinjects a supervised target.
  - Cazaux et al.'s gains are about 1–2% MSE or nothing.
  - No paper compares RL against a supervised fine-tune on the *same* metric-aligned loss. That is the comparison that matters.
- The [LLM-TS] successes rely on RL working over a *discrete text policy* where no gradient path from output to reward exists, plus side information (text, reasoning). Neither condition holds for a numeric TSFM with ground truth and a smooth metric.

### Gaps
- No paper was found applying RL/GRPO to TimesFM (any version), or to any quantile-only TSFM head.
- TimeRFT's sampling procedure for MOIRAI-MoE and the TimeSFT loss specifics were not visible in the fetched portion. No paper isolates "RL vs SFT with the identical (MSE) objective".
- GRPO-TST's quantitative results could not be retrieved (paywall).

---

## Q2. DPO and preference optimization: is there a time-series analog, and does it make sense with ground truth?

### Takeaway
No peer-reviewed DPO-for-forecasting paper was found, only a small hobby library. With ground truth available, preference pairs are a lossy, ranking-only re-encoding of the supervised signal, and DPO also needs policy log-likelihoods that a quantile head lacks. DPO makes sense only when the "preference" comes from a judgment that has no closed-form metric. That is not our situation.

### Cited Findings
- DPO reparameterizes the RLHF reward through the policy's log-likelihood ratio against a reference, removing the explicit reward model. It is defined for policies with tractable log-likelihoods (LLMs). — [Rafailov et al. 2023, arXiv 2305.18290](https://arxiv.org/abs/2305.18290)
- **nanoDPO** is an open-source library advertising "DPO for time series data". It uses a causal Transformer/LSTM and trains on *synthetic* preference data (`simulate_dpo_dataset_noise`). It is not peer-reviewed, and its mapping from preferences to numeric forecasts is unclear. — [nanoDPO docs](https://nanodpo.readthedocs.io/en/stable/README.html)
- A search for "DPO time series forecasting" surfaced only LLM-token DPO variants, such as temporal-decay DPO for language models, and no forecasting paper. — [arXiv 2502.14340](https://arxiv.org/pdf/2502.14340)

### Inferences
- Pairs could be built synthetically: two sampled or perturbed forecasts, labelled by which has the higher 南网 score against ground truth. That throws away the *magnitude* and *direction* of the error that the supervised gradient already provides. The pairwise Bradley–Terry likelihood recovers at best the ordering of the metric.
- Preference learning is meaningful only for a non-measurable notion of "good forecast", such as dispatcher or 调度 preferences about ramp shape. Even then, the standard move is to write that preference as a differentiable penalty, such as a ramp or shape term, as DILATE-style losses do.

### Gaps
- No empirical DPO-vs-supervised comparison on numeric forecasting exists to cite.

---

## Q3. Direct optimization of the evaluation metric: surrogate losses, DFL, and evidence that metric-trained beats pinball + post-hoc decision

### Takeaway
When the target metric is differentiable almost everywhere, the literature's tool of choice is a gradient-trained metric or decision loss, not RL. This holds for decision-focused learning (DFL) in energy, value-oriented forecasting, and metric-aligned losses. The closest analog to our setup (Beichter et al.) fine-tuned a TSFM with LoRA on a decision-cost surrogate *via gradients* and beat MSE fine-tuning by 9.45% on the target cost, while MSE/MAE got worse. For 南网 ≈ daily RMSE, theory also says the median is the wrong functional. That likely explains why tuning p* gave about +1 point.

### Cited Findings
**Decision-focused fine-tuning of a TSFM [DFL] (Beichter et al., arXiv 2503.01936)**
- **Setup.** Moirai, LoRA/DoRA (rank 8, α=32, <0.7% of 92M params), AdamW at lr 1e-4, 5 epochs. Ausgrid data from 300 buildings, prosumption = load - PV, 42-h horizon. The training signal is an ensemble of 5 surrogate networks approximating daily dispatch cost, and its gradient updates only the adapters. There is no RL and no differentiable solver. — [arXiv 2503.01936](https://arxiv.org/html/2503.01936v1)
- **Results.** Average daily cost was €12.75 for DFF (local, DoRA), versus €14.08 for the best MSE fine-tune, €17.53 for the MAE fine-tune and €15.99 for zero-shot. This is a **9.45%** improvement over the best prediction-focused fine-tune. Forecast MSE/MAE *worsened* (MAE 0.46 vs 0.36 kW). Published in *Energy and AI* 21 (2025) 100533. — [arXiv 2503.01936](https://arxiv.org/html/2503.01936v1); [KIT listing](https://publikationen.bibliothek.kit.edu/1000182410)

**Value-oriented renewable forecasting [DFL]**
- Zhang et al. (arXiv 2309.00803) train renewable forecasts on the downstream operating cost through a bilevel program, where the upper-level objective serves as the loss. This lowered operating cost versus expected-value forecasts. — [arXiv 2309.00803](https://arxiv.org/pdf/2309.00803)
- The follow-up (arXiv 2405.09004) derives an analytical, piecewise-linear loss and its gradient for day-ahead/real-time markets. The authors note that pinball and SPO losses were designed for single-stage problems. — [arXiv 2405.09004](https://arxiv.org/html/2405.09004v3)
- Value-oriented forecast reconciliation and combination (arXiv 2501.16086) found that optimizing forecast quality when estimating pool weights "does not always translate into better decisions". — [arXiv 2501.16086](https://arxiv.org/html/2501.16086v1)
- Hirsch and Ziel (2026) show that quantile-based trading strategies are not proper scoring rules, because distorted forecasts can earn more. This is a warning that optimizing a decision metric can reward miscalibrated quantiles. — [arXiv 2604.19580](https://arxiv.org/pdf/2604.19580)
- A benchmark package for energy forecasting argues that forecasts should minimize downstream dispatch cost and provides customizable loss functions for that purpose. — [arXiv 2307.07191](https://arxiv.org/html/2307.07191v2)

**Point-forecast theory**
- Gneiting (JASA 2011) shows that an error measure must be *consistent* for the functional being reported. Squared error, and Bregman scores generally, elicit the **mean**. Generalized piecewise-linear scores (pinball, absolute error) elicit **quantiles**, and the median is the optimum for MAE. Mismatching the two can give "grossly misguided inferences". — [Gneiting 2011, arXiv 0912.0902](https://arxiv.org/pdf/0912.0902)

### Inferences
- **The 南网 metric is effectively a reweighted MSE.** Per day d, S_d = 1 - sqrt((1/n_d)·Σ_t (ŷ_t - y_t)²)/Cap over daytime t. Its gradient is ∂S_d/∂ŷ_t = -(ŷ_t - y_t)/(n_d·Cap·RMSE_d). That is a *per-day reweighted MSE gradient* with weight 1/RMSE_d: unlike plain MSE, it relatively *upweights days that are already accurate*. It is differentiable everywhere except RMSE_d = 0 (add ε) and at any clip (score floored at 0, where gradients vanish on terrible days). Being smooth, it can be used directly as a loss: it needs no RL and no surrogate network. This is easier than Beichter's case, where the decision cost needed a learned surrogate.
- **The median is the wrong functional.** Daily RMSE is (approximately) mean-eliciting, while q0.5 is MAE-optimal. PV error distributions are skewed (cloud intrusions under clear-sky-like forecasts), so mean ≠ median. The p* tuning that gave +1 point is a crude one-parameter way to move from the median toward the conditional mean plus day weighting. Two cleaner decision layers follow:
  - (i) Take the mean of the quantile function: a trapezoid-weighted average of the 9 quantiles plus a tail correction.
  - (ii) Learn the decision layer by gradient on S: a softmax over quantiles per horizon step or per clear-sky regime, or a tiny MLP over quantiles and clear-sky features, trained on the 南网 loss with the TSFM frozen.
- **A joint-loss fine-tune is the natural next step.** Use L = pinball(9q) + λ·(1 - S̄) on the decision-layer output, with LoRA as now and fp32 being fine. Pinball keeps the quantiles calibrated for downstream probabilistic use, and Hirsch–Ziel shows why that guard matters. The metric term aligns the point output. This is the "train on the metric" lever that TimeRFT and Beichter both effectively exploit, without RL's variance.
- **The expected gain is smaller than Beichter's 9.45%.** Their target cost was far from MSE, while 南网 is close to RMSE. The residual misalignment is median vs mean, the 1/RMSE_d day weighting, daytime masking and cap normalization. The magnitude of the p* gain (about 1 point) is a reasonable prior for the headroom.

### Gaps
- No published study trains a PV forecaster directly on a 南网/两个细则-style daily accuracy score and compares it with pinball plus a post-hoc quantile choice. This is an open empirical question for our data.
- The exact 南网 clipping and threshold rules (e.g., floors, 考核 thresholds) were not researched here. If discontinuous thresholds exist, they matter for the gradient argument; see Q5.

---

## Q4. Variance and sample efficiency of policy gradients vs supervised gradients for continuous outputs with ground truth; failure modes

### Takeaway
With a smooth reward and a differentiable model, the policy gradient is a score-function (REINFORCE) estimator. It is a noisy, zeroth-order Monte Carlo estimate of the same gradient that backpropagating the metric gives exactly (the pathwise estimator). For a 192-dim output estimated from G=8 scalar rewards per context, it is strictly dominated. RL also optimizes the wrong object (the expected score of a *random sample*) and pushes toward variance collapse.

### Cited Findings
- Mohamed et al. (JMLR 2020) survey three gradient estimators: score function, pathwise and measure-valued. The score-function estimator needs no differentiable path but typically has high variance and requires control variates. The pathwise estimator differentiates through the sample and is the low-variance choice when differentiability holds. *(The variance comparison is the survey's well-known thesis, but the fetched page was only the abstract.)* — [arXiv 1906.10652](https://arxiv.org/abs/1906.10652v2); [JMLR v21](https://www.jmlr.org/papers/v21/19-346.html)
- In supervised learning each (x, y) gives a deterministic gradient. REINFORCE uses Monte Carlo returns that vary between rollouts, and course material lists high variance, slow convergence and learning-rate difficulty as its core problems. — [Berkeley CS285 lec 5](https://rail.eecs.berkeley.edu/deeprlcourse-fa20/static/slides/lec-5.pdf)
- Ilyas et al. found that in practice deep policy-gradient gradient estimates are poorly correlated with the true gradient at typical sample sizes. Learned critics fit their regression targets yet are about 50% off from the true value function. — [A Closer Look at Deep Policy Gradients, arXiv 1811.02553](https://arxiv.org/pdf/1811.02553)

**Failure modes in the forecasting-RL papers**
- Accuracy-only rewards over-smooth forecasts, which is why TimeRFT needs a variability reward. RFT underperforms in very low-data regimes, and cost grows with group size (3.6x SFT time at G=24). — [arXiv 2605.00015](https://arxiv.org/html/2605.00015)
- Cazaux et al. report the following failure modes:
  - Critic/backbone complexity mismatch: CMAPPO MSE went from 0.202 to 0.324.
  - A need to freeze at least half the encoder.
  - Sensitivity to reward design: an unbounded reward was worse.
  - Overfitting when RL reuses the supervised data.
  — [arXiv 2603.20063](https://arxiv.org/html/2603.20063v1)
- With absolute-error rewards, GRPO groups often have zero reward variance and therefore zero advantage (27.97% of groups in PostTime), which wastes samples. — [arXiv 2605.29401](https://arxiv.org/html/2605.29401v1)

### Inferences
- **First-principles comparison.** Take a Gaussian policy a ~ N(μ, σ²I) over D daytime steps (D ≈ 96 of 192) and a smooth reward R. REINFORCE estimates ∇_μ E[R] as E[(R(a) - b)(a - μ)]/σ². That is a randomized finite-difference estimate of ∇R, and its variance grows roughly linearly with D/G. Backprop of R at μ gives the exact ∇R (the pathwise gradient, zero estimator noise) at the cost of one forward/backward pass. GRPO's group of 8 samples buys a noisy projection of the same direction at 8x the forward cost.
- **Objective mismatch.** RL maximizes E_{a~π}[S(a)], while production scores S(decision(quantiles)). For a concave reward like 1 - RMSE, E[S(a)] < S(μ), so the policy gradient systematically drives σ → 0. Without heavy KL or entropy terms this collapses the probabilistic head, which is the over-smoothing symptom TimeRFT patches.
- **Data regime.** Ground truth is available for every training sample, so RL's main advantage, learning without labels or from an environment, is absent. It reduces to "supervised learning with a worse gradient estimator".
- **Hardware.** V100 fp32 with a 200M model makes G-fold sampling plus reference-model KL expensive (about 2–4x SFT wall-clock by TimeRFT's numbers) for no expected gain.

### Gaps
- No published controlled measurement of policy-gradient vs pathwise-gradient variance for a TSFM was found. The argument above is first-principles plus generic RL evidence.

---

## Q5. Other LLM-style alignment ideas transplanted to forecasting (reward models, best-of-N, self-training, distillation, model selection)

### Takeaway
Most are either meaningless here or reduce to standard ensembling or stacking under another name. Two ideas are meaningful. (1) Black-box (zeroth-order) tuning of the *few-parameter decision layer* when the metric has genuine discontinuities: this is what p* tuning already is. (2) Learned model or ensemble selection by regime, a "reward model" that predicts which member will score best, implemented as a stacker such as LightGBM rather than an LLM.

### Cited Findings
- REATS uses GRPO to make an LLM output *ensemble combination weights* rewarded by MSE. In effect it is an RL-trained stacker. — [arXiv 2608.10149](https://arxiv.org/pdf/2608.10149)
- PostTime's LLM keeps the TSFM prior unchanged in about 17–23% of cases, so a learned "revise or keep" selector is a large part of its behavior. — [arXiv 2605.29401](https://arxiv.org/html/2605.29401v1)
- Value-oriented forecast combination learns linear-pool weights by minimizing expected decision cost directly, by optimization rather than RL. — [arXiv 2501.16086](https://arxiv.org/html/2501.16086v1)
- A 2025 Applied Energy paper improves downstream PV-battery operation by iterating the *training data* (task-oriented learning) instead of the loss. This is a non-RL alternative for aligning forecasts with downstream value. — [IDEAS/Applied Energy 2025](https://ideas.repec.org/a/eee/appene/v378y2025ipas0306261924021366.html)
- Chronos tokenizes values into bins with a categorical output, so it natively *is* a token policy, unlike TimesFM's quantile head. That makes LLM-style RL far more natural for Chronos-type models. — [Chronos, arXiv 2403.07815](https://arxiv.org/abs/2403.07815); [TimesFM, arXiv 2310.10688](https://arxiv.org/abs/2310.10688)

### Inferences
- **Reward model + best-of-N at inference.** Ground truth is unknown at inference, so the reward model would have to predict forecast error from inputs. A model that accurately predicts the error of candidate forecasts can be folded into a better forecaster, or used as a per-sample member selector. The meaningful, cheap version is a **stacker**, e.g. LightGBM on [member forecasts, quantile spread, clear-sky index, NWP features, station meta], trained to minimize the 南网 loss. An LLM reward model adds nothing.
- **Best-of-N sampling from a single TSFM** has no selection signal without a reward model. Averaging samples just estimates the mean, which is the mean-functional point from Q3.
- **Self-training / distillation.** Distilling a multi-model or multi-seed ensemble into one TimesFM LoRA is legitimate and cheap. It targets variance reduction and serving cost, not metric alignment. Pseudo-labelling on unlabelled periods has little value because history labels arrive with a one-day lag anyway.
- **Where RL or zeroth-order methods do belong:**
  - (a) Sequential decisions with real dynamics, such as storage dispatch, bidding, or 考核 penalties that depend on past states, where the reward is non-differentiable and delayed.
  - (b) A genuinely discontinuous metric, such as a threshold penalty below a daily accuracy floor, or count-of-days-passing rules.

  Even for (b), the pragmatic tool is grid search, Bayesian optimization or CMA-ES over the handful of decision-layer parameters (p*, per-horizon or per-regime quantile weights, night-zero thresholds), applied to a *frozen* TSFM. Alternatively, smooth the threshold with a sigmoid and keep backprop.
- **Exception: discrete text policies.** LLM-reviser setups like PostTime that inject text context (weather bulletins, curtailment notices) are where RL on a discrete text policy is natural. That is a separate architecture decision and is outside "fine-tuning TimesFM".

### Gaps
- No literature evaluates a reward-model / best-of-N scheme for numeric PV forecasting.

---

## Q6. Verdict table (TimesFM-3 + 9 quantiles + LoRA + 南网 daily-RMSE score)

### Takeaway
None of PPO, GRPO, REINFORCE, DPO or RLHF-style reward fine-tuning is recommended for this setup. All except DPO are *feasible* with extra engineering, but each is dominated by backpropagating a differentiable 南网 surrogate. Decision-layer optimization is the right home for any remaining non-differentiable pieces.

### Cited Findings
- RL/GRPO on numeric forecasters gives small or zero gains: MSE 0.202 → 0.198, 0.120 → 0.118, 0.124 → 0.124. — [arXiv 2603.20063](https://arxiv.org/html/2603.20063v1)
- TimeRFT reports about 10% over *NLL-trained* SFT, adds a ground-truth reference to groups, needs anti-smoothing rewards, and costs 1.7–3.6x SFT. — [arXiv 2605.00015](https://arxiv.org/html/2605.00015)
- Gradient-based decision-focused LoRA fine-tuning of a TSFM gave a 9.45% improvement in the target metric over MSE fine-tuning. — [arXiv 2503.01936](https://arxiv.org/html/2503.01936v1)
- MSE elicits the mean and pinball elicits quantiles, so choose the functional that matches the score. — [Gneiting 2011](https://arxiv.org/pdf/0912.0902)

### Inferences

| Method | Feasible on TimesFM-3 quantile head? | Likely benefit for 南网 score | Recommended alternative |
|---|---|---|---|
| PPO (critic + clipped ratio) | Technically yes, but needs a density head or quantile-function sampling, plus a 192-dim critic. Cazaux et al. saw critic/backbone mismatch. | Low to negative. High variance, critic bias, collapse of the quantile spread. | Backprop the 南网 surrogate loss. |
| GRPO | Yes with a sampling head. G=8 means 8x the forward cost plus a reference model, on V100 fp32. | Low. Expected at most what an MSE/metric-loss fine-tune gives. TimeRFT's gain is confounded by NLL-vs-MSE. | Joint loss pinball + λ(1 - S̄) under LoRA. |
| REINFORCE / score-function PG | Yes (simplest). | Low. Strictly dominated by the pathwise gradient when the reward is smooth. | Same as above. |
| RLHF-style learned reward model + PG | Yes, but the reward model would just re-learn a known closed-form metric. | None. The metric is known exactly. | Use the metric itself as the loss. |
| DPO / preference pairs | Awkward. It needs policy log-likelihoods, which a quantile head lacks, and pairs discard error magnitude. No forecasting literature exists. | None with ground truth. | Supervised metric loss. Use a shape or ramp penalty if a "preference" is really a shape requirement. |
| Best-of-N + reward model (inference) | Yes. | Only if built as a per-sample member selector. | A LightGBM/MLP stacker over member forecasts and regime features, trained on the 南网 loss. |
| Self-training / distillation | Yes. | Moderate for variance and serving cost, not for metric alignment. | Distill a multi-seed or multi-model ensemble into one LoRA if serving cost matters. |
| Decision-layer tuning (black-box / zeroth-order) | Yes; it already exists (p*). | Proven: about +1 point in another model. | Extend it: mean-of-quantile-function, per-horizon or per-regime p*, or a learnable quantile-weighting layer trained by gradient on 南网. Use grid search or BO only for genuinely discontinuous rules. |
| Direct metric fine-tune (gradient) | Yes. Mask nights, normalize by cap, add ε inside sqrt, handle clipping. | Most likely positive. Expected order: similar to the p* gain, possibly additive with it. | This is the recommended path. Keep pinball for calibration. Sweep λ. Report with the 5-seed protocol. |

**Concrete experiment order (cheapest first):**
1. Score mean-of-quantiles vs median vs global p* vs per-station/per-horizon p* on existing checkpoints. This needs no training.
2. Train a frozen-TSFM decision layer (quantile weights, optionally conditioned on clear-sky index and horizon) by gradient on the 南网 loss over the validation split.
3. Run a LoRA fine-tune with L = pinball + λ·(1 - S̄_daily) and a small λ sweep. Monitor quantile coverage so that λ does not wreck calibration.

Revisit RL only if the operational score turns out to contain path-dependent or discrete penalties that cannot be smoothed. Even then, apply it to the decision layer, not to TimesFM weights.

### Gaps
- No direct empirical evidence exists for the size of the gain from metric-loss fine-tuning on 南网-scored PV. It must be measured locally.
- TimesFM-3's internal decoding details (whether any stochastic sampling path exists) were not verified in this note.
