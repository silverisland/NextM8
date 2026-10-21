# Industrial / practitioner fine-tuning techniques: which ones fit TimesFM-3 + LoRA for PV forecasting

Scope: techniques from NLP, vision, LLM practice, tabular and forecasting competitions, judged against our setting. TimesFM-3 has about 330.7M parameters. We adapt it with LoRA r8 on q/k/v/o in both attention types, plus an unfrozen output head (about 4M trainable). Training runs in fp32 on V100s for about 4000 steps at global batch 16 with AdamW lr 1e-4 and a cosine schedule. The loss is capacity-normalised 9-quantile pinball, and early stopping uses the 南网 daily metric. The data covers 28 stations with 1–2 years of 15-min data, split by time, and some test stations are unseen. We use 3 seeds and a noise band.

Already planned, so treated as baseline here: EMA 0.998, smoothed early stopping, lr sweep, WiSE-FT, LP-style head-only and resblock-only runs, full FT at tiny lr, seed ensembles, refit on train∪valid, the p* decision rule, and the clear-sky ceiling.

Citation note: sources marked "(not re-fetched)" are well-known papers cited from prior knowledge, with their canonical arXiv URL, and were not opened in this session. Every numeric effect size below comes from a source that was fetched or searched in this session unless it says otherwise.

---

## Q0. Catalogue summary: verdicts across all questions

### Takeaway
Few of the items below are likely to beat our ±noise band on their own. That matches what M8 showed. The cheap ones worth adding are:
- LoRA on the MLP/all linear layers instead of attention-only (strongest industrial evidence).
- A higher lr for LoRA than for the head (LoRA+ / "LoRA lr ≈ 10× full-FT lr").
- An output-space seed ensemble, which pinball convexity guarantees is no worse than the average member.
- Sorting quantiles to remove crossing.
- Quantile recalibration before p*. This matters because p* reads the quantiles, so calibration changes the point forecast.
- Selection hygiene: rolling-origin folds, re-running the top-k with fresh seeds, and testing once.

Most other items are "later" or "reject". Several popular tricks only make sense for LLM chat or multi-task merging: TIES/DARE, LoRA soups for skill composition, NEFTune, RLHF/DPO.

### Cited Findings (condensed; details under each question)
| Technique | Evidence & effect size | Fit to us | Cost | Verdict |
|---|---|---|---|---|
| LoRA on MLP / all linear (vs attention-only) | Attention-only LoRA "significantly underperforms MLP-only"; adding attention to MLP-only adds nothing ([Thinking Machines, LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)) | High. We are attention-only today, and the `s6_all_linear` variant already exists | 1 config, ~same compute | **Add (priority)** |
| Separate lr for LoRA vs head / LoRA+ | LoRA optimal lr ≈ 10× full-FT lr (fitted 9.8×) ([TM](https://thinkingmachines.ai/blog/lora/)); LoRA+ (B lr = ratio × A lr): up to ~2× faster, ~1–2% better ([Hayou et al. ICML'24](https://proceedings.mlr.press/v235/hayou24a.html)) | Medium–high. Our sweep (1e-5…1e-4) is skewed low for LoRA params | 2–3 runs | **Add** |
| LP-FT | +10% OOD / +1% ID vs full FT on 10 vision shift datasets ([Kumar et al. ICLR'22](https://arxiv.org/abs/2202.10054)) | Medium. Our head is pretrained, not random, so distortion is smaller. Chaining head-only → LoRA is nearly free | ~0 extra | **Add (cheap chain)** |
| LLRD | Its benefit shrinks once basic practice is fixed ([Zhang et al. ICLR'21](https://arxiv.org/pdf/2006.05987)) | Low with r8 LoRA | small | Reject |
| Surgical FT (choose block by shift type) | First block for input-level shift, last layer for output-level shift ([Lee et al. ICLR'23](https://arxiv.org/abs/2210.11466v2)) | Medium. It justifies the planned resblock-only and head-only runs; Auto-RGN can rank layers | already planned | Keep (planned) |
| Constant lr + EMA vs cosine | Playbook: "start with a constant learning rate"; schedule-free matches/beats cosine ([Google tuning playbook](https://github.com/google-research/tuning_playbook); [Defazio et al.](https://arxiv.org/html/2405.15682v4)) | Medium. It makes refit step count less sensitive | 1 run × seeds | **Add (one A/B)** |
| Schedule-free AdamW | Won AlgoPerf 2024 self-tuning track ([Defazio et al.](https://arxiv.org/html/2405.15682v4)) | Low–medium. Mostly a convenience | code change | Later |
| SAM/ASAM | Uniform gains in vision FT; up to large gains in low-data LM FT; ~2× compute ([Foret et al.](https://arxiv.org/pdf/2010.01412); [Fisher-SAM EMNLP'22](https://preview.aclanthology.org/watermark/2022.findings-emnlp.300)); SAM on LoRA params alone "limits its effect" (Bi-LoRA, via search summary) | Low–medium. Evidence is classification; 2× V100 time | 2× | Later |
| Lookahead | (not re-fetched) [Zhang et al. 2019](https://arxiv.org/abs/1907.08610) | Redundant with EMA | small | Reject |
| Grad clipping 1.0 | Hygiene, no accuracy evidence | Guards against bad batches | 0 | Keep as hygiene |
| Batch-size tuning | Playbook: don't tune batch for validation; LoRA "pays a larger penalty for increased batch size" ([playbook](https://github.com/google-research/tuning_playbook); [TM](https://thinkingmachines.ai/blog/lora/)) | Keep 16 | — | Reject |
| EMA / LAWA | EMA/LAWA cut AlgoPerf GPU-hours 612→541/550 ([via LAWA follow-up](https://arxiv.org/pdf/2209.14981)) | Already planned | 0 | Keep (planned) |
| Seed output ensemble | Pinball is convex, so averaging quantiles is ≤ mean member loss (math). M5/AMS winners ensemble ([AMS overview](https://repository.library.noaa.gov/view/noaa/22258/noaa_22258_DS1.pdf)) | High | 3× inference (trivial) | **Add** |
| Uniform/greedy soup | ViT-G 90.94% ImageNet SOTA; no inference cost ([Wortsman et al. ICML'22](https://arxiv.org/pdf/2203.05482)) | Medium. Average ΔW=BA, not the A and B factors | free given seeds | Later (cheap check) |
| WiSE-FT | +4–6 pp OOD over prior work, +2–23 pp on other shifts ([Wortsman et al. CVPR'22](https://arxiv.org/pdf/2109.01903)) | High for unseen stations | free | Keep (planned) |
| TIES / DARE | Built for multi-task interference; TIES+DARE ≈ TIES (93.71 vs 93.83) ([STAR](https://arxiv.org/pdf/2502.10339)) | None (same task) | — | **Reject** |
| L2-SP (head) | "marginal to moderate" gains vs weight decay ([Li et al.](https://vision.cornell.edu/se3/wp-content/uploads/2019/12/ivc_li2020.pdf)) | Low–medium (weight decay on LoRA is already L2-SP-like) | tiny | Later |
| LoRA dropout | Standard PEFT knob, no fetched effect size | Low–medium | 1 run | Later |
| R-Drop | (not re-fetched) [Wu et al. 2021](https://arxiv.org/abs/2106.14448) | 2× forward, classification-centric | 2× | Reject |
| C-Mixup / TS augmentation | C-Mixup +6.6% ID / +5.8% OOD on regression ([Yao et al.](https://arxiv.org/abs/2210.05775)); augmentation degrades channel-independent PatchTST −4.6% ([2026 study](https://arxiv.org/pdf/2605.06032)) | Low. TimesFM is a channel-independent patch model | moderate | Reject (later at most) |
| NWP noise / covariate dropout | No direct evidence. HEFTCom winner covered missing NWP by falling back to other models ([HEFTCom2024](https://arxiv.org/html/2507.01579v2)) | Production robustness, not accuracy | small | Later |
| x-only sample reweighting (station/season balance, recency) | No direct evidence | Medium | small | Later |
| y-dependent weighted pinball (1/max(y,0.2cap)) | Math: changes which quantile is learned | **Conflicts with p*** | small | Reject (use x-only weights) |
| Quantile sorting (rearrangement) | (not re-fetched) Chernozhukov et al. 2010, rearrangement never worsens quantile estimates | High, free | 0 | **Add** |
| HPO: quasi-random → BO, ASHA, PBT | Playbook: quasi-random first, then BO ([playbook](https://github.com/google-research/tuning_playbook)) | Medium. Seed noise dominates | — | Later; selection hygiene: **Add** |
| Online KF/MOS bias correction | NWP GHI: MBE −70–94%, RMSE −8–12% ([Manila KF](https://archium.ateneo.edu/manila-observatory/14/)); "without positive impact on ML methods" for PV ([SIRTA poster](https://sirta.ipsl.polytechnique.fr/documents/JSS2022/posters/JSS2022_SCros.pdf)) | Low–medium for a learned model; useful under drift | small | Later (rolling backtest) |
| Online / test-time adaptation (TAFAS, PROCEED) | Gains claimed, but rankings are protocol-dependent ([TAFAS](https://arxiv.org/html/2501.04970v1); [PROCEED](https://arxiv.org/html/2412.08435v5); [counter-evidence](https://arxiv.org/pdf/2510.18281)) | Medium (deployment protocol) | high eng. | Later |
| Quantile recalibration (conformal / CQR) | (not re-fetched) [Romano et al. 2019](https://arxiv.org/abs/1905.03222) | **Relevant because p* reads quantiles** | small | **Add (diagnose first)** |
| Distill / prune / quantise | Pruning claimed to aid TSFM specialisation ([Less is More](https://arxiv.org/pdf/2505.23195), title only) | Not an accuracy lever here | — | Reject (for now) |

### Inferences
- Of the levers M8 taught us (priors, decision layer, protocol), the decision layer interacts with fine-tuning more than expected. Anything that changes quantile calibration also changes the p* point: WiSE α, ensembling, y-weighted losses, recalibration. Evaluate these on the final p* point metric, not on pinball alone.
- Likely ordering of expected gain: all-linear LoRA ≳ LoRA lr fix > seed ensemble ≈ WiSE > recalibration → p* > the rest. That ordering is an inference with no PV-specific evidence behind it.

### Gaps
- I found no published TimesFM LoRA results on PV. The official repo added a LoRA example on 2026-04-09 but reports no PV numbers ([timesfm repo](https://github.com/google-research/timesfm)).
- I found no industry blogs from Uber, Amazon, Alibaba or Huawei on TSFM fine-tuning practice. The industrial sources used are Google (tuning playbook), Meta FAIR (schedule-free) and Thinking Machines (LoRA).

---

## Q1. Optimisation & schedule (LP-FT, LLRD, discriminative lr, warmup, cosine vs constant + EMA, SAM, Lookahead, clipping, weight decay on LoRA, batch/steps, schedule-free)

### Takeaway
Two of these probably matter: which modules LoRA is placed on, and the lr ratio between LoRA and the head. LP-FT costs almost nothing as a chain from the planned head-only run. Run one constant-lr + EMA vs cosine A/B. LLRD, SAM, Lookahead and batch tuning are low-value for an r8 LoRA on ~1–2 years of data.

### Cited Findings
- **LP-FT.** On 10 distribution-shift datasets, full FT gives about 2% higher ID but about 7% lower OOD accuracy than linear probing. LP-FT gives "1% better ID, 10% better OOD than full fine-tuning" and is better on 5/6 ID and 10/10 OOD datasets. The mechanism is feature distortion: when the head starts fixed or random, lower layers change while the head is being learned. Linear probing does not beat FT where ID ≈ OOD (CIFAR-10.1, ImageNetV2) — [Kumar et al., ICLR 2022](https://arxiv.org/abs/2202.10054); NTK-based analysis of LP-FT for Transformers — [Tomihari & Sato, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/fcc22e5b7d5d2155d994da22d045f0a6-Paper-Conference.pdf).
- **Surgical FT.** Tuning one contiguous block matches or beats full FT. First block works best for input-level shift, later blocks for feature-level shift, the last layer for output-level shift. Auto-RGN (relative gradient norm) picks layers in a single run — [Lee et al., ICLR 2023](https://arxiv.org/abs/2210.11466v2).
- **LLRD / few-sample FT.** Zhang et al. re-evaluated LLRD, Mixout and pretrained weight decay after fixing Adam bias correction, top-layer re-init and training length: "generally, we observe a decrease in their relative impact". Re-initialising the top layers sped convergence and narrowed the seed spread. ELECTRA tunes LLRD in {0.9, 0.8, 0.7} — [Zhang et al., ICLR 2021](https://arxiv.org/pdf/2006.05987); [ELECTRA](https://arxiv.org/pdf/2003.10555).
- **LoRA lr and placement (industry).** The optimal LoRA lr is about 10× that of full FT (fitted 9.8×), and about 15× for very short runs. Attention-only LoRA "significantly underperforms MLP-only LoRA", even at matched parameter count (attention r256 at 0.25B params < MLP r128 at 0.24B). The optimal lr changes <2× between rank 4 and 512. LoRA "pays a larger penalty for increased batch size". Default α=32 with α/r scaling and B=0 init could not be improved. Runs used constant lr with "no warmup or cooldown". B=0 creates an implicit warmup — [Thinking Machines, "LoRA Without Regret"](https://thinkingmachines.ai/blog/lora/). Caveat: these are LLM SFT/RL experiments.
- **LoRA+.** Using a single lr for A and B is suboptimal in wide networks. Setting B's lr to a fixed multiple of A's gives up to ~2× faster convergence and ~1–2% better accuracy at equal cost — [Hayou, Ghosh & Yu, ICML 2024](https://proceedings.mlr.press/v235/hayou24a.html). A secondary source puts the typical ratio at ~16×, which I did not verify in the original — [Chronicals](https://arxiv.org/pdf/2601.02609).
- **Schedules (Google playbook).** "Start with a constant learning rate before adding fancy decay schedules". Then use linear or cosine decay. Schedules tuned for short runs transfer poorly to longer ones. "Never tune the `max_train_steps` number in a study"; "when in doubt, err on the side of training longer". If the best checkpoint is always early, the step budget is too high. Prefer retrospective checkpoint selection over prospective early stopping. Polyak averaging is listed as a remedy for late-training variance. Warmup "very likely" transfers from short to long runs — [google-research/tuning_playbook](https://github.com/google-research/tuning_playbook).
- **Schedule-free AdamW.** It matches or beats cosine schedules of various lengths, needs no stopping time and no extra hyperparameters, and was the basis of the winning AlgoPerf 2024 self-tuning entry. It requires `optimizer.train()/eval()` calls — [Defazio et al., "The Road Less Scheduled"](https://arxiv.org/html/2405.15682v4); [repo README](https://github.com/facebookresearch/schedule_free/blob/main/README.md).
- **SAM.** SAM "uniformly improves" fine-tuning of ImageNet-pretrained EfficientNets — [Foret et al.](https://arxiv.org/pdf/2010.01412). Fisher-mask SAM improves "up to 15.1" (units unclear in the snippet) in limited-data LM fine-tuning — [Zhong et al., EMNLP Findings 2022](https://preview.aclanthology.org/watermark/2022.findings-emnlp.300). SAM doubles compute. The perturbation radius ρ needs tuning — [Agnostic-SAM](https://arxiv.org/pdf/2406.07107). Bi-LoRA (2025, via search summary) says applying SAM only to LoRA parameters limits its effect — [arXiv 2508.19564](https://arxiv.org/abs/2508.19564v1).
- **Batch size.** The playbook says batch size governs throughput, not validation performance, and should not be tuned for it. Changing it means re-tuning lr — [playbook](https://github.com/google-research/tuning_playbook).
- **Lookahead, ASAM, gradient clipping.** No session evidence on fine-tuning accuracy. Canonical refs (not re-fetched): [Lookahead](https://arxiv.org/abs/1907.08610), [ASAM](https://arxiv.org/abs/2102.11600).

### Inferences
- **LoRA placement is the biggest optimisation-side lever we have not tested.** Our q/k/v/o-only config is exactly the attention-only setting that TM found weakest. Run `s6_all_linear` (or MLP-only) at the same rank first, before any optimiser tricks. Expected effect is unknown for TimesFM. In LLMs it is the largest gap TM reports.
- **lr is probably mis-centred for LoRA parameters.** Our sweep {1e-5, 3e-5, 1e-4} is aimed at full FT. Industry evidence puts LoRA's optimum ~10× higher than full FT, so somewhere around 1e-4–1e-3. Suggested design: param groups with LoRA lr ∈ {1e-4, 3e-4, 1e-3} and head lr fixed at 1e-4 (discriminative lr), optionally LoRA+ ratio 4–16 for B. Watch for instability in fp32 with the clear-sky-ceiling targets.
- **LP-FT costs nothing extra.** Initialise the LoRA run from the best head-only checkpoint (the planned LP-style run). Kumar's mechanism is weaker here because our head is pretrained, not random. Capacity-normalised targets and a different quantile usage may still make the initial head mismatch non-trivial. Expect a small effect, mostly lower seed variance.
- **Constant + EMA vs cosine.** Under refit-with-fixed-steps, a constant lr + EMA makes the refit endpoint much less sensitive than cosine, whose decay is tied to the total step count. Run one 3-seed A/B. If they are equal, prefer constant + EMA for robustness of refit.
- **Weight decay on LoRA already acts like L2-SP.** Decaying A and B toward 0 shrinks ΔW toward the pretrained weights. For the unfrozen head, plain weight decay pulls toward 0, not toward the pretrained head. Switch the head to L2-SP or set head wd to 0 (see Q3).
- **Warmup.** LoRA's B=0 already ramps the effective lr. A short warmup (≤5% of steps) is harmless and protects the head. Don't tune it.
- **SAM.** 2× fp32 V100 time for a classification-only evidence base, and Bi-LoRA reports a limited effect on LoRA-only parameters. Defer until cheaper levers are exhausted.

### Gaps
- No session evidence on LP-FT or LoRA placement for time-series foundation models specifically.
- I could not verify the LoRA+ recommended ratio from the primary paper (secondary source says ~16×).
- No effect sizes for gradient clipping or Lookahead in fine-tuning.

---

## Q2. Weight averaging (EMA, SWA, soups, checkpoint averaging, WiSE-FT, LoRA soups / TIES / DARE)

### Takeaway
EMA and WiSE-FT, both planned, are the well-supported ones. A seed output ensemble is guaranteed not to hurt pinball. A uniform "delta soup" of seed LoRAs, averaging ΔW=BA and not the factors, is a free check. TIES, DARE and "LoRA soups" target multi-task skill composition and should be rejected for a single regression task.

### Cited Findings
- **Model soups.** Averaging weights of models fine-tuned with different hyperparameters "often improves accuracy and robustness" at no inference cost, because fine-tuned models often lie in one low-error basin. Greedy soup adds models in order of validation score and keeps each only if validation does not drop. ViT-G reached 90.94% top-1 on ImageNet (then SOTA) — [Wortsman et al., ICML 2022](https://arxiv.org/pdf/2203.05482). Learned-soup variants beat greedy by only 0.17–0.59%, and greedy needs 2N validation evaluations — [MEHL-Soup](https://arxiv.org/pdf/2407.03641).
- **WiSE-FT.** Interpolates zero-shot and fine-tuned weights with α. It gives +4–6 pp under distribution shift over prior work while adding +1.6 pp ImageNet accuracy, +2–23 pp on six further shifts, and +0.8–3.3 pp vs standard FT on transfer datasets. An earlier version reported +8.7 pp OOD for end-to-end FT and +2.1 pp for linear-classifier FT vs standard FT. Varying α "provides a superior performance curve than tuning other hyperparameters like learning rates or epochs". It works beyond CLIP — [Wortsman et al., CVPR 2022](https://arxiv.org/pdf/2109.01903).
- **LAWA / EMA.** Averaging the last k=6 epoch checkpoints saves about 68 GPU-h (ResNet-50/ImageNet) and 30 GPU-h (RoBERTa). k>16 hurts — [Kaddour 2022](https://arxiv.org/pdf/2209.14981). A 2025 AlgoPerf study reports 612 → 550 (LAWA) / 541 (EMA) GPU-h — [via search summary of "When, where and why to average weights", ICML 2025](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICML/2025/when_where_and_why_to_average_weights__973991b6.txt).
- **SWA.** Averages tail checkpoints under a constant or cyclic lr. LAWA is its rolling-window cousin — [LAWA paper / review](https://arxiv.org/pdf/2209.14981). Canonical (not re-fetched): [Izmailov et al. 2018](https://arxiv.org/abs/1803.05407).
- **TIES / DARE.** DARE drops delta parameters at rate p and rescales by 1/(1−p), and can drop 90–99% of deltas with little loss. TIES trims, elects a sign and takes a disjoint mean, targeting sign conflicts across tasks — [DARE, ICML 2024](https://proceedings.mlr.press/v235/yu24p.html). On 8 Flan-T5 models, TIES = 93.83 vs TIES+DARE = 93.71, so DARE did not help — [STAR](https://arxiv.org/pdf/2502.10339). DARE fails at high drop rates or large deltas — [search summary, arXiv 2410.09344](https://arxiv.org/abs/2410.09344v1). PEFT `add_weighted_adapter` supports `ties`/`dare`. Linear averaging of LoRA factors ≠ averaging ΔW. Concatenation with 1/√N scaling gives the exact mean delta — [HF PEFT model merging docs](https://huggingface.co/docs/peft/v0.10.0/en/developer_guides/model_merging).
- **"LoRA Soups"** is skill composition (math + code), not same-task averaging — [Prabhakar et al., COLING 2025 industry](https://arxiv.org/html/2410.13025v2). Entropy-based merge methods "do not apply to regression"; NSC merging targets mixed classification/regression multi-task merges — [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/38149).
- **Industry practice: averaging forecasts.** AMS 2013-14 solar winners blended 13 GBRT models with a Nelder–Mead weight optimiser on MAE. Second place used an ensemble of boosted trees "to reduce overfitting risk", and one all-site model beat per-site models — [AMS contest overview / announcements](https://repository.library.noaa.gov/view/noaa/22258/noaa_22258_DS1.pdf); [AMS headlines](https://headlines.ametsoc.org/2013/12/17/annual-meeting-session-to-feature-solar-energy-contest-winners/). The HEFTCom2024 winner stacked per-NWP-source CatBoost quantile models with a linear quantile-regression meta-model over all 27 quantiles — [Browell et al., HEFTCom2024](https://arxiv.org/html/2507.01579v2).

### Inferences
- **Seed ensemble (output space) is a guaranteed non-loss for pinball.** Pinball loss is convex in the prediction, so averaging the quantile forecasts of K seeds has loss ≤ the mean of the members' losses (Jensen). It also shrinks seed noise, which directly attacks our noise-band problem. Inference cost for 28 stations is negligible, so add it, and evaluate on the p* point.
- **Delta soup vs output ensemble.** For LoRA, average ΔW_k = B_k A_k per layer, or equivalently concatenate the adapters with 1/√N scaling. Do not average A and B separately: different seeds give unaligned factor bases. Because ΔW is small, the network is near-linear in ΔW, which is the same reason WiSE works, so the soup should approach the ensemble. Value: one model for deployment and a mild regulariser. Run it as a free post-hoc check on the 3 seeds. Expect soup ≤ ensemble.
- **Greedy soup over a hyperparameter grid** carries a winner's-curse risk with our small validation set (~months × 28 stations, daily metric). Prefer uniform soups or ensembles of the top configs over greedy selection on the same validation set.
- **WiSE-FT** is the most relevant averaging method for the unseen-station subset (OOD by construction). Select α on a validation split that includes held-out stations, or report the α curve separately for seen and unseen stations. Note that α changes quantile spread, so p* must be recomputed for each α.
- **EMA + WiSE + ensemble stack.** These are approximately additive in the vision literature (soups of WiSE-ed models). The order of operations that preserves meaning: EMA weights → WiSE α → ensemble over seeds.
- **Reject TIES/DARE.** Same-task seed deltas have little sign conflict, so there is nothing for TIES to resolve. DARE's random drop only adds noise. These methods exist for multi-skill LLM merging.

### Gaps
- No session evidence for soups or WiSE in regression or forecasting fine-tuning. All quantified evidence is classification.
- No direct study of same-task LoRA seed merging.

---

## Q3. Regularisation for small data and distribution shift (L2-SP, LoRA dropout, R-Drop, mixup, NWP noise, covariate dropout, quantile label smoothing, stochastic depth)

### Takeaway
With 4M trainable parameters in a low-rank subspace plus early stopping, EMA and WiSE, we are already heavily regularised toward the pretrained model. Extra regularisers have a small expected effect. L2-SP on the head and LoRA dropout are cheap one-run checks. Mixup and augmentation have negative evidence for channel-independent patch models. Noise injection on NWP is a robustness tool, not an accuracy tool.

### Cited Findings
- **L2-SP.** Penalise ‖w − w_pretrained‖² instead of ‖w‖². The authors recommend it as "the baseline penalty for transfer learning tasks" — [Li, Grandvalet & Davoine, ICML 2018](https://arxiv.org/pdf/1802.01483). A follow-up reports "systematic improvements compared to weight decay", "marginal to moderate" in size, at minimal compute — [Li et al., IVC 2020](https://vision.cornell.edu/se3/wp-content/uploads/2019/12/ivc_li2020.pdf).
- **LoRA vs full FT in TSFMs.** LoRA mostly beats full FT on Chronos for building energy, except for HVAC point estimates. The authors credit LoRA's balance of retention and adaptation. r=4 cuts FLOPs by 33% — [Building-energy TSFM study, 2025](https://arxiv.org/pdf/2506.00630). A 100-step LoRA can match or exceed full-shot training, with dataset-dependent benefit. Full-FT results were omitted from the main tables "due to instability" — [Break-even analysis, 2026](https://arxiv.org/pdf/2607.04919). On ICU data, LoRA is often comparable or better than full FT, and full FT may overfit with limited data — [Gupta et al. 2024](https://arxiv.org/html/2409.11302v1). No considerable LoRA gain vs full FT at r32–128 on Chronos-Bolt/TTM — [Less is More, 2025](https://arxiv.org/pdf/2505.23195).
- **C-Mixup (regression mixup).** Samples mixing partners with a Gaussian kernel on label distance. It gives +6.56% ID, +4.76% task generalisation and +5.82% OOD vs the best prior method on 11 datasets — [Yao et al., NeurIPS 2022](https://arxiv.org/abs/2210.05775).
- **Time-series augmentation.** Effects are mixed: gains on average, but Transformer on traffic/electricity got worse — [Wen et al. survey](https://arxiv.org/pdf/2002.12478). A 4,218-run study found the effect "sharply architecture-conditional": channel-mixing models gain (TimesNet +4.3%, iTransformer +2.3%), while channel-independent models degrade (PatchTST −4.6%, DLinear −7.6%, up to −26%) — [2026 preprint](https://arxiv.org/pdf/2605.06032). Online (per-batch) augmentation beats offline on 3,797 series — [Cerqueira et al.](https://arxiv.org/abs/2404.16918).
- **Missing/unreliable NWP in practice.** The HEFTCom2024 winner filled missing NWP inputs with predictions from the models on the remaining sources. Adding a third NWP source (MetCoOp) gave an 8% pinball-score gain on 2023 validation — [HEFTCom2024](https://arxiv.org/html/2507.01579v2).
- **R-Drop, Mixout, stochastic depth, NEFTune.** Not fetched this session. Canonical refs: [R-Drop](https://arxiv.org/abs/2106.14448), [Mixout](https://arxiv.org/abs/1909.11299), [NEFTune](https://arxiv.org/abs/2310.05914). NEFTune is an instruction-tuning (LLM chat) trick. Zhang et al. found Mixout's relative impact shrinks under a fixed baseline recipe — [Zhang et al.](https://arxiv.org/pdf/2006.05987).

### Inferences
- **L2-SP on the head: add as a one-line change.** The head is the only full-rank trainable block. Plain AdamW weight decay pulls it toward 0, which works against the pretrained head. Use wd=0 on the head plus λ‖θ_head − θ_head⁰‖², or simply wd=0. Effect is likely within noise, but it is principled and free.
- **LoRA dropout (0.05–0.1).** A standard PEFT knob with no time-series evidence. One run.
- **Mixup/augmentation: reject for now.** TimesFM is a channel-independent patch decoder like PatchTST, the class that degraded in the 2026 study. Mixing two PV days also mixes clear-sky phase and shape, which breaks the physics prior that M8 showed is the main source of gains. If tried at all, mix only same-station, same-season days with similar clear-sky index (C-Mixup logic), applied to context and target jointly.
- **NWP perturbation / covariate dropout.** Useful only if production NWP has outages or source switches; HEFTCom shows practitioners handle that structurally. Our NWP covariates enter through the M5 covariate path. Covariate dropout (randomly zeroing one NWP source per sample) is cheap insurance for a deployment fallback, but don't expect a metric gain. Later.
- **Quantile "label smoothing" / stochastic depth: reject.** There is no clear analogue for quantile regression, and stochastic depth on a frozen backbone changes the pretrained function.

### Gaps
- No evidence on noise injection to NWP covariates during training for PV.
- No TSFM-specific evidence on LoRA dropout or L2-SP.

---

## Q4. Data strategy (reweighting, curriculum, hard-day oversampling, recency, auxiliary targets, augmentation, rolling fine-tuning)

### Takeaway
One statistical rule decides most of this: **weights or oversampling that depend only on information available at forecast time (station, month, NWP cloudiness, clear-sky) leave the learned conditional quantiles unbiased. Weights that depend on the realised target y (actual power, realised clear-sky index) change which quantile is learned and corrupt p*.** Station balancing and recency weighting are later-stage options. Rolling refits are standard industrial practice but belong to the deployment protocol and must be validated by rolling-origin backtest.

### Cited Findings
- **Pooling across sites works.** In AMS 2013-14, a single model over all 98 sites beat per-site models (2nd place) — [AMS overview](https://repository.library.noaa.gov/view/noaa/22258/noaa_22258_DS1.pdf). M5 winners generally used cross-learning, one model across many series — [M5 commentary](https://towardsdatascience.com/why-you-currently-do-not-need-deep-learning-for-time-series-forecasting-0de57f2bc0ed/).
- **Distribution shift in a live competition.** In HEFTCom2024, teams handled an export cap by rescaling, clipping or retraining on clipped data. Teams forecasting the total directly "found it harder to adapt than those who could apply simple post-processing methods". A cluster over-forecast because training data lacked "similar periods of constrained production". Quantiles were capped at capacity net of outages (REMIT) — [HEFTCom2024](https://arxiv.org/html/2507.01579v2).
- **Weather-type stratification** (sunny/cloudy/rainy modelling) is a common Chinese PV practice — [SJTU journal figure page](https://xuebao.sjtu.edu.cn/article/2021/1006-2467/1006-2467-55-12-1510/img_6.png.html) (figure only; text not read). A CSG/SGCC-style contest format (5 wind + 5 PV, 3 NWP sources, day-ahead 15-min) is described in a Datawhale learning note — [CSDN note](https://blog.csdn.net/m0_54783477) (not a winner write-up).
- **Online / test-time adaptation for forecasting.** TAFAS adapts a pretrained forecaster at test time using partially observed ground truth, with the largest gains under strong shift — [Kim et al., AAAI 2025](https://arxiv.org/html/2501.04970v1). PROCEED points out that online fine-tuning suffers from feedback delay, because labels arrive only after the horizon; FSNet's optimal-vs-practical gap is ~11%–472% relative MSE — [Zhao & Shen](https://arxiv.org/html/2412.08435v5). Another benchmark finds a PROCEED variant far worse than OneNet on ETTh2, so rankings are protocol-dependent — [arXiv 2510.18281](https://arxiv.org/pdf/2510.18281).
- **Evaluation-length warning.** A PV forecasting review found evidence of "cherry picking" and recommends test sets of at least one year — [Renewable & Sustainable Energy Reviews, via search](https://ideas.repec.org/a/eee/rensus/v81y2018ip1p912-928.html).

### Inferences
- **The y-dependent vs x-dependent weighting rule.** Pinball loss weighted by w(x) still has argmin = the conditional τ-quantile of y|x. Weighted by w(y), it estimates the τ-quantile of a tilted distribution ∝ w(y)p(y|x). This is basic M-estimation algebra (my derivation, not from a fetched source). Therefore:
  - OK: station balancing (equal effective days per station), season balancing, recency weighting, oversampling days flagged cloudy **by NWP cloud cover or forecast clear-sky index**.
  - Not OK, while p* consumes the quantiles: oversampling days that are "hard" by realised error or realised clear-sky index, and loss weights 1/max(y, 0.2cap).
- **Recency upweighting** suits refit on train∪valid (drift from soiling, degradation, curtailment policy). Try an exponential day-weight with half-life ~6–12 months as one run in the refit protocol. No published effect size found.
- **Auxiliary targets** (clear-sky index, irradiance) would need a second head. M8 showed priors matter, but here the prior is better injected as an input/envelope (clear-sky ceiling, history day-shape) than as an auxiliary loss. Later.
- **Curriculum.** No evidence for regression FT. Reject.
- **Rolling fine-tuning / refit.** In production we have actuals up to D−1. A monthly LoRA refit, or a simple bias correction (Q7), is the industrial pattern. Evaluate it only with a rolling-origin backtest that respects the D−1 data lag; PROCEED's feedback-delay point applies directly. Treat it as deployment protocol, not model selection.

### Gaps
- I did not find a public CSG/SGCC PV competition winner write-up.
- No source quantifies recency weighting or station balancing gains for PV.

---

## Q5. Loss alignment (metric-direct training, weighted pinball, CRPS vs pinball, crossing penalties, horizon weighting)

### Takeaway
Keep plain capacity-normalised pinball for the quantiles and let p* do the metric alignment, which already gave +1 point in M8. If more alignment is wanted, use x-only weights such as 1/max(clear-sky or NWP-expected power, 0.2cap). These emphasise low-output periods without biasing quantiles. Remove quantile crossing by sorting at inference instead of penalising it in training. CRPS adds nothing over 9-quantile pinball.

### Cited Findings
- **Competition practice: optimise the scoring metric at the right layer.** The HEFTCom2024 winner trained MultiQuantile CatBoost, tuned only the number of boosting iterations, and combined models with linear quantile regression on the pinball score. UI BUD scored competitively despite "substantial bias"; the organisers note calibration and sharpness can trade off under pinball, but "in general calibrated forecasts are preferred". Daytime pinball scores were about 2× overnight — [HEFTCom2024](https://arxiv.org/html/2507.01579v2). AMS 2013-14 used MAE because it penalises extreme forecasts less than RMSE, and the winner optimised blend weights directly on MAE — [AMS overview](https://repository.library.noaa.gov/view/noaa/22258/noaa_22258_DS1.pdf).
- **Quantile rearrangement.** Sorting the predicted quantiles never increases quantile-estimation error (not re-fetched): [Chernozhukov, Fernández-Val & Galichon, Econometrica 2010](https://arxiv.org/abs/0704.3649).

### Inferences
- **Why not train on 1/max(y,0.2cap)-weighted pinball.** See Q4: it shifts the learned quantiles toward low-y regions. p* then computes the "Bayes-optimal point" from the wrong distribution, double-counting the metric's weighting. That is acceptable only if we drop p* and train a dedicated point output directly on the 南网 loss. That option is worth one experiment as a point head next to the quantile head, but it loses the clean separation that made p* work.
- **x-only alignment variant.** Weight each step by 1/max(P_cs·k, 0.2cap), where P_cs is the clear-sky or NWP-implied power known at forecast time. This keeps quantiles valid and reallocates capacity toward the dawn/dusk and overcast regime where the metric's relative errors are large. It is cheap: one loss-weight tensor.
- **CRPS vs pinball.** Average pinball over a dense quantile grid ≈ CRPS/2 (standard identity), so switching gains nothing with 9 quantiles. Reject.
- **Crossing.** Penalties add a hyperparameter. `torch.sort` on the quantile axis at inference is free, and p* should read sorted quantiles. Add.
- **Multi-horizon weighting.** The day-ahead metric weights all 96 points within a day equally (after the max(P, 0.2cap) normalisation). Horizon weights beyond the x-only weight above are unjustified. Reject.

### Gaps
- No fetched source evaluates weighted-pinball vs post-hoc decision rules for relative-error metrics. The argument above is theoretical.

---

## Q6. Hyperparameter search in industry & avoiding the winner's curse

### Takeaway
Our bottleneck is selection noise (seeds × small validation set), not search efficiency. Industrial guidance converges on these steps:
1. Quasi-random exploration, then focused search.
2. Fix the step budget instead of tuning it.
3. Re-run the best trials N times to measure trial variance.
4. Adopt a change only if it beats that variance.
5. Use early-discard of bad seeds to save compute.

ASHA and PBT are overkill at ~4000 steps per run.

### Cited Findings
- **Google tuning playbook.** Use quasi-random search for exploration and Bayesian optimisation afterwards. With fewer than 10 trials, tune only the base lr. Retrain variance can make identical configs look significantly different, so "run the best trial N times to characterize the run-to-run trial variance" and adopt a change only if it beats the baseline beyond that variance and justifies the added complexity. Use isolation plots for fair comparisons. Treat whether to use a regulariser as a scientific choice and its strength as a nuisance hyperparameter — [tuning_playbook](https://github.com/google-research/tuning_playbook).
- **Seed variance.** Hundreds of BERT runs on four GLUE tasks varying only the seed produced "substantially different results". Weight init and data order contribute comparably to variance. Many small-data runs diverge partway, and a simple early-stopping-of-bad-seeds algorithm improves expected performance for a fixed budget — [Dodge et al. 2020](https://arxiv.org/pdf/2002.06305).
- **Adaptive overfitting.** Across ~120 Kaggle competitions, comparing public-holdout vs private ranks found "little evidence of substantial overfitting". Roelofs concludes distribution shift, not holdout reuse, is the bigger risk — [Roelofs et al., NeurIPS 2019](https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning); [thesis](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-102.html).
- **Competition practice.** The HEFTCom2024 winner tuned only the number of boosting iterations. Top teams selected features by train/validation experiments, while lower-ranked teams relied on EDA — [HEFTCom2024](https://arxiv.org/html/2507.01579v2). AMS winners used 3-fold "continuous" (time-contiguous) CV — [AMS overview](https://repository.library.noaa.gov/view/noaa/22258/noaa_22258_DS1.pdf).
- **Tools** (not re-fetched): [ASHA](https://arxiv.org/abs/1810.05934), [PBT](https://arxiv.org/abs/1711.09846), [Optuna/TPE](https://arxiv.org/abs/1907.10902), [Bouthillier et al., "Accounting for variance in ML benchmarks"](https://arxiv.org/abs/2103.03098).

### Inferences
- **Why Kaggle's reassurance doesn't transfer.** Kaggle holdouts are large and i.i.d. with test. Ours is a few months of 28 stations with seasonal shift to test, so both winner's curse and distribution shift apply. Mitigations, roughly in order of cost:
  1. Pre-declare a small number of comparisons per round (we already work in rounds).
  2. Re-run the top-3 configs with 3 *fresh* seeds before declaring a winner (playbook). Report the mean and a seed-paired difference vs baseline.
  3. Add a second time fold. Rolling-origin works well: train on the first N months, validate on the next 2; then shift. Accept a variant only if it wins on both folds. This also checks seasonal robustness.
  4. Check rank stability with Kendall τ of variant rankings across seeds and folds. If τ is low, the "variants" are noise, which M8 round 2 already suggested.
  5. Touch test once per round for the final pick, and keep a seen/unseen station split in all reports.
- **Search algorithm.** With ~15 variants × 3 seeds × 4000 steps, a grid over 1–2 knobs (LoRA lr, placement) plus successive halving on seeds is enough. Dodge-style early discard can kill seeds whose smoothed validation at ~25% of steps is in the bottom tercile. TPE or BO only becomes useful past ~25 trials on a continuous space. PBT would conflict with the "fixed steps for refit" protocol. Reject PBT.
- **Fix steps.** "Never tune max_train_steps" agrees with our refit-at-max-best-step protocol. If best_step is consistently early, reduce the budget instead of tuning it.

### Gaps
- No TSFM-specific study of selection noise. Our own 3-seed noise band is the best available estimate.

---

## Q7. Test-time techniques (TTA, seed/temporal ensembles, online bias correction, conformal recalibration) and deployment compression

### Takeaway
Seed ensembles: add. Quantile recalibration: diagnose, and add if needed, because p* turns any calibration change into a point change, so "conformal doesn't affect the point" is false for us. Online bias correction (KF/MOS) gives large gains on raw NWP, but evidence is null or negative on top of ML models. Keep it as a drift guard and evaluate it by rolling backtest. NWP-perturbation TTA has no evidence. Distillation, pruning and quantisation are not accuracy levers here.

### Cited Findings
- **Kalman-filter post-processing of NWP solar forecasts.** With 3 training days, MBE fell by 70–94% and RMSE by 8–12%. The optimal window is seasonal, and the filter underestimates clear-sky periods after cloudy runs — [Metro Manila WRF-Solar KF](https://archium.ateneo.edu/manila-observatory/14/). KF bias removal plus spatial averaging cut RMSE about 15%, mostly for regional averages — [NRCan](https://www.nrcan.gc.ca/sites/www.nrcan.gc.ca/files/canmetenergy/files/pubs/2011-062.pdf). For PV models, KF "reduces strongly the bias of the physical model, without positive impact on machine learning methods". Relative RMSE got worse for most ML models after correction, on a short validation period — [SIRTA JSS2022 poster](https://sirta.ipsl.polytechnique.fr/documents/JSS2022/posters/JSS2022_SCros.pdf).
- **Probabilistic PV post-processing** with distributional and quantile regression on ensemble PV forecasts for 7 Hungarian plants is the current frontier (EMOS/DRN) — [arXiv 2508.15508](https://arxiv.org/abs/2508.15508v1).
- **Test-time adaptation** for TSF: TAFAS and its successors PETSA (low-rank adapters), DynaTTA and COSA — [arXiv 2605.17250](https://arxiv.org/pdf/2605.17250); [TAFAS](https://arxiv.org/html/2501.04970v1).
- **Calibration matters in competitions.** "In general calibrated forecasts are preferred" — [HEFTCom2024](https://arxiv.org/html/2507.01579v2). Conformal tools (not re-fetched): [CQR, Romano et al. 2019](https://arxiv.org/abs/1905.03222); [Adaptive conformal inference, Gibbs & Candès 2021](https://arxiv.org/abs/2106.00170).
- **Compression.** Structured pruning is claimed to "unlock specialization" of TSFMs (title and abstract only) — [Less is More, 2025](https://arxiv.org/pdf/2505.23195). TimesFM 2.5 dropped from 500M to 200M parameters vs 2.0 — [timesfm repo](https://github.com/google-research/timesfm).

### Inferences
- **Recalibration → p*.** p* is a functional of the predicted quantiles. If the fine-tuned quantiles are over- or under-dispersed, especially on unseen stations or after WiSE, p* moves systematically. Steps:
  1. Compute per-station PIT coverage on validation for each of the 9 levels.
  2. If coverage deviates by more than a few points, fit a monotone per-level (or spread_lo/spread_hi-style) recalibration on validation, which we already have as T3, then recompute p*.
  3. Optionally use ACI for online drift.
  Judge on the 南网 point metric, not on coverage.
- **Online bias correction.** On top of a learned model with history in context, a KF on residuals is likely neutral or harmful in steady state (SIRTA evidence). It does help when a station shifts: new curtailment, capacity change, soiling. The deployment-safe form is a slow residual EMA per station × hour-of-day, applied only when recent bias exceeds a threshold, and backtested with D−1 lag. Later.
- **TTA by NWP perturbation.** No evidence. The quantile head already models uncertainty, and perturbing inputs mostly widens quantiles, which then interacts with p*. Reject.
- **Temporal ensembles.** Averaging forecasts issued at different runs or offsets is possible for intraday (g-series) but not for the single day-ahead issue. Check it only for ultra-short-term.
- **Distillation, pruning, quantisation.** No accuracy rationale at our scale. V100 fp32 inference for 28 stations is cheap. Reject unless deployment latency or memory forces it. If forced, distil the seed ensemble into one model so the ensemble gain is kept.

### Gaps
- No PV study evaluating KF/MOS on top of a fine-tuned TSFM.
- No evidence that NWP-perturbation TTA helps PV forecasting.
- Techniques that only matter for LLM chat/alignment, flagged for exclusion: RLHF/DPO/preference optimisation, NEFTune, instruction templates, sequence packing, LoRA-soup skill composition, TIES/DARE multi-task merges, and SAM variants tuned for token-level LM losses.
