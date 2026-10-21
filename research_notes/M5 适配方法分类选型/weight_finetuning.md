# Weight fine-tuning methods for M5 (TimesFM-3 3.0.2): method-by-method verdicts for 28-station 15-min PV day-ahead

Scope: weight-space adaptation only (full/partial FT, anti-forgetting regularizers, weight interpolation, LoRA targeting, probing, soft prompts, multi-station structures, continued pretraining, memory-saving PEFT). Builds on `reports/M5 基础模型微调方法调研.md` and `research_notes/M5 基础模型微调方法调研/sft_peft.md`. Evidence from those notes is not repeated except where a verdict depends on it.

Evidence labels: **[TSFM]** measured on a time-series foundation model; **[LLM/vision-only]**; **[local code]** read from the installed `timesfm3` 3.0.2 package or `forecast_backbones/m5`; **[abstract-only]** I read only the arXiv abstract; **[estimate]** my own arithmetic, not measured. Searched and read 2026-10-15.

Local code paths cited below:
- timesfm3 package: `/private/tmp/claude-501/-Users-tqa946816-Documents----NextGenPVForecast/56e02fe2-d7e0-4557-984b-fd80d7f01e47/scratchpad/venv314/lib/python3.14/site-packages/timesfm3/torch/` (`model.py`, `transformer.py`, `dense.py`, `configs.py`). Abbreviated `timesfm3/torch/...` below.
- M5: `/Users/tqa946816/Documents/华为/NextGenPVForecast/forecast_backbones/m5/` (`lora.py`, `train.py`, `transformer_trainer.py`, `config/finetune.yaml`).

---

## 0. Verified TimesFM-3 module map and parameter counts (basis for every recipe below)

### Takeaway
I instantiated `TimesFM3Torch()` with default config and counted parameters by name: 330,710,976 parameters in total. **79% of them are attention (seq_attn 131.1M + var_attn 131.1M), the FFN is small (ratio 1: 1280→1280→1280, 65.5M in total), and no Linear layer except `output_head` has a bias.** The last point means BitFit is not applicable. Its stand-in is RMSNorm-gain tuning (163k params).

### Cited Findings
- Exact module paths and sizes [local code, `timesfm3/torch/model.py`, `transformer.py`, `dense.py`, verified by instantiating the model and running `named_parameters()`]:

  | Module path | Shape | Params |
  |---|---|---|
  | `pre_transformer_resblock.hidden_layer` | Linear 192→1280, no bias | 245,760 |
  | `pre_transformer_resblock.output_layer` | Linear 1280→1280 | 1,638,400 |
  | `pre_transformer_resblock.residual_layer` | Linear 192→1280 | 245,760 |
  | `transformer_stack.layers.{0..19}.seq_attn.{query,key,value,out}_proj` | 4 × Linear 1280→1280 | 6,553,600 per layer (+240 for `query_ln`, `key_ln`, `per_dim_scale`) |
  | `transformer_stack.layers.{i}.var_attn.{query,key,value,out}_proj` | same | 6,553,840 per layer |
  | `transformer_stack.layers.{i}.ff0`, `.ff1` | Linear 1280→1280 each | 3,276,800 per layer |
  | `transformer_stack.layers.{i}.{pre,post}_{seq_attn,var_attn,ff}_ln` | RMSNorm(1280) ×6 | 7,680 per layer |
  | `output_head` | Linear 1280→576 (=64×9) **with bias** | 737,856 |
  | One full layer | — | 16,392,160 |
  | Total | — | **330,710,976** |
- The resblock input is 192 numbers per token: `2*(input_patch_len + output_patch_len)` = [32-step patch | 64-step lookahead (covariates only) | 96 mask bits] [local code, `model.py` `set_input_dims(2*(input_patch_len+output_patch_len))`, `_preprocess`].
- `use_bias=False` for the resblock and every transformer Linear. Only `output_head` has `bias=True` [local code, `model.py` default `ResidualBlockConfig(use_bias=False)`, `TransformerConfig(use_bias=False)`].
- `use_rope_seq=True`, `use_rope_var=False`. `qk_norm="rms"`: RMSNorm is applied to q and k *after* RoPE and projection, followed by a learned `per_dim_scale` on q [local code, `transformer.py` `MultiHeadAttention.forward`].
- `seq_attn` runs over `(b·v, n, d)`, causal, with RoPE positions `arange(n)+next_index`. `var_attn` runs over `(b·n, v, d)`, non-causal, without RoPE, so it is permutation-invariant over variates. The order in each layer is seq_attn → var_attn → FFN, with pre- and post-RMSNorm around each and residual adds [local code, `transformer.py` `MixingTransformer.forward`].
- `forward()` calls `transformer_stack(transformer_input, effective_patch_mask)` without `segment_pos`. The output head is applied to all tokens. When `use_iterative_cpm_revin` is on, the head's raw logits are reused to refine the horizon RevIN stats [local code, `model.py` lines ~300–345].
- M5's current trainable set: r8 LoRA on every Linear whose name ends in `query_proj/key_proj/value_proj/out_proj`. That is 160 modules × 20,480 = **3,276,800**, plus `output_head` 737,856, for **4,014,656 (1.21%)** [local code, `lora.py` `inject_lora` matches on the last name segment; `finetune.yaml`].
- M5 optimizer: HF Trainer AdamW, cosine schedule, warmup 0, lr 1e-4, wd 0.01, grad clip 1.0. `ema_decay: null` and `smooth_k: 1` are implemented but switched off [local code, `transformer_trainer.py` `build_training_args`; `finetune.yaml`].

### Inferences
- **The current weight decay does essentially nothing.** Decoupled AdamW decay shrinks weights by ∏(1 − lr_t·wd). With a cosine schedule (mean lr ≈ 5e-5), wd = 0.01 and 4,000 steps, the factor is exp(−0.002) ≈ 0.998, a 0.2% pull. For the LoRA factors this pulls ΔW = BA toward 0, i.e. toward the pretrained weights. For `output_head` it pulls toward 0, *not* toward the pretrained head. Either way it is negligible [estimate].
- Because q and k are RMS-normalized after projection, a LoRA update on `query_proj`/`key_proj` can only change attention *directions* (which tokens attend to which). `value_proj`/`out_proj` change *what* is carried. This is a mechanistic argument for testing v/o-only LoRA (half the params). I found no empirical test of it [inference].
- r8 LoRA parameter counts for candidate target sets [estimate, r·(in+out)]:

  | Target set | LoRA params |
  |---|---|
  | var_attn q/k/v/o only (80 modules) | 1,638,400 |
  | seq_attn q/k/v/o only | 1,638,400 |
  | var_attn v/o only | 819,200 |
  | ff0/ff1 (40 modules) | 819,200 |
  | resblock hidden/output/residual (r8) | 44,032 |
  | all-linear (attn + FFN + resblock) | 4,140,032 |
  | RMSNorm gains + qk_ln + per_dim_scale (full unfreeze) | 163,200 |
  | resblock full unfreeze | 2,129,920 |
  | last 4 layers full unfreeze | 65,568,640 |

### Gaps
- I did not run a forward/backward pass on a V100, so every cost figure below is FLOP arithmetic, not a measurement.

---

## 1. Full and partial fine-tuning and anti-forgetting regularizers (L2-SP, LP-FT, LLRD, WD-to-pretrained, replay, early stopping, EMA/SWA, soups, WiSE-FT)

### Takeaway
I found **no TSFM study that tests L2-SP, LP-FT, layer-wise LR decay, model soups, SWA or WiSE-FT** for forecasting; all of that evidence is vision or LLM. The TSFM-specific anti-overfitting evidence is about **small lr (1e-6 full / 1e-5 LoRA), best-validation checkpointing and multiple seeds**, plus one study showing that **pruning to ~25–33% of layers and then fine-tuning beat full fine-tuning on TimesFM, Chronos, Moirai and Sundial (ETTh1)**. For us the cheapest high-value items are **EMA + smoothed early stopping (already implemented, switch on)** and **WiSE-FT-style interpolation of the LoRA delta and head (training-free)**. L2-SP and LLRD are only relevant inside a full-FT run. LP-FT's mechanism does not apply (our head is pretrained, not random). Replay of generic series does not target our failure mode.

### Cited Findings
- **WiSE-FT (vision) [LLM/vision-only]:** linearly interpolating zero-shot and fine-tuned weights with one α gave "large out-of-distribution accuracy gains while matching or improving in-distribution accuracy", and some α beat both endpoints. The experiments were CLIP/ImageNet shifts — [Wortsman et al., arXiv 2109.01903](https://arxiv.org/pdf/2109.01903). A follow-up cautions that WiSE-FT may not improve ID and OOD at the same time — [arXiv 2408.05749](https://arxiv.org/pdf/2408.05749).
- **Model soups (vision/NLP) [LLM/vision-only]:** averaging weights of several fine-tuned runs from the same init gives ensemble-like accuracy at single-model inference cost. Greedy soups add a run only if validation improves, because a bad member can drag a uniform soup down — [Wortsman et al., arXiv 2203.05482](https://arxiv.org/abs/2203.05482).
- **No forecasting evidence for weight interpolation:** two searches (WiSE-FT/weight interpolation/weight-space ensemble × TimesFM/Chronos/Moirai/forecasting) found no paper applying it to a TSFM. The closest TSFM practice is **output-space** ensembling: AutoGluon's `chronos2_ensemble` preset combines zero-shot and fine-tuned Chronos-2 as the robust option — [AutoGluon docs](https://auto.gluon.ai/dev/tutorials/timeseries/forecasting-chronos.html) (cited in the earlier notes).
- **Few-layer dominance [TSFM]:** on ETTh1, "prune + fine-tune" beat "fine-tune the full model" for every TSFM tested (MSE): Sundial-Large 0.387 vs 0.398, Chronos-Base 0.416 vs 0.424, Moirai-Large 0.425 vs 0.433, **TimesFM 0.410 vs 0.415**. 67–81% of layers were pruned, with 2.3–3.1× speedups. Important layers sit "in the initial and final portions". Pruning without fine-tuning was much worse (Sundial 0.455 vs 0.387) — [Qiu et al., arXiv 2511.07237 (HTML, Tables 9/11)](https://arxiv.org/html/2511.07237v1). Caveats: only ETTh1 MSE is shown; the paper does not compare "fine-tune only dominant layers" against "fine-tune all"; the TimesFM version is not stated in the part I read.
- **Full FT instability [TSFM, abstract-only]:** a 2026 break-even study of Chronos, Moirai and Lag-Llama on 30 datasets kept full FT results out of its main tables because of instability and reported LoRA instead. It also found **LoRA "can actively degrade performance on short series"** and gave the rule "n_train < 700 and non-negligible seasonality → stay zero-shot" — [arXiv 2607.04919](https://arxiv.org/abs/2607.04919); the full-FT instability note comes from the [search snippet](https://arxiv.org/pdf/2607.04919).
- **TimesFM full FT at lr 1e-4 on one household (V100) [TSFM]:** Adam lr 1e-4, wd 0.01, batch 64, early stopping with patience 10. Fine-tuned TimesFM MAE 4.49 vs a 49.5k-param SAMFormer at 4.28 — [arXiv 2510.00742](https://arxiv.org/html/2510.00742v3). This is weak evidence: one series, and the intro's claim that "extended fine-tuning can lead to performance degradation" comes from a citation, not their own experiment.
- **Regularized-FT recipes that worked on TSFMs [TSFM, from the earlier notes]:** Chronos-2 grid load: full FT at lr 1e-6, LoRA at lr 1e-5, ReduceLROnPlateau, best-validation checkpoint, 5 seeds; full FT > LoRA with covariates — [arXiv 2609.06656](https://arxiv.org/html/2609.06656v1). PaCTS baselines: full FT at lr 1e-6, LoRA r8 at lr 1e-5, 5,000 steps, batch 256 — [arXiv 2609.34786](https://arxiv.org/html/2609.34786).
- **Replay [TSFM]:** MixFT used experience replay with pretraining data plus MixUp on Chronos-Bolt/Moirai, but did not ablate replay alone — [arXiv 2603.02840](https://arxiv.org/html/2603.02840v1) (from the earlier notes). R-Tuning (wavelet-augmented replay of model-generated samples + latent consistency) reports up to 46.9% lower MAE on new tasks and up to 5.7% MAE better retention on old tasks, against unspecified baselines — [arXiv 2511.11685](https://arxiv.org/abs/2511.11685) [abstract-only]. An incremental-FT study reports that Time-MoE and Chronos keep improving under incremental fine-tuning while conventional models degrade, with no numbers in the abstract — [arXiv 2504.14677](https://arxiv.org/abs/2504.14677) [abstract-only].
- **LP-FT / L2 penalties / LLRD [LLM/vision-only]:** LP-FT helps because a *randomly initialized* head distorts pretrained features during FT — [generality-forgetting study](https://scispace.com/pdf/speciality-vs-generality-an-empirical-study-on-catastrophic-44oixzw2bg.pdf); Kumar et al. [arXiv 2202.10054](https://arxiv.org/abs/2202.10054). L1/L2/KD penalties reduce generality forgetting in LLM FT (same source). LiNeS argues that shallow layers hold general features and deep layers specialize, so it scales down shallow-layer updates after FT — [ICLR 2025 LiNeS](https://proceedings.iclr.cc/paper_files/paper/2025/file/43ae0b566b802b2d00e525b672794b82-Paper-Conference.pdf). I found no TSFM test of LP-FT, L2-SP or LLRD.
- **EMA/SWA in time series:** the project's own M8 review found no with/without table for TS; TSPP is self-contradictory but 27 of its 34 best configurations chose EMA — `docs/survey_m8_architecture_review.md` §6.4 (local). No TSFM-specific EMA study was found in this session.

### Inferences (verdicts and recipes)
- **EMA + smoothed early stopping → ADOPT (protocol, not a method).** Set `train.ema_decay: 0.998` (≈500-step horizon; 0.999 ≈ 1,000 steps is too slow if the best step is ≤2,000, as in M8's 0.4–0.6 epoch peak), `early_stop.smooth_k: 3`, `eval_every_steps: 250`, 3 seeds, and report the median. Cost: EMA shadows only trainable params (4.0M ≈ 16 MB), so it is free. This directly targets the M8 failure (validation swings of 1.5–3 points between evals). SWA adds nothing over EMA here; skip it.
- **WiSE-FT interpolation → ADOPT as a free post-hoc knob** (evidence is vision-only, so validate on our data). For LoRA it is exact and cheap: W(α) = W₀ + α·(α_lora/r)·BA for every LoRA module, and head(α) = (1−α)·head₀ + α·head_ft. Evaluate α ∈ {0, 0.25, 0.5, 0.75, 1.0} on validation with p* re-selected at each α, separately for seen and unseen stations. Pick α on unseen-station validation if production will add new stations. Cost: 5 evals, no training. Compare it with output-space averaging of zero-shot and LoRA quantiles (AutoGluon-style), which costs 2× inference and is not equivalent for a 20-layer model. Why it fits us: MixFT and PaCTS both show weight-space FT can lose to zero-shot on unseen entities, and the 国能和煦 case (+12.2 for zero-shot M5 over M8) shows zero-shot day-shape skill is worth keeping. Implementation: a `scale` attribute on `LoRALinear` (multiply `self.scaling`) plus a head blend in `load_lora_state_dict`.
- **Greedy soup of 3–5 seed LoRA runs → TRY LATER.** Averaging LoRA factors A and B is *not* the same as averaging ΔW, because the product is bilinear. Average the merged ΔW_i = B_i·A_i (rank ≤ 5r) and the heads instead. Expect no more than the M8 seed-ensemble gain (+0.3–0.5), at single-model inference cost. Only worth it if the deployment cannot run a 3-member output ensemble.
- **Full FT → TRY LATER (stage 5), and only with this recipe:** all 330.7M params; lr 1e-6 (one run at 1e-5 as an upper bound); AdamW wd 0 plus **L2-SP** λ‖θ − θ₀‖² with λ ∈ {1e-3, 1e-2} (on capacity-normalized pinball scale; tune). Use the *pretrained* head as the anchor, not 0. EMA 0.998, smooth_k 3, ≤4,000 steps, 3 seeds, WiSE-FT α sweep at the end. Memory: weights 1.32 GB + grads 1.32 + Adam 2.65 + anchor 1.32 + EMA 1.32 ≈ 8 GB plus small activations (81 tokens/sample) → fits a 16 GB V100 at batch 16 [estimate]. Time: ≈ 3 × 2 × 330M × 81 tokens ≈ 161 GFLOP/sample → 2.6 TFLOP/step at batch 16. At ~40% of 15.7 TFLOPS fp32 peak that is ≈ 0.4 s/step, ≈ 30 min per 4,000 steps per seed [estimate; V100 peak from the [NVIDIA V100 page](https://www.nvidia.com/en-us/data-center/v100/)]. Evidence for and against: grid load full FT at 1e-6 > LoRA with covariates; small-data/lr 1e-4 full FT < zero-shot; full FT unstable in the break-even study. M8: adding capacity did not help. Expected gain: ≈0 ± noise relative to tuned LoRA.
- **Partial FT (last-N / first+last layers) → TRY LATER, folded into the full-FT stage.** The few-layer-dominance result says first and last layers matter most. A cheap variant: unfreeze `transformer_stack.layers.{0,1,18,19}` + resblock + head = 4 × 16.39M + 2.13M + 0.74M ≈ 68.4M params, lr 1e-5, L2-SP. It costs nearly as much as full FT because backprop still runs from layer 19 down to layer 0 (the resblock is at the input). Without the resblock it is "last 2 layers + head" (33.5M), which needs backprop only through 2 layers: ≈ 1/3 of LoRA's step time [estimate].
- **Layer-wise LR decay → TRY LATER, only inside full FT** (decay 0.8 per layer from the top, i.e. layer 0 gets 0.8¹⁹ ≈ 0.014× the head lr). Vision/NLP evidence only. TimesFM's important first layers (few-layer dominance) argue *against* a strong decay at the bottom. Low priority.
- **LP-FT → REJECT.** Its mechanism (a random head distorting features) is absent: `output_head` is pretrained and already outputs 64×9 quantiles in the right space. The head-only/probe run is still useful as a *control* (§3), not as a warm start.
- **Weight decay toward pretrained for LoRA → no action.** It is already implicit (wd on A,B shrinks ΔW toward 0) and negligible at the current settings (see §0). Raising wd to 0.1 gives only a 2% shrink over 4k steps; it is not a meaningful regularizer at these lrs.
- **Replay of synthetic/pretraining-like series → REJECT for LoRA, TRY LATER only as a full-FT stabilizer.** Our risk is not forgetting non-PV skills (production never forecasts non-PV with this model). It is overfitting to 28 stations and losing zero-shot day-shape transfer to unseen ones. Generic replay does not address that. A better "replay" for us is **function-space anchoring to the zero-shot model on our own PV windows** (a pinball/L2 term between the FT and zero-shot quantiles, weight 0.1–0.3) — LwF-style, LLM/vision evidence only. TimesFM-3's pretraining corpus is not public, so literal replay is impossible anyway; synthetic KernelSynth-style series would be the substitute.
- **Early stopping on 南网 after p* → keep**; add smooth_k 3 (above).

### Gaps
- No TSFM paper tests WiSE-FT, soups, L2-SP, LP-FT, LLRD or SWA. Every verdict above transfers vision/LLM mechanisms.
- The few-layer-dominance paper's per-model dominant-layer indices for TimesFM were not visible in the part I read (characters 0–100k of the HTML).
- No measured forgetting or unseen-entity curve exists for TimesFM fine-tuning.

---

## 2. LoRA targeting on TimesFM-3: which modules carry NWP → power?

### Takeaway
No published study ablates where adaptation should go in a TSFM with native covariates: not Chronos-2 group-vs-time attention, not TimesFM-3 var-vs-seq attention. The only layer-level TSFM evidence is general: first and last layers dominate (§1), and FETS found {q} vs {q,k,v,o} gave the same median. **From the code, NWP influence on same-time power has exactly two learnable entry points:** `pre_transformer_resblock` (reads the 64-step NWP lookahead inside each covariate token) and every layer's `var_attn` (moves covariate information into the target token at the same patch). History day-shape is carried by `seq_attn` on the target variate. The current LoRA covers both attentions but not the resblock. Verdict: **adopt "var_attn + resblock + head" as the main candidate**, with a var-only vs seq-only diagnostic pair.

### Cited Findings
- Code path [local code, `model.py` `_preprocess`; `transformer.py` `MixingTransformer.forward`]:
  - Each covariate token = RevIN(patch) ‖ RevIN(next 64 steps) ‖ masks. The target and past-only lookahead is masked by `patch_is_target`. The token passes through `pre_transformer_resblock`.
  - In each layer, `var_attn` attends across `[target, ghi, temp]` at the same patch index (non-causal, no RoPE). Horizon target tokens have masked inputs, so their content comes from seq_attn (target history) and var_attn (covariates).
  - The head reads the target-variate tokens.
- Chronos-2 has the same time/group alternation. Practitioner LoRA configs target q/k/v/o of both attention types, and I found no published ablation of group-only vs time-only — [Chronos-2 paper](https://arxiv.org/html/2510.15821v1); the [search](https://towardsdatascience.com/five-ways-to-fine-tune-chronos-2-the-time-series-foundation-model/) returned no module-level ablation (the article itself returned HTTP 403).
- LoRA on attention q/k/v/o is the standard TSFM target (Lag-Llama/Moirai/Chronos) — [arXiv 2405.10216](https://arxiv.org/pdf/2405.10216). Google's official TimesFM-2.5 example uses `all-linear`, r4 — [timesfm finetuning README](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/finetuning) (earlier notes). The TimesFM README does not say whether this example supports 3.0 — [timesfm README](https://github.com/google-research/timesfm).
- FETS (Chronos-2, energy): targets {q} vs {q,k,v,o} and rank 4 vs 16 left the median NRMSE change in the same −18% to −28% band — [arXiv 2604.22328](https://arxiv.org/abs/2604.22328) (earlier notes).
- TRACE (MOMENT): gated module selection beat plain LoRA only slightly (ETTh1-192 MSE 0.412 vs 0.420) — [arXiv 2503.16991](https://arxiv.org/html/2503.16991) (earlier notes).

### Inferences (recipes)
- **Main candidate L-va: `var_attn` q/k/v/o LoRA r8 + resblock LoRA r8 + `output_head` unfrozen → ADOPT (stage 3).** Trainable: 1,638,400 + 44,032 + 737,856 = **2,420,288 (0.73%)**. Same lr protocol as the baseline (sweep {3e-5, 1e-4}). Cost ≈ the current LoRA: backward still traverses all 20 layers because LoRA sits in every layer and the resblock is at the input. ≈ 2 × forward ≈ 107 GFLOP/sample → ≈ 0.27 s/step at batch 16 → ≈ 18 min per 4,000 steps on one V100 [estimate]. Rationale: it freezes the history/day-shape path (seq_attn), which is the zero-shot skill that won at 国能和煦, while adapting the NWP→power path.
- **Diagnostic pair → ADOPT (one run each, 3 seeds if budget allows):**
  - L-s: seq_attn-only LoRA (1.64M) + head.
  - L-v: var_attn-only LoRA (1.64M) + head.
  - If L-v ≈ current and L-s ≪ current, NWP mapping is where the gain is.
  - If L-s ≈ current, LoRA is mostly re-learning day shape/level (station-specific), which predicts poor unseen-station transfer.
- **Add resblock to the current target set → ADOPT (stage 3).** Add `hidden_layer`, `output_layer`, `residual_layer` to `target_modules`. Those tail names hit only the resblock, since `output_head` has a different name [local code]. Only +44,032 params. This is the only learnable reader of the NWP 64-step lookahead.
- **FFN (`ff0`/`ff1`) → TRY LATER.** +819,200 params. The FFN is small in this model (ratio 1, 10% of params), and "all-linear" is Google's default, but FETS insensitivity and M8's "capacity doesn't help" put the expected gain at ≈ noise.
- **v/o-only LoRA → TRY LATER** (halves params to 819k for var_attn). Motivated by qk-norm (§0). No evidence.
- **Rank sweep, DoRA, AdaLoRA, VeRA, FourierFT → REJECT** (same as the earlier report; FETS insensitivity, M8 noise band).
- **Code change needed:** `inject_lora` matches only the last name segment. Path-scoped targeting needs a full-name regex, e.g. `^transformer_stack\.layers\.\d+\.var_attn\.(query|key|value|out)_proj$`. Optional per-layer selection: `layers\.(0|1|18|19)\.` for a first+last-layer variant motivated by few-layer dominance. The existing `already_wrapped` check and `load_lora_state_dict` key validation keep working, since parameter names are unchanged.

### Gaps
- There is no empirical study of var-vs-seq attention adaptation in any covariate-native TSFM. The proposed diagnostic would be the first evidence, and it is cheap.
- I did not check whether TimesFM-3 was pretrained with covariates in enough volume for var_attn to be already well-shaped (the blog has no numbers). If var_attn is weak at zero-shot, var-LoRA gains could be larger than FETS-style insensitivity suggests.

---

## 3. Linear probing / head-only and head + input-embedding tuning

### Takeaway
Head-only or probe tuning matched LoRA on MOMENT. On TimesFM 1.0, "LP" (input + output residual blocks) was better than zero-shot but worse than full FT (Monash scaled MAE 0.676 vs base 0.694 vs full 0.663). On TimesFM-3 the head is a single Linear(1280→576) and the backbone is deterministic, so **head-only training can run on cached embeddings in minutes**. That makes it the cheapest control for "how much of the LoRA gain is just recalibration". **Head + resblock** costs a full backward pass (the resblock is at the input), so it is not cheaper than LoRA. It is still a useful control because it is TimesFM's own "LP" definition.

### Cited Findings
- TimesFM 1.0 Monash: base 0.694, LP (input + output residual blocks) 0.676, full FT 0.663, ICF 0.643 scaled MAE — [arXiv 2410.24087](https://arxiv.org/html/2410.24087) (earlier report).
- MOMENT: LP ≈ LoRA (ETTh1-192 MSE 0.417 vs 0.420). Large flatten heads overfit, and shrinking them by up to 98% matched or improved results — [TRACE, arXiv 2503.16991](https://arxiv.org/html/2503.16991). MOMENT healthcare: LP 2.622 < prompt 2.918 < LoRA 3.109 MSE — [arXiv 2411.12824](https://arxiv.org/html/2411.12824) (earlier notes).
- TimesFM-3 head: `output_head` = Linear(1280, 64×9, bias), 737,856 params. Raw logits are de-normalized with the per-variate RevIN stats, and with CPM refinement on they also feed the iterative RevIN refinement [local code, `model.py`].

### Inferences (recipes)
- **Head-only (H0) → ADOPT as a control (stage 2).**
  - Train only `output_head` (737,856 params) with the same pinball/cap loss, lr 1e-4 to 1e-3, 1–2k steps, EMA.
  - Cheap path: one frozen forward over all training windows, caching the target-variate horizon tokens' `transformer_output` (6 tokens × 1280 fp32 ≈ 30 KB/window, ≈ 7–8 GB for ~250k windows) plus the RevIN stats. Training is then a 1280→576 regression costing seconds per epoch [estimate].
  - Caveat: with `use_iterative_cpm_revin` the refinement depends on head outputs, so either recompute the refine step in the cached loop or turn CPM refinement off for H0 and for its zero-shot comparison.
  - Interpretation: if H0 ≈ current LoRA, the gain is level/spread recalibration and an output-side corrector (XReg/ELF-style, in the earlier report) could do the same job.
- **Head + resblock (LP, TimesFM-paper definition) → ADOPT as a control (stage 2).** Trainable 2,129,920 + 737,856 = 2,867,776 (0.87%). lr 1e-4 (resblock at 3e-5 if unstable). Cost ≈ LoRA (full backward) [estimate]. It answers whether re-reading the NWP lookahead at the input, without touching attention, captures the NWP gain.
- **Frozen head + LoRA → ADOPT as a control.** The unfrozen 738k-param head may carry both the gain and the overfitting (TRACE's large-head overfitting).
- **Head-only per station (28 heads) → REJECT.** 28 × 738k params trained on 1–2 years each is the per-entity configuration that hurt in FETS/MixFT, and it cannot serve unseen stations.

### Gaps
- No TimesFM-2.5/3 probe-vs-LoRA numbers exist. The MOMENT "LP ≈ LoRA" result comes from a model with a much larger flatten head (9.4M at H=192).

---

## 4. Prompt learning (soft prompts / prefix) for TimesFM-3

### Takeaway
The best TSFM evidence (PaCTS, 2026) shows small in-distribution gains on TimesFM-2.5: MASE −1.1% to −1.5%, CRPS −1.0% to −1.4% on GIFT-Eval. On an unseen benchmark the gains were tiny (Chronos-2 TIME: MASE −0.45%, CRPS −0.54%), but PaCTS was the only method there that beat zero-shot. LoRA/FT numbers are reported for Chronos-2 only. **On TimesFM-3 the cleanest implementation is not a time-axis prefix but extra learnable "virtual variates" on the variate axis**: var_attn has no RoPE and is permutation-invariant, so no positional bookkeeping is needed. Per-station prompts cannot serve unseen stations; a context-statistics generator (PaCTS-style) or a shared prompt can. Verdict: **TRY LATER**. The expected in-distribution gain is below M8's ±0.6 noise band, and its value is unseen-station robustness.

### Cited Findings
- PaCTS on TimesFM-2.5 (GIFT-Eval) [TSFM]:

  | Context | Zero-shot MASE / CRPS | PaCTS MASE / CRPS |
  |---|---|---|
  | 4,096 | 0.717 / 0.514 | 0.706 / 0.507 |
  | 8,192 | 0.709 / 0.506 | 0.700 / 0.500 |
  | 16,384 | 0.706 / 0.503 | 0.698 / 0.498 |

  - PaCTS at 4,096 ≈ zero-shot at 8,192 (MASE −0.42%) with 48% fewer GFLOPs.
  - Prompts are prepended to the patch embeddings; prompt tokens get **position ID 0** and real tokens keep their zero-shot positions.
  - Defaults: M=10 tokens, rank 4, ≤16 segments; AdamW lr 1e-3, effective batch 48, ≤12 epochs.
  - LoRA/full-FT baselines (r8/α16, lr 1e-5; full lr 1e-6; 5,000 steps, batch 256) were run on Chronos-2 only.
  - Ablation (Chronos-2): static prompts 0.707/0.490 → +adaptive 0.704/0.486 → +segment refinement 0.697/0.480. Static prompts "improve CRPS but provide little improvement in MASE".
  - Unseen TIME: 0.661/0.555 vs zero-shot 0.664/0.558, and all five weight-space baselines were worse than zero-shot.
  - Source: [arXiv 2609.34786](https://arxiv.org/html/2609.34786)
- Gen-P-Tuning attaches a learned cross-channel prefix to every channel at each layer of a frozen univariate TSFM (MOMENT, GPT4TS, TimesFM family) — [PMLR v259](https://proceedings.mlr.press/v259/liu25a.html). Black-box cross-prompt adaptation injects prompts into MHSA with the backbone frozen — [arXiv 2510.00487](https://arxiv.org/pdf/2510.00487).
- TimesFM-3 attention facts [local code, `transformer.py`]: seq_attn RoPE positions = `arange(n) + next_index` unless `segment_pos` is passed, and `model.forward` passes none. Causal mask over patches. var_attn: no RoPE, non-causal, over the v axis. Leading fully-masked patches are masked via cumprod.

### Inferences (implementation and verdict)
- **Option A, virtual-variate prompts (recommended for TimesFM-3):**
  - Build P ∈ ℝ^{k×n×1280} (k = 2–4), either one learned vector per prompt broadcast over the n=27 patches, or a learned vector + a learned per-patch-index offset.
  - Concatenate P to `transformer_input` along the variate axis after `pre_transformer_resblock`, with `patch_mask` False for these variates. Drop them before `output_head` (or slice to the target variate) so the RevIN de-normalization and CPM refinement only see real variates.
  - Effect: in every layer the target token at each patch attends to the prompt variates in var_attn (no RoPE, so position is irrelevant). Prompt variates also run through seq_attn over their own constant sequence, which is harmless.
  - Params: k × 1280 = 2.6k–5.1k shared. Per station: 28 × 5,120 ≈ 143k. With a time-of-day offset, ×27.
  - Cost: v grows from 3 to 5–7 variates, i.e. 1.7–2.3× tokens in forward and backward, and backprop runs through all layers → ≈ 2× LoRA step time [estimate].
  - Requires a small fork of `TimesFM3Torch.forward` (or a hook on `pre_transformer_resblock` output plus slicing before the head). `max_variates=32` appears only in config and is not a constraint at 5–7 variates [local code grep].
- **Option B, time-axis prefix (PaCTS-style):** prepend k tokens to each variate's sequence. RoPE is relative, so shifting real tokens by k leaves their mutual geometry unchanged; the prefix just sits k..1 patches "before" the context. To reproduce PaCTS's "position 0 for prompts, zero-shot positions for real tokens", pass `segment_pos` through `transformer_stack`. That needs a code change, because `forward` passes none. More invasive than Option A, and the decode-cache path (`next_index`) must also be handled for inference.
- **Per-station prompts for unseen stations:** not viable directly. Fallbacks: (i) the mean of trained station prompts; (ii) a PaCTS-style generator mapping context statistics (7-day mean/peak, daily-max envelope, autocorrelation at lag 96, NWP-GHI-to-power ratio) → prompt via rank-4 basis (~10–50k params), which works for any station with 7 days of history. A metadata (lat/lon/tracker) → prompt map is not learnable from 28 stations, and one station has no coordinates anyway.
- **Verdict: TRY LATER (stage 5)** as "shared/generated virtual-variate prompt + frozen backbone", evaluated mainly on unseen stations and combinable with L-va LoRA. Expected seen-station gain ≈ 1% relative MASE in PaCTS terms, i.e. ≪ 0.6 南网 points. Do not prioritize over WiSE-FT, which also targets unseen-station robustness at zero cost.

### Gaps
- No prompt-tuning study exists on TimesFM-3 or on any TSFM with past-future covariates. The virtual-variate design is my proposal and has no published evidence.
- PaCTS reports no energy/solar breakdown and no ablation over the number of prompt tokens.

---

## 5. Multi-station structure: clustered/mixture LoRA, station-conditioned adapters, hypernetwork LoRA, meta-learning

### Takeaway
For 28 stations the only structure with TSFM evidence of beating one pooled adapter is **MixFT's K=2 cluster LoRAs with hard routing** (avg rank 2.0 vs shared 3.0 vs zero-shot 2.9 on Chronos-Bolt; unseen datasets). **Per-entity adapters have evidence of harm.** **Hypernetwork-generated LoRA and MAML/Reptile have no TSFM or PV-TSFM evidence** and need far more entities than 28. Verdicts: clustered LoRA = TRY LATER; per-station LoRA/heads = REJECT; hypernetwork LoRA = REJECT; MAML/Reptile = REJECT.

### Cited Findings
- MixFT [TSFM]: Bayesian GMM on embeddings → K sub-domains → one LoRA (r2, α16, lr 5e-5, batch 256, MixUp + replay) each, hard routing at inference. K=2 best on validation; hard routing (rank 1.6) ≫ soft (4.4). Per-dataset LoRAs (rank 6.3) were worse than not fine-tuning — [arXiv 2603.02840](https://arxiv.org/html/2603.02840v1) (earlier notes).
- FETS per-series Chronos-2 LoRA improved only 16/54 series, and the median regressed (NRMSE 0.599 vs 0.472 zero-shot) — [arXiv 2604.22328](https://arxiv.org/abs/2604.22328) (earlier notes).
- Few-shot PV for new stations in the non-TSFM literature is done by transfer from similar mature stations (IDS-Net picks source stations via MMD) or by a single cross-learned global model conditioned on site metadata + NWP — [IDS-Net, arXiv 2507.12745](https://arxiv.org/html/2507.12745); [cold-start residential PV cross-learning](https://www.sciencedirect.com/science/article/abs/pii/S0378779622005648). Stacked-LSTM transfer from source plants gave 12.6% lower RMSE with 1 year of target data — [Transfer learning under data scarcity, PMC9420121](https://pmc.ncbi.nlm.nih.gov/articles/PMC9420121/) (snippet).
- MAML for forecasting exists for short-term *load* with recurrent nets (per-layer meta-learned inner lr), not for TSFMs — [arXiv 2406.05887](https://arxiv.org/pdf/2406.05887) (snippet).
- Hypernetwork → LoRA appears only in LLMs (Zhyper) — [arXiv 2510.19733](https://arxiv.org/pdf/2510.19733) [LLM-only].
- Cold-start PV with TSFMs (Chronos-2, TabPFN-TS, 440 sites, 4 datasets): zero-shot conditioning on physics-informed synthetic histories beat classical baselines by ~1.7–2×, and performance was largely insensitive to the synthetic-history source. Fine-tuning was not tested — [arXiv 2606.07457](https://arxiv.org/abs/2606.07457) [abstract-only].
- The break-even rule "n_train < 700 & seasonal → zero-shot" and "LoRA can degrade on short series" — [arXiv 2607.04919](https://arxiv.org/abs/2607.04919) [abstract-only].

### Inferences (recipes)
- **Pooled single adapter (current) → keep as baseline.**
- **Clustered LoRA K=2 → TRY LATER (stage 5).**
  - Clusters from station metadata (tracker/bifacial vs fixed, or region), or from k-means/GMM on mean-pooled zero-shot `transformer_output` of target tokens over each station's training windows. Use hard routing by station; route unseen stations by embedding.
  - Each cluster LoRA uses the L-va recipe (2.42M params) at the same lr, initialized from the pooled LoRA (warm start halves steps).
  - Cost: K runs × ≈18 min × 3 seeds [estimate].
  - Risk: each cluster has ~14 stations, so the M8 "capacity doesn't help" lesson and seed noise likely swamp it. Run only if per-station deltas from the pooled LoRA show a clear bimodal pattern, e.g. tracker stations losing.
- **Station-ID conditioning → TRY LATER only via §4's prompt variates** (≈5k params per station). Never as per-station LoRA or heads.
- **Per-station LoRA / per-station head → REJECT** (FETS 16/54, MixFT per-dataset rank 6.3; M8 seen/unseen gap).
- **Hypernetwork LoRA → REJECT.** Learning a metadata→ΔW map from 28 examples, one without coordinates, is under-determined. No TSFM evidence.
- **MAML/Reptile → REJECT.** Second-order or many-inner-step meta-training on a 330M model in fp32 on V100 costs ~K× a LoRA run. No TSFM evidence. And the few-shot regime it targets does not match ours: every production station has 1–2 years of history, and genuinely new stations are better served by zero-shot + WiSE-FT-blended pooled LoRA.

### Gaps
- No mixture-of-LoRA study on PV sites, and no leave-station-out evaluation of any fine-tuned TimesFM.

---

## 6. Continued / domain-adaptive pretraining on extra PV series without NWP

### Takeaway
I found **no TSFM study of domain-adaptive continued pretraining on solar/PV** (or on energy generally) followed by fine-tuning. The only TSFM continued-pretraining example is TimesFM-ICF, which used part of the original pretraining corpus to teach in-context learning, with no weights released. For TimesFM-3 there are specific risks: pretraining on target-only PV series would train no covariate pathway, could drift var_attn and the resblock away from covariate use, and costs a full-FT run per pass. Verdict: **REJECT for now**. Pooled multi-station LoRA on our own 28 stations is already the in-domain adaptation.

### Cited Findings
- TimesFM-ICF: continued pretraining on part of the TimesFM pretraining corpus, with 50 in-context examples (5 from the target's own history, 45 from other series) — [Google Research blog](https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/); [arXiv 2410.24087](https://arxiv.org/pdf/2410.24087).
- The ERCOT benchmark says current TSFMs "struggle to capture the domain-specific dynamics of renewable generation" without adaptation, with accuracy improving up to 80% of the training data. For unseen solar sites, TimesFM and Chronos-Bolt lost the least (~1% absolute nMAE) — [arXiv 2604.22077](https://arxiv.org/html/2604.22077v1).
- The clean-energy FM review recommends continual learning to absorb new plants without forgetting, but gives no experiment — [arXiv 2507.23147](https://arxiv.org/html/2507.23147v1).
- SolarFM uses a lightweight pretrained TSFM per climatic factor inside a spatio-temporal pipeline, not DAPT — [SolarFM, Applied Energy](https://www.sciencedirect.com/science/article/abs/pii/S0306261926001844) [abstract-only].
- The cold-start PV study found zero-shot TSFMs largely insensitive to the source of the synthetic history, with no fine-tuning or continued pretraining tested — [arXiv 2606.07457](https://arxiv.org/abs/2606.07457) [abstract-only].
- The TimesFM 3.0 weights are licensed "non-commercial, non-production use" only, and the README has no 3.0 fine-tuning guidance — [timesfm README](https://github.com/google-research/timesfm).

### Inferences
- **DAPT on public PV without NWP → REJECT now.**
  - Mechanism: in TimesFM-3, a target-only series exercises the resblock (with masked lookahead) and seq_attn, but var_attn sees a single variate. The day-shape prior is already strong at zero-shot (国能和煦 +12.2), while the NWP mapping, which is our bottleneck, gets no signal.
  - Cost: full FT at lr ≤1e-5 over millions of windows, i.e. hours to days on V100 fp32 [estimate].
- **If ever revisited (TRY LATER, low):** do it with NWP-like covariates, e.g. public PV with reanalysis GHI (ERA5) as a past-future covariate, mixed 1:1 with our stations, as LoRA (not full FT) at lr 1e-5. Then fine-tune on our 28 stations. Judge only on unseen-station validation.

### Gaps
- No TSFM DAPT-for-PV evidence at all. Public PV datasets with 15-min NWP comparable to ours were not checked in this session.

---

## 7. QLoRA and other memory-saving methods

### Takeaway
**REJECT.** At 330M params in fp32 the whole model is 1.32 GB. Even full FT with AdamW, an L2-SP anchor and EMA fits in ≈ 8 GB (§1). QLoRA's only benefit is memory, and it adds 4-bit NF4 quantization error to the frozen base, which zero-shot and LoRA evaluation then inherit. I found no TSFM QLoRA-vs-LoRA comparison; the general LLM evidence is "comparable to slightly worse".

### Cited Findings
- No TSFM QLoRA benchmark was found. A HAR paper notes that QLoRA "has not yet been widely applied" in that time-series task — [arXiv 2512.17983](https://arxiv.org/pdf/2512.17983). LLM-side comparisons are vendor blogs rating QLoRA "slightly below" or "comparable to" LoRA — [Modal blog](https://modal.com/blog/lora-qlora); [index.dev](https://www.index.dev/blog/top-ai-fine-tuning-tools-lora-vs-qlora-vs-full) [LLM-only, non-peer-reviewed].
- QLoRA design (frozen 4-bit NF4 base + LoRA) — Dettmers et al. [arXiv 2305.14314](https://arxiv.org/abs/2305.14314) [LLM-only].
- Model size 330,710,976 params; the transformer runs on 81 tokens/sample (3 variates × 27 patches) [local code + earlier report].

### Inferences
- QLoRA, 8-bit Adam, gradient checkpointing (`use_remat` exists in config but the torch path is inference-only) → **REJECT**: no memory pressure at batch 16.
- fp16 AMP is the only memory/speed method with a real upside on V100 (tensor cores, ~2–4× in principle), but `value_clip=1e20` and RevIN on raw-unit values make overflow plausible. Accuracy benefit: none. Speed benefit: possible. **Not recommended** without a bit-exact zero-shot parity check against fp32.

### Gaps
- V100 step times were not measured, and fp16 numerical safety for TimesFM-3 is unverified.

---

## 8. Final ranking for our scenario (adopt / try later / reject)

### Takeaway
Two things are worth adopting now, and neither is a new parameterization:
- **The protocol:** EMA, smoothed early stopping, 3 seeds, LoRA lr sweep, all from M8's lessons.
- **Training-free WiSE-FT interpolation**, plus cheap controls that locate the gain: head-only, head+resblock, var-only vs seq-only LoRA, frozen head.

The one targeting change with a code-level rationale is **LoRA on `var_attn` + resblock + head (2.42M params)**. Full/partial FT with L2-SP, clustered LoRA and virtual-variate prompts are second-tier. Hypernetworks, MAML, DAPT, LP-FT, QLoRA and PEFT-family swaps are rejected.

### Cited Findings
- The evidence base for each line is in §§1–7. Key quantitative anchors:
  - FETS configuration insensitivity (−18% to −28% median band) — [arXiv 2604.22328](https://arxiv.org/abs/2604.22328)
  - Grid-load full FT (lr 1e-6) > LoRA with covariates — [arXiv 2609.06656](https://arxiv.org/html/2609.06656v1)
  - TimesFM prune+FT 0.410 vs full FT 0.415 — [arXiv 2511.07237](https://arxiv.org/html/2511.07237v1)
  - PaCTS TimesFM-2.5 −1.1% to −1.5% MASE, unseen −0.45% — [arXiv 2609.34786](https://arxiv.org/html/2609.34786)
  - MixFT K=2 rank 2.0 vs shared 3.0 — [arXiv 2603.02840](https://arxiv.org/html/2603.02840v1)
  - WiSE-FT OOD gains, vision only — [arXiv 2109.01903](https://arxiv.org/pdf/2109.01903)
  - M8 seed noise ±0.6, EMA/smoothing as protocol levers — `docs/survey_m8_architecture_review.md` (local)

### Inferences (ranked table; costs are [estimate] for one V100 fp32, batch 16, ≤4,000 steps, one seed)

| Rank | Method | Verdict | Exact modules (timesfm3 paths) | Trainable params | lr / steps / regularization | Cost | Why (evidence + M8 tie-in) |
|---|---|---|---|---|---|---|---|
| 1 | Protocol: EMA + smooth_k + seeds + lr sweep on the current LoRA | **ADOPT** | unchanged (`*.seq_attn.*_proj`, `*.var_attn.*_proj`, `output_head`) | 4,014,656 | lr {3e-5, 1e-4}; `ema_decay` 0.998; `smooth_k` 3; eval every 250; 3 seeds | ≈18 min/run | M8: protocol was the only lever beyond noise. Amazon/grid-load use LoRA lr 1e-5. EMA is already implemented |
| 2 | WiSE-FT α-interpolation (LoRA delta + head) | **ADOPT** | scale every `LoRALinear.scaling` by α; blend `output_head` with pretrained | 0 (post-hoc) | α ∈ {0, .25, .5, .75, 1}, p* re-selected per α | 5 evals | Vision evidence only, but zero cost. Protects unseen stations (MixFT/PaCTS: weight-space FT < zero-shot on unseen) and keeps zero-shot day-shape skill (国能和煦) |
| 3 | Head-only control | **ADOPT (control)** | `output_head` | 737,856 | lr 1e-4–1e-3, 1–2k steps, EMA; cached embeddings | minutes after one cache pass | MOMENT LP ≈ LoRA. Tells whether the gain is recalibration |
| 4 | Head + resblock (TimesFM "LP") control | **ADOPT (control)** | `pre_transformer_resblock.{hidden,output,residual}_layer`, `output_head` | 2,867,776 | lr 1e-4 | ≈ LoRA | TimesFM-1 LP: 0.694 → 0.676. The resblock is the only reader of the NWP lookahead |
| 5 | LoRA var_attn + resblock + head (L-va) | **ADOPT (stage 3 candidate)** | `transformer_stack.layers.*.var_attn.{query,key,value,out}_proj` r8 + resblock r8 + `output_head` | 2,420,288 | same as rank 1 | ≈18 min | Code path: NWP→power enters only via resblock + var_attn. Freezes the history path (zero-shot strength) |
| 6 | Diagnostics: var-only vs seq-only LoRA; frozen-head LoRA | **ADOPT (one-off)** | `...var_attn...` or `...seq_attn...` only | 1.64M (+0.74M head) | same | ≈18 min each | No published ablation exists; this locates the gain |
| 7 | +FFN (all-linear) or v/o-only LoRA | TRY LATER | `ff0`, `ff1` / `{value,out}_proj` | +819,200 / 819,200 | same | ≈18 min | Google default is all-linear, but FETS insensitivity. qk-norm rationale for v/o has no evidence |
| 8 | Full FT with L2-SP + EMA + WiSE | TRY LATER (stage 5) | all | 330,710,976 | lr 1e-6 (and 1e-5); L2-SP λ ∈ {1e-3, 1e-2} toward θ₀; EMA 0.998; ≤4k steps; 3 seeds | ≈30 min/run, ≈8 GB | Grid load: full FT > LoRA with covariates at 1e-6. Unstable or worse elsewhere. M8: capacity didn't help |
| 9 | Partial FT: first+last layers (+resblock, head) | TRY LATER (with 8) | `transformer_stack.layers.{0,1,18,19}`, resblock, head | ≈68.4M | lr 1e-5, L2-SP | ≈ full FT step time | Few-layer dominance: TimesFM prune+FT 0.410 < full FT 0.415 |
| 10 | Layer-wise LR decay | TRY LATER (only inside 8) | all | — | decay 0.8/layer | — | Vision/NLP only. Dominant first layers argue against strong decay |
| 11 | Seed soup of merged LoRA deltas | TRY LATER | average B_iA_i and heads | 0 | greedy on val | evals only | Soups are vision-only evidence. Expect ≤ M8 seed-ensemble +0.3–0.5 |
| 12 | Clustered LoRA K=2 (hard routing) | TRY LATER (stage 5) | L-va per cluster | 2 × 2.42M | warm start from pooled LoRA | 2× runs | MixFT K=2 beats shared on unseen data; 14 stations/cluster is likely in the noise |
| 13 | Virtual-variate soft prompt (shared or context-generated) | TRY LATER (stage 5, for unseen stations) | new tensor concatenated on the variate axis after `pre_transformer_resblock`, sliced off before `output_head` | 2.6k–50k | lr 1e-3, ≤12 epochs (PaCTS) | ≈2× LoRA step | PaCTS: −1.1% to −1.5% MASE on TimesFM-2.5, unseen −0.45%. No RoPE issue on var axis |
| 14 | Function-space anchoring to zero-shot (LwF) | TRY LATER (with 8) | loss term only | 0 | weight 0.1–0.3 | +1 frozen forward/step | LLM/vision only. A better "replay" for our failure mode than generic series |
| 15 | Replay of synthetic/pretraining-like series | REJECT (for LoRA) | — | — | — | — | Corpus not public. Forgetting non-PV skill is irrelevant to us |
| 16 | LP-FT | REJECT | — | — | — | — | The mechanism needs a random head; ours is pretrained |
| 17 | Per-station LoRA / per-station head | REJECT | — | 28× | — | — | FETS 16/54; MixFT per-dataset rank 6.3 |
| 18 | Hypernetwork-generated LoRA | REJECT | — | — | — | — | LLM-only evidence; 28 stations cannot fit a metadata→ΔW map |
| 19 | MAML / Reptile | REJECT | — | — | — | K× cost | No TSFM evidence; all stations have 1–2 years of history |
| 20 | Continued/DAPT pretraining on public PV without NWP | REJECT (now) | — | 330.7M | — | hours to days | No TSFM evidence; trains no covariate path |
| 21 | QLoRA / 8-bit Adam / checkpointing | REJECT | — | — | — | — | No memory pressure (≈8 GB even for full FT); quantization error, no accuracy upside |
| 22 | DoRA / VeRA / IA3 / AdaLoRA / FourierFT / rank sweep | REJECT | — | — | — | — | Earlier report; FETS insensitivity; M8 noise band |
| — | BitFit | N/A | — | 576 | — | — | TimesFM-3 has no biases except `output_head`. Use RMSNorm-gain tuning (163,200 params) as the ultra-light control if wanted |

- **Adoption rule (from the earlier report, unchanged):** a retrained change needs 3-seed mean gain ≥ δ = max(0.6, 2σ_pair), ≥2/3 seeds positive, and no unseen-station loss > 0.3. Post-hoc changes (WiSE α, soups) need ≥0.3 with a day-block bootstrap CI excluding 0.
- **Expected outcome [inference]:** controls 3–6 will most likely show that most of the LoRA gain sits in head recalibration plus var_attn, and WiSE α ≈ 0.5–0.75 will be best on unseen stations. Ranks 8–13 are unlikely to clear δ given M8's noise band and FETS insensitivity, which is why they are deferred, not dropped.

### Gaps
- All effect sizes are transferred from Chronos-2, Chronos-T5, MOMENT, TimesFM 1.0/2.5, vision or LLMs. None is from TimesFM-3, PV with NWP covariates, or the 南网 metric.
- V100 step times, memory and fp16 safety are FLOP estimates, not measurements.
- Licensing (3.0 weights non-commercial, non-production) may make all M5 weight fine-tuning moot for production; this is outside the method question but is a gating item — [timesfm README](https://github.com/google-research/timesfm).
