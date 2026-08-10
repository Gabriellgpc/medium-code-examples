# SNet: architecture, losses, evaluation — research memo

Status: research complete, components **decided** (RF-DETR + WASB, §3.5), **no training
code written yet**. Every number below is sourced; where I could not verify something I
say so instead of guessing.

Scope: decide the backbone/head layout, the loss per head, and the evaluation
protocol for the model that will back `core/snet.py`.

---

## 1. What the field actually does (evidence)

### 1.1 The GSR leaderboard is flat at the top and crowded in the middle

| Edition | Baseline | Winner | Winner's stack |
|---|---|---|---|
| 2024 | 23.36 | 63.81 (Constructor.Tech) | YOLOv5m + DeepSORT-in-pitch-coords + SegFormer calib |
| 2025 | 29.01 | 63.90 (KIST-GSR) | YOLO-X + Deep-EIoU + OSNet ReID + LLaMA-3.2-Vision for identity |

14 teams, 66 submissions in 2025; 12 beat the baseline [1][2][3].

Two things matter for us:

- **The top moved 0.09 points in a year.** The headroom is not at the top of the
  leaderboard, it is in the cost of getting there. Nobody in either edition reports
  edge latency.
- **RF-DETR is already on this leaderboard.** Playbox & MIXI took 4th (61.64) with
  RF-DETR as the detector, explicitly "outperforming the YOLO series" [3]. Our
  existing detector choice is not a compromise; it is what a top-5 team picked.

### 1.2 GS-HOTA, exactly

From the benchmark paper [1]:

$$\mathrm{Sim}_{\text{GS-HOTA}}(P,G) = \mathrm{LocSim}(P,G) \times \mathrm{IdSim}(P,G)$$

$$\mathrm{LocSim}(P,G) = e^{\ln(0.05)\frac{\|P-G\|_2^2}{\tau^2}}, \qquad \tau = 5\ \text{m}$$

$$\mathrm{IdSim}(P,G) = \begin{cases} 1 & \text{if role, team and jersey all match} \\ 0 & \text{otherwise}\end{cases}$$

then the standard HOTA integral $\int \sqrt{\mathrm{DetA}_\alpha \cdot \mathrm{AssA}_\alpha}$ over
$\alpha \in [0.05, 0.95]$ in 0.05 steps.

Three consequences that shape the architecture:

1. **Distance is in metres on the pitch, not pixels.** A calibration error is a
   detection error. There is no way to be good at GSR with a weak homography.
2. **IdSim is a hard AND gate.** One wrong attribute turns the detection into a false
   positive outright. Jersey number is scored for players and goalkeepers; it is
   ignored for referees and when never visible in the 30 s clip.
3. **The ball is not scored.** The dataset annotates it, but the benchmark drops it:
   *"we remove the ball as it spends significant time in the air"*, because the
   foot-point-on-pitch assumption fails for an airborne object [1]. Referee
   sub-roles are ignored too.

Point 3 is the single most load-bearing fact in this memo. **Our ball work cannot be
justified by GS-HOTA and must carry its own metric.**

### 1.3 Where the baseline's points actually go

The benchmark's own oracle ablation — every module replaced by ground truth except
the one under test *and everything downstream of it* — on the GSR test set [1]:

| Module under test (+ downstream) | GS-HOTA |
|---|---|
| Team side heuristic | 92.00 |
| ReID (PRTreID) | 87.42 |
| Jersey number (MMOCR) | 56.75 |
| Camera calibration (TVCalib) | 51.39 |
| Pitch localization (TVCalib) | 49.99 |
| Detection (YOLOv8) | 35.28 |
| **Full baseline** | **22.26** |

And the attribute ablation, same table [1]: with pitch projection and all attributes
disabled the metric collapses to plain HOTA on image-space IoU and reads **57.64**;
turning on pitch projection alone drops it to **42.65**; the full metric is **22.26**.
(Test 22.26, valid 18.05, challenge 23.36.)

Read that as a budget: **projecting to the pitch costs ~15 points, and identifying who
each person is costs ~20 more.** Detection and calibration are the two modules whose
failure propagates furthest.

The baseline is also offline and slow: ~11 minutes per 30 s sequence on an A100, with
pitch localization at 2.9 FPS and calibration at 7.6 FPS [1].

### 1.4 Ball detection: WASB is the baseline to beat

WASB (BMVC 2023) [4] is the strongest published sports-ball tracker and its design is
directly reusable:

- **Backbone**: small HRNet, 4 stages of high-resolution modules. Critically, the
  stem's strides are *removed* so features stay at input resolution instead of the
  usual 1/4 — this is the whole trick for a ball that is a handful of pixels.
- **MIMO**: $N = 3$ consecutive frames concatenated on the channel axis ($H\times W\times 3N$
  in) producing $N$ heatmaps at the same $H \times W$. Input is **288 × 512**.
- **1.5 M parameters.**

Soccer results at $\tau = 4$ px [4, Table 2]:

| Method | Params | F1 | Acc | AP | FPS |
|---|---|---|---|---|---|
| DeepBall | 0.1 M | 44.5 | 92.7 | 26.3 | 44.6 |
| TrackNetV2 | 11.3 M | 86.6 | 97.7 | 77.2 | 60.0 |
| MonoTrack | 2.9 M | 84.6 | 97.4 | 76.6 | 56.0 |
| **WASB (step 3)** | **1.5 M** | **88.3** | **97.9** | **83.6** | **55.7** |

Note TrackNetV2 gets 86.6 F1 with 7.5× the parameters. High resolution beats capacity
here, which is exactly the right shape for an edge model.

### 1.5 Calibration: accurate exists, cheap does not

| Method | Accuracy | Cost |
|---|---|---|
| TVCalib (GSR baseline) | JaC@5 52.9 | 7.6 FPS (A100) [1] |
| NBJW [6] | JaC@5 73.7 | HRNetV2-w48 |
| PnLCalib [5] | JaC@5 80.6 (SN22-test-center) | 164 ms baseline → 439 ms at its best config, RTX 2080 Ti |
| BroadTrack [7] | JaC@5 75.25 reinit; +15% Jaccard tracked | **16 FPS on two RTX 4090s** |

JaC@5/JaC@10 = percentage of field elements reprojected within 5/10 px. Careful with
resolution conventions: sn-calibration reports at 960×540, BroadTrack's tracking table
at HD 1920×1080 [7].

PnLCalib's recipe: modified **HRNetV2-w48** encoder, two networks — one for keypoint
heatmaps (Gaussian peak per keypoint + a background channel), one for line-extremity
heatmaps (two peaks per line + a boundary channel) — **l2 heatmap loss**, then a
non-linear refinement minimising a blend of point and line reprojection error with
$\alpha = 0.6$ [5]. BroadTrack's ablation says the single biggest win was adding
**radial distortion $k_1$** to the camera model (JaC@5 42.09 → 54.1), ahead of both
optical flow and the tripod constraint [7].

**This is the gap worth attacking.** BroadTrack needs two 4090s to hit 16 FPS. I did
not find a published sports-field registration method reporting latency on an
integrated GPU.

### 1.6 The 2025 winners lean on models we cannot ship

Identity — the ~20-point half of the budget — is where the leaderboard now spends its
compute: LLaMA-3.2-Vision for open-set role/jersey/colour generation (1st place),
Qwen2-VL-Instruct with chain-of-thought and a zoom tool (7th), CLIP-ReID embeddings
(4th) [3]. None of that runs on an iGPU.

That is a scoping fact, not a defeat: it means an edge GSR number will be attribute-
limited by construction, and the honest move is to report it that way rather than
pretend to compete head-on.

---

## 2. Which SoccerNet tasks belong on one backbone

Group by **input granularity × output geometry**, not by topic. Two tasks share a
backbone when they consume the same tensor and want features at the same scale.

### Group A — full frame, dense spatial output → **one shared trunk** ✅

| Task | Output | Fit |
|---|---|---|
| Camera Calibration | keypoint heatmaps | native |
| Field Localization | line/segment heatmaps | native — same features as calibration; PnLCalib already runs both off one encoder style [5] |
| Tracking (detection stage) | boxes / centre heatmaps | native |
| Ball (not a listed task; inside GSR) | centre heatmap, needs 3 frames | native *if* resolution is preserved |
| Monocular Depth Estimation | dense per-pixel | plausible auxiliary head, different target domain |
| Camera Shot Segmentation | per-frame class | nearly free — global pool + linear |

**These are all heatmap or dense-map problems on the same 1920×1080 frame.** That is
the real reason they compose, and it is why the head zoo stays cheap.

### Group B — person crops → **a second, small trunk** ✅

| Task | Fit |
|---|---|
| Re-Identification | native |
| Jersey Number Recognition | native, but wants a higher-res torso crop |
| team affiliation, role (GSR sub-tasks) | native |

Already proven: **PRTreID** solves ReID + team affiliation + role classification on a
single shared backbone with multi-task supervision — deep metric learning (triplet)
for identity and team, focal loss for the 4-class role head, built on the part-based
BPBreID [1][8]. This group is a *different input* (crops, not frames), so it cannot
share Group A's trunk.

### Group C — long temporal context → **head on frozen Group-A features** ⚠️

Action Spotting, Ball Action Spotting, Replay Grounding, Dense Video Captioning. These
need seconds-to-minutes of context. Standard practice is a temporal neck over
per-frame features, so they can reuse Group A as a frozen extractor — but they are a
separate training stage, not a joint head.

### Group D — does not fit ❌

**Multi-View Foul Recognition**: multiple synchronised clips as input. Different input
structure entirely.

### Not a task, a composition

**Game State Reconstruction = Group A (geometry + detection) + Group B (identity) +
tracking + the homography.** It is the integration test, not a head.

### Verdict for SNet v1

Train **Group A minus depth and shot-segmentation**: detection + ball + pitch
keypoints/lines. That is exactly the three exports `core/gamestate.py` already
produces, which was the right call. Group B is a separate small net for a later phase.

---

## 3. The architecture decision, and the one thing that constrains it

### The binding constraint is input resolution

Everything else is negotiable; this is not.

| Consumer | What it needs |
|---|---|
| Ball | ~15 px object in 1080p. WASB deletes its stem strides to keep features at input scale [4] |
| Pitch keypoints | sub-5-px accuracy to score JaC@5 |
| Player detection | tolerant — players are the large objects here |

At WASB's 288×512 the 1920×1080 frame is scaled by 0.267, so a 15 px ball becomes
**~4 px** — and WASB still reaches 88.3 F1 there. So a shared input in the
**288×512 to 384×640** band is defensible for ball and pitch simultaneously.

**Unmeasured, and it should not be guessed:** the player bounding-box height
distribution in SN-GSR. It decides whether detection survives at that resolution or
needs its own scale. Measure it in the first Kaggle session, from the `detection.json`
the prepare step already writes.

### Recommended: one high-resolution trunk, three heatmap heads

The elegant version, and it is genuinely coherent rather than merely tidy: **ball,
pitch keypoints, and player centres are all Gaussian-heatmap regression.** An
HRNet-style trunk that preserves resolution serves all three with one loss family.

```
        3 frames (t-2, t-1, t) ─┐
                                ├─► HRNet-small trunk, stem strides removed
                                │   (WASB's Fig. 3c layout, ~1.5–4 M params)
                                └─► shared high-res features @ H×W
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
   ball head                    detection head                   pitch head
   N heatmaps                   centre heatmap + size            keypoint heatmaps
   (WASB)                       + offset (CenterNet)             + line heatmaps
```

**Honest cost of that elegance:** a centre-heatmap detector is weaker than RF-DETR in
crowded scenes, and crowds are the whole problem in a penalty box. We would be trading
away a detector we have already benchmarked and quantised.

### So: v1 is two networks, not one

| Net | Content | Why |
|---|---|---|
| **A — geometry + ball** | HRNet-small trunk, 3-frame input, heads: ball heatmap, pitch keypoints, pitch lines | New. Genuinely shares a backbone; all three heads are high-res heatmaps |
| **B — players** | existing RF-DETR, already INT8/OpenVINO-benchmarked on the iGPU | Do not throw away a working, measured component to satisfy a diagram |

This is compatible with the `SNet` contract already written: task flags select *which
compiled IR runs*, and a facade over two IRs satisfies that exactly. `predict()` does
not promise one graph.

Then **v2 attempts the merge, and the merge is itself the experiment**: does folding
detection into the shared trunk cost mAP, and how much iGPU latency does it buy? That
is a real result with a numbers-first answer, which is the kind of thing this site
publishes. Deciding it by assertion now would waste it.

---

## 3.5 Component choice: why RF-DETR and WASB survive the 2026 check

Both picks were made before this review. Re-examined against current work:

### Detector → **RF-DETR, confirmed, and for a better reason than before**

RF-DETR now has a paper (ICLR 2026) [11]. It is not "another YOLO alternative": it is a
**weight-sharing NAS that discovers an accuracy-latency Pareto curve for a target
dataset**, by fine-tuning a pre-trained base and then evaluating thousands of
configurations *without retraining each one*.

That is precisely our situation — we are fine-tuning on SoccerNet against a fixed iGPU
latency budget. The mechanism is aimed at the problem we actually have.

| Candidate | Verdict |
|---|---|
| **RF-DETR** | Nano: **48.0 AP** COCO, beating **D-FINE nano by 5.3 AP at similar latency** [11]. Apache-2.0 through Large. Already INT8/OpenVINO-measured on our iGPU. 4th place in GSR 2025 chose it over the YOLO series [3] |
| D-FINE | Strong (D-FINE-L 54.0 AP @ 8.07 ms) and Apache-2.0, but loses the nano tier badly and offers no target-dataset NAS |
| DEIM | ~54.7 AP @ ~124 FPS (L) on T4; same objection |
| RT-DETRv4 / Le-DETR | Newer, plausibly better, but each paper re-benchmarks its rivals on its own hardware — the latency columns are not comparable across papers, and I will not rank them from those tables |
| YOLO26 / Ultralytics | **AGPL.** Rules itself out for us |

Switching would mean discarding a component we have already quantised and measured, to
gain nothing the NAS does not already give. **Keep it.**

### Ball → **WASB stays the base, but it is no longer uncontested**

Three methods published since WASB beat it:

| Method | Idea | Result | Sport tested |
|---|---|---|---|
| **BlurBall** [12] | HRNet + SE attention, jointly predicts ball **and motion-blur** orientation/length; quality focal loss; PCA on the heatmap for blur params | F1 **96.52** vs WASB 95.58; 1.49 M params, 79 FPS; trajectory error 84.4±136.6 → **53.0±87.1 px** | table tennis only |
| **TOTNet** [13] | 3D convolutions, **visibility-weighted loss**, occlusion augmentation | RMSE 37.30 → **7.19**; fully-occluded accuracy 0.63 → **0.80** | tennis, badminton, table tennis |
| TrackNetV4 / V5 / V6 | motion-attention maps; residual spatio-temporal refinement; V6 targets "lightweight and robust" (ICMR 2026) | — | racquet sports |

**The hole in all of them: none evaluates on soccer.** WASB is the only method in this
group with a published soccer benchmark [4]. Soccer broadcast is the hardest case in
the set — smallest ball relative to frame, panning and zooming camera, long distances —
so a table-tennis win does not transfer by assumption.

WASB is also **MIT-licensed with released weights** [4], so it is reusable without the
licensing problem that rules out the Ultralytics path.

**Decision: WASB as the base recipe, plus one graft (the second was measured away).**

1. **TOTNet's visibility-weighted loss** — kept. Near-zero cost, because
   `build_ball_track` already exports a per-frame `visible` flag and the ball is absent
   in **5.89%** of train frames (§6.4).
2. ~~BlurBall's blur-centre relabeling~~ — **dropped.** M2 (§6.5) measured the
   convention rather than assuming it: SN-GSR already annotates the ball at the centre
   of its streak (median offset −0.08 on a −1…+1 scale), so there is nothing to fix.
   The blur in this footage is mild anyway — a 15 px ball moving 20 px per frame barely
   produces a streak, which is not the table-tennis setting BlurBall was built for.

And the honest framing, which is itself the interesting question: *the 2024–2026
improvements to sports-ball tracking are all validated on racquet sports; do they
transfer to broadcast soccer?* That is a real, publishable experiment with a numeric
answer, and we are set up to run it.

### One lead worth ten minutes before committing

**RF-DETR ships Apache-2.0 segmentation and keypoint variants** (RF-DETR-Seg Nano→2XL,
RF-DETR Keypoint in preview); only XL/2XL *detection* are PML 1.0 [11]. Pitch lines are
a segmentation problem and pitch landmarks are a keypoint problem, both in a family we
already deploy and quantise.

Unverified: the keypoint variant is aimed at human pose, and I have not confirmed it is
trainable on field landmarks or that it reaches JaC@5 territory. Worth a check before
building an HRNet pitch head from scratch — it could collapse Net A and Net B further
than planned, or it could be a dead end.

### Not changed

Pitch registration still has no cheap published option. PnLCalib and NBJW both use
**HRNetV2-w48**, and BroadTrack needs two RTX 4090s for 16 FPS [5][6][7]. Our plan of a
small shared HRNet trunk is unvalidated at JaC@5 — that is a real risk, not a formality,
and it is the first thing to measure.

---

## 4. Losses

### Ball head — WASB, adopted wholesale

Real-valued Gaussian target instead of a binary disk [4, Eq. 2]:

$$y^{real}_{\mathbf p} = \begin{cases}\min\!\left(C\exp\left(-\frac{\|\mathbf p - \mathbf p^{GT}\|^2}{d^2}\right), 1\right) & \text{if } \|\mathbf p - \mathbf p^{GT}\| \le d \\ 0 & \text{otherwise}\end{cases}$$

with a focal-style loss that accepts non-binary targets [4, Eq. 3]:

$$L = \sum_{\mathbf p} \left[-|y_{\mathbf p} - \sigma_{\mathbf p}|^{\beta}\left\{(1-y_{\mathbf p})\log(1-\sigma_{\mathbf p}) + y_{\mathbf p}\log \sigma_{\mathbf p}\right\}\right]$$

This reduces to ordinary focal loss when the target is binary. Published settings:
$d = 2.5$, $c_{min} = 0.7$, Adam, 30 epochs, batch 8, trained from scratch [4].

Plus **HLSM** (hard-to-localise sample mining): run inference at epoch 20, find frames
where the prediction is far from GT, and apply the real-valued target *only* to those.
The paper is explicit that applying it everywhere did not help [4].

### Pitch head

PnLCalib uses plain **l2 heatmap regression** for both its keypoint and line networks
[5]. Given the ball head is already a Gaussian-focal formulation, trying the same
focal form for keypoints is a cheap and worthwhile ablation — but l2 is the published
baseline and should be what we run first.

### Detection head

Keep RF-DETR's DETR-family objective (Hungarian matching + classification + L1 +
GIoU) in v1, since we are keeping the model. If v2 folds detection into the trunk,
it becomes CenterNet-style focal + L1 size/offset — same family as everything else.

### Balancing the heads

Do **not** hand-tune scalars. The gradient signal is wildly unbalanced: from our own
prepared data, the ball is **5.6% of all boxes** (40,931 of 731,555 in the train split)
against 82% for players.

Default to **Kendall uncertainty weighting** — one learned $\log \sigma_t$ per task,
loss $\sum_t \frac{1}{2\sigma_t^2}\mathcal L_t + \log \sigma_t$, where the log term stops the
weights collapsing to zero [9]. GradNorm [10] is the fallback if that under-serves the
ball head. Note the winning GSR team of 2024 instead ran **Optuna/TPE** over fixed
weights for its calibration loss [2]; that is a valid third option and cheaper to
reason about, at the cost of a hyperparameter search.

---

## 5. Evaluation — three protocols, never mixed

**This is the part most likely to go wrong**, because GS-HOTA looks like it covers
everything and does not.

### 5.1 Players / tracking → GS-HOTA
- SN-GSR-2025 test, $\tau = 5$ m, via the official evaluator (TrackLab / sn-gamestate).
- Also report **image-space HOTA / MOTA / IDF1** for continuity with the published
  article — and compute HOTA this time, rather than listing a tool we never ran.
- Report GS-HOTA **with and without the jersey attribute**. The winners buy jersey
  accuracy with VLMs we cannot ship; hiding that behind one aggregate number would
  misrepresent both their result and ours.

### 5.2 Ball → WASB protocol, never GS-HOTA
- F1 / Accuracy / AP at $\tau = 4$ px, plus a sweep over $\tau$ (WASB's Fig. 6 shape).
- Target to beat: **88.3 F1 / 83.6 AP** on their soccer split [4]. Our numbers will be
  on SN-GSR, which is a *different dataset* — so this is a reference point, not a
  head-to-head, and must be labelled as such.

### 5.3 Calibration → JaC
- JaC@5 / JaC@10 + completeness rate, plus mean and median reprojection error.
- State the frame size every time. sn-calibration convention is 960×540; BroadTrack's
  tracking table is 1920×1080 [7]. Comparing across the two without saying so would be
  a silent 2× error.

### 5.4 Edge → the repo's existing protocol
Per-head iGPU latency, FP32 vs INT8, warmup discarded, median of N runs, with the full
reproducibility block from the published article.

**Reference costs on much larger hardware** (not comparable, but they frame the claim):
PnLCalib 164–439 ms on an RTX 2080 Ti [5]; BroadTrack 16 FPS on two RTX 4090s [7]; the
GSR baseline ~11 min per 30 s clip on an A100 [1].

---

## 6. Data on hand

From `output/gsr_stats/` (SN-GSR-2025, 750 frames per sequence):

| Split | Sequences | Frames | Boxes | Ball visible |
|---|---|---|---|---|
| train | 57 | 42,750 | 731,555 | 94.67% |
| valid | 58 | 43,500 | 748,267 | 92.91% |
| test | 49 | 36,750 | 563,234 | 91.87% |

Class balance (train): player 82.7%, referee 8.4%, ball 5.6%, goalkeeper 3.2%.

Ball absent in 5–8% of frames — which is why `Prediction.ball` returning `None` had to
be distinguishable from "not requested", and why those frames stay in the ball export
rather than being dropped.

---

## 6.4 M1 — player box geometry (measured)

Kaggle kernel `condados/snet-gsr-measurements`, train split: 57 sequences, 42,750
frames, 731,555 boxes, all 1920×1080. Artifact:
`output/gsr_measurements/train_measurements.json`.

Native pixel sizes, before any resize:

| Class | n | p1 | p10 | **median** | p90 | p99 |
|---|---|---|---|---|---|---|
| ball (longest side) | 40,931 | 7 | 10 | **15** | 22 | 32 |
| player (height) | 604,971 | 44 | 65 | **100** | 142 | 184 |
| referee (height) | 61,546 | 30 | 41 | **88** | 134 | 175 |
| goalkeeper (height) | 24,107 | 35 | 56 | **79** | 116 | 143 |

Median height / % of boxes under 12 px, at each candidate input resolution:

| Resolution | scale | player | referee | goalkeeper | ball (longest) |
|---|---|---|---|---|---|
| 288×512 | ×0.267 | 26.7 px / **1.1%** | 23.5 px / **12.2%** | 21.1 px / 3.4% | 4.0 px |
| 384×640 | ×0.333 | 33.3 px / 0.4% | 29.3 px / 6.8% | 26.3 px / 1.1% | 5.0 px |
| 512×896 | ×0.467 | 46.7 px / 0.2% | 41.1 px / 0.3% | 36.9 px / 0.2% | 7.0 px |
| 640×1088 | ×0.567 | 56.7 px / 0.1% | 49.9 px / 0.2% | 44.8 px / 0.1% | 8.5 px |

**Verdict: the v2 single-backbone merge is viable.** At WASB's own 288×512 the
median player is still 27 px tall and only 1.1% of players fall under 12 px. The
fear that detection could not survive at ball-head resolution was wrong.

**The weak class is the referee, not the player.** Referees have a much longer
tail (p10 = 41 px native against the player's 65) because assistants stand at the
far touchline, so 12.2% of them drop under 12 px at 288×512. That halves to 6.8%
at 384×640, which is why **384×640 is the recommended shared input** rather than
288×512: it costs ~1.4× the pixels and buys back the referee class.

Also confirmed against the earlier estimate: ball median 15 px, visible in
**94.11%** of frames.

## 6.5 M2 — ball annotation convention (measured)

240 fast frames sampled (displacement ≥ 12 px between consecutive annotated
frames), 219 measured, 12 with no blob found. Offset is signed along the motion
direction and normalised by the streak's half-length: **0 = centre of the streak,
+1 = leading tip, −1 = trailing tip.**

| Statistic | Value |
|---|---|
| offset along motion, **median** | **−0.08** |
| offset, p25 / p75 | −0.63 / +0.04 |
| streak vs motion angle, median / p90 | 10.5° / 63.2° |
| ball speed, median / p90 | 20.5 / 38.2 px per frame |
| correlation(speed, annotated box size) | 0.46 |

**Verdict: SN-GSR already labels the ball at the centre. BlurBall's relabeling
graft is dropped** — there is nothing for it to fix. The 0.46 correlation between
speed and annotated box size says the same thing a second way: the box grows with
the streak rather than tracking one end of it.

Two honesty notes:

- The p90 streak-vs-motion angle of 63° means my bright-blob detector locks onto
  something that is *not* a motion streak in a sizeable minority of samples — a
  shirt, a line marking, a boot. Those failures are visible in the contact sheet
  (`m2_ball_crops.png`, red = annotation, green = detected centroid) and they are
  what produces the −1.29 p10 tail. **The median is the trustworthy statistic
  here; the mean (−0.355) is dragged by detector failures and should not be
  quoted.**
- The deeper reason the distinction does not matter: at a median 20.5 px per frame
  with a 15 px ball, the streak is barely longer than the ball itself. This
  footage is not motion-blurred the way table-tennis footage is, which is the
  setting BlurBall was built for.

## 6.6 M3 — the keypoint error budget (measured, and the synthetic run it confirms)

Two runs, and they agree. `scripts/keypoint_budget_synthetic.py` answered this
from geometry alone before the dataset was available (15 look-at cameras, no
radial distortion, artifact `output/keypoint_budget_synthetic.json`). The Kaggle
run then repeated it on **300 real frames**, deriving a ground-truth homography
per frame from athlete foot points against their annotated `bbox_pitch` in metres,
inverting it to place the 33 pitch landmarks in the image, and perturbing those.

The GT homographies are sound: **median fit residual 0.09 m, p90 0.17 m.**

Cell = median player-position error / % of players beyond GS-HOTA's 5 m tolerance,
**measured on real frames**:

| σ (px) | k=4 | k=6 | k=8 | k=12 |
|---|---|---|---|---|
| 0.0 | 0.00 m — **33.6%** | 0.00 m — 3.5% | 0.00 m — 0.0% | 0.00 m — 0.0% |
| 0.5 | 1.04 m — 37.2% | 0.08 m — 1.7% | 0.07 m — 0.5% | 0.05 m — 0.0% |
| 1.0 | 1.63 m — 39.5% | 0.15 m — 3.4% | 0.12 m — 0.6% | 0.14 m — 0.0% |
| 2.0 | 3.77 m — 46.2% | 0.31 m — 5.0% | 0.26 m — 0.4% | 0.22 m — 0.0% |
| 3.0 | 5.08 m — 50.4% | 0.45 m — 5.6% | **0.39 m — 2.1%** | 0.35 m — 0.1% |
| 5.0 | 6.57 m — 53.6% | 0.75 m — 8.6% | 0.62 m — 3.7% | 0.56 m — 0.3% |
| 8.0 | 9.02 m — 60.5% | 1.26 m — 12.6% | 1.01 m — 7.2% | 1.00 m — 3.1% |

**Visible landmarks per frame: median 9 of 33 (p10 = 5, p90 = 12)** — the
quantitative form of the complaint that "only a small subset of field markings is
visible" [2]. The synthetic sweep guessed a median of 10 with a range to 17, so it
was optimistic at the top end but right in the middle.

Synthetic vs measured, on the two cells that matter: k=8 at σ=3 was predicted
0.40 m / 1.6% and measured **0.39 m / 2.1%** — close enough to trust the method.
k=4 at σ=0 was predicted 22.1% and measured **33.6%**, so reality is *worse* than
the model, and the finding below is stronger than it first looked.

### Finding 1: four keypoints are unsafe at *any* accuracy

The k=4 column loses **33.6%** of players past 5 m with **perfect, zero-noise
keypoints**. That is not numerical noise; it is the landmark set's own geometry.
Verified directly against `LANDMARKS`:

- **Eight landmarks lie on each goal line** (`corner_tl`, `corner_bl`,
  `l_pen_top_goalline`, `l_pen_bottom_goalline`, `l_goal_top_goalline`,
  `l_goal_bottom_goalline`, `l_post_top`, `l_post_bottom`), and eight more on the
  other; five on the halfway line.
- **11.0% of all 4-subsets of the 33 landmarks contain a collinear triple**
  (4,518 of 40,920) — and far more than that once visibility restricts the set to
  one penalty area, which is the common broadcast crop.

A collinear triple makes the homography undetermined, and a 4-point DLT passes
exactly through its own points regardless, so the failure is invisible to any
residual check. **The pitch head must emit many keypoints and the solver must
never fall back to a minimal set.**

### Finding 2: JaC@5 is a stricter bar than GS-HOTA actually needs

With k ≥ 8, **σ = 3 px costs 0.39 m median and puts only 2.1% of players past 5 m**.
Even σ = 8 px holds 1.01 m median. The pitch head does not need sub-pixel accuracy
to serve GS-HOTA.

That is because JaC@5 is a 5-**pixel** reprojection criterion while GS-HOTA is a
5-**metre** one, and they are not the same bar. It reframes §3.5's stated risk: the
worry that a small HRNet trunk cannot reach HRNetV2-w48's JaC@5 may be measuring
the wrong thing. **Optimise the pitch head for metres, report JaC for
comparability.**

Caveat that keeps this honest: the table above is per-frame and independent. It
says nothing about temporal jitter, and a homography that wobbles frame to frame
will wreck association even while every single frame passes. That is a separate
measurement, and it is why every 2025 entrant added optical-flow smoothing [3].

---

## 7. Open questions before the first training run

M1–M3 are **done** (`kaggle/measure_gsr.py`, run as Kaggle kernel
`condados/snet-gsr-measurements`; artifacts under `output/gsr_measurements/`).
What they settled: shared input **384×640**, detection *can* share the trunk,
pitch head needs **≥8 keypoints at ~2–3 px**, BlurBall's graft dropped.

Still open:

1. **Temporal stability of the homography** — not covered by M3, which is per-frame
   and independent. A homography that wobbles between frames wrecks association even
   when every single frame passes, and it is why every 2025 entrant added optical-flow
   smoothing [3]. Needs its own metric before any of §6.6 can be called sufficient.
2. **Which HRNet width.** WASB used the small design at 1.5 M params for the ball
   alone; three heads may want more.
3. **Do pitch-line and keypoint heads share a decoder** or need separate ones. PnLCalib
   uses two *separate networks* [5]; folding them is untested by us.
4. **RF-DETR's Apache-2.0 Seg/Keypoint variants** for the pitch head (§3.5) — still
   unverified for field landmarks rather than human pose.
5. **Weights remain private** until written KAUST permission — unchanged, and no
   training artefact should be published before that.

---

## References

1. V. Somers, V. Joos, A. Cioppa, S. Giancola, S. A. Ghasemzadeh, F. Magera, B. Standaert, A. M. Mansourian, X. Zhou, S. Kasaei, B. Ghanem, A. Alahi, M. Van Droogenbroeck, C. De Vleeschouwer. *SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap*. CVPRW (CVsports), 2024. <https://arxiv.org/abs/2404.11335>
2. A. Golubev et al. *From Broadcast to Minimap: Achieving State-of-the-Art SoccerNet Game State Reconstruction*. 2025. <https://arxiv.org/abs/2504.06357>
3. SoccerNet Team. *SoccerNet 2025 Challenges Results*. 2025. <https://arxiv.org/abs/2508.19182>
4. S. Tarashima, M. A. Haq, Y. Wang, N. Tagawa. *Widely Applicable Strong Baseline for Sports Ball Detection and Tracking*. BMVC, 2023. <https://arxiv.org/abs/2311.05237> · code <https://github.com/nttcom/WASB-SBDT>
5. M. Gutiérrez-Pérez, A. Agudo. *PnLCalib: Sports Field Registration via Points and Lines Optimization*. CVIU, 2026. <https://arxiv.org/abs/2404.08401> · code <https://github.com/mguti97/PnLCalib>
6. M. Gutiérrez-Pérez, A. Agudo. *No Bells, Just Whistles: Sports Field Registration by Leveraging Geometric Properties*. CVPRW (CVsports), 2024. <https://github.com/mguti97/No-Bells-Just-Whistles>
7. F. Magera, T. Hoyoux, O. Barnich, M. Van Droogenbroeck. *BroadTrack: Broadcast Camera Tracking for Soccer*. WACV, 2025. <https://arxiv.org/abs/2412.01721>
8. A. M. Mansourian, V. Somers, C. De Vleeschouwer, S. Kasaei. *Multi-task Learning for Joint Re-identification, Team Affiliation, and Role Classification for Sports Visual Tracking* (PRTreID). MMSports @ ACM MM, 2023. <https://arxiv.org/abs/2401.09942>
9. A. Kendall, Y. Gal, R. Cipolla. *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*. CVPR, 2018. <https://arxiv.org/abs/1705.07115>
10. Z. Chen, V. Badrinarayanan, C.-Y. Lee, A. Rabinovich. *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*. ICML, 2018. <https://arxiv.org/abs/1711.02257>

11. I. Robinson, P. Robicheaux, M. Popov, D. Ramanan, N. Peri. *RF-DETR: Neural Architecture Search for Real-Time Detection Transformers*. ICLR, 2026. <https://arxiv.org/abs/2511.09554> · code <https://github.com/roboflow/rf-detr>
12. *BlurBall: Joint Ball and Motion Blur Estimation for Table Tennis Ball Tracking*. 2025. <https://arxiv.org/abs/2509.18387>
13. *TOTNet: Occlusion-Aware Temporal Tracking for Robust Ball Detection in Sports Videos*. 2025. <https://arxiv.org/abs/2508.09650> · code <https://github.com/AugustRushG/TOTNet>

### Not verified

- SoccerMaster (CVPR 2026 oral, <https://arxiv.org/abs/2512.11016>) claims a unified
  soccer vision foundation model beating task-specific experts. Both the CVF and arXiv
  full texts refused to fetch, so I have **only the abstract**. Directly relevant to
  the one-backbone question — worth reading before v2.
- Broadcast2Pitch (WACV 2026) — CVF returned 403. Unread.
- GS-HOTA attribute-ablation table: I read the endpoints reliably (all-off 57.64,
  pitch-only 42.65, full 22.26) but could not resolve the middle rows' checkmark
  columns with confidence, so they are not quoted here.
- **"Tracking the Blur: Accurate Ball Trajectory Detection in Broadcast Sports Videos"**
  (Chao, Nguyen, Jamsrandorj, Oo, Mun, Park, Park, Kim; MMSports @ ACM MM, 2024,
  <https://doi.org/10.1145/3689061.3689075>). ACM paywall, **unread** — but it is the
  one recent ball paper aimed squarely at *broadcast soccer*, and Chao, Oo and Kim are
  the same authors who won GSR 2025 as KIST-GSR [3]. **Highest-value thing to obtain
  before writing the ball head.**
- "Real-time Localization of a Soccer Ball from a Single Camera"
  (<https://arxiv.org/abs/2506.07981>) claims centimetre accuracy on CPU, but on a
  proprietary 6K Russian Premier League dataset with no code — not reproducible by us.
- RF-DETR Keypoint's suitability for *field* landmarks (rather than human pose):
  unverified, see §3.5.
