"""Losses for the three heads, plus the weighting that stops one from eating the rest.

The heads are wildly unbalanced in how much signal they carry: a frame has ~17
boxes, 9 visible landmarks and at most one ball, and the ball is absent entirely
in 5.89% of frames. Hand-tuned scalars would be guesswork, so the multi-task
weighting is learned (Kendall et al.), and the ball's absence is handled
explicitly rather than by hoping the focal term absorbs it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Positive pixels in one ball Gaussian at WASB's d = 2.5 (measured: 21 on the
# integer grid). Used as the normalisation floor so a frame with no ball is
# scaled like a frame with one.
BALL_POS_FLOOR = 21.0


def wasb_focal_loss(
    logits: torch.Tensor, target: torch.Tensor, beta: float = 2.0, eps: float = 1e-4,
    pos_floor: float = BALL_POS_FLOOR,
) -> torch.Tensor:
    """WASB Eq. 3: focal loss generalised to real-valued targets.

    ``L = sum_p -|y_p - s_p|^beta * [ (1 - y_p) log(1 - s_p) + y_p log s_p ]``

    where ``s`` is the sigmoid of the logit. When the target is binary this
    reduces exactly to ordinary focal loss; the difference here is that ``y`` is
    the graded Gaussian from ``targets.ball_gaussian``, so the modulating factor
    ``|y - s|^beta`` measures distance from a *soft* target and keeps pushing on
    pixels that are nearly right but not quite — which is the whole point of the
    real-valued map.

    **Normalised by the number of positive pixels, not by pixel count.** A ball
    Gaussian covers about 21 pixels of a 384x640 map, so a pixel-mean weights the
    entire positive signal at 1e-4 and the loss reads 0.0004 against the detection
    head's 8.0 — measured, not hypothetical. Uncertainty weighting cannot rescue
    that: log-variance moves far too slowly to close four orders of magnitude.
    Dividing by the positive count is what CenterNet does for the same reason.

    The floor matters as much as the division. When the ball is out of frame the
    target is all zeros and the positive count is zero, so a ``clamp(min=1)``
    would divide a quarter-million negative terms by one and blow the loss up on
    exactly the 5.89% of frames we most want the head to handle calmly. Clamping
    at the *typical* positive count instead keeps present and absent frames on one
    scale.
    """
    s = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    modulating = (target - s).abs().pow(beta)
    log_terms = (1.0 - target) * torch.log(1.0 - s) + target * torch.log(s)
    n_pos = target.gt(0).float().sum().clamp(min=pos_floor)
    return -(modulating * log_terms).sum() / n_pos


def centernet_focal_loss(
    logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """CornerNet/CenterNet penalty-reduced focal loss.

    Normalised by the number of positive peaks, not by pixel count: with one
    object in tens of thousands of pixels, a pixel-mean would make the positive
    term vanish.
    """
    s = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    pos = target.ge(1.0 - 1e-6).float()
    neg = 1.0 - pos

    pos_loss = -((1 - s).pow(alpha) * torch.log(s)) * pos
    neg_loss = -((1 - target).pow(beta) * s.pow(alpha) * torch.log(1 - s)) * neg
    n_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 at the centre pixels only. ``mask`` is (B, H, W), pred/target (B, C, H, W)."""
    m = mask.unsqueeze(1).expand_as(pred)
    n = m.sum().clamp(min=1.0)
    return (torch.abs(pred - target) * m).sum() / n


def ball_loss(
    logits: torch.Tensor, target: torch.Tensor, visible: torch.Tensor,
    absent_weight: float = 1.0, beta: float = 2.0,
) -> torch.Tensor:
    """WASB focal, with our visibility weighting.

    The ball genuinely leaves frame in 5.89% of SN-GSR frames (measured), and
    those frames are *supervision*, not padding: an all-zero target teaches the
    head to report absence, which is a thing the tracker needs it to do. So they
    are kept, at a weight that can be tuned down if the head starts suppressing
    real balls.

    This formulation is ours. TOTNet reports a "visibility-weighted loss" for the
    same problem, but that paper was not available to read, so nothing here is
    inherited from it beyond the idea.
    """
    per_sample = torch.stack([
        wasb_focal_loss(logits[i : i + 1], target[i : i + 1], beta=beta)
        for i in range(logits.shape[0])
    ])
    weights = torch.where(visible > 0, torch.ones_like(per_sample),
                          torch.full_like(per_sample, absent_weight))
    return (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)


def keypoint_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None,
    use_focal: bool = True,
) -> torch.Tensor:
    """Gaussian-heatmap loss for the pitch landmarks.

    **Focal by default, not PnLCalib's l2 — because l2 here is degenerate.** The
    target is 33 channels of 96x160 that is about 99.95% zeros, so predicting
    nothing scores a mean-squared error of 6e-05. Measured, not hypothetical: the
    12-epoch run reported exactly that, and Kendall weighting responded by driving
    this task's weight to 55,652 to bring its contribution into line. Learned loss
    weighting does not protect against a badly-normalised loss; it chases one, and
    what it amplified was mostly noise.

    Normalising by the positive count is the same fix the ball head needed, and the
    focal form is the one already used for the ball and the detection centres, so
    all three heads now share a scale. ``use_focal=False`` keeps the published l2
    for comparison.

    ``valid`` (B,) marks which *frames* carry pitch supervision at all, and it is
    not the same thing as a landmark being out of frame. An out-of-frame landmark
    with an all-zero target is correct supervision: the head should say "not
    here". A frame where the homography could not be fitted has no ground truth,
    and supervising it as all-zero would actively teach the head that a pitch
    full of landmarks contains none. Those frames are dropped instead.
    """
    if valid is not None:
        keep = valid.view(-1) > 0
        if not bool(keep.any()):
            return logits.sum() * 0.0
        logits, target = logits[keep], target[keep]

    if not use_focal:
        return F.mse_loss(torch.sigmoid(logits), target)
    # ~9 visible landmarks at sigma=2 is a few hundred positive pixels; the
    # ball-sized floor would over-weight a frame where none are supervised.
    return wasb_focal_loss(logits, target, beta=2.0, pos_floor=200.0)


class UncertaintyWeighting(nn.Module):
    """Kendall et al.: learn one log-variance per task instead of tuning scalars.

    ``L = sum_t exp(-s_t) * L_t + s_t`` with ``s_t = log(sigma_t^2)`` learned. The
    trailing ``+ s_t`` is what stops the trivial solution of driving every weight
    to zero, and it is the reason this is a principled scheme rather than just
    "more parameters".
    """

    def __init__(self, tasks: tuple[str, ...]):
        super().__init__()
        self.tasks = tasks
        self.log_var = nn.Parameter(torch.zeros(len(tasks)))

    def forward(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        total = 0.0
        weights = {}
        for i, name in enumerate(self.tasks):
            if name not in losses:
                continue
            s = self.log_var[i]
            total = total + torch.exp(-s) * losses[name] + s
            weights[name] = float(torch.exp(-s).detach())
        return total, weights


def compute_losses(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], heads: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Per-head losses, ready for the weighting module."""
    losses: dict[str, torch.Tensor] = {}
    if "ball" in heads and "ball_heat" in batch:
        losses["ball"] = ball_loss(outputs["ball"], batch["ball_heat"], batch["ball_visible"])
    if "detection" in heads and "det_heat" in batch:
        losses["detection"] = (
            centernet_focal_loss(outputs["detection"], batch["det_heat"])
            + masked_l1(outputs["det_size"], batch["det_size"], batch["det_mask"])
            + 1.0 * masked_l1(outputs["det_offset"], batch["det_offset"], batch["det_mask"])
        )
    if "pitch" in heads and "kp_heat" in batch:
        losses["pitch"] = keypoint_loss(
            outputs["pitch"], batch["kp_heat"], batch.get("kp_valid")
        )
    return losses
