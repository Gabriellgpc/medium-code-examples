"""The SNet trunk and heads, in PyTorch.

Defaults are the Step 0 winner: ``stem_stride=4`` with ``head_upsample=4`` at
384x640, measured at **30.6 ms / 32.7 FPS** on the target Iris Xe against
**214.3 ms / 4.7 FPS** for the WASB-faithful full-resolution trunk at the same
input and the *same output resolution* — 7x faster for an identical heatmap grid.
Whether it is as accurate is untested and is what Step 1 exists to find out.

One high-resolution HRNet-style trunk feeding three heads, because ball centres,
player centres and pitch landmarks are all Gaussian-heatmap regression problems on
the same frame — that is what makes the backbone sharing real rather than tidy.

The design knob that matters is ``stem_stride``. HRNet's original stem reduces the
input to 1/4 before the high-resolution modules ever run; WASB removes those
strides so a ball a handful of pixels wide survives (their Fig. 3, options (a) 1/4,
(b) 1/2, (c) 1/1). Full resolution is what makes WASB accurate and also what makes
it expensive, so this is left as a parameter rather than a decision: Step 0 of the
training plan measures the accuracy/latency trade-off on the actual target device
instead of guessing it.

Nothing here is trained yet. The point of this module existing before any training
script is that an architecture which cannot meet the latency budget should be found
out in an afternoon on the iGPU, not after thirty GPU-hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

BN_MOMENTUM = 0.1


def conv3x3(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin: int, cout: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(cin, cout, stride)
        self.bn1 = nn.BatchNorm2d(cout, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(cout, cout)
        self.bn2 = nn.BatchNorm2d(cout, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin: int, cout: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(cout, cout, stride)
        self.bn2 = nn.BatchNorm2d(cout, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(cout, cout * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class FusionModule(nn.Module):
    """One HRNet stage: parallel branches, then every-to-every resolution exchange.

    The exchange is the whole point of HRNet — each branch keeps its own scale but
    is repeatedly informed by the others, so the highest-resolution branch stays
    semantically rich instead of being a shallow shortcut.
    """

    def __init__(self, channels: list[int], blocks: int):
        super().__init__()
        self.channels = channels
        self.branches = nn.ModuleList(
            nn.Sequential(*[BasicBlock(c, c) for _ in range(blocks)]) for c in channels
        )
        n = len(channels)
        self.fuse = nn.ModuleList()
        for i in range(n):
            row = nn.ModuleList()
            for j in range(n):
                if j > i:
                    # Coarser -> finer: cheap 1x1 then upsample at forward time.
                    row.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 1, bias=False),
                        nn.BatchNorm2d(channels[i], momentum=BN_MOMENTUM),
                    ))
                elif j == i:
                    row.append(nn.Identity())
                else:
                    # Finer -> coarser: strided 3x3s, one per octave of difference.
                    ops: list[nn.Module] = []
                    cur = channels[j]
                    for k in range(i - j):
                        last = k == i - j - 1
                        out_c = channels[i] if last else cur
                        ops += [
                            conv3x3(cur, out_c, stride=2),
                            nn.BatchNorm2d(out_c, momentum=BN_MOMENTUM),
                        ]
                        if not last:
                            ops.append(nn.ReLU(inplace=True))
                        cur = out_c
                    row.append(nn.Sequential(*ops))
            self.fuse.append(row)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        xs = [branch(x) for branch, x in zip(self.branches, xs, strict=True)]
        out = []
        for i in range(len(xs)):
            acc = None
            for j, x in enumerate(xs):
                y = self.fuse[i][j](x)
                if j > i:
                    y = F.interpolate(y, size=xs[i].shape[-2:], mode="nearest")
                acc = y if acc is None else acc + y
            out.append(self.relu(acc))
        return out


@dataclass
class SNetConfig:
    """Everything Step 0 sweeps over."""

    width: int = 18                    # channels of the highest-resolution branch
    stem_stride: int = 4               # 1 = WASB Fig 3(c), 2 = (b), 4 = original HRNet
    stage_blocks: tuple[int, ...] = (2, 2, 2)   # blocks per branch, stages 2..4
    stage_modules: tuple[int, ...] = (1, 1, 1)  # fusion modules per stage
    head_channels: int = 32
    n_classes: int = 3                 # player, referee, goalkeeper (ball has its own head)
    n_keypoints: int = 33              # LANDMARKS; see TRAINING-DESIGN section 8.1 on expanding it
    n_lines: int = 26                  # distinct pitch line types in SN-GSR
    in_frames: int = 3                 # temporal window stacked on channels
    head_upsample: int = 4             # ball head upsample (see Head docstring)
    # Pitch keypoints need no decoder. Measured: soft-argmax on a sigma=2 Gaussian
    # at trunk stride decodes to a median 0.66 native px (p90 1.22), comfortably
    # inside the 2-3 px budget of section 6.6 — so the expensive full-resolution
    # path buys nothing here, and the target drops from 32.4 MB per sample to 2.0.
    pitch_upsample: int = 1
    heads: tuple[str, ...] = field(default=("ball", "detection", "pitch"))


class SNetBackbone(nn.Module):
    """Multi-resolution trunk. Returns the fused highest-resolution feature map."""

    def __init__(self, cfg: SNetConfig):
        super().__init__()
        self.cfg = cfg
        cin = 3 * cfg.in_frames

        # Original HRNet halves twice here. We split the reduction across the two
        # convs so stem_stride=2 puts one stride on each and stays symmetric.
        s1 = 2 if cfg.stem_stride >= 2 else 1
        s2 = 2 if cfg.stem_stride >= 4 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(cin, 64, 3, stride=s1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=s2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_bottleneck_layer(64, 32, 2)   # -> 128 channels
        c1 = 32 * Bottleneck.expansion

        w = cfg.width
        widths = [w, w * 2, w * 4, w * 8]
        self.transitions = nn.ModuleList()
        self.stages = nn.ModuleList()
        prev = [c1]
        for stage in range(3):                                  # stages 2, 3, 4
            target = widths[: stage + 2]
            self.transitions.append(self._make_transition(prev, target))
            self.stages.append(nn.Sequential(*[
                FusionModule(target, cfg.stage_blocks[stage])
                for _ in range(cfg.stage_modules[stage])
            ]))
            prev = target
        self.out_channels = sum(widths)

    @staticmethod
    def _make_bottleneck_layer(cin: int, cout: int, blocks: int) -> nn.Sequential:
        downsample = nn.Sequential(
            nn.Conv2d(cin, cout * Bottleneck.expansion, 1, bias=False),
            nn.BatchNorm2d(cout * Bottleneck.expansion, momentum=BN_MOMENTUM),
        )
        layers = [Bottleneck(cin, cout, downsample=downsample)]
        layers += [Bottleneck(cout * Bottleneck.expansion, cout) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    @staticmethod
    def _make_transition(prev: list[int], target: list[int]) -> nn.ModuleList:
        ops = nn.ModuleList()
        for i, c in enumerate(target):
            if i < len(prev):
                if prev[i] != c:
                    ops.append(nn.Sequential(
                        conv3x3(prev[i], c),
                        nn.BatchNorm2d(c, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    ops.append(nn.Identity())
            else:
                # New, coarser branch: strided from the deepest existing one.
                ops.append(nn.Sequential(
                    conv3x3(prev[-1], c, stride=2),
                    nn.BatchNorm2d(c, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                ))
        return ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(self.stem(x))
        xs = [x]
        for transition, stage in zip(self.transitions, self.stages, strict=True):
            nxt = []
            for i, op in enumerate(transition):
                nxt.append(op(xs[i] if i < len(xs) else xs[-1]))
            xs = stage(nxt)
        size = xs[0].shape[-2:]
        return torch.cat(
            [xs[0]] + [F.interpolate(x, size=size, mode="bilinear", align_corners=False)
                       for x in xs[1:]],
            dim=1,
        )


class Head(nn.Module):
    """3x3 conv, optional upsampling refinement, then a 1x1 to the output channels.

    ``upsample`` exists because Step 0 measured a hard conflict: a trunk that keeps
    full resolution the way WASB does costs 216 ms on the target iGPU (4.6 FPS),
    while a trunk at stride 2 or 4 is fast but emits a heatmap too coarse for the
    keypoint budget in TRAINING-DESIGN section 6.6. Upsampling inside the head is
    the cheap way out: the expensive multi-resolution exchange stays at low
    resolution and only a thin refinement runs at the output scale.

    This is precisely what WASB argued against, and their argument was about
    feature quality, not latency — they had no edge constraint. So the accuracy
    cost of the decoder is a real open question, and a thing to measure rather
    than assume away.
    """

    def __init__(
        self, cin: int, mid: int, cout: int, bias_init: float | None = None, upsample: int = 1
    ):
        super().__init__()
        self.upsample = upsample
        self.project = nn.Sequential(
            nn.Conv2d(cin, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.refine = (
            nn.Sequential(
                nn.Conv2d(mid, mid, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            if upsample > 1
            else nn.Identity()
        )
        self.out = nn.Conv2d(mid, cout, 1)
        if bias_init is not None:
            # Heatmap heads start pessimistic: with one ball in ~250k pixels, a
            # neutral init spends the first epochs unlearning "everything is ball".
            nn.init.constant_(self.out.bias, bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        if self.upsample > 1:
            x = F.interpolate(
                x, scale_factor=self.upsample, mode="bilinear", align_corners=False
            )
            x = self.refine(x)
        return self.out(x)


class SNetModel(nn.Module):
    """Trunk plus the selected heads. Outputs are logits at the trunk's stride."""

    def __init__(self, cfg: SNetConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or SNetConfig()
        self.backbone = SNetBackbone(cfg)
        c, mid, up = self.backbone.out_channels, cfg.head_channels, cfg.head_upsample

        self.heads = nn.ModuleDict()
        if "ball" in cfg.heads:
            self.heads["ball"] = Head(c, mid, 1, bias_init=-4.6, upsample=up)
        if "detection" in cfg.heads:
            # Boxes are large objects; they do not need the ball head's resolution,
            # so detection stays at the trunk stride and costs nothing extra.
            self.heads["detection"] = Head(c, mid, cfg.n_classes, bias_init=-4.6)
            self.heads["det_size"] = Head(c, mid, 2)
            self.heads["det_offset"] = Head(c, mid, 2)
        if "pitch" in cfg.heads:
            pu = cfg.pitch_upsample
            self.heads["pitch"] = Head(c, mid, cfg.n_keypoints, upsample=pu)
            self.heads["pitch_lines"] = Head(c, mid, cfg.n_lines, upsample=pu)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        return {name: head(feats) for name, head in self.heads.items()}

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
