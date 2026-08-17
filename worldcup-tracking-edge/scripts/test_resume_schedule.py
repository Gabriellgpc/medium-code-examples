"""Check that `--resume` reproduces an uninterrupted learning-rate schedule.

`train_snet.py --resume` exists so a five-hour run can survive the OOM killer, but
a resume that gets the schedule wrong is worse than a crash: it produces a
plausible checkpoint trained on the wrong learning rates, which is exactly the
failure mode section 9.10 already paid for once.

This is the check behind the numbers quoted in `save_state`'s docstring. It uses a
toy parameter and a 5-epoch schedule rather than the real trainer, because the
property under test belongs to OneCycleLR's state, not to SNet.

    uv run python scripts/test_resume_schedule.py
"""

from __future__ import annotations

import warnings

import torch

STEPS_PER_EPOCH = 100
EPOCHS = 5
RESUME_AT = 2  # epochs completed before the simulated kill


def build() -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.OneCycleLR]:
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, total_steps=EPOCHS * STEPS_PER_EPOCH)
    return opt, sched


def trace(sched, n: int) -> list[float]:
    out = []
    for _ in range(n):
        out.append(sched.get_last_lr()[0])
        sched.step()
    return out


def main() -> None:
    # The trainer steps the optimizer first; stepping only the scheduler here is
    # what the toy harness does, and the warning is noise rather than a finding.
    warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")

    _, ref_sched = build()
    reference = trace(ref_sched, EPOCHS * STEPS_PER_EPOCH)

    _, first = build()
    resumed = trace(first, RESUME_AT * STEPS_PER_EPOCH)
    state = first.state_dict()

    _, second = build()
    second.load_state_dict(state)
    resumed += trace(second, (EPOCHS - RESUME_AT) * STEPS_PER_EPOCH)

    cut = RESUME_AT * STEPS_PER_EPOCH
    worst = max(abs(a - b) for a, b in zip(reference, resumed, strict=True))

    print(f"steps compared:       {len(reference)}")
    print(f"lr at resume point:   correct {reference[cut]:.6e}  "
          f"resumed {resumed[cut]:.6e}")
    print(f"lr at final step:     correct {reference[-1]:.6e}  "
          f"resumed {resumed[-1]:.6e}")
    print(f"max abs difference:   {worst:.3e}")

    # The alternative this guards against: rebuilding the scheduler for however
    # many epochs are left, which silently starts a second one-cycle.
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1e-3)
    naive = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, total_steps=(EPOCHS - RESUME_AT) * STEPS_PER_EPOCH)
    print(f"\nrebuilt-for-remaining would resume at {naive.get_last_lr()[0]:.6e} "
          f"instead of {reference[cut]:.6e}, then warm back up to 1e-3 — a second "
          "high-lr phase where the tail should be annealing")

    assert worst == 0.0, f"schedule diverged by {worst}"
    print("\nPASS: resumed schedule is identical to the uninterrupted one")


if __name__ == "__main__":
    main()
