"""Check the WASB ball counting on hand-built cases with known answers.

`classify` returned a single label until 2026-08-17, so a prediction farther than
tau counted as FP only. The frame then appeared in neither TP nor FN and dropped
out of recall's denominator, which inflated recall exactly when the model was
firing badly — the direction that flatters. On the converged checkpoint that was
recall 0.422 where 0.351 was true.

The property that has to hold, and that the old code broke:

    TP + FN == the number of frames that contain a ball

Every case below is small enough to verify by reading it.

    uv run python scripts/test_ball_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soccernet_tracking_edge.core.snet_eval import ball_metrics, classify  # noqa: E402

TAU = 4.0


def rec(pred, truth, score=0.9):
    return {"pred": pred, "score": score, "truth": truth}


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got}, want {want}")
    return ok


def main() -> None:
    ok = True

    print("classify(), one frame at a time:")
    ok &= check("hit inside tau", classify((10, 10), (11, 11), TAU), ("TP",))
    ok &= check("fire far from truth", classify((10, 10), (99, 99), TAU),
                ("FN", "FP"))
    ok &= check("silent, ball present", classify(None, (11, 11), TAU), ("FN",))
    ok &= check("silent, no ball", classify(None, None, TAU), ("TN",))
    ok &= check("fire, no ball", classify((10, 10), None, TAU), ("FP",))
    ok &= check("exactly at tau is a hit", classify((10, 10), (14, 10), TAU), ("TP",))

    # 5 frames: the first three contain a ball, the last two do not.
    records = [
        rec((10, 10), (11, 11)),    # TP
        rec((10, 10), (99, 99)),    # far fire  -> FN + FP
        rec(None, (11, 11)),        # FN
        rec((10, 10), None),        # FP
        rec(None, None),            # TN
    ]
    m = ball_metrics(records, tau=TAU)
    print("\nball_metrics() over 5 frames (3 with a ball):")
    ok &= check("TP", m["TP"], 1)
    ok &= check("FN", m["FN"], 2)
    ok &= check("FP", m["FP"], 2)
    ok &= check("TN", m["TN"], 1)
    ok &= check("TP + FN == frames containing a ball", m["TP"] + m["FN"], 3)

    n_ball = sum(1 for r in records if r["truth"] is not None)
    ok &= check("TP + FN equals the ball-bearing frame count", m["TP"] + m["FN"],
                n_ball)
    ok &= check("recall denominator matches ball frames",
                round(m["TP"] / (m["TP"] + m["FN"]), 4), m["recall"])
    ok &= check("precision", m["precision"], round(1 / 3, 4))
    # 5 frames, of which TP=1 and TN=1 are correct.
    ok &= check("accuracy is over frames, not counters", m["accuracy"], 0.4)

    print("\nthe regression this file exists for:")
    all_far = [rec((0, 0), (500, 500)) for _ in range(10)]
    m2 = ball_metrics(all_far, tau=TAU)
    ok &= check("10 far fires -> recall 0.0 (was 0.0/0 -> 0.0 by luck)",
                m2["recall"], 0.0)
    ok &= check("10 far fires -> FN 10 (old code gave 0)", m2["FN"], 10)

    half = [rec((10, 10), (11, 11)) for _ in range(5)] + \
           [rec((0, 0), (500, 500)) for _ in range(5)]
    m3 = ball_metrics(half, tau=TAU)
    ok &= check("5 hits + 5 far fires -> recall 0.5 (old code said 1.0)",
                m3["recall"], 0.5)

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
