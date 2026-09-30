"""Ch. 17 --- Abstraction and Interfaces: the point-in-time join, worked.

Runs the chapter's worked example end to end on its own two tables:

  1. The naive join (current snapshot), the availability-keyed as-of join, and
     the event-time as-of join that people write *after* learning about as-of
     joins. It reproduces the NAIVE and AS-OF columns of tab:asof_join and the
     value 3 that the event-time join leaks into the 2026-03-17 label.
  2. Offline AUC on the five labels: 1.00 with the leak, 0.33 without.
  3. The contract test from the listing, re-deriving each feature from the raw
     order log as of the row's own feature_ts. It passes on the as-of join and
     fails on the naive one.
  4. The config-driven feature test from the "separated" listing: schema checks
     pass for a 7-day and a 30-day activity window alike; a fixed example does not.

Standard library + NumPy, so the SQL is written out as the Python it means.
Run:  python3 ch17_point_in_time.py
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

D = date.fromisoformat

# =============================================================================
# The chapter's tables
# =============================================================================

# tab:asof_features --- user_features: append-only history of orders_last_30d.
# (user_id, orders_last_30d, feature_ts, available_ts)
USER_FEATURES = [
    (7, 2, D("2026-03-01"), D("2026-03-02")),
    (7, 3, D("2026-03-14"), D("2026-03-18")),   # four days in the pipeline
    (7, 5, D("2026-03-24"), D("2026-03-25")),
    (12, 0, D("2026-03-01"), D("2026-03-02")),
    (23, 4, D("2026-03-01"), D("2026-03-02")),
]

# The five labeled moments; y = "placed an order in the following week".
LABELS = [
    (7, D("2026-03-10"), 1),
    (12, D("2026-03-10"), 0),
    (23, D("2026-03-10"), 0),
    (7, D("2026-03-17"), 1),
    (23, D("2026-03-21"), 0),
]

# tab:asof_join, as printed in the chapter.
BOOK_NAIVE = [5, 0, 4, 5, 4]
BOOK_ASOF = [2, 0, 4, 2, 4]

# A raw order log consistent with the feature table: the nightly job counts
# orders in the 30 days up to and including feature_ts, and each order lands
# (becomes queryable) on the day its row's available_ts says.
# (user_id, order_date, landed_date)
RAW_ORDERS = [
    (7, D("2026-02-20"), D("2026-03-02")), (7, D("2026-03-01"), D("2026-03-02")),
    (7, D("2026-03-12"), D("2026-03-18")),                       # the order the 03-10 label predicts
    (7, D("2026-03-20"), D("2026-03-25")), (7, D("2026-03-22"), D("2026-03-25")),
    (7, D("2026-03-23"), D("2026-03-25")),
    (23, D("2026-02-10"), D("2026-03-02")), (23, D("2026-02-15"), D("2026-03-02")),
    (23, D("2026-02-22"), D("2026-03-02")), (23, D("2026-02-28"), D("2026-03-02")),
]


@dataclass(frozen=True)
class Row:
    user_id: int
    label_ts: date
    y: int
    orders_last_30d: int | None
    feature_ts: date | None
    available_ts: date | None


# =============================================================================
# The three joins
# =============================================================================

def naive_join(labels, features):
    """SELECT ... JOIN user_features_current: the latest row, whenever it landed."""
    out = []
    for uid, lts, y in labels:
        f = max((r for r in features if r[0] == uid), key=lambda r: r[2])
        out.append(Row(uid, lts, y, f[1], f[2], f[3]))
    return out


def asof_join(labels, features, on="available_ts"):
    """ROW_NUMBER() ... ORDER BY feature_ts DESC, available_ts DESC, WHERE <on> < label_ts.

    on="available_ts" is the chapter's query. on="feature_ts" is the event-time
    version: it looks like an as-of join and passes review, and it still leaks.
    """
    col = {"feature_ts": 2, "available_ts": 3}[on]
    out = []
    for uid, lts, y in labels:
        cands = [r for r in features if r[0] == uid and r[col] < lts]
        if not cands:
            out.append(Row(uid, lts, y, None, None, None))
            continue
        f = max(cands, key=lambda r: (r[2], r[3]))
        out.append(Row(uid, lts, y, f[1], f[2], f[3]))
    return out


def auc(scores, y):
    """P(score_pos > score_neg) + 0.5 P(tie): the Mann-Whitney form of ROC AUC."""
    s, y = np.asarray(scores, float), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0) + 0.5 * (diff == 0)))


# =============================================================================
# The contract test from the listing
# =============================================================================

def orders_last_30d(raw, user_id, as_of):
    lo = as_of - timedelta(days=29)
    return sum(1 for u, d, _ in raw if u == user_id and lo <= d <= as_of)


def test_no_future_leakage(rows, raw_log):
    """Listing: the join's time predicate, then re-derive as of the row's OWN feature_ts."""
    failures = []
    for r in rows:
        if r.available_ts is None:
            continue
        if not r.available_ts < r.label_ts:
            failures.append((r, "available_ts >= label_ts"))
            continue
        landed = [o for o in raw_log if o[2] <= r.available_ts]
        truth = orders_last_30d(landed, r.user_id, as_of=r.feature_ts)
        if r.orders_last_30d != truth:
            failures.append((r, f"value {r.orders_last_30d} != re-derived {truth}"))
    return failures


# =============================================================================
# The config-driven feature test from the "separated" listing
# =============================================================================

@dataclass(frozen=True)
class FeatureConfig:
    active_window_days: int


def compute_features(rows, cfg: FeatureConfig):
    return [{"engagement": (r["sessions_30d"] or 0) / 30,
             "is_active": r["last_login_days"] < cfg.active_window_days,
             "tenure_days": r["tenure_days"]} for r in rows]


def schema_ok(feats):
    return all(isinstance(f["is_active"], bool) and f["engagement"] >= 0 for f in feats)


def test_active_window_is_config_driven():
    row = [{"last_login_days": 10, "sessions_30d": 3, "tenure_days": 90}]

    def is_active(days):
        return compute_features(row, FeatureConfig(days))[0]["is_active"]

    assert schema_ok(compute_features(row, FeatureConfig(7)))
    assert schema_ok(compute_features(row, FeatureConfig(30)))   # schema can't tell them apart
    assert not is_active(7)
    assert is_active(30)


# =============================================================================

def main():
    y = [lab[2] for lab in LABELS]
    naive = naive_join(LABELS, USER_FEATURES)
    asof = asof_join(LABELS, USER_FEATURES, on="available_ts")
    event = asof_join(LABELS, USER_FEATURES, on="feature_ts")

    print("user  label_ts    y  naive  as-of(available)  as-of(event time)")
    for n, a, e in zip(naive, asof, event):
        star = "*" if n.orders_last_30d != a.orders_last_30d else " "
        leak = "  <- not readable on the label date" if e.orders_last_30d != a.orders_last_30d else ""
        print(f"{n.user_id:>4}{star} {n.label_ts}  {n.y}  {n.orders_last_30d:>5}  "
              f"{a.orders_last_30d:>16}  {e.orders_last_30d:>17}{leak}")

    assert [r.orders_last_30d for r in naive] == BOOK_NAIVE
    assert [r.orders_last_30d for r in asof] == BOOK_ASOF
    print("\nNAIVE and AS-OF columns match tab:asof_join exactly.")

    print(f"\noffline AUC on the five labels:  naive {auc([r.orders_last_30d for r in naive], y):.2f}"
          f"   as-of {auc([r.orders_last_30d for r in asof], y):.2f}"
          f"   event-time {auc([r.orders_last_30d for r in event], y):.2f}")

    for r in USER_FEATURES:   # the raw log really does reproduce the feature table
        landed = [o for o in RAW_ORDERS if o[2] <= r[3]]
        assert orders_last_30d(landed, r[0], r[2]) == r[1], r

    print("\ncontract test (listing), re-deriving from the raw order log:")
    for name, rows in (("as-of (available_ts)", asof), ("event-time as-of", event), ("naive", naive)):
        fails = test_no_future_leakage(rows, RAW_ORDERS)
        print(f"  {name:<22} {'PASS' if not fails else f'FAIL ({len(fails)} rows)'}")
        for r, why in fails:
            print(f"      user {r.user_id} label {r.label_ts}: {why}")

    test_active_window_is_config_driven()
    print("\ntest_active_window_is_config_driven: PASS "
          "(schema passes for 7 and 30 days; the fixed example pins which one ships)")


if __name__ == "__main__":
    main()
