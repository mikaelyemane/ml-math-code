"""Ch. 18 --- Testing and Versioning: the chapter's tests, runnable.

Three of the chapter's four test layers on concrete code, plus the data manifest:

  1. Unit test: bucketize_age, checked on both sides of every boundary
     (the parametrized listing).
  2. Behavioral invariance test: the name-swap listing, run against two toy
     credit models. One reads only the financial fields and passes; the other
     has picked up a name shortcut and fails, with the per-name scores in the
     assertion message.
  3. Data manifest: pin a content hash, not a filename, so a silent upstream
     edit is detected instead of trained on.

Run either way:
    python3 ch18_testing.py          # runs every test and prints a report
    pytest ch18_testing.py           # the same tests, collected by pytest
NumPy and the standard library only; pytest is optional.
"""
from __future__ import annotations

import hashlib
import io
import math
import re

try:
    import pytest
except ImportError:          # pytest is optional: the file runs its own tests below
    pytest = None


# =============================================================================
# 1. Unit test: a feature transform
# =============================================================================

def bucketize_age(age):
    if age is None:
        return "missing"
    if age < 0:
        return "invalid"
    if age <= 17:
        return "0-17"
    if age <= 34:
        return "18-34"
    if age <= 64:
        return "35-64"
    return "65+"


# Both sides of every boundary: off-by-one errors in
# feature engineering hide exactly there.
AGE_CASES = [
    (0, "0-17"), (17, "0-17"), (18, "18-34"),
    (34, "18-34"), (35, "35-64"), (64, "35-64"),
    (65, "65+"),        # one past the last boundary
    (-1, "invalid"),    # impossible value
    (None, "missing"),  # nulls happen
]


def _param(*args, **kw):
    return pytest.mark.parametrize(*args, **kw) if pytest else (lambda f: f)


@_param("age, bucket", AGE_CASES)
def test_bucketize_age(age, bucket):
    assert bucketize_age(age) == bucket


def buggy_bucketize_age(age):
    """The classic slip: `< 17` where the spec says 17 is still a minor."""
    if age is None:
        return "missing"
    if age < 0:
        return "invalid"
    if age < 17:
        return "0-17"
    if age <= 34:
        return "18-34"
    if age <= 64:
        return "35-64"
    return "65+"


# =============================================================================
# 2. Behavioral invariance test: the model as a black box
# =============================================================================

APP = ("{name}, credit score 720, annual income "
       "$85,000, requesting a $15,000 auto loan.")
NAMES = ["James Smith", "Lakisha Washington",
         "Wei Chen", "Maria Garcia"]


def _fields(text):
    score = int(re.search(r"credit score (\d+)", text).group(1))
    income, loan = (int(x.replace(",", "")) for x in re.findall(r"\$([\d,]+)", text))
    return score, income, loan


class FinancialModel:
    """Scores only the financial fields."""

    def predict_proba(self, text):
        score, income, loan = _fields(text)
        z = 0.02 * (score - 650) + 3.0 * (1 - loan / (0.4 * income)) - 1.0
        p = 1 / (1 + math.exp(-z))
        return {"approved": p, "denied": 1 - p}


class ShortcutModel(FinancialModel):
    """The same model after training on data where a name token correlated with
    the historical label. Nothing in its aggregate metrics has to look wrong."""

    NAME_WEIGHTS = {"Washington": -0.9, "Garcia": -0.4}

    def predict_proba(self, text):
        score, income, loan = _fields(text)
        z = 0.02 * (score - 650) + 3.0 * (1 - loan / (0.4 * income)) - 1.0
        z += sum(w for tok, w in self.NAME_WEIGHTS.items() if tok in text)
        p = 1 / (1 + math.exp(-z))
        return {"approved": p, "denied": 1 - p}


if pytest:
    @pytest.fixture
    def model():
        return FinancialModel()


def test_invariant_to_name(model):  # pytest fixture
    # A name carries no financial information, so no
    # choice of name should move the prediction.
    texts = [APP.format(name=n) for n in NAMES]
    p = [model.predict_proba(t)["approved"] for t in texts]
    assert max(p) - min(p) < 0.02, dict(zip(NAMES, p))


def test_directional_income(model=None):
    """Directional expectation: more income, same everything else, never lowers approval."""
    model = model or FinancialModel()
    base = APP.format(name="Wei Chen")
    richer = base.replace("$85,000", "$120,000")
    assert model.predict_proba(richer)["approved"] >= model.predict_proba(base)["approved"]


# =============================================================================
# 3. Data manifest: content hash, not filename
# =============================================================================

def content_hash(rows):
    """sha256 over a canonical serialization (sorted rows, fixed separators)."""
    buf = io.StringIO()
    for row in sorted(rows):
        buf.write("\x1f".join(map(str, row)) + "\n")
    return "sha256:" + hashlib.sha256(buf.getvalue().encode()).hexdigest()


def test_manifest_detects_silent_edit():
    snapshot = [(1001, 42.50, 0), (1002, 13.99, 0), (1003, 980.00, 1)]
    manifest = {"dataset": "transactions_train_v47", "row_count": len(snapshot),
                "content_hash": content_hash(snapshot)}
    # Upstream "corrects" one label in place. Same table name, same row count.
    upstream_today = [(1001, 42.50, 0), (1002, 13.99, 1), (1003, 980.00, 1)]
    assert len(upstream_today) == manifest["row_count"]               # the filename-and-count check passes
    assert content_hash(upstream_today) != manifest["content_hash"]  # the hash does not
    assert content_hash(list(reversed(snapshot))) == manifest["content_hash"]  # row order is not a change


# =============================================================================

def _run(name, fn, *args):
    try:
        fn(*args)
        print(f"  PASS  {name}")
        return True
    except AssertionError as err:
        print(f"  FAIL  {name}: {err}")
        return False


def main():
    print("1. Unit test, both sides of every boundary")
    ok = all(_run(f"bucketize_age({a!r}) == {b!r}", test_bucketize_age, a, b) for a, b in AGE_CASES)
    missed = [(a, b) for a, b in AGE_CASES if buggy_bucketize_age(a) != b]
    print(f"  the same cases against an off-by-one version catch it at: {missed}")

    print("\n2. Invariance test (name swap) and a directional test")
    ok &= _run("FinancialModel: invariant to name", test_invariant_to_name, FinancialModel())
    passed_shortcut = _run("ShortcutModel: invariant to name (expected to FAIL)",
                           test_invariant_to_name, ShortcutModel())
    ok &= not passed_shortcut
    ok &= _run("FinancialModel: more income never lowers approval", test_directional_income)

    print("\n3. Data manifest")
    ok &= _run("content hash catches an in-place label edit", test_manifest_detects_silent_edit)

    print("\nall checks behaved as expected" if ok else "\nUNEXPECTED RESULT above")


if __name__ == "__main__":
    main()
