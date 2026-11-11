"""critique.py: JSON validation of the fresh reviewer's and the blind judge's answers."""
from b2r import critique as C

GOOD = {
    "a_has_b_lacks": [{"what": "A shows a view through the opening.", "kind": "exterior", "bbox": [10, 20, 200, 180],
                       "severity": 5, "fix_hint": "Add geometry outside the opening."},
                      {"what": "A's table top is dark.", "kind": "material", "bbox": [0, 0, 5, 5], "severity": 2,
                       "fix_hint": "Darken the table top."}],
    "b_has_a_lacks": [{"what": "B has hard light patches.", "bbox": [30, 40, 90, 100], "severity": 3}],
    "overall": "Layout matches; the exterior is the main gap.",
}
JUDGE = {"closer_to": "reference_2", "score_0_to_10": 8.5, "reasons": "same palette", "photo_or_cg": "CG on a close look",
         "cg_giveaways": [{"what": "flat light", "bbox": [0, 0, 100, 50], "severity": 3}],
         "passes_as_real": ["floor texture"], "verdict": "Mostly yes."}


def test_valid_critique_passes():
    assert C.validate_critique(GOOD, (320, 240)) == []
    assert C.validate_critique({"a_has_b_lacks": [], "b_has_a_lacks": [], "overall": "same"}) == []


def test_bad_critiques_are_rejected():
    import copy
    cases = []
    b = copy.deepcopy(GOOD); b["a_has_b_lacks"][0]["kind"] = "colour"; cases.append(("kind", b))
    b = copy.deepcopy(GOOD); b["a_has_b_lacks"][0]["severity"] = 6; cases.append(("severity", b))
    b = copy.deepcopy(GOOD); b["a_has_b_lacks"][0]["severity"] = 2.5; cases.append(("severity", b))
    b = copy.deepcopy(GOOD); b["a_has_b_lacks"][0]["bbox"] = [200, 20, 10, 180]; cases.append(("x1 > x0", b))
    b = copy.deepcopy(GOOD); b["a_has_b_lacks"][0]["bbox"] = [10, 20, 400, 180]; cases.append(("outside", b))
    b = copy.deepcopy(GOOD); b["a_has_b_lacks"][0]["bbox"] = [10, 20, 200]; cases.append(("bbox", b))
    b = copy.deepcopy(GOOD); del b["a_has_b_lacks"][0]["fix_hint"]; cases.append(("fix_hint", b))
    b = copy.deepcopy(GOOD); del b["overall"]; cases.append(("overall", b))
    b = copy.deepcopy(GOOD); b["b_has_a_lacks"] = {}; cases.append(("must be a list", b))
    b = copy.deepcopy(GOOD); b["extra"] = 1; cases.append(("unexpected keys", b))
    for needle, bad in cases:
        errs = C.validate_critique(bad, (320, 240))
        assert errs and any(needle in e for e in errs), (needle, errs)
    assert C.validate_critique([1, 2]) == ["critique must be a JSON object"]


def test_judge_validation():
    import copy
    assert C.validate_judge(JUDGE, (320, 240)) == []
    for key, val, needle in (("closer_to", "candidate", "closer_to"), ("score_0_to_10", 11, "score"),
                             ("passes_as_real", "floor", "passes_as_real"), ("verdict", "", "verdict")):
        b = copy.deepcopy(JUDGE)
        b[key] = val
        errs = C.validate_judge(b, (320, 240))
        assert any(needle in e for e in errs), (key, errs)
    b = copy.deepcopy(JUDGE)
    del b["cg_giveaways"]
    assert any("missing keys" in e for e in C.validate_judge(b))


def test_prompts_name_the_shape():
    p = C.CRITIQUE_PROMPT.format(w=640, h=480)
    for k in ("a_has_b_lacks", "b_has_a_lacks", "overall", "fix_hint", "severity", "bbox", "640 x 480") + C.CRITIQUE_KINDS:
        assert k in p, k
    j = C.JUDGE_PROMPT.format(w=640, h=480)
    for k in ("closer_to", "score_0_to_10", "cg_giveaways", "passes_as_real", "verdict", "reference_1", "candidate"):
        assert k in j, k
