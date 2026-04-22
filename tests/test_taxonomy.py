from social_corrections.taxonomy import ALL_LABELS, FAILURE_TYPES, canonicalize


def test_all_labels_present():
    assert len(ALL_LABELS) == 6
    assert "Overly Harsh or Judgmental Language" in ALL_LABELS
    assert "Overconfidence / Lack of Uncertainty" in ALL_LABELS


def test_canonicalize_by_label():
    assert canonicalize("Lack of Acknowledgment") == "Lack of Acknowledgment"


def test_canonicalize_by_short():
    assert canonicalize("harsh") == "Overly Harsh or Judgmental Language"
    assert canonicalize("cold") == "Emotional Mismatch (Too Cold or Abrupt)"


def test_canonicalize_case_insensitive():
    assert canonicalize("lack of acknowledgment") == "Lack of Acknowledgment"


def test_canonicalize_raises_on_unknown():
    import pytest

    with pytest.raises(KeyError):
        canonicalize("not a real label")


def test_descriptions_nonempty():
    for ft in FAILURE_TYPES.values():
        assert ft.description
        assert ft.short
