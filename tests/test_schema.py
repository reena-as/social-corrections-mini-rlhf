import json
import tempfile
from pathlib import Path

from social_corrections.data import CorrectionPair, load_correction_pairs
from social_corrections.data.build_jsonl import build


def test_correction_pair_roundtrip():
    p = CorrectionPair(
        user="Is this right?",
        bad="No.",
        better="It looks close, but there may be a small issue. Let's check together.",
        failure_type="harsh",  # short form -> should canonicalize
        source="seed",
        pair_id="seed-0001",
    )
    assert p.failure_type == "Overly Harsh or Judgmental Language"
    sft = p.to_sft_example(system="You are helpful.")
    assert sft["messages"][0]["role"] == "system"
    assert sft["messages"][-1]["role"] == "assistant"
    assert sft["messages"][-1]["content"].startswith("It looks close")

    dpo = p.to_dpo_example(system="You are helpful.")
    assert dpo["chosen"].startswith("It looks close")
    assert dpo["rejected"] == "No."
    assert dpo["prompt_messages"][-1] == {"role": "user", "content": "Is this right?"}


def test_load_seed_dataset(tmp_path: Path):
    # Write a minimal seed file
    seed = [
        {"user": "hi", "bad": "wrong", "better": "not quite, let's see",
         "failure_type": "Overly Harsh or Judgmental Language"},
        {"user": "ok?", "bad": "no", "target_better": "close, try again",
         "failure_type": "Lack of Acknowledgment"},
    ]
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(seed), encoding="utf-8")
    pairs = load_correction_pairs(path)
    assert len(pairs) == 2
    assert pairs[0].better == "not quite, let's see"
    # target_better should have been picked up
    assert pairs[1].better == "close, try again"


def test_build_jsonl_splits(tmp_path: Path):
    pairs = [
        CorrectionPair(
            user=f"q{i}", bad=f"bad{i}", better=f"good{i}",
            failure_type="Lack of Acknowledgment",
            pair_id=f"pid-{i:04d}",
        )
        for i in range(40)
    ]
    counts = build(pairs, tmp_path)
    # Every pair appears once in SFT and once in DPO
    sft_total = counts["train_sft"] + counts["val_sft"] + counts["test_sft"]
    dpo_total = counts["train_dpo"] + counts["val_dpo"] + counts["test_dpo"]
    assert sft_total == 40
    assert dpo_total == 40
    # Deterministic splits roughly match 80/10/10. Allow a little slack.
    assert counts["train_sft"] >= 25
    # Files exist
    for split in ("train", "val", "test"):
        assert (tmp_path / f"{split}_sft.jsonl").exists()
        assert (tmp_path / f"{split}_dpo.jsonl").exists()


def test_build_jsonl_deterministic_splits(tmp_path: Path):
    """Same pair_id always lands in same bucket across runs."""
    pairs = [
        CorrectionPair(
            user=f"q{i}", bad=f"bad{i}", better=f"good{i}",
            failure_type="Lack of Acknowledgment",
            pair_id=f"pid-{i:04d}",
        )
        for i in range(20)
    ]
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    build(pairs, out1)
    build(pairs, out2)
    for split in ("train", "val", "test"):
        a = (out1 / f"{split}_sft.jsonl").read_text(encoding="utf-8")
        b = (out2 / f"{split}_sft.jsonl").read_text(encoding="utf-8")
        assert a == b
