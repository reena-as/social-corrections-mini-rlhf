"""Politeness classifier trained on the Stanford Politeness Corpus.

We keep this simple on purpose: the classifier is a scalar politeness score
used as an additional evaluation signal, not a research artifact of its own.
A scikit-learn logistic regression over word unigrams+bigrams reaches ~75-80%
accuracy on the Wikipedia/StackExchange splits, which is more than enough
resolution to compare our four systems.

Train:
    python -m social_corrections.evaluation.politeness_classifier train \\
        --corpus data/raw/stanford_politeness.csv \\
        --out-path models/politeness_clf.pkl

Predict:
    python -m social_corrections.evaluation.politeness_classifier predict \\
        --model-path models/politeness_clf.pkl \\
        --texts "some text" "another"

Corpus format: CSV with columns ``text`` and ``score`` (a real-valued politeness
rating from the annotators). We binarize at the median for the LR target.
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

from ..utils import ensure_dir


@dataclass
class PolitenessPrediction:
    text: str
    prob_polite: float

    @property
    def label(self) -> str:
        return "polite" if self.prob_polite >= 0.5 else "impolite"


class PolitenessClassifier:
    """Thin wrapper around a scikit-learn pipeline. Lazy imports so the module
    is safe to import in environments without sklearn.
    """

    def __init__(self, pipeline=None):
        self._pipeline = pipeline

    # ---- training ----
    @classmethod
    def train(
        cls,
        texts: list[str],
        labels: list[int],
        random_state: int = 0,
    ) -> "PolitenessClassifier":
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore

        pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=2000, C=1.0, random_state=random_state
                    ),
                ),
            ]
        )
        pipe.fit(texts, labels)
        return cls(pipe)

    @classmethod
    def train_from_csv(
        cls, csv_path: str, text_col: str = "text", score_col: str = "score"
    ) -> "PolitenessClassifier":
        import pandas as pd  # type: ignore

        df = pd.read_csv(csv_path)
        if text_col not in df.columns or score_col not in df.columns:
            raise ValueError(
                f"CSV must contain {text_col!r} and {score_col!r} columns; "
                f"got {list(df.columns)}"
            )
        median = df[score_col].median()
        labels = (df[score_col] > median).astype(int).tolist()
        texts = df[text_col].astype(str).tolist()
        return cls.train(texts, labels)

    # ---- inference ----
    def predict(self, texts: list[str]) -> list[PolitenessPrediction]:
        if self._pipeline is None:
            raise RuntimeError("Classifier is not trained. Call .train(...) or load().")
        probs = self._pipeline.predict_proba(texts)[:, 1]
        return [PolitenessPrediction(text=t, prob_polite=float(p)) for t, p in zip(texts, probs)]

    def mean_politeness(self, texts: list[str]) -> float:
        if not texts:
            return 0.0
        preds = self.predict(texts)
        return sum(p.prob_polite for p in preds) / len(preds)

    # ---- serialization ----
    def save(self, path: str) -> None:
        ensure_dir(Path(path).parent)
        with open(path, "wb") as f:
            pickle.dump(self._pipeline, f)

    @classmethod
    def load(cls, path: str) -> "PolitenessClassifier":
        with open(path, "rb") as f:
            pipe = pickle.load(f)
        return cls(pipe)


def _cmd_train(args) -> None:
    clf = PolitenessClassifier.train_from_csv(args.corpus, args.text_col, args.score_col)
    clf.save(args.out_path)
    print(f"Saved classifier to {args.out_path}")


def _cmd_predict(args) -> None:
    clf = PolitenessClassifier.load(args.model_path)
    for pred in clf.predict(args.texts):
        print(f"{pred.prob_polite:.3f}\t{pred.label}\t{pred.text}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Politeness classifier CLI.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train on Stanford Politeness CSV.")
    p_train.add_argument("--corpus", required=True)
    p_train.add_argument("--text-col", default="text")
    p_train.add_argument("--score-col", default="score")
    p_train.add_argument("--out-path", default="models/politeness_clf.pkl")
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict", help="Score one or more texts.")
    p_pred.add_argument("--model-path", default="models/politeness_clf.pkl")
    p_pred.add_argument("--texts", nargs="+", required=True)
    p_pred.set_defaults(func=_cmd_predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
