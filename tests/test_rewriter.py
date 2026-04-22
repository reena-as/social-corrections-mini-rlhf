from social_corrections.rule_based import RuleBasedRewriter, rewrite_response


def test_full_sentence_replacement():
    r = RuleBasedRewriter()
    # Smart apostrophe variant
    assert "I see what you" in r("That\u2019s wrong.")
    # ASCII apostrophe variant
    assert "I see what you" in r("That's wrong.")


def test_harsh_word_softening():
    r = RuleBasedRewriter()
    out = r("Your approach is wrong and bad.")
    assert "wrong" not in out.lower()  # should be softened
    # Either 'bad' is softened to 'could be improved', or full-sentence replace fired
    assert "may not be correct" in out or "could be improved" in out


def test_hedging_replacement():
    r = RuleBasedRewriter()
    out = r("This is definitely the best way to solve it.")
    # 'definitely' -> 'likely' and 'best way' -> 'a good way'
    assert "definitely" not in out.lower()
    assert "best way" not in out.lower()


def test_acknowledgment_prepended_on_short():
    r = RuleBasedRewriter()
    out = r("No.")
    # Short/abrupt reply should get an acknowledgment prefix
    assert any(phrase.lower() in out.lower() for phrase in [
        "i see", "i understand", "thanks for", "of course"
    ])


def test_deterministic_with_seed():
    r1 = RuleBasedRewriter(seed=0)
    r2 = RuleBasedRewriter(seed=0)
    text = "Wrong."
    assert r1(text) == r2(text)


def test_default_rewriter_callable():
    out = rewrite_response("That's wrong.")
    assert isinstance(out, str)
    assert out.strip()
