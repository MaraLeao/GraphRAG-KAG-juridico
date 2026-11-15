from graphrag_juridico.retrieval import diversify, interleave, lucene, rrf


def test_interleave_puts_shared_chunks_first_then_alternates():
    lgpd = ["l1", "shared", "l2"]
    dignidade = ["d1", "d2", "d3", "shared"]

    assert interleave([lgpd, dignidade]) == ["shared", "l1", "d1", "d2", "l2", "d3"]


def test_interleave_handles_empty_lists():
    assert interleave([]) == []
    assert interleave([[], ["a"]]) == ["a"]


def test_lucene_escapes_special_characters():
    assert lucene("Lei 13.709/2018: o que diz?") == r"Lei 13.709\/2018\: o que diz\?"


def test_rrf_rewards_items_present_in_several_rankings():
    fused = rrf(["a", "b", "c"], ["c", "d"], ["c"])

    assert fused[0] == "c"
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_ignores_empty_rankings():
    assert rrf([], ["x", "y"], []) == ["x", "y"]


def test_diversify_limits_chunks_per_article_and_keeps_order():
    rows = [{"id": f"{art}:{i}", "article_id": art} for art, i in
            [("a1", 1), ("a1", 2), ("a1", 3), ("a2", 1), ("a3", 1)]]

    chosen = diversify(rows, k=3, per_article=2)

    assert [r["id"] for r in chosen] == ["a1:1", "a1:2", "a2:1"]
