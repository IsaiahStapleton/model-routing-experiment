from eval.sweep import cheapest_adequate, evaluate, gpu_seconds


def test_gpu_seconds_uses_measured_throughput():
    # 336 tokens on a 33.6 tok/s tier is 10 seconds
    assert round(gpu_seconds("gpt-oss-120b", 336), 2) == 10.0
    assert round(gpu_seconds("qwen3-1.7b", 93.1 * 2), 1) == 2.0


def test_cheapest_adequate_prefers_small_tier():
    labels = {"qwen3-1.7b": True, "qwen3-30b-a3b": True, "gpt-oss-120b": True}
    assert cheapest_adequate(labels) == "qwen3-1.7b"


def test_cheapest_adequate_skips_inadequate():
    labels = {"qwen3-1.7b": False, "qwen3-30b-a3b": True, "gpt-oss-120b": True}
    assert cheapest_adequate(labels) == "qwen3-30b-a3b"


def test_cheapest_adequate_none_when_all_fail():
    labels = {"qwen3-1.7b": False, "qwen3-30b-a3b": False, "gpt-oss-120b": False}
    assert cheapest_adequate(labels) is None


def test_evaluate_counts_under_and_over_routing():
    items = [{"id": "a"}, {"id": "b"}]
    labels = {
        "a": {"qwen3-1.7b": True, "qwen3-30b-a3b": True, "gpt-oss-120b": True},
        "b": {"qwen3-1.7b": False, "qwen3-30b-a3b": True, "gpt-oss-120b": True},
    }
    costs = {("a", "gpt-oss-120b"): 100, ("b", "qwen3-1.7b"): 1}
    # route a to the biggest tier (over-route), b to the smallest (under-route)
    chooser = lambda item: "gpt-oss-120b" if item["id"] == "a" else "qwen3-1.7b"
    result = evaluate(items, labels, costs, chooser)
    assert result["n"] == 2
    assert result["accuracy"] == 0.5
    assert result["under_route"] == 0.5
    assert result["over_route"] == 0.5


def test_evaluate_ignores_items_with_no_adequate_tier():
    items = [{"id": "a"}]
    labels = {"a": {"qwen3-1.7b": False, "qwen3-30b-a3b": False, "gpt-oss-120b": False}}
    result = evaluate(items, labels, {}, lambda item: "qwen3-1.7b")
    assert result["n"] == 0
