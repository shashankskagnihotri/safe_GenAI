from __future__ import annotations

from pathlib import Path

from hierasafe_flow.steering.concept_graph import ConceptHierarchy, compose_concept_prompt


def test_concept_graph_loads_safe_siblings() -> None:
    graph = ConceptHierarchy.from_yaml_file(Path("configs/concept_hierarchies/all_safety.yaml"))
    assert graph.name == "all_safety"
    assert len(graph.pairs) >= 2
    first = graph.pairs[0]
    assert graph.safe_sibling_for(first.unsafe_concept) == first.safe_sibling_concept
    assert first.parent in graph.by_parent()


def test_compose_concept_prompt() -> None:
    prompt = compose_concept_prompt("a portrait", "fully clothed")
    assert "a portrait" in prompt
    assert "fully clothed" in prompt

