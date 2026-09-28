"""With no textbook, the synthesis model fills the gaps, labelled and capped.

With a textbook, only the textbook fills gaps.
"""

from __future__ import annotations

from studysynth.models import Admitted, RetrievalOutcome
from studysynth.pipeline.graph import Nodes
from studysynth.pipeline.render import tag_provenance
from studysynth.store import Catalogue

from .conftest import StubLlm, make_concept, make_draft, make_gap


class RecordingLlm(StubLlm):
    def __init__(self) -> None:
        super().__init__({"synthesize_chapter": {"text": "# Doc"}})
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, job: str, effort: str | None = None) -> str:
        self.prompts.append(prompt)
        return super().complete(prompt, job=job, effort=effort)


def synthesis_prompt(settings, admitted_gap: str | None, has_textbook: bool) -> str:
    llm = RecordingLlm()
    nodes = Nodes(settings, llm, None, Catalogue(settings.paths.catalogue))
    filled = make_gap("gap-filled")
    open_gap = make_gap("gap-open").model_copy(update={"question": "Why is the base case needed?"})
    admitted = (
        [
            Admitted(
                candidate_id="c",
                gap_id=admitted_gap,
                text="Book text",
                citation="[C: pages 1-2]",
                page_start=1,
                page_end=2,
                token_estimate=10,
                necessity=0.9,
            )
        ]
        if admitted_gap
        else []
    )
    nodes.synthesize(
        {
            "chapter": "induction",
            "concepts": [make_concept()],
            "gaps": [filled, open_gap],
            "drafts": [make_draft()],
            "retrieval": RetrievalOutcome(admitted=admitted, has_textbook=has_textbook),
        }
    )
    return llm.prompts[0]


def open_gaps(prompt: str) -> str:
    return prompt.split("OPEN GAPS:")[1].split("UNVERIFIED CLAIMS:")[0].strip()


def test_without_a_textbook_the_model_fills_every_gap(settings):
    prompt = synthesis_prompt(settings, admitted_gap=None, has_textbook=False)
    assert "Why is the base case needed?" in open_gaps(prompt)
    assert open_gaps(prompt).count("- Strong induction:") == 2
    assert f"at most {settings.gap_fill.max_words} words" in prompt


def test_with_a_textbook_only_the_textbook_fills_gaps(settings):
    """Even a gap the gate left unfilled is not handed to the model."""
    prompt = synthesis_prompt(settings, admitted_gap="gap-filled", has_textbook=True)
    assert open_gaps(prompt) == ""


def test_gap_filling_can_be_switched_off(settings):
    settings.gap_fill.enabled = False
    assert open_gaps(synthesis_prompt(settings, admitted_gap=None, has_textbook=False)) == ""


def test_model_explanations_are_labelled_in_the_rendered_document():
    html = tag_provenance("[M: model explanation, not from your sources] The base case anchors it.")
    assert '<span class="src-m">' in html and html.endswith("</span>")
