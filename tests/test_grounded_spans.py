from types import SimpleNamespace

import torch

from my_own_ai.corpus_eval import evaluate_cases
from my_own_ai.functional_modules import Research
from my_own_ai.grounding import GroundedProposal, SPAN_FORMAT, span_prompt
from my_own_ai.modules import CheckpointStore, ModuleController, default_modules
from my_own_ai.proposal_training import encode_example
from my_own_ai.reasoning import END_PROPOSAL, LocalProposer, ProposalRequest, Stage, Status
from my_own_ai.retrieval import build_index


def test_span_prompt_preserves_exact_relevant_excerpt():
    text = "An unrelated introductory sentence. " * 6 + "An amber lantern signals that the west ferry is closed."
    request = ProposalRequest("research", "grounded_analysis", Stage.CONSTRUCT, "Draft",
                              "What does the amber lantern signal about the west ferry?",
                              {"evidence": [{"text": text}]}, "Use the signal condition.", "", "")
    prompt, choices = span_prompt(request, 512)
    assert len(prompt) == 384
    assert "amber lantern signals" in choices[1]
    assert choices[1] in text
    assert choices[1].encode() in bytes(prompt)


def test_v2_targets_train_the_choice_and_inference():
    row = {"id": "ferry_construct", "group": "ferry", "stage": "construct",
           "question": "What should a west ferry passenger do when the lantern is amber?",
           "passages": ["A blue flag announces a maintenance crew.",
                        "An amber lantern signals that the west ferry is closed."],
           "plan": "Use the correct signal.", "candidate": "", "feedback": "",
           "answer": "Premises:\n- [2] the west ferry is closed\nInference: A passenger must choose another route."}
    _, targets = encode_example(row, 512, SPAN_FORMAT)
    expected = list(b"2 A passenger must choose another route.") + [END_PROPOSAL]
    assert targets[targets != -100].tolist() == expected


def test_model_selects_an_available_excerpt_and_stops_at_end_marker():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(context_size=512)
            self.proposal_format = SPAN_FORMAT
            self.output = iter([ord("2"), *b" The bridge is the available route.", END_PROPOSAL])

        def forward(self, inputs):
            token = next(self.output)
            logits = torch.zeros((1, inputs.size(1), 256))
            logits[0, -1, ord("9")] = 2
            logits[0, -1, token] = 3 if token != ord("2") else 1
            return logits, None

    sources = [{"text": "The construction crew raises a blue flag."},
               {"text": "The bridge remains open when the ferry is closed."}]
    request = ProposalRequest("research", "grounded_analysis", Stage.CONSTRUCT, "Draft",
                              "Which route is still open when the ferry is closed?",
                              {"evidence": sources}, "Compare the routes.", "", "")
    result = LocalProposer(Toy()).propose(request)
    assert result.passage == 2 and result.finished
    assert result.quote in sources[1]["text"]
    assert result.inference.strip() == "The bridge is the available route."


def test_unfinished_inference_cannot_complete_and_correction_keeps_real_quote(tmp_path):
    class Engine:
        def propose(self, request):
            if request.stage == Stage.FORMULATE:
                return "Find a usable route."
            excerpt = "the bridge remains open"
            return GroundedProposal(1, excerpt, "The bridge is the remaining route.",
                                    request.stage == Stage.CORRECT)

    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "routes.txt").write_text("When the ferry closes, the bridge remains open.")
    index = tmp_path / "corpus.sqlite"
    build_index(folder, index)
    with ModuleController(CheckpointStore(tmp_path / "state"),
                          default_modules(corpus_index=index), Engine()) as core:
        core.switch("research")
        state = core.start("grounded_analysis", "What remains open when the ferry closes?")
        for _ in range(3):
            state = core.step()
        assert state["status"] == Status.REQUIRES_CORRECTION
        core.step()
        assert core.step()["status"] == Status.COMPLETED
        assert "- [1] the bridge remains open" in core.progress()["candidate"]


def test_audit_reports_real_source_locations_and_rejects_ungrounded_prose(tmp_path):
    class Engine:
        def propose(self, request):
            if request.stage == Stage.FORMULATE:
                return "Trace the permit back to the charter."
            quote = "The charter gives the recorder authority to renew permits."
            return GroundedProposal(1, quote, "The recorder retains the charter authority.", True)

    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "charter.txt").write_text("The charter gives the recorder authority to renew permits.")
    index = tmp_path / "corpus.sqlite"
    build_index(folder, index)
    cases = [{"id": "permit", "question": "Who may renew the permit under the charter?",
              "expected_sources": ["charter.txt"], "inference_terms": ["recorder"]}]
    raw = evaluate_cases(index, cases, Engine())
    assert "answer" not in raw["cases"][0]
    audited = evaluate_cases(index, cases, Engine(), include_answers=True)["cases"][0]
    assert audited["passed"] and "recorder retains" in audited["answer"]
    assert audited["citations"] == [{"source": "charter.txt", "locator": "line 1, passage 1"}]

    researcher = Research(folder)
    quote = "The charter gives the recorder authority to renew permits."
    raw = researcher.render_grounded(GroundedProposal(1, quote, "Tempera llacada gherton plonden.", True),
                                     {"evidence": [{"text": quote}]})
    assert raw.startswith("Inference lacks a term from the selected evidence:")
