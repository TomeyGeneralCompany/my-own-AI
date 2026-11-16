"""Use the original local language model for module proposals, without weight updates."""
import json
from .foundation_contract import FoundationContract, immutable_inference
from .language_inference import prompt_tokens
from .tokenizer import PAD, BOS, EOS, USER, ASSISTANT, UNK


class BPEProposer:
    """Development adapter; loading weights does not confer owner qualification."""

    def __init__(self, model, tokenizer, tokens=128):
        if type(tokens) is not int or not 1 <= tokens <= 4096:
            raise ValueError("Invalid output token budget")
        if any(p.device.type != "cpu" for p in model.parameters()):
            raise ValueError("Module runtime requires a CPU model")
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.foundation = FoundationContract()
        self.tokenizer = tokenizer
        self.tokens = tokens

    def propose(self, request):
        import torch
        # No normative wrapper, external inference or retrieved answer substitution.
        # Preserve all fields in the request; a non-fitting request is visible to
        # the caller rather than silently dropping the owner's objective.
        question = json.dumps({
            "module": request.module, "operation": request.operation,
            "stage": request.stage.value, "procedure": request.procedure,
            "objective": request.objective, "foundation": self.foundation.record(), "context": request.context,
            "plan": request.plan, "candidate": request.candidate,
            "feedback": request.feedback
        }, ensure_ascii=False, separators=(",", ":"))
        capacity = self.model.config.context_size
        history = prompt_tokens(question, self.tokenizer, capacity)
        generated = []
        with immutable_inference(self.model):
            for _ in range(min(self.tokens, capacity-len(history))):
                logits, _ = self.model(torch.tensor([history], dtype=torch.long))
                scores = logits[0,-1].clone()
                scores[[PAD,BOS,USER,ASSISTANT,UNK]]=float("-inf")
                token=int(scores.argmax())
                if token==EOS:
                    break
                generated.append(token)
                history.append(token)
        # Do not substitute text when raw output is invalid.
        return self.tokenizer.decode(generated)
