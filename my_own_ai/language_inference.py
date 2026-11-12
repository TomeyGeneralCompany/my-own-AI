"""Shared chat formatting and true original-model autoregressive decoding."""
import math
import torch
from .tokenizer import BOS, EOS, USER, ASSISTANT, PAD
from .language_training import load_language


def prompt_tokens(question, tokenizer, context_size):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must contain text")
    prompt = [BOS, USER] + tokenizer.encode(question) + [EOS, ASSISTANT]
    if len(prompt) >= context_size:
        raise ValueError("Question exceeds model context window")
    return prompt


@torch.inference_mode()
def answer_question(checkpoint, tokenizer_path, question, *,
                    max_tokens=128, temperature=0., top_p=1., seed=42, threads=2):
    if not (1 <= max_tokens <= 4096 and 0 <= temperature <= 4
            and 0 < top_p <= 1 and isinstance(seed, int)):
        raise ValueError("Invalid decoding settings")
    torch.set_num_threads(threads)
    model, tokenizer = load_language(checkpoint, tokenizer_path)
    prompt = prompt_tokens(question, tokenizer, model.config.context_size)
    history = list(prompt)
    generated = []
    generator = torch.Generator().manual_seed(seed)
    reason = "max_tokens"
    for _ in range(min(max_tokens, model.config.context_size - len(prompt))):
        logits, _ = model(torch.tensor([history], dtype=torch.long))
        scores = logits[0, -1].clone()
        # Never emit structural role tokens as conversational words.
        scores[[PAD, BOS, USER, ASSISTANT, 261]] = float("-inf")
        if temperature == 0:
            token = int(scores.argmax())
        else:
            probabilities = torch.softmax(scores / temperature, dim=-1)
            values, indexes = probabilities.sort(descending=True)
            remove = values.cumsum(0) - values > top_p
            values[remove] = 0
            values /= values.sum()
            token = int(indexes[torch.multinomial(values, 1, generator=generator)])
        if token == EOS:
            reason = "eos"
            break
        generated.append(token)
        history.append(token)
    # Tokenizer byte fallback may end on partial UTF-8; report an error rather
    # than hide malformed output by silently replacing unknown characters.
    try:
        rendered = tokenizer.decode(generated)
        decoding_valid = True
    except UnicodeDecodeError:
        rendered = bytes(b for i in generated for b in tokenizer.vocab[i]).decode(
            "utf-8", errors="replace")
        decoding_valid = False
    return {
        "question": question, "answer": rendered, "raw_token_ids": generated,
        "prompt_token_ids": prompt, "tokenizer_sha256": tokenizer.fingerprint(),
        "stopped_by": reason, "valid_utf8": decoding_valid,
        "model_generated": True, "evaluated_for_truth": False,
        "settings": {"max_tokens": max_tokens, "temperature": temperature,
                     "top_p": top_p, "seed": seed},
    }
