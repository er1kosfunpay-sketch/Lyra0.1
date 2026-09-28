"""Shared chat formatting and generation defaults (train/inference parity).

Canonical training format produced by ``lyra.data.PackedTextDataset`` is::

    <ROLE> body <END> <ROLE> body <END> ... <ASSISTANT>

Every history turn contributes its role marker, its (sanitized) body and a
closing ``<END>``; the prompt ends with a bare ``<ASSISTANT>`` trigger. All
inference entry points MUST build prompts through :func:`format_chat` so the
model always sees exactly the pattern it was trained on.
"""
import torch

GENERATION_DEFAULTS = {
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.08,
    "max_new_tokens": 256,
    "eos_token": "<END>",
}

VALID_ROLES = ("system", "user", "assistant", "tool")


def format_chat(messages, tokenizer, add_assistant_trigger=True):
    """Encode a message list into the canonical ``<ROLE> body <END>`` id stream."""
    from .conversations import clean_text
    ids = []
    for m in messages:
        role = str(m.get("role", "user")).lower()
        if role not in VALID_ROLES:
            raise ValueError(f"Unknown chat role {m.get('role')!r}; expected one of {VALID_ROLES}.")
        ids.append(tokenizer.id(f"<{role.upper()}>"))
        ids.extend(tokenizer.encode(clean_text(m.get("content", ""))))
        ids.append(tokenizer.id("<END>"))
    if add_assistant_trigger:
        ids.append(tokenizer.id("<ASSISTANT>"))
    return ids


def fit_history_to_context(messages, tokenizer, context_length, reserve=256):
    """Drop oldest user/assistant turns (keeping system) until the prompt fits.

    Returns a (possibly trimmed) message list whose formatted prompt leaves
    ``reserve`` tokens free for generation inside ``context_length``.
    """
    messages = list(messages)
    budget = max(1, context_length - max(1, reserve))
    while len(messages) > 1 and len(format_chat(messages, tokenizer)) > budget:
        drop = 1 if messages[0].get("role") == "system" else 0
        del messages[drop:drop + 2]
        if not messages:
            break
    prompt = format_chat(messages, tokenizer)
    if len(prompt) > budget:
        prompt = prompt[-budget:]
    return messages, prompt


@torch.inference_mode()
def generate_response(model, tokenizer, messages, device, max_new_tokens=None,
                      temperature=None, top_k=None, top_p=None, repetition_penalty=None):
    """Generate one assistant reply for ``messages``; returns decoded text."""
    d = GENERATION_DEFAULTS
    max_new_tokens = d["max_new_tokens"] if max_new_tokens is None else max_new_tokens
    temperature = d["temperature"] if temperature is None else temperature
    top_k = d["top_k"] if top_k is None else top_k
    top_p = d["top_p"] if top_p is None else top_p
    repetition_penalty = d["repetition_penalty"] if repetition_penalty is None else repetition_penalty
    model.eval()
    prompt = format_chat(messages, tokenizer)
    ctx = model.config.context_length
    if len(prompt) >= ctx:
        prompt = prompt[-(ctx - 1):]
    x = torch.tensor([prompt], device=device)
    out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                         eos_token_id=tokenizer.id(d["eos_token"]))
    gen = out[0, len(prompt):].tolist()
    end = tokenizer.id(d["eos_token"])
    if end in gen:
        gen = gen[:gen.index(end)]
    return tokenizer.decode(gen)
