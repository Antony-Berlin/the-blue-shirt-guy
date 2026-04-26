"""local_client.py — OpenAI-compatible wrapper around a local HF model.

Exposes `client.chat.completions.create(model=..., messages=..., ...)` so that
`inference.run_tool_loop` can use either the HF router or a local model
without code changes.
"""

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Lightweight response objects mimicking openai's API
# ---------------------------------------------------------------------------

@dataclass
class _Message:
    content: str
    role: str = "assistant"

@dataclass
class _Choice:
    message: _Message
    index: int = 0
    finish_reason: str = "stop"

@dataclass
class _ChatCompletion:
    choices: List[_Choice]
    id: str = "local"
    model: str = "local"

@dataclass
class _Completions:
    """Mimics `client.chat.completions`."""
    _generate: Any = None  # callable

    def create(self, *, model: str = "", messages: List[dict] = None,
               temperature: float = 0.7, max_tokens: int = 2048,
               stream: bool = False, **kwargs) -> _ChatCompletion:
        return self._generate(
            messages=messages or [],
            temperature=temperature,
            max_tokens=max_tokens,
        )


@dataclass
class _Chat:
    completions: _Completions = None


class LocalClient:
    """Drop-in replacement for ``openai.OpenAI`` that runs a local HF model.

    Usage::

        client = LocalClient(model, tokenizer)
        # works identically to:
        # client = OpenAI(base_url=..., api_key=...)
        resp = client.chat.completions.create(model="ignored", messages=[...])
    """

    def __init__(self, model, tokenizer, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chat = _Chat(completions=_Completions(_generate=self._generate))

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _generate(self, messages: List[dict], temperature: float, max_tokens: int) -> _ChatCompletion:
        # Apply chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            # Fallback: concatenate
            text = "\n".join(
                f"{m.get('role','user')}: {m.get('content','')}" for m in messages
            )
            text += "\nassistant:"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][input_len:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return _ChatCompletion(
            choices=[_Choice(message=_Message(content=response_text))],
            model="local",
        )
