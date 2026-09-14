"""A DSPy LM for endpoints that silently discard the `system` role.

Some Foundry deployments accept a `system` message, return HTTP 200, and throw
it away. Measured on Azure's `Phi-4-mini-instruct` deployment (2026-09-10):

    messages                              prompt_tokens   reply
    user "What is the capital of France?"       10         "The capital of France is Paris. It is not only..."
    system <one-word rule> + same user          10         byte-identical to the above
    developer <one-word rule> + same user       10         byte-identical to the above
    same rule inlined into the user turn        21         "Paris"
    system 1000 tok + user 1000 tok           1010         == user-only, so system was never counted

So the content is not merely ignored by the model, it never reaches it: the
token counter proves it was dropped in transport. The model obeys the identical
instruction perfectly from the `user` role.

This is not a cosmetic problem for this project. DSPy's ChatAdapter puts the
signature instructions, every field definition, the `Literal[...]` label
enumeration and our format contract in the **system** message. A model that
drops it receives a bare unlabelled sentence: no task, no label set, no
`[[ ## field ## ]]` markers. The row would score near chance with a 100% parse
failure rate, and would look like a weak model rather than a broken transport.

litellm cannot express this. `supports_system_messages: False` plus
`modify_params = True` was verified NOT to merge on a generic `openai/` route
(prompt_tokens stayed at 10), because the folding logic only runs for providers
with a bespoke prompt template. Hence a wrapper, following the same precedent
as `ollama_lm.OllamaLM`.

The fold is content-preserving: same text, same order, one role. It is still a
deviation from what every other row on the chart sends, so any model needing it
carries the fact in its `notes` and should carry it on the chart too.
"""

from __future__ import annotations

from typing import Any

import dspy

# Blank line, not a header. A header ("System:", "Instructions:") would be new
# text the optimizer never wrote and the other rows never saw, and on a format
# contract that says "never add or rename a field" it is exactly the kind of
# stray token that talks a small model into wrapping its answer.
JOIN = "\n\n"


def fold_system_into_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Move `system`/`developer` content into the first user turn.

    Order is preserved: system content leads, then the user's own text. If
    there is no user turn at all the system content becomes one, so a
    system-only request still says something rather than nothing.
    """
    carried = [m["content"] for m in messages
               if m.get("role") in ("system", "developer") and m.get("content")]
    if not carried:
        return messages

    rest = [m for m in messages if m.get("role") not in ("system", "developer")]
    prefix = JOIN.join(carried)

    for i, m in enumerate(rest):
        if m.get("role") == "user":
            merged = dict(m)
            merged["content"] = f"{prefix}{JOIN}{m.get('content', '')}".rstrip()
            return rest[:i] + [merged] + rest[i + 1:]

    return [{"role": "user", "content": prefix}] + rest


class SystemRoleFoldingLM(dspy.LM):
    """`dspy.LM` that rewrites messages before dispatch.

    Both `forward` and `aforward` are overridden: DSPy picks between them by
    call path, and folding only the sync one would leave an async run silently
    sending the prompt the endpoint discards -- the exact failure this exists
    to prevent, reintroduced on a path nobody tests.
    """

    def forward(self, prompt: str | None = None,
                messages: list[dict[str, Any]] | None = None, **kwargs):
        if messages:
            messages = fold_system_into_user(messages)
        return super().forward(prompt=prompt, messages=messages, **kwargs)

    async def aforward(self, prompt: str | None = None,
                       messages: list[dict[str, Any]] | None = None, **kwargs):
        if messages:
            messages = fold_system_into_user(messages)
        return await super().aforward(prompt=prompt, messages=messages, **kwargs)
