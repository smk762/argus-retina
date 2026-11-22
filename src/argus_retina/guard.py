"""SingGuard's request and generation, without the model: what ``argus_retina.singguard`` sends and reads back.

Kept apart from the server so it is tested without torch or transformers. Moved from argus-cull's ``singguard/``
(``inputs.py`` and ``parse.py``), with agent-composer's text-only checks, conversations and prompt-injection guards
(``rag-chat/app/guard_policy.py``), so one server takes both callers' requests (argus-retina#9).

**The generation.** The model writes a verdict on the first line and the matched rule inside an ``<answer>`` tag. In
``fast-slow`` the first line is the fast first-token signal and the tag is the final judgment after reasoning. The
tag decides; a ``[Step 3] Final Judgment`` that concludes safe or unsafe stands when the tag is missing; the first
line only stands in when neither is present. In ``slow`` there is no first line at all (the server prefills the
first step heading), and the explanation starts at ``[Step 1]``. A generation with neither a verdict line, a Final
Judgment, nor a tag is a failed check, not a pass. An ``<answer>`` that is neither Safe nor a title in the active
policy is an unreadable check: unsafe, with no rule, so it is not recorded as the rule that fired. When no policy
text was sent, the titles are SingGuard's own headings.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import re
import warnings
from typing import Any

log = logging.getLogger("argus_retina.singguard")

# fast: a verdict and rule. fast-slow: SingGuard's own mode, a first-line verdict, then the reasoning. slow: this
# server's variant of fast-slow with the first line left out (see SLOW_PREFILL). cascade: fast, then slow for what fast
# flagged; slow's verdict stands. The same four modes as argus_cortex.moderation.THINKING_MODES.
THINKING = ("fast", "fast-slow", "slow", "cascade")
ROLES = ("user", "assistant")

# In fast-slow the model writes the verdict before it has reasoned, and its reasoning then defers to that line ("the
# hint says unsafe, so unsafe") even when its own Step 2 found no rule matched. Slow prefills the assistant turn with
# the first step heading, so the first thing the model decides is the <answer> at the end.
SLOW_PREFILL = "[Step 1] Content Summary\n"

# A request's image: the base64 text, then the decoded pixel count. 80 MP takes a 61 MP camera frame and stays under
# Pillow's own bomb warning (about 89 MP).
MAX_ENCODED_CHARS = 64 * 1024 * 1024
MAX_PIXELS = 80_000_000
# What the model sees: argus_cortex's SingGuardClient sends at most this edge already. The processor would otherwise
# take up to 16 MP, about 16k visual tokens, and a large original could exhaust the GPU.
MAX_EDGE = 1536

# --- the request ---------------------------------------------------------------------------------------------------


def decode_image(encoded: str, max_pixels: int = MAX_PIXELS, max_edge: int = MAX_EDGE) -> Any:
    """The request's image (raw base64 or a data URL), upright and at most *max_edge* px on its long side.

    Raises ValueError for anything that is not a bounded, single-frame image.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError  # noqa: PLC0415 - in the singguard image

    text = encoded.strip()
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    if len(text) > MAX_ENCODED_CHARS:
        raise ValueError("image_base64 is too large")
    try:
        raw = base64.b64decode("".join(text.split()), validate=True)
        with warnings.catch_warnings():
            # Pillow only warns between its limit and twice it; treat that as refused too.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(raw))
            width, height = image.size
            if width * height > max_pixels:
                raise ValueError(f"image is {width}x{height}, over {max_pixels} pixels")
            if getattr(image, "n_frames", 1) > 1:
                # A verdict on frame 0 would not describe the frames after it: the caller splits them.
                raise ValueError(f"image has {image.n_frames} frames; SingGuard judges one")
            if image.format == "JPEG":
                image.draft("RGB", (max_edge, max_edge))  # decode at a reduced scale when it can
            image.load()
    except (
        binascii.Error,
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("image_base64 is not an image") from exc
    image = ImageOps.exif_transpose(image)  # as a viewer sees it
    image.thumbnail((max_edge, max_edge))
    return image


# ``<|im_start|>``, ``<|im_end|>``, ``<|vision_start|>`` and friends. The tokenizer turns one written in the text into
# the control token, so it would close the user turn and let the text forge an assistant verdict, a new system block
# or an image slot. The bars may also be look-alikes: DeepSeek's control tokens use the fullwidth bar.
_BARS = "|｜￨∣"  # | ｜ ￨ ∣
_SPECIAL_TOKEN = re.compile(rf"<[{_BARS}]([^<>{_BARS}\n]{{0,64}})[{_BARS}]>")
# SingGuard's template writes the conversation as one transcript, a ``[user]: `` or ``[assistant]: `` line per turn:
# a line opening with one in a message's text would read as a new turn.
_ROLE_MARKER = re.compile(r"^(\s*)\[\s*(user|assistant|system)\s*\]\s*:", re.IGNORECASE | re.MULTILINE)


def neutralize(text: str) -> str:
    """*text* with control tokens and role markers made plain characters: the model still reads what was written.

    ``<|im_end|>`` becomes ``< |im_end| >`` (fullwidth bars would not be inert: they are DeepSeek's own control-token
    form) and ``[assistant]:`` at the start of a line becomes ``(assistant):``. The same rules as agent-composer's
    ``guard_policy.neutralize_special_tokens`` and ``neutralize_role_markers``.
    """
    text = _SPECIAL_TOKEN.sub(lambda m: f"< |{m.group(1)}| >", text)
    return _ROLE_MARKER.sub(lambda m: f"{m.group(1)}({m.group(2)}):", text)


def messages(
    text: str,
    *,
    role: str = "user",
    conversation: list[tuple[str, str]] | None = None,
    image: Any = None,
) -> list[dict[str, Any]]:
    """The chat messages for one check: *conversation*'s turns, then *text* as *role*, every text neutralized.

    The image sits on a user turn, so it is judged with the prompt beside it: the new turn when *role* is ``user``,
    else the latest user turn in *conversation* (a model's reply about an image is judged in the context of the prompt
    and the image), else a user turn of its own before the reply, which keeps its assistant role. Raises ValueError
    for a check with nothing to judge.
    """
    if role not in ROLES:
        raise ValueError(f"input_role must be one of {', '.join(ROLES)}")
    turns = [(r, c) for r, c in conversation or []]
    for r, _ in turns:
        if r not in ROLES:
            raise ValueError(f"conversation roles must be one of {', '.join(ROLES)}")
    has_user = any(r == "user" for r, _ in turns)
    image_at: int | None = None
    if image is not None and role == "user":
        turns.append(("user", text))
        image_at = len(turns) - 1
    else:
        if image is not None and not has_user:  # a reply about an image with no prompt: the image on its own turn
            turns.append(("user", ""))
            image_at = len(turns) - 1
        if text.strip() or image is None:
            turns.append((role, text))
    if image is not None and image_at is None:
        image_at = max(i for i, (r, _) in enumerate(turns) if r == "user")
    if image is None and not any(c.strip() for _, c in turns):
        raise ValueError("there is nothing to check: send input text or image_base64")
    chat: list[dict[str, Any]] = []
    for i, (r, c) in enumerate(turns):
        content: list[dict[str, Any]] = []
        if i == image_at:
            content.append({"type": "image", "image": image})
        if c.strip() or not content:
            content.append({"type": "text", "text": neutralize(c.strip())})
        chat.append({"role": r, "content": content})
    return chat


def template_kwargs(thinking: str, policy: str | None) -> dict[str, str]:
    """The chat template's ``thinking_type`` and ``policy``. Slow is fast-slow's prompt with the first step prefilled."""
    kwargs = {"thinking_type": "fast-slow" if thinking == "slow" else thinking}
    if policy and policy.strip():
        kwargs["policy"] = neutralize(policy.strip())
    return kwargs


# --- the generation ------------------------------------------------------------------------------------------------

_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_HEADING = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
# SingGuard's own headings. A default check sends no policy text, so these are the titles an <answer> may name when the
# request did not bring its own.
DEFAULT_TITLES = (
    "A. Sexual Content Risk",
    "B. Real-World Crimes & Public Safety",
    "C. Unethical Behavior",
    "D. Cybersecurity & Information Manipulation",
    "E. Agent Safety",
    "F. Politically Sensitive Content",
    "G. Animal Abuse",
)
_UNREADABLE = (
    "SingGuard's <answer> is not Safe and not a title in the active policy. "
    "The check is unsafe so an unreadable result is not treated as safe."
)
_LATE_VERDICT_LINE = re.compile(r"^\s*\**\s*(safe|unsafe)\s*\**\s*[.!]?\s*$", re.IGNORECASE)
_STEP3_HEADING = re.compile(r"^\[step 3\]", re.IGNORECASE)
_CONCLUDE = re.compile(
    r"\b(?:therefore[,:]?\s+)?(?:(?:the\s+)?(?:content|image)\s+is|"
    r"it\s+is(?:\s+classified\s+as)?|classified\s+as|conclude(?:s|d)?|"
    r"final judgment[:\s]+)\s+(safe|unsafe)\b",
    re.IGNORECASE,
)
# A 2048-token slow write-up is about 8k characters; argus_cortex keeps 16384 too.
EXPLANATION_MAX = 16384
# Minimum length of a shortened title ("B. Real" for "B. Real-World Crimes & Public Safety").
_MIN_SHORT_TITLE = 4
# What may follow a full title when the answer runs on past it ("B. Crimes, since ...").
_RUN_ON = re.compile(r"\s*[,;:(\-–—.]")


class NoVerdict(ValueError):
    """The generation carried no verdict."""


# The first line's verdict: the whole first word after any "Safety:" or "Verdict:" label, not a prefix of it.
_FIRST_WORD = re.compile(r"[\W_]*(?:(?:safety|verdict)\s*:)?[\W_]*([a-z]+)")


def _first_line_verdict(raw: str) -> bool | None:
    lines = raw.strip().splitlines()
    if not lines:
        return None
    first = lines[0].strip().lower()
    if "unsafe" in first:  # anywhere on the line: "Safety: Unsafe" is not a pass
        return False
    word = _FIRST_WORD.match(first)
    return True if word and word.group(1) == "safe" else None  # "safety concern", "safeguard": no verdict


def _late_judgment(raw: str) -> bool | None:
    """Verdict after reasoning: a standalone safe/unsafe line in Step 3, else its prose."""
    found: bool | None = None
    concluded: bool | None = None
    in_step3 = False
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _STEP3_HEADING.match(stripped):
            in_step3 = True
            continue
        if not in_step3:
            continue
        late = _LATE_VERDICT_LINE.match(stripped)
        if late:
            found = late.group(1).lower() == "safe"
            continue
        conclude = _CONCLUDE.search(stripped)
        if conclude:
            concluded = conclude.group(1).lower() == "safe"
    return found if found is not None else concluded


def _titles(policy: str | None) -> list[str]:
    """Rule headings an unsafe ``<answer>`` may name. No policy text means SingGuard's own.

    ``### Safe`` is not a rule, and as a title any prose opening with "safe" would match it.
    """
    text = (policy or "").strip()
    if not text:
        return list(DEFAULT_TITLES)
    headings = (match.group(1).strip() for match in _HEADING.finditer(text))
    return [heading for heading in headings if heading.casefold() != "safe"]


def _rule_title(answer: str, policy: str | None) -> str | None:
    """The rule to record, or None when *answer* is not a title of *policy*.

    A shortened heading matches (``B. Real-World Crimes`` for the full title) when it is cut at a word boundary and at
    least ``_MIN_SHORT_TITLE`` characters long, and the caller's wording is kept for that. An exact match uses the
    heading, as does an answer that runs on past it after punctuation (``..., since``), so prose is never recorded as a
    rule. A fragment ("B", "No"), a cut mid-word, or an answer that fits more than one heading names no rule: it is
    unreadable, not whichever heading came first. The same rule as agent-composer's ``guard_policy._lookup_rule``.
    """
    wanted = answer.casefold()
    if not wanted:
        return None
    titles = _titles(policy)
    for title in titles:
        if title.casefold() == wanted:
            return title
    candidates: list[str] = []
    for title in titles:
        name = title.casefold()
        shortened = len(wanted) >= _MIN_SHORT_TITLE and name.startswith(wanted) and not name[len(wanted)].isalnum()
        runs_on = wanted.startswith(name) and _RUN_ON.match(wanted, len(name)) is not None
        if shortened:
            candidates.append(answer)
        elif runs_on:
            candidates.append(title)
    return candidates[0] if len(candidates) == 1 else None


def parse_generation(text: str, policy: str | None = None) -> tuple[bool, str, str, bool | None]:
    """Return ``(safe, rule, explanation, fast_safe)`` from one generation.

    ``fast_safe`` is the first-line verdict, or None when the first line was not one. Raises :class:`NoVerdict` when
    there is neither an ``<answer>`` tag, a Step 3 conclusion, nor a first-line verdict, and when an unsafe verdict names
    no rule. An ``<answer>`` that is neither Safe nor a title in the active policy is unsafe with an empty rule:
    unreadable, not a named rule, and not a pass.
    """
    raw = text.strip()
    if not raw:
        raise NoVerdict("SingGuard returned an empty generation")
    fast_safe = _first_line_verdict(raw)
    match = _ANSWER.search(raw)
    answer = " ".join(match.group(1).split()) if match else ""
    # The explanation is everything before the tag, less the verdict line when there is one.
    body = raw[: match.start()] if match else raw
    if fast_safe is not None:
        body = body.split("\n", 1)[1] if "\n" in body else ""
    explanation = body.strip()[:EXPLANATION_MAX]
    if answer:
        safe = answer.casefold() == "safe"
    elif (late := _late_judgment(raw)) is not None:
        safe = late
    elif fast_safe is not None:
        safe = fast_safe
    else:
        raise NoVerdict("SingGuard returned no verdict")
    if fast_safe is not None and fast_safe != safe:
        log.info("fast verdict %s overturned by answer %r", "safe" if fast_safe else "unsafe", answer)
    if safe:
        return True, "Safe", explanation, fast_safe
    if not answer:
        raise NoVerdict("SingGuard returned an unsafe verdict with no rule")
    rule = _rule_title(answer, policy)
    if rule is None:
        explanation = f"{_UNREADABLE}\n\n{explanation}" if explanation else _UNREADABLE
        return False, "", explanation, fast_safe
    return False, rule, explanation, fast_safe


def to_response(text: str, model: str, elapsed_ms: int, policy: str | None = None) -> dict[str, Any]:
    """The ``/api/guard/check`` body ``argus_cortex.moderation.SingGuardClient`` parses (``thinking`` added by the
    server)."""
    safe, rule, explanation, fast_safe = parse_generation(text, policy)
    categories: list[dict[str, str]] = []
    if not safe and rule:
        categories.append({"code": rule, "label": rule, "why": explanation or "-"})
    return {
        "safe": safe,
        "verdict": "safe" if safe else "unsafe",
        "rule": rule,
        "explanation": explanation,
        "categories": categories,
        # The first-line signal, kept so a reader can see when reasoning overturned it.
        "fast_verdict": None if fast_safe is None else ("safe" if fast_safe else "unsafe"),
        "model": model,
        "elapsed_ms": elapsed_ms,
    }
