"""Fixed-window rolling transcription for Qwen3-ASR's vLLM wrapper.

Unlike Qwen3ASRModel.streaming_transcribe(), this class never sends an
ever-growing waveform to the model. Each decode receives only the most recent
``window_seconds`` of 16 kHz mono PCM. A short committed transcript is kept
outside the model request; the current window's transcript remains revisable.

The decoder interface is intentionally small so the windowing and transcript
stitching can be tested without loading a large ASR model.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import time
import unicodedata
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


SAMPLE_RATE = 16_000
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\s]")


@dataclass(frozen=True)
class WindowUpdate:
    """Snapshot emitted after one bounded-window decode."""

    text: str
    window_text: str
    audio_start_sec: float
    audio_end_sec: float
    window_duration_sec: float
    decode_wall_sec: float
    final: bool = False


Decoder = Callable[[np.ndarray, int, Optional[str], str], str]


def _spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    matches = list(_TOKEN_RE.finditer(text))
    return [m.group(0) for m in matches], [(m.start(), m.end()) for m in matches]


def _join_text(left: str, right: str) -> str:
    """Join two transcript fragments without damaging English/CJK spacing."""
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    if unicodedata.category(right[0]).startswith("P"):
        return left + right
    if left[-1].isspace() or right[0].isspace():
        return left + right
    # Chinese, Japanese and Korean text is conventionally written without
    # spaces between adjacent characters. Keep a separator only between two
    # Latin/digit fragments.
    if _is_cjk(left[-1]) or _is_cjk(right[0]):
        return left + right
    if left[-1].isalnum() and right[0].isalnum():
        return left + " " + right
    if left[-1] in ",.!?;:%" and right[0].isalnum():
        return left + " " + right
    return left + right


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x3400 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0x3040 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
    )


class FixedWindowStreamingASR:
    """Re-decode a bounded rolling audio window and maintain a growing transcript.

    Args:
        model: Qwen3ASRModel instance using the vLLM backend.
        language: Optional forced language, e.g. ``"English"`` or ``"Chinese"``.
        window_seconds: Maximum audio duration sent in any ASR request.
        update_seconds: New audio duration between normal decode updates.
        context: Optional fixed textual context. This is passed unchanged to
            every request and must not contain the accumulating transcript.
        decoder: Optional test/custom decoder. Signature is
            ``decoder(audio_float32, sample_rate, language, context) -> text``.

    ``feed`` accepts arbitrary-sized chunks and returns zero or more updates.
    ``finish`` decodes the final tail and returns a final update. The ``text``
    property is the complete rolling transcript, including committed history.
    """

    def __init__(
        self,
        model=None,
        *,
        language: Optional[str] = None,
        window_seconds: float = 30.0,
        update_seconds: float = 2.0,
        context: str = "",
        decoder: Optional[Decoder] = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Qwen3-ASR expects {SAMPLE_RATE} Hz audio, got {sample_rate}")
        if window_seconds <= 0 or update_seconds <= 0:
            raise ValueError("window_seconds and update_seconds must be positive")
        if update_seconds > window_seconds:
            raise ValueError("update_seconds must not exceed window_seconds")
        if decoder is None and model is None:
            raise ValueError("Provide either a Qwen model or a decoder callback")

        self.model = model
        self.language = language
        self.context = context or ""
        self.sample_rate = sample_rate
        self.window_samples = int(round(window_seconds * sample_rate))
        self.hop_samples = int(round(update_seconds * sample_rate))
        if self.window_samples < 1 or self.hop_samples < 1:
            raise ValueError("window_seconds and update_seconds are too small")
        self.decoder = decoder or self._decode_qwen

        self._ring = np.empty((0,), dtype=np.float32)
        self._pending = np.empty((0,), dtype=np.float32)
        self._total_samples = 0
        self._previous_start_sample: Optional[int] = None
        self._previous_window_text = ""
        self._committed_text = ""
        self._text = ""
        self._closed = False
        self._final_update: Optional[WindowUpdate] = None
        self.fallback_alignment_count = 0
        self.repeated_overlap_insertions_removed = 0
        self.decode_count = 0

    @property
    def text(self) -> str:
        """Current complete transcript; the trailing window can be revised."""
        return self._text

    @property
    def buffered_audio_seconds(self) -> float:
        return len(self._ring) / self.sample_rate

    @property
    def buffered_audio_samples(self) -> int:
        return len(self._ring)

    @property
    def audio_seconds(self) -> float:
        return self._total_samples / self.sample_rate + len(self._pending) / self.sample_rate

    def feed(self, pcm16k: np.ndarray) -> List[WindowUpdate]:
        """Add audio and decode every complete update interval."""
        if self._closed:
            raise RuntimeError("Cannot feed audio after finish()")
        if pcm16k is None:
            raise ValueError("pcm16k must not be None")
        x = np.asarray(pcm16k)
        if x.ndim > 1:
            if x.shape[-1] == 1:
                x = x.reshape(-1)
            else:
                raise ValueError("Expected mono audio; downmix stereo before calling feed()")
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)

        updates: List[WindowUpdate] = []
        cursor = 0
        while cursor < len(x):
            needed = self.hop_samples - len(self._pending)
            take = min(needed, len(x) - cursor)
            self._pending = np.concatenate((self._pending, x[cursor : cursor + take]))
            cursor += take
            if len(self._pending) == self.hop_samples:
                self._append_audio(self._pending)
                self._pending = np.empty((0,), dtype=np.float32)
                updates.append(self._decode_update(final=False))
        return updates

    def warmup(self, seconds: float = 1.0) -> float:
        """Run one discarded request to absorb first-inference initialization.

        Call this after model construction and before opening the microphone if
        first-partial latency matters. The dummy audio does not enter the live
        ring buffer or transcript.
        """
        if self._closed:
            raise RuntimeError("Cannot warm up after finish()")
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        samples = np.zeros(int(round(seconds * self.sample_rate)), dtype=np.float32)
        started = time.perf_counter()
        self.decoder(samples, self.sample_rate, self.language, self.context)
        return time.perf_counter() - started

    def finish(self) -> WindowUpdate:
        """Flush any partial hop, perform a final bounded decode, and close."""
        if self._final_update is not None:
            return self._final_update
        if self._closed:
            raise RuntimeError("Stream is already closed")
        had_pending_audio = bool(len(self._pending))
        if had_pending_audio:
            self._append_audio(self._pending)
            self._pending = np.empty((0,), dtype=np.float32)
        self._closed = True
        if len(self._ring) == 0:
            self._final_update = WindowUpdate(
                text=self._text,
                window_text="",
                audio_start_sec=0.0,
                audio_end_sec=0.0,
                window_duration_sec=0.0,
                decode_wall_sec=0.0,
                final=True,
            )
        elif not had_pending_audio and self._previous_start_sample is not None:
            # The latest complete update already decoded exactly this audio.
            # Mark it final without paying for a duplicate full-window pass.
            self._final_update = WindowUpdate(
                text=self._text,
                window_text=self._previous_window_text,
                audio_start_sec=max(0, self._total_samples - len(self._ring)) / self.sample_rate,
                audio_end_sec=self._total_samples / self.sample_rate,
                window_duration_sec=len(self._ring) / self.sample_rate,
                decode_wall_sec=0.0,
                final=True,
            )
        else:
            self._final_update = self._decode_update(final=True)
        return self._final_update

    def _append_audio(self, samples: np.ndarray) -> None:
        n = len(samples)
        self._total_samples += n
        if n >= self.window_samples:
            self._ring = np.array(samples[-self.window_samples :], dtype=np.float32, copy=True)
        else:
            self._ring = np.concatenate((self._ring, samples))[-self.window_samples :].copy()

    def _decode_qwen(
        self, audio: np.ndarray, sample_rate: int, language: Optional[str], context: str
    ) -> str:
        result = self.model.transcribe(
            audio=(audio, sample_rate),
            context=context,
            language=language,
        )
        if not result:
            return ""
        return str(result[0].text or "")

    def _decode_update(self, *, final: bool) -> WindowUpdate:
        audio = self._ring
        start_sample = self._total_samples - len(audio)
        start = start_sample / self.sample_rate
        end = self._total_samples / self.sample_rate

        t0 = time.perf_counter()
        window_text = self.decoder(audio, self.sample_rate, self.language, self.context).strip()
        decode_wall = time.perf_counter() - t0
        self.decode_count += 1

        if self._previous_start_sample is None or start_sample <= self._previous_start_sample:
            # The window is still growing from time zero; revise the whole
            # transcript rather than committing any part of it.
            self._text = _join_text(self._committed_text, window_text)
        else:
            delta_samples = start_sample - self._previous_start_sample
            old_duration = max(1, self._total_samples - delta_samples - self._previous_start_sample)
            expected_fraction = delta_samples / old_duration
            window_text = self._remove_spurious_overlap_insertions(
                self._previous_window_text,
                window_text,
                expected_fraction=expected_fraction,
            )
            dropped = self._select_dropped_prefix(
                self._previous_window_text,
                window_text,
                expected_fraction=expected_fraction,
            )
            self._committed_text = _join_text(self._committed_text, dropped)
            self._text = _join_text(self._committed_text, window_text)

        self._previous_start_sample = start_sample
        self._previous_window_text = window_text
        return WindowUpdate(
            text=self._text,
            window_text=window_text,
            audio_start_sec=start,
            audio_end_sec=end,
            window_duration_sec=len(audio) / self.sample_rate,
            decode_wall_sec=decode_wall,
            final=final,
        )

    def _select_dropped_prefix(
        self, old_text: str, new_text: str, *, expected_fraction: float
    ) -> str:
        """Find the old-window prefix that has slid out of the next window.

        Qwen does not return word timestamps. We therefore align successive
        window hypotheses and commit the old prefix before the strongest early
        overlap. If alignment is too weak, estimate the dropped token count
        from the audio-window shift and mark the fallback for diagnostics.
        """
        old_tokens, old_spans = _spans(old_text)
        new_tokens, _ = _spans(new_text)
        if not old_tokens:
            return ""
        if not new_tokens:
            # If the model emits nothing for this window, preserve the previous
            # transcript rather than silently deleting already-heard words.
            return old_text

        expected = max(0.0, min(1.0, expected_fraction)) * len(old_tokens)
        old_keys = [x.casefold() for x in old_tokens]
        new_keys = [x.casefold() for x in new_tokens]
        matcher = SequenceMatcher(a=old_keys, b=new_keys, autojunk=False)
        early_limit = max(2, min(len(new_tokens) // 3 + 1, 12))
        candidates = [
            block
            for block in matcher.get_matching_blocks()
            if block.size >= min(3, len(old_tokens), len(new_tokens))
            and block.b <= early_limit
        ]
        if candidates:
            block = max(
                candidates,
                key=lambda b: b.size - 0.55 * b.b - 0.45 * abs(b.a - expected),
            )
            cut = block.a
            # A repeated phrase can also sit inside the aligned block itself:
            # Qwen may restart with the suffix of a phrase that was partly
            # outside the new audio window. Compare against a bounded lookahead
            # at the new window's left edge, even when block.b == 0.
            new_lookahead = max(block.b, max(12, int(len(new_tokens) * 0.28)))
            overlap = self._suffix_prefix_overlap(
                old_tokens[:cut],
                new_tokens[: min(len(new_tokens), new_lookahead)],
            )
            cut -= overlap
            return self._token_prefix(old_text, old_tokens, old_spans, cut)

        self.fallback_alignment_count += 1
        drop_count = int(round(expected))
        if drop_count <= 0:
            return ""
        new_lead_count = min(len(new_tokens), max(3, int(round(expected * 2))))
        overlap = self._suffix_prefix_overlap(
            [x.casefold() for x in old_tokens[:drop_count]],
            [x.casefold() for x in new_tokens[:new_lead_count]],
        )
        return self._token_prefix(old_text, old_tokens, old_spans, drop_count - overlap)

    @staticmethod
    def _token_prefix(text: str, tokens: Sequence[str], spans: Sequence[Tuple[int, int]], count: int) -> str:
        count = max(0, min(count, len(tokens)))
        if count == 0:
            return ""
        if count == len(tokens):
            return text.rstrip()
        return text[: spans[count][0]].rstrip()

    @staticmethod
    def _suffix_prefix_overlap(old_tokens: Sequence[str], new_tokens: Sequence[str]) -> int:
        """Length of an exact repeated tail/head token sequence; ignore singletons."""
        old_trailing_punctuation = 0
        while old_trailing_punctuation < len(old_tokens) and unicodedata.category(
            old_tokens[-old_trailing_punctuation - 1][-1]
        ).startswith("P"):
            old_trailing_punctuation += 1
        new_leading_punctuation = 0
        while new_leading_punctuation < len(new_tokens) and unicodedata.category(
            new_tokens[new_leading_punctuation][0]
        ).startswith("P"):
            new_leading_punctuation += 1
        old_core = old_tokens[: len(old_tokens) - old_trailing_punctuation]
        new_core = new_tokens[new_leading_punctuation:]
        limit = min(len(old_core), len(new_core))
        for count in range(limit, 1, -1):
            if list(old_core[-count:]) == list(new_core[:count]):
                return count + old_trailing_punctuation
        return 0

    def _remove_spurious_overlap_insertions(
        self, old_text: str, new_text: str, *, expected_fraction: float
    ) -> str:
        """Drop repeated phrases newly hallucinated near a sliding-window edge.

        Windowed audio can begin mid-word or mid-phrase. Qwen occasionally
        repeats a phrase at that left edge even though the previous overlapping
        decode contained it only once. We compare successive hypotheses and
        remove only exact insertions that repeat nearby overlap text; insertions
        close to the new right edge are retained as possible new speech.
        """
        old_tokens, _ = _spans(old_text)
        new_tokens, new_spans = _spans(new_text)
        if len(old_tokens) < 3 or len(new_tokens) < 3:
            return new_text

        old_keys = [x.casefold() for x in old_tokens]
        new_keys = [x.casefold() for x in new_tokens]
        matcher = SequenceMatcher(a=old_keys, b=new_keys, autojunk=False)
        # Only inspect the earlier overlap portion. The rightmost portion is
        # where newly arrived audio should add genuinely new words.
        left_limit = max(3, min(len(new_tokens) - 1, int(len(new_tokens) * 0.28)))
        removals: List[Tuple[int, int]] = []
        for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
            if tag != "insert" or new_start > left_limit or new_end - new_start < 2:
                continue
            phrase = new_keys[new_start:new_end]
            lookback = max(32, len(phrase) * 3)
            earlier_new = new_keys[max(0, new_start - lookback) : new_start]
            earlier_old = old_keys[max(0, old_start - lookback) : old_start]
            if self._contains_subsequence(earlier_new, phrase) or self._contains_subsequence(earlier_old, phrase):
                removals.append((new_spans[new_start][0], new_spans[new_end - 1][1]))

        if not removals:
            return new_text
        pieces = []
        cursor = 0
        for start, end in removals:
            pieces.append(new_text[cursor:start])
            cursor = end
        pieces.append(new_text[cursor:])
        cleaned = "".join(pieces)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
        self.repeated_overlap_insertions_removed += len(removals)
        return cleaned

    @staticmethod
    def _contains_subsequence(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
        width = len(phrase)
        return any(list(tokens[i : i + width]) == list(phrase) for i in range(len(tokens) - width + 1))


__all__ = ["FixedWindowStreamingASR", "WindowUpdate", "SAMPLE_RATE"]
