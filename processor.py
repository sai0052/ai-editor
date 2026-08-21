"""
Core processing logic for the AI silence/filler-word editor.

Pipeline:
1. Transcribe the media with word-level timestamps (faster-whisper).
2. Find "junk" spans: long silences between words + filler words (um, uh, like...),
   each independently toggleable by the caller.
3. Invert junk spans against the full timeline to get the spans to keep.
4. Cut the keep spans out with ffmpeg and concatenate them back together.
"""

import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field

from faster_whisper import WhisperModel

# ---- Config -----------------------------------------------------------

FILLER_WORDS = {
    "um", "umm", "uh", "uhh", "erm", "er",
    "like", "you know", "i mean", "sort of", "kind of", "basically",
}

DEFAULT_SILENCE_GAP_THRESHOLD = 0.6   # seconds of silence between words to cut
PADDING = 0.06                        # seconds of breathing room kept around each spoken word

# "ultrafast" trades a larger output file size for much quicker encoding —
# worth it here since users care more about turnaround time than file size.
FFMPEG_PRESET = "ultrafast"

_MODEL = None


def get_model(size: str = "tiny"):
    """Lazily load the whisper model once per process."""
    global _MODEL
    if _MODEL is None:
        # compute_type="int8" keeps this usable on CPU-only machines.
        _MODEL = WhisperModel(size, device="cpu", compute_type="int8")
    return _MODEL


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class ProcessingResult:
    output_path: str
    transcript: str
    removed_seconds: float
    kept_seconds: float
    removed_fillers: list = field(default_factory=list)


def transcribe(input_path: str, model_size: str = "tiny") -> tuple[list[Word], float]:
    """Run whisper with word-level timestamps. Returns (words, total_duration)."""
    model = get_model(model_size)
    segments, info = model.transcribe(
        input_path,
        word_timestamps=True,
        vad_filter=True,  # skip pure-silence stretches during decoding
    )

    words: list[Word] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            words.append(Word(text=w.word.strip().lower(), start=w.start, end=w.end))

    duration = info.duration if info and info.duration else (words[-1].end if words else 0.0)
    return words, duration


def _is_filler(word_text: str) -> bool:
    cleaned = word_text.strip(".,!?").lower()
    return cleaned in FILLER_WORDS


def find_keep_segments(
    words: list[Word],
    total_duration: float,
    remove_fillers: bool = True,
    remove_silences: bool = True,
    silence_threshold: float = DEFAULT_SILENCE_GAP_THRESHOLD,
):
    """
    Build a list of spans to CUT (filler words, if enabled, and long silences,
    if enabled), merge overlapping cuts, then invert against the full timeline
    to get the spans to KEEP. Returns (keep_segments, removed_fillers_metadata).

    Both remove_fillers and remove_silences can be toggled independently:
    - both True: original full behavior
    - remove_silences False: keeps natural pauses, only strips filler words
    - remove_fillers False: keeps filler words, only strips long dead air
    - both False: returns the whole clip untouched (single keep segment)
    """
    cut_spans = []
    removed_fillers = []
    prev_end = 0.0

    for w in words:
        if remove_fillers and _is_filler(w.text):
            cut_spans.append((w.start, w.end))
            removed_fillers.append({"word": w.text, "start": w.start, "end": w.end})

        gap = w.start - prev_end
        if remove_silences and gap > silence_threshold:
            cut_spans.append((prev_end, w.start))

        prev_end = max(prev_end, w.end)

    # Trailing dead air after the last word — only trimmed if silence removal is on.
    if remove_silences and words:
        trailing_gap = total_duration - prev_end
        if trailing_gap > silence_threshold:
            cut_spans.append((prev_end, total_duration))

    # Merge overlapping/adjacent cut spans.
    cut_spans.sort()
    merged_cuts = []
    for s, e in cut_spans:
        if merged_cuts and s <= merged_cuts[-1][1]:
            merged_cuts[-1] = (merged_cuts[-1][0], max(merged_cuts[-1][1], e))
        else:
            merged_cuts.append((s, e))

    # Shrink each cut inward by PADDING so we don't clip right up against speech.
    padded_cuts = []
    for s, e in merged_cuts:
        s2, e2 = s + PADDING, e - PADDING
        if e2 > s2:
            padded_cuts.append((s2, e2))

    # Invert the cuts against [0, total_duration] to get what to keep.
    keep = []
    cursor = 0.0
    for s, e in padded_cuts:
        if s > cursor:
            keep.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < total_duration:
        keep.append((cursor, total_duration))

    if not keep:
        # Cuts covered the entire clip (e.g. very aggressive settings on a short
        # file) — fall back to keeping everything rather than an empty video.
        keep = [(0.0, total_duration)]

    return keep, removed_fillers


def cut_and_concat(input_path: str, keep_segments: list[tuple], output_path: str):
    """Use ffmpeg to extract each keep segment and concatenate them."""
    if not keep_segments:
        raise ValueError("Nothing left to keep — check your silence/filler thresholds.")

    with tempfile.TemporaryDirectory() as tmp:
        clip_paths = []
        for i, (start, end) in enumerate(keep_segments):
            clip_path = os.path.join(tmp, f"clip_{i:04d}.mp4")
            duration = max(0.01, end - start)
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-i", input_path,
                "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                clip_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            clip_paths.append(clip_path)

        # Build ffmpeg concat list file
        list_path = os.path.join(tmp, "concat_list.txt")
        with open(list_path, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p}'\n")

        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path,
        ]
        subprocess.run(concat_cmd, check=True, capture_output=True)


def remap_words_to_new_timeline(
    words: list[Word], keep_segments: list[tuple], remove_fillers: bool = True
) -> list[Word]:
    """
    The cut/concatenated video has a new, shorter timeline. Take the
    original words (excluding filler words only if they were actually
    removed) and figure out where they land in that new timeline, so
    captions line up with the edited video instead of the original.
    """
    remapped = []
    cumulative = 0.0

    for seg_start, seg_end in keep_segments:
        seg_len = seg_end - seg_start
        for w in words:
            if remove_fillers and _is_filler(w.text):
                continue
            # Word falls inside this keep segment if it overlaps it.
            if w.start >= seg_start and w.start < seg_end:
                new_start = cumulative + max(0.0, w.start - seg_start)
                new_end = cumulative + min(seg_len, w.end - seg_start)
                remapped.append(Word(text=w.text, start=new_start, end=new_end))
        cumulative += seg_len

    return remapped


def _format_srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def build_srt(remapped_words: list[Word], words_per_caption: int = 6, max_gap: float = 1.2) -> str:
    """Group remapped words into short caption cues and render as an .srt string."""
    if not remapped_words:
        return ""

    cues = []
    current: list[Word] = []

    for w in remapped_words:
        if current and (
            len(current) >= words_per_caption
            or (w.start - current[-1].end) > max_gap
        ):
            cues.append(current)
            current = []
        current.append(w)
    if current:
        cues.append(current)

    lines = []
    for i, cue in enumerate(cues, start=1):
        start = _format_srt_timestamp(cue[0].start)
        end = _format_srt_timestamp(cue[-1].end)
        text = " ".join(w.text for w in cue)
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    return "\n".join(lines)


def _escape_ffmpeg_path(path: str) -> str:
    """ffmpeg's subtitles filter needs forward slashes and an escaped colon
    (Windows drive letters like C:\\ trip up the filter graph parser otherwise)."""
    normalized = path.replace("\\", "/")
    normalized = normalized.replace(":", "\\:")
    return normalized


def burn_captions(video_path: str, srt_path: str, output_path: str):
    """Burn an .srt file into the video as hardcoded (visible) captions."""
    escaped = _escape_ffmpeg_path(srt_path)
    style = (
        "FontName=Arial,FontSize=14,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,"
        "Alignment=2,MarginV=30"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles='{escaped}':force_style='{style}'",
        "-preset", FFMPEG_PRESET,
        "-c:a", "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def process_file(
    input_path: str,
    output_dir: str,
    model_size: str = "tiny",
    add_captions: bool = True,
    remove_fillers: bool = True,
    remove_silences: bool = True,
    silence_threshold: float = DEFAULT_SILENCE_GAP_THRESHOLD,
) -> ProcessingResult:
    os.makedirs(output_dir, exist_ok=True)
    job_id = uuid.uuid4().hex[:8]
    cut_output_path = os.path.join(output_dir, f"cut_{job_id}.mp4")

    words, duration = transcribe(input_path, model_size=model_size)
    keep_segments, removed_fillers = find_keep_segments(
        words,
        duration,
        remove_fillers=remove_fillers,
        remove_silences=remove_silences,
        silence_threshold=silence_threshold,
    )
    cut_and_concat(input_path, keep_segments, cut_output_path)

    kept_seconds = sum(end - start for start, end in keep_segments)
    transcript = " ".join(
        w.text for w in words if not (remove_fillers and _is_filler(w.text))
    )

    final_output_path = cut_output_path

    if add_captions:
        remapped_words = remap_words_to_new_timeline(
            words, keep_segments, remove_fillers=remove_fillers
        )
        srt_text = build_srt(remapped_words)
        if srt_text:
            srt_path = os.path.join(output_dir, f"captions_{job_id}.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_text)

            captioned_path = os.path.join(output_dir, f"edited_{job_id}.mp4")
            burn_captions(cut_output_path, srt_path, captioned_path)
            final_output_path = captioned_path

    return ProcessingResult(
        output_path=final_output_path,
        transcript=transcript,
        removed_seconds=round(duration - kept_seconds, 2),
        kept_seconds=round(kept_seconds, 2),
        removed_fillers=removed_fillers,
    )