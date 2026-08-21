"""
Core processing logic for the AI silence/filler-word editor.

Pipeline:
1. Transcribe the media with word-level timestamps (faster-whisper).
2. Find "junk" spans: long silences between words + filler words (um, uh, like...).
3. Invert junk spans into "keep" spans.
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

SILENCE_GAP_THRESHOLD = 0.6   # seconds of silence between words to cut
PADDING = 0.06                # seconds of breathing room kept around each spoken word

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


def find_keep_segments(words: list[Word], total_duration: float):
    """
    Walk through words, drop filler words and long silent gaps,
    and return a list of (start, end) tuples to keep, plus metadata
    about what was cut.
    """
    keep = []
    removed_fillers = []
    cursor = 0.0

    for i, w in enumerate(words):
        gap = w.start - cursor
        is_filler = _is_filler(w.text)

        if is_filler:
            removed_fillers.append({"word": w.text, "start": w.start, "end": w.end})
            # Skip this word entirely; move cursor past it.
            cursor = w.end
            continue

        if gap > SILENCE_GAP_THRESHOLD:
            # Cut the silence: keep up to `cursor`, then jump to just before this word.
            if cursor > 0:
                keep.append((max(0.0, cursor - PADDING), cursor + PADDING))
            start = max(0.0, w.start - PADDING)
            end = w.end + PADDING
            keep.append((start, end))
        else:
            # Extend / merge into the previous keep segment.
            if keep and keep[-1][1] >= w.start - PADDING:
                keep[-1] = (keep[-1][0], w.end + PADDING)
            else:
                keep.append((max(0.0, w.start - PADDING), w.end + PADDING))

        cursor = w.end

    # Trailing silence after the last word gets dropped automatically
    # (we simply don't extend `keep` past the last word's end).

    # Merge any overlapping/adjacent segments produced above.
    keep.sort()
    merged = []
    for seg in keep:
        if merged and seg[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)

    return merged, removed_fillers


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


def remap_words_to_new_timeline(words: list[Word], keep_segments: list[tuple]) -> list[Word]:
    """
    The cut/concatenated video has a new, shorter timeline. Take the
    original (non-filler) words and figure out where they land in that
    new timeline, so captions line up with the edited video instead of
    the original.
    """
    remapped = []
    cumulative = 0.0

    for seg_start, seg_end in keep_segments:
        seg_len = seg_end - seg_start
        for w in words:
            if _is_filler(w.text):
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
) -> ProcessingResult:
    os.makedirs(output_dir, exist_ok=True)
    job_id = uuid.uuid4().hex[:8]
    cut_output_path = os.path.join(output_dir, f"cut_{job_id}.mp4")

    words, duration = transcribe(input_path, model_size=model_size)
    keep_segments, removed_fillers = find_keep_segments(words, duration)
    cut_and_concat(input_path, keep_segments, cut_output_path)

    kept_seconds = sum(end - start for start, end in keep_segments)
    transcript = " ".join(w.text for w in words if not _is_filler(w.text))

    final_output_path = cut_output_path

    if add_captions:
        remapped_words = remap_words_to_new_timeline(words, keep_segments)
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