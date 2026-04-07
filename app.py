from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st
import whisper
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}
FONT_NAME = "Montserrat"
SUBTITLE_COLOR_HEX = "&H00FFFF00"
OUTLINE_COLOR_HEX = "&H00000000"
DEFAULT_WRAP_WIDTH = 24
DEFAULT_MARGIN_V = 120
WHISPER_MODEL_NAME = os.getenv("ASHENGLISH_WHISPER_MODEL", "medium")
MAX_WORDS_PER_LINE = 4
MIN_WORDS_PER_LINE = 2
MAX_GAP_SECONDS = 0.55
MAX_UPLOAD_MB = 200
LOW_CONFIDENCE_THRESHOLD = 0.55
DICTIONARY_PATH = Path(__file__).with_name("assamese_dictionary.txt")
CORRECTIONS_PATH = Path(__file__).with_name("corrections.json")


@dataclass
class WordCue:
    start: float
    end: float
    text: str


@dataclass
class SubtitleLine:
    start: float
    end: float
    assamese_text: str
    romanized_text: str
    confidence: float
    review_status: str
    edited_text: str = ""


def normalize_assamese_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?।])", r"\1", text)
    return text


def normalize_romanized_text(text: str) -> str:
    cleaned = " ".join(text.split())
    replacements = {
        "x": "h",
        "X": "H",
        "v": "w",
        "V": "W",
        "~n": "ny",
        "~N": "ng",
        "aa": "a",
        "ii": "i",
        "uu": "u",
    }
    for src, dest in replacements.items():
        cleaned = cleaned.replace(src, dest)
    cleaned = re.sub(r"\s+([?.!,])", r"\1", cleaned)
    return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned


def romanize_assamese_text(text: str) -> str:
    romanized = transliterate(text, sanscript.BENGALI, sanscript.OPTITRANS)
    return normalize_romanized_text(romanized)


def format_ass_timestamp(seconds: float) -> str:
    total_cs = max(0, round(seconds * 100))
    hours, remainder = divmod(total_cs, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def format_srt_timestamp(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def escape_ass_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def build_wrapped_ass_text(text: str, width: int) -> str:
    wrapped_lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return r"\N".join(escape_ass_text(line) for line in wrapped_lines) or escape_ass_text(text)


def parse_vocab_hints(raw_text: str) -> list[str]:
    return [normalize_assamese_text(line) for line in raw_text.splitlines() if normalize_assamese_text(line)]


def parse_corrections(raw_text: str) -> dict[str, str]:
    corrections: dict[str, str] = {}
    for line in raw_text.splitlines():
        if "=>" not in line:
            continue
        wrong, right = line.split("=>", 1)
        wrong = normalize_assamese_text(wrong)
        right = normalize_assamese_text(right)
        if wrong and right:
            corrections[wrong] = right
    return corrections


def parse_dictionary_text(raw_text: str) -> set[str]:
    return {
        normalize_assamese_text(line)
        for line in raw_text.splitlines()
        if normalize_assamese_text(line)
    }


def load_dictionary_words() -> set[str]:
    if not DICTIONARY_PATH.exists():
        return set()
    return parse_dictionary_text(DICTIONARY_PATH.read_text(encoding="utf-8"))


def load_correction_rules_file() -> dict[str, str]:
    if not CORRECTIONS_PATH.exists():
        return {}
    try:
        data = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    corrections: dict[str, str] = {}
    for wrong, right in data.items():
        wrong_clean = normalize_assamese_text(str(wrong))
        right_clean = normalize_assamese_text(str(right))
        if wrong_clean and right_clean:
            corrections[wrong_clean] = right_clean
    return corrections


def load_uploaded_dictionary(uploaded_file) -> set[str]:
    if uploaded_file is None:
        return set()
    return parse_dictionary_text(uploaded_file.getvalue().decode("utf-8", errors="ignore"))


def load_uploaded_corrections(uploaded_file) -> dict[str, str]:
    if uploaded_file is None:
        return {}
    try:
        data = json.loads(uploaded_file.getvalue().decode("utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return {}
    corrections: dict[str, str] = {}
    for wrong, right in data.items():
        wrong_clean = normalize_assamese_text(str(wrong))
        right_clean = normalize_assamese_text(str(right))
        if wrong_clean and right_clean:
            corrections[wrong_clean] = right_clean
    return corrections


def apply_strict_corrections(text: str, corrections: dict[str, str]) -> str:
    corrected = normalize_assamese_text(text)
    for wrong, right in sorted(corrections.items(), key=lambda item: len(item[0]), reverse=True):
        corrected = corrected.replace(wrong, right)
    return corrected


def dictionary_match_ratio(text: str, dictionary_words: set[str]) -> float:
    words = [normalize_assamese_text(word) for word in text.split() if normalize_assamese_text(word)]
    if not words:
        return 0.0
    matched = sum(1 for word in words if word in dictionary_words)
    return matched / len(words)


def finalize_review_status(confidence: float, dictionary_ratio: float) -> str:
    if confidence < LOW_CONFIDENCE_THRESHOLD or dictionary_ratio < 0.5:
        return "Review needed"
    return "Looks good"


def assamese_validity_score(text: str) -> float:
    if not text:
        return 0.0
    assamese_chars = len(re.findall(r"[\u0980-\u09FF]", text))
    total_chars = len(text)
    return assamese_chars / total_chars if total_chars else 0.0


@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return whisper.load_model(WHISPER_MODEL_NAME)


def ensure_ffmpeg_exists() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg was not found on PATH.")


def extract_and_preprocess_audio(video_path: Path, audio_path: Path) -> None:
    ensure_ffmpeg_exists()
    filters = ",".join(
        [
            "highpass=f=120",
            "lowpass=f=7600",
            "afftdn",
            "dynaudnorm",
            "loudnorm",
        ]
    )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        filters,
        str(audio_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Audio preprocessing failed.")


def transcribe_audio(audio_path: Path, vocab_hints: list[str], corrections: dict[str, str]) -> list[WordCue]:
    model = load_whisper_model()
    initial_prompt = " ".join(vocab_hints[:25]) if vocab_hints else None
    result = model.transcribe(
        str(audio_path),
        language="as",
        task="transcribe",
        word_timestamps=True,
        fp16=False,
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
    )

    word_cues: list[WordCue] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            text = apply_strict_corrections(word.get("word", "").strip(), corrections)
            text = normalize_assamese_text(text)
            start = word.get("start")
            end = word.get("end")
            if text and start is not None and end is not None:
                word_cues.append(WordCue(start=float(start), end=float(end), text=text))
    return word_cues


def group_word_cues(words: list[WordCue]) -> list[list[WordCue]]:
    lines: list[list[WordCue]] = []
    current: list[WordCue] = []
    for word in words:
        if not current:
            current = [word]
            continue
        gap = word.start - current[-1].end
        should_break = (
            len(current) >= MAX_WORDS_PER_LINE
            or gap > MAX_GAP_SECONDS
            or current[-1].text.endswith(("।", ".", "!", "?"))
        )
        if should_break and len(current) >= MIN_WORDS_PER_LINE:
            lines.append(current)
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(current)
    return lines


def build_subtitle_lines(
    audio_path: Path,
    vocab_hints: list[str],
    corrections: dict[str, str],
    dictionary_words: set[str],
) -> list[SubtitleLine]:
    word_cues = transcribe_audio(audio_path, vocab_hints, corrections)
    if not word_cues:
        return []

    subtitles: list[SubtitleLine] = []
    for grouped_words in group_word_cues(word_cues):
        assamese_text = normalize_assamese_text(" ".join(word.text for word in grouped_words))
        confidence = max(0.1, min(0.99, assamese_validity_score(assamese_text)))
        dictionary_ratio = dictionary_match_ratio(assamese_text, dictionary_words)
        romanized_text = romanize_assamese_text(assamese_text)
        subtitles.append(
            SubtitleLine(
                start=grouped_words[0].start,
                end=grouped_words[-1].end,
                assamese_text=assamese_text,
                romanized_text=romanized_text,
                confidence=confidence,
                review_status=finalize_review_status(confidence, dictionary_ratio),
                edited_text=romanized_text,
            )
        )
    return subtitles


def write_ass_file(subtitles: list[SubtitleLine], output_path: Path, wrap_width: int, margin_v: int) -> None:
    ass_lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: ReelSub,{FONT_NAME},72,{SUBTITLE_COLOR_HEX},{SUBTITLE_COLOR_HEX},"
            f"{OUTLINE_COLOR_HEX},&H64000000,-1,0,0,0,100,100,0,0,1,3,0,2,90,90,{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for subtitle in subtitles:
        rendered_text = subtitle.edited_text or subtitle.romanized_text
        ass_lines.append(
            "Dialogue: 0,"
            f"{format_ass_timestamp(subtitle.start)},"
            f"{format_ass_timestamp(subtitle.end)},"
            f"ReelSub,,0,0,0,,{build_wrapped_ass_text(rendered_text, wrap_width)}"
        )
    output_path.write_text("\n".join(ass_lines), encoding="utf-8")


def write_srt_file(subtitles: list[SubtitleLine], output_path: Path) -> None:
    lines = []
    for index, subtitle in enumerate(subtitles, start=1):
        rendered_text = subtitle.edited_text or subtitle.romanized_text
        lines.extend(
            [
                str(index),
                f"{format_srt_timestamp(subtitle.start)} --> {format_srt_timestamp(subtitle.end)}",
                rendered_text,
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_burned_video(source_video: Path, subtitles_ass: Path, output_video: Path) -> None:
    ensure_ffmpeg_exists()
    subtitles_path = subtitles_ass.as_posix().replace(":", r"\:")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_video),
        "-vf",
        f"ass='{subtitles_path}'",
        "-c:a",
        "copy",
        str(output_video),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "FFmpeg failed to render subtitles.")


def save_uploaded_video(uploaded_file, workspace: Path) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise ValueError("Please upload an MP4, MOV, M4V, or WEBM video.")
    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Please upload a video smaller than {MAX_UPLOAD_MB} MB.")
    video_path = workspace / f"input{suffix}"
    video_path.write_bytes(uploaded_file.getbuffer())
    return video_path


def build_editor_dataframe(subtitles: list[SubtitleLine]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Start": f"{line.start:.2f}",
                "End": f"{line.end:.2f}",
                "Assamese": line.assamese_text,
                "Roman Assamese": line.romanized_text,
                "Confidence": f"{line.confidence:.2f}",
                "Review": line.review_status,
                "Final Subtitle": line.edited_text or line.romanized_text,
            }
            for line in subtitles
        ]
    )


def restore_subtitles_from_editor(editor_df: pd.DataFrame) -> list[SubtitleLine]:
    subtitles: list[SubtitleLine] = []
    for row in editor_df.to_dict(orient="records"):
        assamese_text = normalize_assamese_text(str(row["Assamese"]))
        romanized_text = normalize_romanized_text(str(row["Roman Assamese"]))
        final_text = " ".join(str(row["Final Subtitle"]).split())
        subtitles.append(
            SubtitleLine(
                start=float(row["Start"]),
                end=float(row["End"]),
                assamese_text=assamese_text,
                romanized_text=romanized_text,
                confidence=float(row["Confidence"]),
                review_status=str(row.get("Review", "Review needed")),
                edited_text=final_text or romanized_text,
            )
        )
    return subtitles


def reset_state_for_new_upload(uploaded_name: str) -> None:
    previous_name = st.session_state.get("uploaded_name")
    if previous_name != uploaded_name:
        for key in (
            "draft_editor_df",
            "rendered_video_bytes",
            "rendered_srt_text",
            "subtitle_json",
            "draft_render_settings",
        ):
            st.session_state.pop(key, None)
        st.session_state["uploaded_name"] = uploaded_name


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(255, 214, 10, 0.16), transparent 30%),
                radial-gradient(circle at bottom left, rgba(0, 0, 0, 0.18), transparent 35%),
                linear-gradient(180deg, #0d0d0f 0%, #18181d 55%, #101114 100%);
            color: #f6f6ef;
        }
        .block-container {
            max-width: 1080px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def persist_render_results(output_video: Path, srt_path: Path, subtitles: list[SubtitleLine]) -> None:
    st.session_state["rendered_video_bytes"] = output_video.read_bytes()
    st.session_state["rendered_srt_text"] = srt_path.read_text(encoding="utf-8")
    st.session_state["subtitle_json"] = json.dumps(
        [
            {
                "start": line.start,
                "end": line.end,
                "assamese": line.assamese_text,
                "roman_assamese": line.romanized_text,
                "final_subtitle": line.edited_text,
                "confidence": line.confidence,
            }
            for line in subtitles
        ],
        ensure_ascii=False,
        indent=2,
    )


def render_editor_panel() -> pd.DataFrame | None:
    draft_df = st.session_state.get("draft_editor_df")
    if draft_df is None:
        return None
    st.subheader("Review And Edit Ashenglish")
    edited_df = st.data_editor(
        draft_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=["Start", "End", "Assamese", "Confidence", "Review"],
    )
    st.session_state["draft_editor_df"] = edited_df
    review_needed_count = int((edited_df["Review"] == "Review needed").sum()) if "Review" in edited_df else 0
    if review_needed_count:
        st.warning(f"{review_needed_count} subtitle lines are flagged for manual review.")
    else:
        st.success("All subtitle lines passed the current dictionary/confidence checks.")
    return edited_df


def main() -> None:
    st.set_page_config(page_title="Ashenglish Reels Generator", page_icon="🎬", layout="wide")
    inject_custom_css()

    st.title("Ashenglish Reels Generator")
    st.caption("Assamese-only transcription first, Roman Assamese subtitles second, manual subtitle correction before final render.")

    with st.sidebar:
        st.header("Render Controls")
        wrap_width = st.slider("Subtitle width", min_value=12, max_value=30, value=DEFAULT_WRAP_WIDTH)
        margin_v = st.slider("Bottom margin", min_value=80, max_value=220, value=DEFAULT_MARGIN_V, step=10)
        st.caption(f"ASR model: `{WHISPER_MODEL_NAME}`")
        st.caption(f"Upload limit: `{MAX_UPLOAD_MB} MB`")
        custom_dictionary_file = st.file_uploader(
            "Optional custom dictionary",
            type=["txt"],
            help="Upload a larger Assamese dictionary text file to improve consistency for this session.",
        )
        custom_corrections_file = st.file_uploader(
            "Optional corrections JSON",
            type=["json"],
            help="Upload exact correction mappings for this session.",
        )

    vocab_hints_text = st.text_area("Assamese vocabulary hints", placeholder="One Assamese word or phrase per line")
    corrections_text = st.text_area("Strict correction rules", placeholder="wrong word => correct word")

    uploaded_file = st.file_uploader("Upload Assamese MP4 for reels subtitles", type=["mp4", "mov", "m4v", "webm"])
    if uploaded_file is None:
        st.info("Upload a video to start.")
        return

    reset_state_for_new_upload(uploaded_file.name)
    st.video(uploaded_file)

    if st.button("Generate Assamese Draft", type="primary", use_container_width=True):
        with tempfile.TemporaryDirectory(prefix="ashenglish_draft_") as tmp_dir:
            workspace = Path(tmp_dir)
            try:
                vocab_hints = parse_vocab_hints(vocab_hints_text)
                corrections = {
                    **load_correction_rules_file(),
                    **load_uploaded_corrections(custom_corrections_file),
                    **parse_corrections(corrections_text),
                }
                dictionary_words = {
                    *load_dictionary_words(),
                    *load_uploaded_dictionary(custom_dictionary_file),
                    *parse_dictionary_text(vocab_hints_text),
                }

                input_video = save_uploaded_video(uploaded_file, workspace)
                audio_path = workspace / "preprocessed.wav"
                with st.spinner("Extracting and preprocessing audio..."):
                    extract_and_preprocess_audio(input_video, audio_path)
                with st.spinner(f"Transcribing Assamese speech with Whisper {WHISPER_MODEL_NAME}..."):
                    subtitles = build_subtitle_lines(audio_path, vocab_hints, corrections, dictionary_words)

                if not subtitles:
                    st.error("No Assamese speech could be transcribed from this video.")
                    return

                st.session_state["draft_editor_df"] = build_editor_dataframe(subtitles)
                st.session_state["draft_render_settings"] = {
                    "uploaded_file_name": uploaded_file.name,
                    "uploaded_file_bytes": uploaded_file.getvalue(),
                    "wrap_width": wrap_width,
                    "margin_v": margin_v,
                }
                st.success("Draft ready. Review and edit the subtitle lines below.")
            except Exception as exc:
                st.error(str(exc))

    edited_df = render_editor_panel()
    draft_settings = st.session_state.get("draft_render_settings")
    if edited_df is not None and draft_settings:
        if st.button("Render Edited Reel", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory(prefix="ashenglish_render_") as tmp_dir:
                workspace = Path(tmp_dir)
                try:
                    suffix = Path(draft_settings["uploaded_file_name"]).suffix.lower() or ".mp4"
                    input_video = workspace / f"input{suffix}"
                    input_video.write_bytes(draft_settings["uploaded_file_bytes"])

                    subtitles = restore_subtitles_from_editor(edited_df)
                    ass_path = workspace / "subtitles.ass"
                    srt_path = workspace / "subtitles.srt"
                    output_video = workspace / "ashenglish_reel.mp4"

                    write_ass_file(subtitles, ass_path, int(draft_settings["wrap_width"]), int(draft_settings["margin_v"]))
                    write_srt_file(subtitles, srt_path)
                    with st.spinner("Burning styled subtitles into the final reel..."):
                        render_burned_video(input_video, ass_path, output_video)

                    persist_render_results(output_video, srt_path, subtitles)
                    st.success("Rendered reel ready.")
                except Exception as exc:
                    st.error(str(exc))

    rendered_video_bytes = st.session_state.get("rendered_video_bytes")
    rendered_srt_text = st.session_state.get("rendered_srt_text")
    subtitle_json = st.session_state.get("subtitle_json")
    if rendered_video_bytes:
        st.subheader("Final Output")
        st.video(rendered_video_bytes)
        st.download_button("Download MP4", data=rendered_video_bytes, file_name="ashenglish_reel.mp4", mime="video/mp4", use_container_width=True)
        if rendered_srt_text:
            st.download_button("Download SRT", data=rendered_srt_text, file_name="ashenglish_reel.srt", mime="application/x-subrip", use_container_width=True)
        if subtitle_json:
            st.download_button("Download JSON", data=subtitle_json, file_name="ashenglish_reel.json", mime="application/json", use_container_width=True)


if __name__ == "__main__":
    main()
