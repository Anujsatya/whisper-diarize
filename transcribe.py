import sys
import os
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass

# ── torchaudio compatibility shims (APIs removed in 2.3+) ──────────────────
import torchaudio

if not hasattr(torchaudio, 'AudioMetaData'):
    @dataclass
    class _AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str
    torchaudio.AudioMetaData = _AudioMetaData

if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: "soundfile"

if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda *a, **kw: None
# ───────────────────────────────────────────────────────────────────────────

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_dominant_speaker(seg_start, seg_end, speaker_segments):
    overlap = {}
    for sp_start, sp_end, spk in speaker_segments:
        o_start = max(seg_start, sp_start)
        o_end = min(seg_end, sp_end)
        if o_end > o_start:
            overlap[spk] = overlap.get(spk, 0) + (o_end - o_start)
    if not overlap:
        return "SPEAKER_UNKNOWN"
    return max(overlap, key=overlap.get)


def transcribe_with_diarization(input_path, hf_token, model_size="medium"):
    input_file = Path(input_path)
    output_path = Path.home() / "Downloads" / (input_file.stem + "_transcript.txt")

    print(f"Loading Whisper model '{model_size}' (CPU mode)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print("Transcribing... (~30 min for a 1-hour meeting on CPU)")
    segments_gen, info = model.transcribe(input_path, beam_size=5, word_timestamps=False)
    segments = list(segments_gen)
    print(f"Detected language: {info.language} | Duration: {format_time(info.duration)}")

    print("Extracting audio for diarization (ffmpeg)...")
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-vn", tmp_wav.name],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    print("Running speaker diarization (pyannote, ~10 min)...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    diarization = pipeline(tmp_wav.name)
    os.unlink(tmp_wav.name)

    speaker_segments = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    print(f"Writing transcript to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"File: {input_file.name}\n")
        f.write(f"Duration: {format_time(info.duration)}\n")
        f.write("=" * 70 + "\n\n")

        current_speaker = None
        current_lines = []
        current_start = None

        for seg in segments:
            speaker = get_dominant_speaker(seg.start, seg.end, speaker_segments)
            text = seg.text.strip()
            if not text:
                continue
            if speaker != current_speaker:
                if current_lines and current_speaker:
                    f.write(f"[{format_time(current_start)}] {current_speaker}:\n")
                    f.write(" ".join(current_lines) + "\n\n")
                current_speaker = speaker
                current_lines = [text]
                current_start = seg.start
            else:
                current_lines.append(text)

        if current_lines and current_speaker:
            f.write(f"[{format_time(current_start)}] {current_speaker}:\n")
            f.write(" ".join(current_lines) + "\n\n")

    print(f"\nDone! Saved to:\n  {output_path}")
    return str(output_path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python transcribe.py <video.mp4> <hf_token>")
        sys.exit(1)

    input_path = sys.argv[1]
    hf_token = sys.argv[2]
    model_size = sys.argv[3] if len(sys.argv) > 3 else "medium"

    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    transcribe_with_diarization(input_path, hf_token, model_size)
