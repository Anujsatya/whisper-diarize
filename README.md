# Whisper Diarize

A small CLI script that transcribes a video/audio file and labels who said what, by combining [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (speech-to-text) with [pyannote.audio](https://github.com/pyannote/pyannote-audio) (speaker diarization).

Useful for turning a meeting recording into a readable, speaker-attributed transcript without needing any cloud transcription service.

## How it works

1. Transcribes the input file with `faster-whisper` (runs on CPU, `int8` for speed).
2. Extracts mono 16kHz audio with `ffmpeg` and runs `pyannote/speaker-diarization-3.1` over it to detect speaker turns.
3. Merges the two: each transcribed segment is attributed to whichever speaker had the most overlap with it.
4. Writes a timestamped, speaker-labeled transcript to `~/Downloads/<input_filename>_transcript.txt`.

## Requirements

- Python 3.10+
- [`ffmpeg`](https://ffmpeg.org/) on your `PATH`
- A free Hugging Face account + access token with access granted to [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) (accept the model's terms on the model page first)

```bash
pip install -r requirements.txt
```

## Usage

```bash
python transcribe.py <video_or_audio_file> <your_huggingface_token> [model_size]
```

- `model_size` — any faster-whisper model size (`tiny`, `base`, `small`, `medium` (default), `large-v3`, etc.). Bigger = more accurate, slower on CPU.
- Your Hugging Face token is passed as a plain argument — it is **never** read from a file or stored by this script. Treat your shell history accordingly (e.g. pass it via an env var and `$MY_HF_TOKEN` if you don't want it in `history`).

Example:

```bash
python transcribe.py meeting.mp4 hf_xxxxxxxxxxxx small
```

A 1-hour recording takes roughly ~30 minutes to transcribe plus ~10 minutes to diarize on CPU; a GPU will be dramatically faster (swap `device="cpu"` for `device="cuda"` in `transcribe_with_diarization`).

## Output format

```
File: meeting.mp4
Duration: 42:17
======================================================================

[00:00] SPEAKER_00:
Hey, thanks for joining — let's get started.

[00:04] SPEAKER_01:
Sounds good, I pulled up the doc.
```

## License

MIT
