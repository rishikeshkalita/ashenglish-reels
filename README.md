# Ashenglish Reels Generator

Streamlit app for Assamese reels subtitling with a strict Assamese-first pipeline:

- Upload a video
- Extract and preprocess 16kHz mono audio with FFmpeg
- Transcribe Assamese speech into Assamese script only
- Romanize Assamese into Ashenglish
- Review and edit subtitle lines before final render
- Export styled MP4 plus SRT

## Accuracy Rules

- Accuracy is prioritized over speed
- Assamese is transcribed in Assamese script first
- The app does not translate Assamese into English for subtitle generation
- Roman Assamese is phonetic transliteration, not translation
- Exact corrections are only applied from user-provided rules
- If the model is uncertain, the editable review step is where you fix subtitle text before burn-in

## Models

- Primary ASR: `openai/whisper-large-v3`

The app currently defaults to Whisper `medium` for free hosting, and you can override it with the `ASHENGLISH_WHISPER_MODEL` environment variable.

Recommended defaults:

- Free CPU hosting: `medium`
- Stronger paid/GPU hosting: `large-v3`

## Features

- FFmpeg audio preprocessing
  - 16kHz mono conversion
  - noise reduction
  - normalization
- Reels-style subtitle grouping
  - 2 to 4 words per line
  - word-timestamp-based chunking
- Manual subtitle review table
- Styled overlay video
  - `Montserrat`
  - bright yellow text
  - black outline
  - center-bottom placement
- Downloads
  - final MP4
  - SRT
  - JSON subtitle data

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install FFmpeg and ensure `ffmpeg` and `ffprobe` are available on `PATH`.

## Run

```powershell
streamlit run app.py
```

Optional:

```powershell
$env:ASHENGLISH_WHISPER_MODEL="medium"
streamlit run app.py
```

Use `medium` on weaker machines. Keep `large-v3` for the highest local accuracy.

## Free Hosting Recommendation

For a free public launch, use:

- GitHub for the code
- Hugging Face Spaces with `Docker`
- Free CPU hardware
- Default model: `medium`

This is the most realistic free setup for this app. It avoids load on your laptop, but heavy videos may still be slow on free CPU.

## Public Launch

### Option 1: Docker host

This repo already includes a `Dockerfile`.

```powershell
docker build -t ashenglish-reels .
docker run -p 8501:8501 ashenglish-reels
```

### Option 2: Hugging Face Spaces

1. Push this repo to GitHub.
2. Create a new Docker Space on Hugging Face.
3. Point the Space at the repo.
4. Upgrade hardware if `large-v3` is too slow on CPU.

Files to upload:

- `app.py`
- `requirements.txt`
- `README.md`
- `Dockerfile`
- `packages.txt`
- `runtime.txt`
- `.dockerignore`
- `.streamlit/config.toml`

### Option 3: Render / Railway

1. Create a new Docker web service.
2. Connect the repo.
3. Expose port `8501`.
4. Start command is already handled by the `Dockerfile`.

## Workflow

1. Upload the Assamese video.
2. Add Assamese vocabulary hints if needed.
3. Add optional exact correction rules in the form `wrong => correct`.
4. Click `Generate Assamese Draft`.
5. Review Assamese text and Roman Assamese output.
6. Edit the `Final Subtitle` column if anything is wrong.
7. Click `Render Edited Reel`.

## Notes

- Whisper large-v3 is heavy and may be slow on CPU.
- Background music removal is limited to lightweight audio filtering in this version; very noisy videos may still need manual subtitle cleanup.
- For public traffic, GPU-backed hosting is strongly recommended if you keep `large-v3`.

## Market-Ready Next Steps

For a later paid version, the best next upgrades are:

- user accounts
- saved projects
- subtitle history
- team workspaces
- watermark-free premium export
- subscription billing
- queued background jobs instead of one-request processing
