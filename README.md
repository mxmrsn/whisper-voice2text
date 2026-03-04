# Whisper Dictation Utility

A lightweight local dictation utility using `faster-whisper`.

## Requirements
- NVIDIA GPU (NVIDIA RTX PRO 2000 detected)
- CUDA 11+ and cuDNN 8/9 binaries (handled automatically by the virtual environment)
- Python 3.10+

## Usage
1. Open a terminal in this folder.
2. Run the utility:
   ```powershell
   .\venv\Scripts\python.exe dictate.py
   ```
3. Wait for the "Ready!" and "Model test successful!" messages.
4. **Hold `Ctrl + Alt`** while speaking.
5. **Release `Ctrl + Alt`** to transcribe. Transcription is hardware-accelerated via your GPU.

## Configuration
You can edit `dictate.py` to change:
- `MODEL_SIZE`: "tiny", "base", "small", "medium", "large-v3" (default is "base")
- `HOTKEY`: Change the activation keys.
- `DEVICE`: "cuda" or "cpu"
