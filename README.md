# whisper Dictation Utility

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
3. Wait for the "whisper" window to show "Status: Ready".
4. **Hold `Ctrl + Alt`** while speaking. The window will turn red and status will change to "Recording...".
5. **Release `Ctrl + Alt`** to transcribe. The text will be typed out and copied to your clipboard automatically.
6. **History**: You can see recent snippets in the history list and click the 📋 button to re-copy them.
7. **Close the window** to exit the program.

## Shortcuts
You can create a project-local shortcut by running:
```powershell
powershell.exe -ExecutionPolicy Bypass -File create_shortcut.ps1
```
The shortcut will be created in the `build/` folder as `whisper.lnk`.

## Configuration
You can edit `dictate.py` to change:
- `MODEL_SIZE`: "tiny", "base", "small", "medium", "large-v3" (default is "large-v3")
- `HOTKEY`: Change the activation keys.
- `DEVICE`: "cuda" or "cpu"
