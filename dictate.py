import sounddevice as sd
import numpy as np
import keyboard
import pyautogui
from faster_whisper import WhisperModel
import io
from scipy.io.wavfile import write
import sys
import os

# CUDA/cuDNN Path Setup for Windows
def setup_cuda_paths():
    if sys.platform == "win32":
        # Get the path to the current virtual environment's site-packages
        venv_base = os.path.dirname(os.path.dirname(sys.executable))
        site_packages = os.path.join(venv_base, "Lib", "site-packages")
        nvidia_base = os.path.join(site_packages, "nvidia")
        
        if os.path.exists(nvidia_base):
            # Check for various nvidia library subfolders
            for lib in ["cudnn", "cublas"]:
                bin_path = os.path.join(nvidia_base, lib, "bin")
                if os.path.exists(bin_path):
                    print(f"Adding {bin_path} to DLL search path...")
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                    try:
                        os.add_dll_directory(bin_path)
                    except Exception as e:
                        print(f"Error adding {bin_path}: {e}")

setup_cuda_paths()

# Configuration
MODEL_SIZE = "base"
DEVICE = "cuda" 
COMPUTE_TYPE = "float16" # float16 is faster on GPU
HOTKEY = "ctrl+alt"

def load_model():
    try:
        # Try loading on GPU first
        print(f"Attempting to load Whisper model '{MODEL_SIZE}' on {DEVICE}...")
        return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"GPU loading failed or library missing: {e}")
        print("Falling back to CPU for reliable transcription...")
        return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

model = load_model()

# Test the model immediately to catch DLL errors before starting the main loop
print("Testing model with silent audio...")
try:
    # Transcribe 1 second of silence
    model.transcribe(np.zeros(16000, dtype=np.float32))
    print("Model test successful!")
except Exception as e:
    print(f"Model test failed: {e}")
    if "cudnn" in str(e).lower() or "cuda" in str(e).lower():
        print("Switching to CPU due to late-loading library error...")
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

print(f"Ready! Press and hold '{HOTKEY}' to record. Release to transcribe and type.")

recording = []
is_recording = False

def callback(indata, frames, time, status):
    if is_recording:
        recording.append(indata.copy())

def start_recording():
    global is_recording, recording
    if not is_recording:
        print("Recording...")
        recording = []
        is_recording = True

def stop_recording():
    global is_recording
    if is_recording:
        is_recording = False
        print("Stopped recording. Transcribing...")
        process_recording()

def process_recording():
    global model
    if not recording:
        return
    
    # Concatenate all recorded chunks
    audio_data = np.concatenate(recording, axis=0)
    audio_data = audio_data.flatten()
    
    try:
        segments, info = model.transcribe(audio_data, beam_size=5)
        text = "".join(segment.text for segment in segments).strip()
    except Exception as e:
        print(f"Transcription failed: {e}")
        if "cudnn" in str(e).lower() or "cuda" in str(e).lower():
            print("CUDA error detected during transcription. Falling back to CPU...")
            model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            segments, info = model.transcribe(audio_data, beam_size=5)
            text = "".join(segment.text for segment in segments).strip()
        else:
            raise e
    
    if text:
        print(f"Transcribed: {text}")
        pyautogui.write(text + " ")
    else:
        print("No speech detected.")

# Setup InputStream
fs = 16000 # Whisper expects 16kHz
with sd.InputStream(samplerate=fs, channels=1, callback=callback):
    while True:
        # Wait for the hotkey
        if keyboard.is_pressed(HOTKEY):
            start_recording()
            while keyboard.is_pressed(HOTKEY):
                sd.sleep(10)
            stop_recording()
        sd.sleep(50)
