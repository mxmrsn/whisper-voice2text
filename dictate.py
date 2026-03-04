import sounddevice as sd
import numpy as np
import keyboard
import pyautogui
from faster_whisper import WhisperModel
import io
from scipy.io.wavfile import write
import sys
import os
import tkinter as tk
from tkinter import messagebox
import threading

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

root = None
app = None

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
        # Add to history and clipboard
        if app:
            root.after(0, lambda: app.add_history_item(text))
            root.after(0, lambda: app.copy_to_clipboard(text))
    else:
        print("No speech detected.")

# Setup InputStream
fs = 16000 # Whisper expects 16kHz

class WhisperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("whisper")
        self.root.geometry("400x400")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Header
        self.header_frame = tk.Frame(root)
        self.header_frame.pack(fill="x", pady=5)
        
        self.label = tk.Label(self.header_frame, text="whisper", font=("Helvetica", 14, "bold"))
        self.label.pack()
        
        self.status_label = tk.Label(self.header_frame, text="Status: Initializing...", fg="blue")
        self.status_label.pack()
        
        self.info_label = tk.Label(self.header_frame, text=f"Hotkey: {HOTKEY} | Hold to record", font=("Helvetica", 9))
        self.info_label.pack()

        # History List
        tk.Label(root, text="History:", font=("Helvetica", 10, "bold")).pack(anchor="w", padx=10)
        
        self.history_container = tk.Frame(root)
        self.history_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(self.history_container)
        self.scrollbar = tk.Scrollbar(self.history_container, orient="vertical", command=self.canvas.yview)
        self.history_frame = tk.Frame(self.canvas)

        self.history_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.history_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.running = True
        self.model_ready = False
        self.thread = threading.Thread(target=self.hotkey_loop, daemon=True)
        self.thread.start()
        
        # Check model status
        self.root.after(100, self.check_status)

    def check_status(self):
        if hasattr(self, 'model_ready') and self.model_ready:
            self.status_label.config(text="Status: Ready", fg="green")
        else:
            self.root.after(500, self.check_status)

    def add_history_item(self, text):
        item_frame = tk.Frame(self.history_frame, bd=1, relief="flat", pady=2)
        item_frame.pack(fill="x", expand=True)
        
        # Truncate text for display if very long
        display_text = (text[:60] + '...') if len(text) > 63 else text
        text_label = tk.Label(item_frame, text=display_text, font=("Helvetica", 9), anchor="w")
        text_label.pack(side="left", padx=5)
        
        copy_btn = tk.Button(item_frame, text="📋", font=("Helvetica", 8), command=lambda: self.copy_to_clipboard(text))
        copy_btn.pack(side="right", padx=5)
        
        # Tooltip-like effect or highlight
        item_frame.bind("<Enter>", lambda e: item_frame.config(bg="#f0f0f0"))
        item_frame.bind("<Leave>", lambda e: item_frame.config(bg="SystemButtonFace"))

    def copy_to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_label.config(text="Status: Copied to clipboard!", fg="green")
        self.root.after(2000, lambda: self.status_label.config(text="Status: Ready", fg="green"))

    def hotkey_loop(self):
        with sd.InputStream(samplerate=fs, channels=1, callback=callback):
            self.model_ready = True
            while self.running:
                if keyboard.is_pressed(HOTKEY):
                    self.root.after(0, self.update_ui_recording)
                    start_recording()
                    while keyboard.is_pressed(HOTKEY) and self.running:
                        sd.sleep(10)
                    stop_recording()
                    self.root.after(0, self.update_ui_ready)
                sd.sleep(50)

    def update_ui_recording(self):
        self.root.config(bg="#ffcccc")
        self.header_frame.config(bg="#ffcccc")
        self.label.config(bg="#ffcccc")
        self.status_label.config(text="Status: Recording...", fg="red", bg="#ffcccc")
        self.info_label.config(bg="#ffcccc")

    def update_ui_ready(self):
        self.root.config(bg="SystemButtonFace")
        self.header_frame.config(bg="SystemButtonFace")
        self.label.config(bg="SystemButtonFace")
        self.status_label.config(text="Status: Ready", fg="green", bg="SystemButtonFace")
        self.info_label.config(bg="SystemButtonFace")

    def on_closing(self):
        self.running = False
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperApp(root)
    root.mainloop()
