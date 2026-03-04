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
from tkinter import messagebox, ttk
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
model = None

# Configuration
MODEL_SIZE = "large-v3"
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

# Model will be loaded asynchronously in the App

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

# Design System (Catppuccin-inspired)
BG_COLOR = "#1e1e2e"
SURFACE_COLOR = "#313244"
TEXT_COLOR = "#cdd6f4"
ACCENT_COLOR = "#89b4fa"
RECORD_COLOR = "#f38ba8"
SUCCESS_COLOR = "#a6e3a1"
FONT_MAIN = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 12, "bold")
FONT_UI = ("Segoe UI", 9)

class WhisperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("whisper")
        self.root.geometry("400x550")
        self.root.configure(bg=BG_COLOR)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Window Icon
        try:
            icon_img = tk.PhotoImage(file="icon.png")
            self.root.iconphoto(True, icon_img)
        except Exception as e:
            print(f"Could not load window icon: {e}")
        
        # Styles
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TProgressbar", 
                        thickness=10,
                        troughcolor=BG_COLOR, 
                        background=ACCENT_COLOR, 
                        bordercolor=BG_COLOR, 
                        lightcolor=ACCENT_COLOR, 
                        darkcolor=ACCENT_COLOR)
        
        # Header
        self.header_frame = tk.Frame(root, bg=BG_COLOR, pady=20)
        self.header_frame.pack(fill="x")
        
        self.label = tk.Label(self.header_frame, text="whisper", font=("Segoe UI", 24, "bold"), fg=ACCENT_COLOR, bg=BG_COLOR)
        self.label.pack()
        
        self.status_label = tk.Label(self.header_frame, text="Preparing...", font=FONT_UI, fg=ACCENT_COLOR, bg=BG_COLOR)
        self.status_label.pack(pady=(5, 10))
        
        # Progress Bar
        self.progress = ttk.Progressbar(self.header_frame, style="TProgressbar", orient="horizontal", length=300, mode="indeterminate")
        self.progress.pack(pady=5)
        self.progress.start(10)
        
        self.info_label = tk.Label(self.header_frame, text=f"Hotkey: {HOTKEY} | Hold to record", font=FONT_UI, fg="#7f849c", bg=BG_COLOR)
        self.info_label.pack(pady=10)

        # History Section
        tk.Label(root, text="RECENT", font=("Segoe UI", 9, "bold"), fg="#585b70", bg=BG_COLOR).pack(anchor="w", padx=20, pady=(10, 5))
        
        self.history_container = tk.Frame(root, bg=BG_COLOR)
        self.history_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        self.canvas = tk.Canvas(self.history_container, bg=BG_COLOR, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.history_container, orient="vertical", command=self.canvas.yview)
        self.history_frame = tk.Frame(self.canvas, bg=BG_COLOR)

        self.history_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.history_frame, anchor="nw", width=340)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Handle mousewheel
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        # Footer branding
        self.footer = tk.Label(root, text="by smoosewellington", font=("Segoe UI", 8, "italic"), fg="#45475a", bg=BG_COLOR)
        self.footer.pack(side="bottom", pady=5)
        
        self.running = True
        self.model_ready = False
        
        # Start background model loading
        self.load_thread = threading.Thread(target=self.async_load_model, daemon=True)
        self.load_thread.start()
        
        # Start hotkey thread (will wait for model_ready)
        self.hotkey_thread = threading.Thread(target=self.hotkey_loop, daemon=True)
        self.hotkey_thread.start()

    def async_load_model(self):
        global model
        self.root.after(0, lambda: self.status_label.config(text=f"Loading {MODEL_SIZE} model..."))
        
        try:
            model = load_model()
            
            # Test the model
            self.root.after(0, lambda: self.status_label.config(text="Testing model..."))
            try:
                model.transcribe(np.zeros(16000, dtype=np.float32))
            except Exception as e:
                if "cudnn" in str(e).lower() or "cuda" in str(e).lower():
                    self.root.after(0, lambda: self.status_label.config(text="CUDA Error, falling back to CPU...", fg=RECORD_COLOR))
                    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            
            self.model_ready = True
            self.root.after(0, self.on_model_loaded)
            
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Error: {str(e)[:40]}...", fg=RECORD_COLOR))
            self.root.after(0, lambda: self.progress.stop())

    def on_model_loaded(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.status_label.config(text="Ready", fg=SUCCESS_COLOR)

    def add_history_item(self, text):
        card = tk.Frame(self.history_frame, bg=SURFACE_COLOR, padx=15, pady=12, highlightthickness=1, highlightbackground="#45475a")
        card.pack(fill="x", pady=5)
        
        # Text area
        display_text = (text[:120] + '...') if len(text) > 123 else text
        msg_label = tk.Label(card, text=display_text, font=FONT_UI, fg=TEXT_COLOR, bg=SURFACE_COLOR, wraplength=250, justify="left", anchor="w")
        msg_label.pack(side="left", fill="x", expand=True)
        
        # GitHub-style copy button
        copy_btn = tk.Button(
            card, text="📋", font=("Segoe UI Symbol", 10), 
            command=lambda: self.copy_to_clipboard(text),
            bg=SURFACE_COLOR, fg=ACCENT_COLOR, activebackground=ACCENT_COLOR, 
            activeforeground=SURFACE_COLOR, bd=0, cursor="hand2"
        )
        copy_btn.pack(side="right", padx=(10, 0))
        
        # Hover效果
        card.bind("<Enter>", lambda e: card.config(highlightbackground=ACCENT_COLOR))
        card.bind("<Leave>", lambda e: card.config(highlightbackground="#45475a"))

    def copy_to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        old_status = self.status_label.cget("text")
        old_fg = self.status_label.cget("fg")
        self.status_label.config(text="Copied!", fg=SUCCESS_COLOR)
        self.root.after(1500, lambda: self.status_label.config(text=old_status, fg=old_fg))

    def hotkey_loop(self):
        # Wait for model to load
        while not self.model_ready and self.running:
            sd.sleep(100)
            
        with sd.InputStream(samplerate=fs, channels=1, callback=callback):
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
        self.animate_bg(RECORD_COLOR)
        self.status_label.config(text="Recording...", fg=BG_COLOR)
        self.label.config(fg=BG_COLOR)
        self.info_label.config(fg=BG_COLOR)

    def update_ui_ready(self):
        self.animate_bg(BG_COLOR)
        self.status_label.config(text="Ready", fg=SUCCESS_COLOR)
        self.label.config(fg=ACCENT_COLOR)
        self.info_label.config(fg="#7f849c")

    def animate_bg(self, color):
        self.root.config(bg=color)
        self.header_frame.config(bg=color)
        self.label.config(bg=color)
        self.status_label.config(bg=color)
        self.info_label.config(bg=color)
        self.history_container.config(bg=color)
        self.canvas.config(bg=color)
        self.history_frame.config(bg=color)

    def on_closing(self):
        self.running = False
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperApp(root)
    root.mainloop()
