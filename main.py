import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import os
import io
import subprocess
import pysrt
import webvtt
import time
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
import soundfile

# === Configuration ===

class Config:
    debug_mode = False
    output_format = "wav"

class SubsDictKeys:
    start_ms = "start_ms"
    end_ms = "end_ms"
    duration_ms = "duration_ms"
    speed_factor = "speed_factor"
    TTS_FilePath = "tts_filepath"
    TTS_FilePath_Trimmed = "tts_filepath_trimmed"

workingFolder = "workingFolder"
OUTPUT_FOLDER = "output"
os.makedirs(workingFolder, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
config = Config()

# === Audio Helpers ===

def trim_clip(inputSound: AudioSegment) -> AudioSegment:
    start_trim = detect_leading_silence(inputSound)
    end_trim = detect_leading_silence(inputSound.reverse())
    return inputSound[start_trim:len(inputSound) - end_trim]

def insert_audio(canvas, audioToOverlay, startTimeMs):
    return canvas.overlay(audioToOverlay, position=int(startTimeMs))

def create_canvas(canvasDuration, frame_rate=48000):
    return AudioSegment.silent(duration=canvasDuration, frame_rate=frame_rate)

def get_speed_factor(subsDict, trimmedAudio, desiredDuration, num):
    audio = AudioSegment.from_file(trimmedAudio, format="wav")
    rawDuration = audio.duration_seconds * 1000
    subsDict[num][SubsDictKeys.speed_factor] = rawDuration / float(desiredDuration)
    return subsDict

def stretch_with_ffmpeg(audioInput, speed_factor):
    speed_factor = max(0.5, min(speed_factor, 100.0))
    command = ['ffmpeg', '-i', 'pipe:0', '-filter:a', f'atempo={speed_factor}', '-f', 'wav', 'pipe:1']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate(input=audioInput.getvalue())
    if process.returncode != 0:
        raise Exception(f'ffmpeg error: {err.decode()}')
    return out

def stretch_audio_clip(audioFileToStretch, speedFactor, num):
    stretched_audio = stretch_with_ffmpeg(audioFileToStretch, speedFactor)
    return AudioSegment.from_file(io.BytesIO(stretched_audio), format="wav")

def time_str_to_ms(time_str):
    h, m, s = time_str.split(":")
    s, ms = s.split(".")
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)

def load_subtitles(file_path):
    subsDict = {}
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".srt":
        subs = pysrt.open(file_path)
        for i, sub in enumerate(subs, 1):
            subsDict[i] = {
                SubsDictKeys.start_ms: int(sub.start.ordinal),
                SubsDictKeys.end_ms: int(sub.end.ordinal),
                SubsDictKeys.duration_ms: int(sub.end.ordinal - sub.start.ordinal),
                SubsDictKeys.TTS_FilePath: os.path.join(workingFolder, f"segment_{i}.mp3")
            }

    elif ext == ".vtt":
        subs = webvtt.read(file_path)
        for i, caption in enumerate(subs, 1):
            start = time_str_to_ms(caption.start)
            end = time_str_to_ms(caption.end)
            subsDict[i] = {
                SubsDictKeys.start_ms: start,
                SubsDictKeys.end_ms: end,
                SubsDictKeys.duration_ms: end - start,
                SubsDictKeys.TTS_FilePath: os.path.join(workingFolder, f"segment_{i}.mp3")
            }
    else:
        raise Exception("Unsupported subtitle format. Use .srt or .vtt")
    return subsDict

# === Core Processing Logic with Pause/Stop ===

def process_audio_with_progress(srt_path, audio_path, output_path, log_callback, progress_callback, pause_check):
    log_callback("Loading subtitles...")
    subsDict = load_subtitles(srt_path)

    full_audio = AudioSegment.from_file(audio_path)
    total_steps = len(subsDict) * 4 + 1
    current_step = 0

    def step(msg=None):
        nonlocal current_step
        current_step += 1
        percent = (current_step / total_steps) * 100
        progress_callback(percent)
        if msg:
            log_callback(msg)
        pause_check()

    log_callback("Segmenting audio...")
    for key, value in subsDict.items():
        pause_check()
        segment = full_audio[value[SubsDictKeys.start_ms]:value[SubsDictKeys.end_ms]]
        segment.export(value[SubsDictKeys.TTS_FilePath], format="mp3")
        step(f"Segmented {key} / {len(subsDict)}")

    log_callback("Trimming silence...")
    virtualTrimmedFileDict = {}
    for key, value in subsDict.items():
        pause_check()
        rawClip = AudioSegment.from_file(value[SubsDictKeys.TTS_FilePath], format="mp3")
        trimmedClip = trim_clip(rawClip)
        tempTrimmedFile = io.BytesIO()
        trimmedClip.export(tempTrimmedFile, format="wav")
        virtualTrimmedFileDict[key] = tempTrimmedFile
        step(f"Trimmed {key} / {len(subsDict)}")

    log_callback("Calculating speed factors...")
    for key in subsDict:
        pause_check()
        subsDict = get_speed_factor(subsDict, virtualTrimmedFileDict[key], subsDict[key][SubsDictKeys.duration_ms], key)
        step(f"Speed factor {key} / {len(subsDict)}")

    total_duration = max(sub[SubsDictKeys.end_ms] for sub in subsDict.values())
    canvas = create_canvas(total_duration)

    log_callback("Stretching and inserting audio...")
    for key in subsDict:
        pause_check()
        try:
            stretchedClip = stretch_audio_clip(virtualTrimmedFileDict[key], subsDict[key][SubsDictKeys.speed_factor], key)
            canvas = insert_audio(canvas, stretchedClip, subsDict[key][SubsDictKeys.start_ms])
        except Exception as e:
            log_callback(f"Error in segment {key}: {str(e)}")
        step(f"Processed {key} / {len(subsDict)}")

    log_callback("Exporting final audio...")
    pause_check()
    canvas = canvas.set_channels(2)
    try:
        canvas.export(output_path, format=os.path.splitext(output_path)[1][1:], bitrate="192k")
        log_callback(f"Audio saved to: {output_path}")
    except Exception as e:
        backup_path = output_path + ".bak"
        canvas.export(backup_path, format="wav", bitrate="192k")
        log_callback(f"Export failed. Backup saved to {backup_path}")
        log_callback(f"Error: {str(e)}")
    step("Done!")

# === GUI ===

class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voiceover Sync Tool")
        self.srt_path = ""
        self.audio_path = ""
        self.output_path = ""
        self.output_format = "wav"
        self.paused = False
        self.stop_requested = False
        self.worker_thread = None

        self.build_ui()

    def build_ui(self):
        frame = tb.Frame(self.root, padding=10)
        frame.pack(fill=BOTH, expand=True)

        tb.Label(frame, text="1. Select Subtitles (.srt or .vtt)").pack(anchor=W)
        tb.Button(frame, text="Browse Subtitle File", command=self.browse_srt).pack(fill=X)

        tb.Label(frame, text="2. Select AI Voiceover (.mp3)").pack(anchor=W)
        tb.Button(frame, text="Browse Audio File", command=self.browse_audio).pack(fill=X)

        tb.Label(frame, text="3. Choose Output Format").pack(anchor=W)
        self.format_var = tk.StringVar(value="wav")
        tb.OptionMenu(frame, self.format_var, "wav", "mp3", "wav").pack(fill=X)

        self.start_btn = tb.Button(frame, text="Start", bootstyle=SUCCESS, command=self.start_thread)
        self.start_btn.pack(fill=X, pady=(10, 0))

        self.pause_btn = tb.Button(frame, text="Pause", bootstyle=WARNING, command=self.toggle_pause, state="disabled")
        self.pause_btn.pack(fill=X, pady=(5, 0))

        self.stop_btn = tb.Button(frame, text="Stop", bootstyle=DANGER, command=self.request_stop, state="disabled")
        self.stop_btn.pack(fill=X)

        self.progress = tb.Progressbar(frame, maximum=100, bootstyle=INFO, mode='determinate')
        self.progress.pack(fill=X, pady=10)

        self.log = tk.Text(frame, height=15, wrap=tk.WORD)
        self.log.pack(fill=BOTH, expand=True)

    def set_button_states(self, running=False, paused=False, finished=False):
        self.start_btn.config(state="disabled" if running else "normal")
        self.pause_btn.config(state="normal" if running else "disabled")
        self.stop_btn.config(state="normal" if running else "disabled")
        self.pause_btn.config(text="Continue" if paused else "Pause")

    def browse_srt(self):
        path = filedialog.askopenfilename(filetypes=[("Subtitle Files", "*.srt *.vtt")])
        if path:
            self.srt_path = path
            self.log_message(f"Selected Subtitle File: {path}")

    def browse_audio(self):
        path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3")])
        if path:
            self.audio_path = path
            self.log_message(f"Selected Audio File: {path}")

    def log_message(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def update_progress(self, value):
        self.progress["value"] = value
        self.root.update()

    def toggle_pause(self):
        self.paused = not self.paused
        self.set_button_states(running=True, paused=self.paused)

    def request_stop(self):
        self.stop_requested = True
        self.progress["value"] = 0
        if os.path.exists(self.output_path):
            try:
                os.remove(self.output_path)
                self.log_message("Stopped. Output file deleted.")
            except Exception as e:
                self.log_message(f"Could not delete output: {e}")
        self.set_button_states(running=False, finished=True)

    def wait_if_paused(self):
        while self.paused and not self.stop_requested:
            time.sleep(0.1)
        if self.stop_requested:
            raise Exception("Processing stopped by user.")

    def start_thread(self):
        if not self.srt_path or not self.audio_path:
            messagebox.showwarning("Missing Input", "Please select both a subtitle and audio file.")
            return
        self.output_format = self.format_var.get()
        self.output_path = os.path.join(OUTPUT_FOLDER, f"final_output.{self.output_format}")

        self.progress["value"] = 0
        self.paused = False
        self.stop_requested = False
        self.log_message("Starting processing...")
        self.set_button_states(running=True)

        self.worker_thread = threading.Thread(target=self.process)
        self.worker_thread.start()

    def process(self):
        try:
            process_audio_with_progress(
                self.srt_path,
                self.audio_path,
                self.output_path,
                log_callback=self.log_message,
                progress_callback=self.update_progress,
                pause_check=self.wait_if_paused
            )
            self.log_message("✅ Finished.")
        except Exception as e:
            self.log_message(f"⚠️ {str(e)}")
        finally:
            self.set_button_states(finished=True)

# === Run GUI ===

if __name__ == "__main__":
    app = tb.Window(themename="darkly")
    app.iconbitmap("icon.ico")
    AudioApp(app)
    app.mainloop()