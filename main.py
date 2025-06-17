import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import os, sys
import io
import subprocess
import pysrt
import webvtt
import time
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
import soundfile
import whisper
from pysrt import open as srt_open
from datetime import datetime
import json

# === Helper Functions ===
def get_base_path():
    # If running as a PyInstaller bundle
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

# === Configuration ===

class Config:
    debug_mode = False
    MIN_SPEED = 0.6
    MAX_SPEED = 1.6

class SubsDictKeys:
    start_ms = "start_ms"
    end_ms = "end_ms"
    duration_ms = "duration_ms"
    speed_factor = "speed_factor"
    TTS_FilePath = "tts_filepath"
    TTS_FilePath_Trimmed = "tts_filepath_trimmed"

BASE_DIR = get_base_path()
workingFolder = os.path.join(BASE_DIR, "workingFolder")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
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
    speed_factor = max(0.6, min(speed_factor, 1.6))
    command = ['ffmpeg', '-i', 'pipe:0', '-filter:a', f'atempo={speed_factor}', '-f', 'wav', 'pipe:1']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate(input=audioInput.getvalue())
    if process.returncode != 0:
        raise Exception(f'ffmpeg error: {err.decode()}')
    return out

def stretch_audio_clip(audioFileToStretch, speedFactor, num):
    speedFactor = max(Config.MIN_SPEED, min(speedFactor, Config.MAX_SPEED))
    command = ['ffmpeg', '-i', 'pipe:0', '-filter:a', f"atempo={speedFactor}", '-f', 'wav', 'pipe:1']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate(input=audioFileToStretch.getvalue())
    if process.returncode != 0:
        raise Exception(f"FFmpeg error: {err.decode()}")
    return AudioSegment.from_file(io.BytesIO(out), format="wav")

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

def simulate_progress(progress_callback, done_flag):
    for i in range(50):  # simulate up to 40%
        if done_flag.is_set():
            break
        progress_callback(5 + (i * (35 / 50)))
        time.sleep(0.1)

def transcribe_audio_with_whisper(audio_path, json_path=None, progress_callback=None, log_callback=None):
    model = whisper.load_model("medium")
    done_flag = threading.Event()

    if progress_callback:
        threading.Thread(target=simulate_progress, args=(progress_callback, done_flag), daemon=True).start()

    result = model.transcribe(audio_path, word_timestamps=False, verbose=False)
    done_flag.set()

    if progress_callback:
        progress_callback(40)  # mark transcription complete

    detected_lang = result.get("language", "unknown")
    if log_callback:
        log_callback(f"Detected language: {detected_lang}")

    # Save with timestamp and language in metadata
    if json_path:
        result["meta"] = {
            "detected_language": detected_lang,
            "created": datetime.now().isoformat()
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        if log_callback:
            log_callback(f"Transcript saved to: {json_path}")

    return result['segments'], detected_lang

def load_whisper_json(json_path):
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data['segments']

def trim_clip(inputSound: AudioSegment) -> AudioSegment:
    start_trim = detect_leading_silence(inputSound)
    end_trim = detect_leading_silence(inputSound.reverse())
    return inputSound[start_trim:len(inputSound) - end_trim]

def stretch_audio_clip(audioFileToStretch, speedFactor, num):
    speedFactor = max(0.5, min(speedFactor, 2.0))
    command = ['ffmpeg', '-i', 'pipe:0', '-filter:a', f"atempo={speedFactor}", '-f', 'wav', 'pipe:1']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate(input=audioFileToStretch.getvalue())
    if process.returncode != 0:
        raise Exception(f"FFmpeg error: {err.decode()}")
    return AudioSegment.from_file(io.BytesIO(out), format="wav")

def align_whisper_segments_to_subs(whisper_segments, subsDict, audio):
    aligned_audio_clips = []
    seg_index = 0
    whisper_len = len(whisper_segments)

    for sub_index in subsDict:
        target_duration = subsDict[sub_index]['duration_ms']
        collected = []
        collected_duration = 0

        while seg_index < whisper_len:
            seg = whisper_segments[seg_index]
            seg_start = int(seg['start'] * 1000)
            seg_end = int(seg['end'] * 1000)
            duration = seg_end - seg_start

            if collected_duration + duration > target_duration and collected:
                break

            collected.append((seg_start, seg_end))
            collected_duration += duration
            seg_index += 1

        if not collected:
            raise Exception(f"No whisper segments collected for subtitle {sub_index}")

        first_start = collected[0][0]
        last_end = collected[-1][1]
        clip = audio[first_start:last_end]
        aligned_audio_clips.append(clip)

    return aligned_audio_clips

def process_audio_with_progress(srt_path, audio_path, output_path, log_callback, progress_callback, pause_check, transcript_path=None):
    log_callback("Loading English subtitles...")
    subs = srt_open(srt_path)
    subsDict = {
        i + 1: {
            "start_ms": int(sub.start.ordinal),
            "end_ms": int(sub.end.ordinal),
            "duration_ms": int(sub.end.ordinal - sub.start.ordinal)
        } for i, sub in enumerate(subs)
    }

    log_callback("Loading voiceover audio and trimming silence...")
    full_audio = AudioSegment.from_file(audio_path)
    trimmed_audio = trim_clip(full_audio)
    trimmed_path = os.path.abspath(os.path.join(workingFolder, "temp_trimmed.wav"))
    trimmed_audio.export(trimmed_path, format="wav")

    if transcript_path and os.path.exists(transcript_path):
        log_callback(f"Loading Whisper transcription from file: {transcript_path}")
        whisper_segments = load_whisper_json(transcript_path)
        progress_callback(40)
    else:
        log_callback("Transcribing audio with Whisper...")
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        json_out = os.path.join(workingFolder, f"{base_name}_transcript.json")
        whisper_segments, detected_lang = transcribe_audio_with_whisper(
            trimmed_path, json_out,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        if hasattr(log_callback, '__self__'):
            setattr(log_callback.__self__, "detected_language", detected_lang)

    log_callback(f"Aligning {len(whisper_segments)} Whisper segments to {len(subsDict)} subtitles...")
    whisper_clips = align_whisper_segments_to_subs(whisper_segments, subsDict, trimmed_audio)

    log_callback("Building final audio canvas...")
    canvas_duration = max(s["end_ms"] for s in subsDict.values())
    canvas = AudioSegment.silent(duration=canvas_duration)

    speed_up_mode = False
    remaining_overflow = 0

    for i, key in enumerate(subsDict, 1):
        pause_check()
        target_duration = subsDict[key]["duration_ms"]
        clip = whisper_clips[i - 1]
        trimmed = trim_clip(clip)
        clip_duration = len(trimmed)

        # Calculate natural speed factor
        natural_speed = clip_duration / target_duration

        # Default to natural speed
        speed_factor = natural_speed

        if natural_speed > Config.MAX_SPEED:
            # Clip too long, even at max speed
            speed_factor = Config.MAX_SPEED
            actual_duration = clip_duration / speed_factor
            overflow = actual_duration - target_duration

            if overflow > 0:
                remaining_overflow += overflow
                speed_up_mode = True
                log_callback(f"Segment {i} too long - will speed up following segments to catch up ({remaining_overflow:.0f}ms overflow)")

        elif speed_up_mode and remaining_overflow > 0:
            # Try to catch up by speeding up slightly more than needed
            effective_target = max(50, target_duration - remaining_overflow)
            speed_factor = min(Config.MAX_SPEED, clip_duration / effective_target)
            actual_duration = clip_duration / speed_factor
            time_saved = target_duration - actual_duration

            remaining_overflow = max(0, remaining_overflow - time_saved)
            log_callback(f"Segment {i} catching up - saved {time_saved:.0f}ms, remaining overflow: {remaining_overflow:.0f}ms")

            if remaining_overflow <= 0:
                speed_up_mode = False
                log_callback(f"✓ Caught up at segment {i}")

        else:
            # We're fine: natural speed within limits
            speed_up_mode = False
            remaining_overflow = 0

        # Clamp speed before usage and logging
        speed_factor = max(Config.MIN_SPEED, min(speed_factor, Config.MAX_SPEED))

        temp_buf = io.BytesIO()
        trimmed.export(temp_buf, format="wav")
        stretched = stretch_audio_clip(temp_buf, speed_factor, i)

        canvas = canvas.overlay(stretched, position=subsDict[key]["start_ms"])
        progress_callback((i / len(subsDict)) * 100)
        log_callback(f"Segment {i}/{len(subsDict)} processed at {speed_factor:.2f}x speed")

    log_callback("Exporting final synced audio...")
    output_format = os.path.splitext(output_path)[1][1:]
    
    export_params = {
        "format": output_format,
        "bitrate": "192k"
    }
    
    if output_format == "mp3":
        export_params.update({
            "codec": "libmp3lame",
            "parameters": ["-q:a", "0"]
        })
    
    canvas.export(output_path, **export_params)
    log_callback(f"✅ Finished. Output saved as {output_format.upper()} to: {output_path}")

    # Clean up temporary file
    if os.path.exists(trimmed_path):
        try:
            os.remove(trimmed_path)
            log_callback("Cleaned up temporary files.")
        except Exception as e:
            log_callback(f"Could not clean up temporary file: {e}")
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
        self.use_existing_transcript = tk.BooleanVar(value=False)
        self.transcript_path = ""

        self.build_ui()

    def build_ui(self):
        frame = tb.Frame(self.root, padding=10)
        frame.pack(fill=BOTH, expand=True)

        tb.Label(frame, text="1. Select Subtitles (.srt or .vtt)").pack(anchor=W)
        tb.Button(frame, text="Browse Subtitle File", command=self.browse_srt).pack(fill=X)

        tb.Label(frame, text="2. Select AI Voiceover (.mp3)").pack(anchor=W)
        tb.Button(frame, text="Browse Audio File", command=self.browse_audio).pack(fill=X)

        tb.Checkbutton(
            frame,
            text="Use existing Whisper transcript (.json)",
            variable=self.use_existing_transcript,
            command=self.toggle_transcript_option
        ).pack(anchor=W)

        self.transcript_btn = tb.Button(
            frame,
            text="Browse Transcript File",
            command=self.browse_transcript,
            state="disabled"
        )
        self.transcript_btn.pack(fill=X)

        tb.Label(frame, text="3. Choose Output Format").pack(anchor=W)
        self.format_var = tk.StringVar(value="wav")
        tb.OptionMenu(frame, self.format_var, "wav", "mp3").pack(fill=X)

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

    def toggle_transcript_option(self):
        if self.use_existing_transcript.get():
            self.transcript_btn.config(state="normal")
        else:
            self.transcript_btn.config(state="disabled")
            self.transcript_path = ""

    def browse_transcript(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if path:
            self.transcript_path = os.path.abspath(path)
            self.log_message(f"Selected Transcript File: {path}")

    def set_button_states(self, running=False, paused=False, finished=False):
        self.start_btn.config(state="disabled" if running else "normal")
        self.pause_btn.config(state="normal" if running else "disabled")
        self.stop_btn.config(state="normal" if running else "disabled")
        self.pause_btn.config(text="Continue" if paused else "Pause")

    def browse_srt(self):
        path = filedialog.askopenfilename(filetypes=[("Subtitle Files", "*.srt *.vtt")])
        if path:
            self.srt_path = os.path.abspath(path)
            self.log_message(f"Selected Subtitle File: {path}")

    def browse_audio(self):
        path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3")])
        if path:
            self.audio_path = os.path.abspath(path)
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

        self.log.delete("1.0", tk.END)  # ← clear the text window
        self.log_message("Processing stopped.")
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
        self.output_path = os.path.abspath(os.path.join(OUTPUT_FOLDER, f"final_output.{self.output_format}"))

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
                pause_check=self.wait_if_paused,
                transcript_path=self.transcript_path if self.use_existing_transcript.get() else None
            )
            self.log_message("✅ Finished.")
            if hasattr(self, "detected_language") and self.detected_language:
                self.log_message(f"🈯 Detected language: {self.detected_language}")
        except Exception as e:
            self.log_message(f"⚠️ {str(e)}")
        finally:
            self.set_button_states(finished=True)

# === Run GUI ===

if __name__ == "__main__":
    app = tb.Window(themename="darkly")
    AudioApp(app)
    app.mainloop()