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

# === FFmpeg Path Configuration ===
def get_ffmpeg_path():
    local_ffmpeg = os.path.join(BASE_DIR, "ffmpeg_bin", "ffmpeg.exe")
    if os.path.exists(local_ffmpeg):
        return local_ffmpeg
    return "ffmpeg"  # fallback to system path

# === Configuration ===

class Config:
    debug_mode = False
    MIN_SPEED = 0.7
    MAX_SPEED = 1.25

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
    start_trim = detect_leading_silence(inputSound, silence_thresh=-55)
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
    command = [get_ffmpeg_path(), '-i', 'pipe:0', '-filter:a', f'atempo={speed_factor}', '-f', 'wav', 'pipe:1']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate(input=audioInput.getvalue())
    if process.returncode != 0:
        raise Exception(f'ffmpeg error: {err.decode()}')
    return out

def stretch_audio_clip(audioFileToStretch, speedFactor, num):
    speedFactor = max(Config.MIN_SPEED, min(speedFactor, Config.MAX_SPEED))
    command = [get_ffmpeg_path(), '-i', 'pipe:0', '-filter:a', f"atempo={speedFactor}", '-f', 'wav', 'pipe:1']
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

def transcribe_audio_with_whisper(audio_path, json_path=None, progress_callback=None, log_callback=None):
    model = whisper.load_model("medium")

    if progress_callback:
        progress_callback(5)

    result = model.transcribe(audio_path, word_timestamps=False, verbose=False)

    if progress_callback:
        progress_callback(35)  # mark transcription complete

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
    command = [get_ffmpeg_path(), '-i', 'pipe:0', '-filter:a', f"atempo={speedFactor}", '-f', 'wav', 'pipe:1']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate(input=audioFileToStretch.getvalue())
    if process.returncode != 0:
        raise Exception(f"FFmpeg error: {err.decode()}")
    return AudioSegment.from_file(io.BytesIO(out), format="wav")

def align_whisper_segments_to_subs(whisper_segments, subsDict, audio, log_callback):
    import numpy as np
    from difflib import SequenceMatcher

    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    whisper_len = len(whisper_segments)
    subtitle_len = len(subsDict)
    whisper_durations = [int((s['end'] - s['start']) * 1000) for s in whisper_segments]

    text_weight = 0.2
    duration_weight = 0.2
    index_alignment_weight = 0.6

    anchors = []
    w_idx = 0
    max_span = 6

    for sub_idx in sorted(subsDict.keys()):
        subtitle_text = subsDict[sub_idx].get("text", "").strip()
        sub_duration = subsDict[sub_idx]['duration_ms']
        best_score = 0.0
        best_match = None

        for span in range(1, max_span + 1):
            if w_idx + span > whisper_len:
                break
            segment_slice = whisper_segments[w_idx:w_idx + span]
            combined_text = " ".join(seg["text"] for seg in segment_slice)
            whisper_duration = sum(whisper_durations[w_idx:w_idx + span])

            text_score = similarity(combined_text, subtitle_text)
            duration_score = 1 - abs(sub_duration - whisper_duration) / sub_duration if sub_duration > 0 else 0
            index_alignment_score = 1 - abs((sub_idx / subtitle_len) - ((w_idx + span // 2) / whisper_len))

            combined_score = (
                text_weight * text_score +
                duration_weight * duration_score +
                index_alignment_weight * index_alignment_score
            )

            if combined_score > best_score:
                best_score = combined_score
                best_match = (w_idx, w_idx + span)

            if combined_score >= 0.92:
                break

        if best_score >= 0.45:
            anchors.append((sub_idx, best_match[0], best_match[1]))
            log_callback(f"✔ Subtitle {sub_idx} ⟶ segments {best_match[0]}–{best_match[1] - 1} (score={best_score:.2f})")
            w_idx = best_match[1]

    all_segment_times = [None] * whisper_len
    for i in range(len(anchors)):
        sub_idx, w_start, w_end = anchors[i]
        sub_start = subsDict[sub_idx]['start_ms']
        sub_end = subsDict[sub_idx]['end_ms']

        total_w_dur = sum(whisper_durations[w_start:w_end])
        if total_w_dur == 0:
            continue
        cum_durs = np.cumsum([0] + whisper_durations[w_start:w_end])
        for j, idx in enumerate(range(w_start, w_end)):
            rel_pos = cum_durs[j] / total_w_dur
            aligned_time = int(sub_start + rel_pos * (sub_end - sub_start))
            all_segment_times[idx] = aligned_time

    last_known = None
    for i in range(whisper_len):
        if all_segment_times[i] is not None:
            last_known = i
        elif last_known is not None:
            next_known = next((j for j in range(i + 1, whisper_len) if all_segment_times[j] is not None), None)
            if next_known is not None:
                time_start = all_segment_times[last_known]
                time_end = all_segment_times[next_known]
                dur = sum(whisper_durations[last_known:next_known])
                if dur == 0:
                    continue
                cum_durs = np.cumsum([0] + whisper_durations[last_known + 1:next_known])
                for j, idx in enumerate(range(last_known + 1, next_known)):
                    rel = cum_durs[j] / dur
                    all_segment_times[idx] = int(time_start + rel * (time_end - time_start))

    for i in range(whisper_len):
        if all_segment_times[i] is None:
            prev_time = all_segment_times[i - 1] if i > 0 and all_segment_times[i - 1] is not None else 0
            all_segment_times[i] = prev_time + whisper_durations[i]

    aligned_audio_clips = []
    for i in range(whisper_len):
        start_ms = all_segment_times[i]
        end_ms = all_segment_times[i + 1] if i + 1 < whisper_len and all_segment_times[i + 1] is not None else start_ms + whisper_durations[i]
        aligned_audio_clips.append(audio[start_ms:end_ms])

    return aligned_audio_clips

def process_audio_with_progress(srt_path, audio_path, output_path, log_callback, progress_callback, pause_check, transcript_path=None):
    log_callback("Loading English subtitles...")
    subs = srt_open(srt_path)
    subsDict = {
        i + 1: {
            "start_ms": int(sub.start.ordinal),
            "end_ms": int(sub.end.ordinal),
            "duration_ms": int(sub.end.ordinal - sub.start.ordinal),
            "text": sub.text.strip()
        } for i, sub in enumerate(subs)
    }
    progress_callback(2)

    # replaced section for json transcript working always
    log_callback("Loading voiceover audio...")
    full_audio = AudioSegment.from_file(audio_path)

    if transcript_path and os.path.exists(transcript_path):
        log_callback("Using full audio (no trim) to match saved transcript.")
        trimmed_audio = full_audio  # Use original audio to match JSON timestamps
    else:
        log_callback("Trimming audio for fresh transcription...")
        trimmed_audio = trim_clip(full_audio)

    unique_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trimmed_path = os.path.abspath(os.path.join(workingFolder, f"temp_trimmed_{unique_id}.wav"))
    trimmed_audio.export(trimmed_path, format="wav")
    progress_callback(5)

    if transcript_path and os.path.exists(transcript_path):
        log_callback(f"Loading Whisper transcription from file: {transcript_path}")
        whisper_segments = load_whisper_json(transcript_path)
        progress_callback(35)
    else:
        log_callback("Transcribing audio with Whisper...")
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        json_out = os.path.join(workingFolder, f"{base_name}_transcript_{unique_id}.json")
        whisper_segments, detected_lang = transcribe_audio_with_whisper(
            trimmed_path, json_out,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        if hasattr(log_callback, '__self__'):
            setattr(log_callback.__self__, "detected_language", detected_lang)

    log_callback(f"Aligning {len(whisper_segments)} Whisper segments to {len(subsDict)} subtitles...")
    whisper_clips = align_whisper_segments_to_subs(whisper_segments, subsDict, trimmed_audio, log_callback)

    log_callback("Building final audio canvas...")
    progress_callback(45)
    canvas_duration = max(s["end_ms"] for s in subsDict.values())
    canvas = AudioSegment.silent(duration=canvas_duration)
    actual_timeline_position = 0

    speed_up_mode = False
    remaining_overflow = 0

    for i, key in enumerate(subsDict, 1):
        pause_check()
        target_duration = subsDict[key]["duration_ms"]
        clip = whisper_clips[i - 1]
        trimmed = clip.fade_in(10)
        clip_duration = len(trimmed)

        # Calculate natural speed factor
        natural_speed = clip_duration / target_duration

        # ADAPTIVE ADJUSTMENT: try to reschedule next segment closer to avoid pauses
        next_sub = subsDict.get(i + 1)
        if (
            next_sub and
            abs(natural_speed - Config.MIN_SPEED) < 0.01 and  # fully slowed down
            next_sub["start_ms"] > subsDict[key]["end_ms"] + 200  # noticeable silence
        ):
            original_gap = next_sub["start_ms"] - subsDict[key]["end_ms"]
            saved_gap = int(original_gap * 0.8)  # leave 20% breathing room

            # Move the next segment earlier
            subsDict[i + 1]["start_ms"] = subsDict[key]["end_ms"] + 100
            subsDict[i + 1]["end_ms"] = subsDict[i + 1]["start_ms"] + subsDict[i + 1]["duration_ms"]

            log_callback(
                f"🪄 Adaptive timing: shifted segment {i + 1} earlier by {original_gap - saved_gap}ms "
                f"to reduce speed-up pressure."
            )

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
        stretched = stretched.fade_in(15)

        # Get potential next segment start
        next_sub = subsDict.get(i + 1)
        gap_to_next = None
        if next_sub:
            gap_to_next = next_sub["start_ms"] - (actual_timeline_position + len(stretched))

        # Apply short fade-out if at min speed and room to breathe
        if (
            abs(speed_factor - Config.MIN_SPEED) < 0.01 and
            gap_to_next is not None and
            gap_to_next >= 60  # Only fade if at least 60ms gap before next
        ):
            fade_duration = min(80, int(gap_to_next * 0.6))  # Soft, short fade
            stretched = stretched.fade_out(fade_duration)
            log_callback(f"Segment {i} faded out over {fade_duration}ms to smooth ending.")

        # Prevent overlap
        start_time = max(subsDict[key]["start_ms"], actual_timeline_position)
        end_time = start_time + len(stretched)
        actual_timeline_position = end_time

        canvas = canvas.overlay(stretched, position=start_time)

        if start_time > subsDict[key]["start_ms"]:
            log_callback(f"Segment {i} delayed by {start_time - subsDict[key]['start_ms']}ms to avoid overlap.")

        progress_callback(45 + (i / len(subsDict)) * 50)
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
    
    progress_callback(97)
    canvas.export(output_path, **export_params)
    log_callback(f"✅ Finished. Output saved as {output_format.upper()} to: {output_path}")
    progress_callback(100)

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
            text="Use existing AI Voiceover Whisper transcript (.json)",
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
        # Get detected language if available (fallback to 'unknown')
        lang = getattr(self, "detected_language", "unknown")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        filename = f"output-{lang}-{timestamp}.{self.output_format}"
        self.output_path = os.path.abspath(os.path.join(OUTPUT_FOLDER, filename))

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