# AI Voiceover Synchronizer

A GUI-based Python tool that synchronizes AI-generated voiceovers with subtitle timing from `.srt` or `.vtt` files. Good for automated dubbing, voiceover alignment, or multilingual content syncing. It's far from being perfect - but it's the best I could do on such short time.

---

## ✨ Features

- ✅ Supports `.srt` and `.vtt` subtitle formats  
- ✅ Outputs `.wav` or `.mp3` audio  
- ✅ Auto-trims silence, calculates timing, stretches voiceover audio  
- ✅ Clean, modern GUI with:
  - Start / Pause / Continue / Stop controls
  - Live progress bar and log output  
- ✅ Works offline using FFmpeg and Pydub  
- ✅ Uses Whisper for automatic voiceover language detection and transcription
- ✅ Multi-threaded to keep UI responsive  

---

## 📦 Installation

### 🟢 Option 1: Download `.exe` (no Python required)

1. Download the latest `.exe` from the [Releases](https://github.com/yourusername/ai-voiceover-sync/releases) page.
2. Extract the `.zip` file if needed.
3. Run `AI Voiceover Synchronizer.exe`

✅ No Python or pip needed  
🛑 You **still need FFmpeg** installed separately (see below)

---

### 🧪 Option 2: Run from Python source

#### 1. Clone the repository

"""
git clone https://github.com/yourusername/ai-voiceover-sync.git
cd ai-voiceover-sync
"""

#### 2. Install dependencies

Python 3.10.11 is recommended. NOT WORKING ON NEWEST.

"""
pip install -r requirements.txt
"""

Or manually:

"""
pip install pysrt pydub soundfile ttkbootstrap webvtt-py
"""

#### 3. Install FFmpeg

Make sure `ffmpeg` is installed and in your system `PATH`.

- [Download FFmpeg](https://ffmpeg.org/download.html)

Test it with:

"""
ffmpeg -version
"""

---

## 🚀 Usage

"""
python main.py
"""

Or double-click `AI Voiceover Synchronizer.exe` if using the standalone version.

### Then:
1. Select a subtitle file (`.srt` or `.vtt`) from the original (English) video, 
2. Select an AI voiceover audio (`.mp3`)  ,
3. Choose desired output format (`.wav` or `.mp3`),  
4. Click **Start** to begin,
5. Depending on the lenght of the video, takes approx. 2 hours for 40min video from my initial testing, may be faster on powerful hardware.

🟡 You can pause/resume the process  
🔴 Click **Stop** to cancel and clear progress/output  

Final audio is saved in the `output/` folder.

---

## 🙌 Acknowledgments

- [Pydub](https://github.com/jiaaro/pydub) — audio processing  
- [FFmpeg](https://ffmpeg.org/) — audio stretching  
- [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap) — modern Tkinter UI  
- [pysrt](https://github.com/byroot/pysrt) and [webvtt-py](https://github.com/glut23/webvtt-py) — subtitle support

---
