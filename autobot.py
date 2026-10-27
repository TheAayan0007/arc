"""
autobot.py - Modern PyQt6 Autonomous AI OS Agent (A.R.C)
Upgrades:
1. Voice Input (🎙️ Microphone Button) -> Transcribes spoken commands and runs them step-by-step.
2. Proactive Clarification Dialog (`prompt_dialog` / `ask_user`):
   - If a command is ambiguous (e.g. "check my browser"), the AI opens a dialog asking "Which browser, sir?",
     sends your answer back to the AI Brain, and continues the step-by-step plan.
3. Self-Regenerating Step-by-Step Planner & Diagnostic Mode Selector (Text vs. Image):
   - Breaks every task into numbered steps and executes them one by one in the visible bottom terminal.
   - If a step fails in the terminal or screen (or when the AI needs context), asks the user whether to send
     diagnostic context to the AI Brain as **Text** (terminal/logs) or **Image** (live screenshot + OCR),
     then regenerates the failed step and proceeds.
"""
import os
import re
import io
import ast
import sys
import wave
import uuid
import glob
import time
import json
import hmac
import base64
import shutil
import socket
import getpass
import difflib
import hashlib
import platform
import tempfile
import threading
import subprocess
import webbrowser
from urllib.parse import quote
from datetime import datetime, timedelta

from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QObject, QProcess, QPoint,
    QPropertyAnimation, QEasingCurve
)
from PyQt6.QtGui import (
    QAction, QImage, QPixmap, QFont, QGuiApplication,
    QKeySequence, QShortcut, QTextCursor
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTextEdit, QLabel, QDockWidget,
    QDialog, QFormLayout, QComboBox, QCheckBox, QFileDialog,
    QMessageBox, QInputDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QListWidget, QListWidgetItem,
    QAbstractItemView, QFrame, QMenu, QScrollArea, QSizePolicy,
    QStackedWidget, QGraphicsOpacityEffect
)

try:
    import psutil
except ImportError:
    psutil = None

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import pyautogui
    pyautogui.FAILSAFE = True
except ImportError:
    pyautogui = None

try:
    import requests
except ImportError:
    requests = None

try:
    from send2trash import send2trash
except ImportError:
    send2trash = None

try:
    import pyperclip
except ImportError:
    pyperclip = None

try:
    import cv2
except ImportError:
    cv2 = None

from screen_vision import ScreenReader, VISION_OK, VISION_ERR
from web_agent import WebAgent, SELENIUM_OK
from system_monitor import SystemMonitorWidget
from camera_tracker import CameraTrackerWidget

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "agent_settings.json")
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
UPLOADS_DIR = os.path.join(BASE_DIR, "chat_uploads")
SCRIPTS_DIR = os.path.join(BASE_DIR, "arc_scripts")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SCRIPTS_DIR, exist_ok=True)

MAX_ITERATIONS = 20
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MOD = "command" if platform.system() == "Darwin" else "ctrl"

PIP_PACKAGE_MAP = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "fitz": "PyMuPDF",
    "fpdf": "fpdf2",
    "reportlab": "reportlab",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    "pypdf": "pypdf",
    "PyPDF2": "PyPDF2",
    "pyautogui": "pyautogui",
    "pyperclip": "pyperclip",
    "edge_tts": "edge-tts",
    "speech_recognition": "SpeechRecognition",
}


# ==========================================
# 0. BACKEND CONSOLE STREAM TEE
# ==========================================
class BackendConsoleCapture(QObject):
    new_log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.buffer = []
        self._lock = threading.Lock()
        self._installed = False

    def install(self):
        if self._installed:
            return
        self._installed = True
        try:
            self._orig_stdout_fd = os.dup(1)
            self._orig_stderr_fd = os.dup(2)
            r_fd, w_fd = os.pipe()
            os.dup2(w_fd, 1)
            os.dup2(w_fd, 2)
            os.close(w_fd)

            def _reader():
                while True:
                    try:
                        data = os.read(r_fd, 4096)
                        if not data:
                            break
                        try:
                            os.write(self._orig_stderr_fd, data)
                        except Exception:
                            pass
                        text = data.decode("utf-8", errors="replace")
                        with self._lock:
                            self.buffer.append(text)
                            if len(self.buffer) > 600:
                                self.buffer = self.buffer[-600:]
                        self.new_log_signal.emit(text)
                    except Exception:
                        break

            threading.Thread(target=_reader, daemon=True).start()
        except Exception:
            pass


CONSOLE_CAPTURE = BackendConsoleCapture()


class BackendConsoleWidget(QWidget):
    def __init__(self, capture: BackendConsoleCapture, parent=None):
        super().__init__(parent)
        self.capture = capture
        self.setMinimumSize(460, 200)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        top = QHBoxLayout()
        lbl = QLabel(f"─$ python {os.path.basename(__file__)}  (LIVE BACKEND CONSOLE / STDERR & STDOUT)")
        lbl.setStyleSheet("color: #00ffe7; font-family: 'Consolas', 'Monospace'; font-size: 11px; font-weight: bold;")
        btn_clear = QPushButton("Clear Console")
        btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear.setStyleSheet("""
            QPushButton { background: rgba(0,255,231,0.12); border: 1px solid #00ffe7; color: #00ffe7;
                          padding: 3px 10px; border-radius: 5px; font-size: 11px; font-weight: bold; }
            QPushButton:hover { background: rgba(0,255,231,0.28); }
        """)
        top.addWidget(lbl)
        top.addStretch()
        top.addWidget(btn_clear)
        lay.addLayout(top)

        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setStyleSheet("""
            QTextEdit {
                background-color: #040811;
                color: #b8d4f2;
                border: 1px solid #143558;
                border-radius: 6px;
                font-family: 'Fira Code', 'JetBrains Mono', 'Consolas', 'Monospace';
                font-size: 12px;
                padding: 8px;
            }
        """)
        btn_clear.clicked.connect(self.txt.clear)
        lay.addWidget(self.txt, 1)

        header_banner = f"┌──({getpass.getuser() or 'kali'}㉿{socket.gethostname() or 'Kali'})-[{os.getcwd()}]\n└─$ python {os.path.basename(__file__)}\n"
        self.txt.insertPlainText(header_banner)
        with self.capture._lock:
            if self.capture.buffer:
                self.txt.insertPlainText("".join(self.capture.buffer))

        self.capture.new_log_signal.connect(self._append_text)

    def _append_text(self, text: str):
        c = self.txt.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        c.insertText(text)
        self.txt.setTextCursor(c)
        self.txt.ensureCursorVisible()


# ==========================================
# 1. OS DETECTION, HUMAN NEURAL TTS, VOICE INPUT (STT), & CONFIG
# ==========================================
def get_os_info():
    p = platform.system().lower()
    return "Windows" if "windows" in p else "Linux" if "linux" in p else "macOS" if "darwin" in p else "Unknown"


CURRENT_OS = get_os_info()


def check_internet_available(timeout: float = 2.0) -> bool:
    for host, port in (("1.1.1.1", 53), ("8.8.8.8", 53)):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


class SpeechEngine:
    """Human-Like Natural Neural Text-to-Speech Engine (edge-tts with native fallback)."""
    _proc = None
    _lock = threading.Lock()
    VOICE = "en-US-AriaNeural"

    @classmethod
    def is_speaking(cls) -> bool:
        with cls._lock:
            return cls._proc is not None and cls._proc.poll() is None

    @classmethod
    def stop(cls):
        with cls._lock:
            if cls._proc is not None and cls._proc.poll() is None:
                try:
                    cls._proc.terminate()
                except Exception:
                    pass
            cls._proc = None
        if CURRENT_OS == "Linux" and shutil.which("spd-say"):
            subprocess.Popen(["spd-say", "-C"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @classmethod
    def _play_audio_file(cls, filepath: str) -> bool:
        players = []
        if CURRENT_OS == "Linux":
            if shutil.which("mpv"):
                players.append(["mpv", "--no-video", "--really-quiet", filepath])
            if shutil.which("ffplay"):
                players.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath])
            if shutil.which("cvlc"):
                players.append(["cvlc", "--play-and-exit", "--quiet", filepath])
            if shutil.which("mpg123"):
                players.append(["mpg123", "-q", filepath])
        elif CURRENT_OS == "macOS":
            players.append(["afplay", filepath])
        elif CURRENT_OS == "Windows":
            if shutil.which("ffplay"):
                players.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath])

        for cmd in players:
            try:
                with cls._lock:
                    cls._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                cls._proc.wait()
                return True
            except Exception:
                continue
        return False

    @classmethod
    def speak(cls, text: str, voice: str = None):
        clean = re.sub(r"[`*_#>\[\]]", " ", (text or "")).strip()
        if not clean:
            return
        cls.stop()
        chosen_voice = voice or cls.VOICE

        def _worker():
            if check_internet_available(1.5):
                tmp_mp3 = os.path.join(tempfile.gettempdir(), f"arc_tts_{int(time.time()*1000)}.mp3")
                generated = False
                try:
                    if shutil.which("edge-tts"):
                        subprocess.run(
                            ["edge-tts", "--voice", chosen_voice, "--text", clean, "--write-media", tmp_mp3],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20
                        )
                        generated = os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 100
                    else:
                        import asyncio
                        import edge_tts
                        asyncio.run(edge_tts.Communicate(clean, chosen_voice).save(tmp_mp3))
                        generated = os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 100
                except Exception:
                    generated = False

                if generated:
                    try:
                        if cls._play_audio_file(tmp_mp3):
                            return
                    finally:
                        try:
                            os.remove(tmp_mp3)
                        except OSError:
                            pass

            try:
                if CURRENT_OS == "Windows":
                    safe_txt = clean.replace("'", "''").replace('"', '`"')
                    ps_cmd = (
                        "Add-Type -AssemblyName System.Speech; "
                        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        "$s.Rate = 0; "
                        "try { $s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female) } catch {}; "
                        f"$s.Speak('{safe_txt}')"
                    )
                    with cls._lock:
                        cls._proc = subprocess.Popen(
                            ["powershell", "-NoProfile", "-Command", ps_cmd],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                    cls._proc.wait()
                    return

                if CURRENT_OS == "macOS":
                    with cls._lock:
                        cls._proc = subprocess.Popen(["say", "-v", "Samantha", clean],
                                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    cls._proc.wait()
                    return

                for tts_cmd in (
                    ["spd-say", "-w", "-r", "-5", "-p", "5", clean],
                    ["espeak-ng", "-v", "en-us+f3", "-s", "165", clean],
                    ["espeak", "-v", "en+f3", "-s", "165", clean]
                ):
                    if shutil.which(tts_cmd[0]):
                        with cls._lock:
                            cls._proc = subprocess.Popen(tts_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        cls._proc.wait()
                        return
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()


class VoiceInputEngine:
    """
    Captures spoken voice commands from the user's microphone and transcribes them into text.
    Uses OS audio recording (`arecord` / `ffmpeg` / `speech_recognition`) + Groq Whisper or Google STT.
    """
    @staticmethod
    def record_and_transcribe(config: dict, duration_sec: int = 5) -> tuple:
        """Returns (transcribed_text, error_or_status_msg)."""
        wav_path = os.path.join(tempfile.gettempdir(), f"arc_voice_{int(time.time())}.wav")
        recorded = False

        # 1. Try Python speech_recognition first if installed
        try:
            import speech_recognition as sr
            rec = sr.Recognizer()
            with sr.Microphone() as source:
                rec.adjust_for_ambient_noise(source, duration=0.4)
                audio = rec.listen(source, timeout=5, phrase_time_limit=duration_sec + 3)
            with open(wav_path, "wb") as f:
                f.write(audio.get_wav_data())
            recorded = os.path.exists(wav_path) and os.path.getsize(wav_path) > 500
        except Exception:
            recorded = False

        # 2. Fallback to native Linux `arecord` or `ffmpeg`
        if not recorded:
            try:
                if CURRENT_OS == "Linux" and shutil.which("arecord"):
                    subprocess.run(
                        ["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(duration_sec), wav_path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=duration_sec + 3
                    )
                    recorded = os.path.exists(wav_path) and os.path.getsize(wav_path) > 500
                elif shutil.which("ffmpeg"):
                    inp_fmt = "alsa" if CURRENT_OS == "Linux" else "dshow" if CURRENT_OS == "Windows" else "avfoundation"
                    inp_dev = "default" if CURRENT_OS == "Linux" else ":0"
                    subprocess.run(
                        ["ffmpeg", "-y", "-f", inp_fmt, "-i", inp_dev, "-t", str(duration_sec), "-ar", "16000", "-ac", "1", wav_path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=duration_sec + 4
                    )
                    recorded = os.path.exists(wav_path) and os.path.getsize(wav_path) > 500
            except Exception:
                recorded = False

        if not recorded:
            return "", "Could not record from microphone. Ensure a mic is connected (or `arecord` / `SpeechRecognition` is installed)."

        try:
            # Transcribe via Groq Whisper if Groq API key is configured
            groq_key = (
                config.get("api_keys_by_provider", {}).get("Groq (Official SDK)", "").strip()
                or (config.get("api_key", "").strip() if config.get("provider") == "Groq (Official SDK)" else "")
            )
            if groq_key and requests and check_internet_available():
                with open(wav_path, "rb") as f:
                    r = requests.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {groq_key}"},
                        files={"file": ("voice.wav", f, "audio/wav")},
                        data={"model": "whisper-large-v3-turbo"},
                        timeout=25
                    )
                if r.status_code == 200:
                    txt = r.json().get("text", "").strip()
                    if txt:
                        return txt, "Transcribed via Groq Whisper."

            # Fallback to SpeechRecognition Google Web Speech API
            import speech_recognition as sr
            rec = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = rec.record(source)
            txt = rec.recognize_google(audio_data)
            return txt.strip(), "Transcribed via Google Speech Recognition."
        except Exception as e:
            return "", f"Voice transcription failed: {e}"
        finally:
            try:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except OSError:
                pass


PROVIDER_MODELS = {
    "Groq (Official SDK)": [
        "qwen/qwen3.8-27b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ],
    "Gemini": ["gemini-2.5-flash", "gemini-2.5-pro"],
    "OpenAI (ChatGPT)": ["gpt-4o-mini", "gpt-4o", "o3-mini"],
    "Grok (xAI)": ["grok-2-latest", "grok-2-vision-latest"],
    "Local (Ollama)": ["gemma4:e4b", "llava:latest", "qwen2.5-coder:14b", "llama3.2:3b"],
    "Local GGUF File (llama.cpp)": ["custom-local-gguf"],
}

IMAGE_PROVIDERS = [
    "Pollinations AI (Free Flux API)",
    "OpenAI DALL-E 3 (API)",
    "Local Stable Diffusion WebUI API (127.0.0.1:7860)",
    "Local Diffusers Model (.safetensors / Folder)"
]

ENV_KEYS = {
    "Groq (Official SDK)": "GROQ_API_KEY",
    "Gemini": "GEMINI_API_KEY",
    "OpenAI (ChatGPT)": "OPENAI_API_KEY",
    "Grok (xAI)": "XAI_API_KEY"
}

DEFAULT_CONFIG = {
    "provider": "Groq (Official SDK)",
    "model_name": "qwen/qwen3.8-27b",
    "api_key": "",
    "api_keys_by_provider": {
        "Groq (Official SDK)": "",
        "Gemini": "",
        "OpenAI (ChatGPT)": "",
        "Grok (xAI)": ""
    },
    "local_model_path": "",
    "ollama_url": "http://localhost:11434",
    "password_record": None,
    "superuser_mode": False,
    "image_provider": "Pollinations AI (Free Flux API)",
    "image_model": "flux",
    "image_api_key": "",
    "local_image_model_path": "",
    "sd_webui_url": "http://127.0.0.1:7860",
    "tts_voice": "en-US-AriaNeural"
}


def hash_password(pw, salt=None):
    salt = salt or os.urandom(16).hex()
    dig = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()
    return {"salt": salt, "hash": dig}


def verify_password(pw, rec):
    if not rec:
        return False
    test = hash_password(pw, rec["salt"])["hash"]
    return hmac.compare_digest(test, rec["hash"])


def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except Exception:
        pass


def load_config():
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    old_pw = cfg.pop("security_password", None)
    if old_pw and not cfg.get("password_record"):
        cfg["password_record"] = hash_password(old_pw)
    prov = cfg["provider"]
    if not cfg.get("api_key") and os.environ.get(ENV_KEYS.get(prov, ""), ""):
        cfg["api_key"] = os.environ[ENV_KEYS[prov]]
    SpeechEngine.VOICE = cfg.get("tts_voice", "en-US-AriaNeural")
    save_config(cfg)
    return cfg


class MemoryStore:
    """Manages persistent timestamped memories in memory.json."""
    def __init__(self, filepath=MEMORY_FILE):
        self.filepath = filepath
        self._lock = threading.RLock()
        if not os.path.exists(self.filepath):
            self._write([])

    def _read(self):
        with self._lock:
            if not os.path.exists(self.filepath):
                return []
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, list) else []
            except Exception:
                return []

    def _write(self, items):
        with self._lock:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=4, ensure_ascii=False)

    def get_all(self):
        return self._read()

    def add_memory(self, content: str, key: str = "note", dt: datetime = None):
        dt = dt or datetime.now()
        items = self._read()
        entry = {
            "id": int(time.time() * 1000) + len(items),
            "key": key.strip() or "user_memory",
            "content": content.strip(),
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M:%S"),
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S")
        }
        items.append(entry)
        self._write(items)
        return entry

    def update_memory(self, idx: int, key: str, content: str, date_str: str, time_str: str):
        items = self._read()
        if 0 <= idx < len(items):
            items[idx]["key"] = key.strip()
            items[idx]["content"] = content.strip()
            items[idx]["date"] = date_str.strip()
            items[idx]["time"] = time_str.strip()
            items[idx]["timestamp"] = f"{date_str.strip()} {time_str.strip()}"
            self._write(items)

    def delete_memory(self, idx: int):
        items = self._read()
        if 0 <= idx < len(items):
            items.pop(idx)
            self._write(items)

    def erase_all(self):
        self._write([])


class ChatHistoryStore:
    """Multi-conversation persistent store in chat_history.json."""
    def __init__(self, filepath=CHAT_HISTORY_FILE):
        self.filepath = filepath
        self._lock = threading.RLock()
        self._ensure_initialized()

    def _new_chat_dict(self, title="New Chat", messages=None):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "id": uuid.uuid4().hex[:12],
            "title": title,
            "pinned": False,
            "created_at": now,
            "updated_at": now,
            "messages": messages or []
        }

    def _ensure_initialized(self):
        with self._lock:
            if not os.path.exists(self.filepath):
                first = self._new_chat_dict("General Chat")
                self._write_raw({"active_chat_id": first["id"], "chats": [first]})
                return
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    first = self._new_chat_dict("Previous Chat", messages=data[-250:])
                    self._write_raw({"active_chat_id": first["id"], "chats": [first]})
                elif isinstance(data, dict) and "chats" in data:
                    if not data["chats"]:
                        first = self._new_chat_dict("New Chat")
                        data["chats"] = [first]
                        data["active_chat_id"] = first["id"]
                        self._write_raw(data)
                    elif not data.get("active_chat_id"):
                        data["active_chat_id"] = data["chats"][0]["id"]
                        self._write_raw(data)
                else:
                    first = self._new_chat_dict("New Chat")
                    self._write_raw({"active_chat_id": first["id"], "chats": [first]})
            except Exception:
                first = self._new_chat_dict("New Chat")
                self._write_raw({"active_chat_id": first["id"], "chats": [first]})

    def _read_raw(self) -> dict:
        with self._lock:
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "chats" in data:
                    return data
            except Exception:
                pass
            first = self._new_chat_dict("New Chat")
            return {"active_chat_id": first["id"], "chats": [first]}

    def _write_raw(self, data: dict):
        with self._lock:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def get_chats(self) -> list:
        data = self._read_raw()
        chats = data.get("chats", [])
        return sorted(
            chats,
            key=lambda c: (1 if c.get("pinned") else 0, c.get("updated_at", "")),
            reverse=True
        )

    def get_active_chat(self) -> dict:
        data = self._read_raw()
        active_id = data.get("active_chat_id")
        for c in data.get("chats", []):
            if c.get("id") == active_id:
                return c
        if data.get("chats"):
            data["active_chat_id"] = data["chats"][0]["id"]
            self._write_raw(data)
            return data["chats"][0]
        first = self._new_chat_dict("New Chat")
        data["chats"] = [first]
        data["active_chat_id"] = first["id"]
        self._write_raw(data)
        return first

    def set_active_chat(self, chat_id: str) -> dict:
        data = self._read_raw()
        for c in data.get("chats", []):
            if c.get("id") == chat_id:
                data["active_chat_id"] = chat_id
                self._write_raw(data)
                return c
        return self.get_active_chat()

    def create_chat(self, title: str = "New Chat") -> dict:
        data = self._read_raw()
        new_c = self._new_chat_dict(title=title)
        data.setdefault("chats", []).insert(0, new_c)
        data["active_chat_id"] = new_c["id"]
        self._write_raw(data)
        return new_c

    def delete_chat(self, chat_id: str) -> dict:
        data = self._read_raw()
        chats = [c for c in data.get("chats", []) if c.get("id") != chat_id]
        if not chats:
            chats = [self._new_chat_dict("New Chat")]
        data["chats"] = chats
        if data.get("active_chat_id") == chat_id:
            s_chats = sorted(
                chats,
                key=lambda c: (1 if c.get("pinned") else 0, c.get("updated_at", "")),
                reverse=True
            )
            data["active_chat_id"] = s_chats[0]["id"]
        self._write_raw(data)
        return self.get_active_chat()

    def rename_chat(self, chat_id: str, new_title: str):
        if not new_title.strip():
            return
        data = self._read_raw()
        for c in data.get("chats", []):
            if c.get("id") == chat_id:
                c["title"] = new_title.strip()
                break
        self._write_raw(data)

    def toggle_pin_chat(self, chat_id: str) -> bool:
        data = self._read_raw()
        new_state = False
        for c in data.get("chats", []):
            if c.get("id") == chat_id:
                c["pinned"] = not bool(c.get("pinned", False))
                new_state = c["pinned"]
                break
        self._write_raw(data)
        return new_state

    def get_all(self) -> list:
        return self.get_active_chat().get("messages", [])

    def add_message(self, sender: str, text: str, ts: str = None, attachment: dict = None):
        ts = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = self._read_raw()
        active_id = data.get("active_chat_id")
        target = None
        for c in data.get("chats", []):
            if c.get("id") == active_id:
                target = c
                break
        if target is None:
            target = self._new_chat_dict("New Chat")
            data.setdefault("chats", []).insert(0, target)
            data["active_chat_id"] = target["id"]

        entry = {"sender": sender, "text": text, "timestamp": ts}
        if attachment:
            entry["attachment"] = attachment

        msgs = target.setdefault("messages", [])
        msgs.append(entry)
        target["messages"] = msgs[-250:]
        target["updated_at"] = ts

        if sender == "user" and target.get("title") in ("New Chat", "General Chat"):
            clean_txt = re.sub(r"\s+", " ", text).strip()
            if clean_txt:
                target["title"] = clean_txt[:34] + ("..." if len(clean_txt) > 34 else "")

        self._write_raw(data)
        return entry

    def clear(self, chat_id: str = None):
        data = self._read_raw()
        cid = chat_id or data.get("active_chat_id")
        for c in data.get("chats", []):
            if c.get("id") == cid:
                c["messages"] = []
                c["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break
        self._write_raw(data)


# ==========================================
# 2. WHATSAPP / TELEGRAM CHAT BUBBLES + RIGHT-CLICK (COPY, SPEAK)
# ==========================================
class ChatBubbleWidget(QWidget):
    """Single message bubble with inline attachments and Right-Click Context Menu ('Copy', 'Speak')."""
    def __init__(self, text: str, is_user: bool = True, timestamp: str = "", attachment: dict = None, parent=None):
        super().__init__(parent)
        self.raw_text = (text or "").strip()
        self.msg_lbl = None

        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 4, 10, 4)

        self.bubble = QFrame()
        self.bubble.setMaximumWidth(580)
        self.bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)
        self.bubble.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.bubble.customContextMenuRequested.connect(
            lambda pos: self._show_text_context_menu(self.bubble.mapToGlobal(pos))
        )

        b_lay = QVBoxLayout(self.bubble)
        b_lay.setContentsMargins(13, 8, 13, 7)
        b_lay.setSpacing(5)

        sender_lbl = QLabel("YOU" if is_user else "A.R.C AI")
        sender_lbl.setStyleSheet(
            f"color: {'#00ffe7' if is_user else '#ffc300'}; font-size: 10px; font-weight: bold; border: none; background: transparent;"
        )
        b_lay.addWidget(sender_lbl)

        if attachment and isinstance(attachment, dict):
            att_type = attachment.get("type", "file")
            att_path = attachment.get("path", "")
            att_name = attachment.get("name") or os.path.basename(att_path)
            att_size = attachment.get("size_str", "")

            if att_type in ("image", "screenshot", "camera", "generated_image") and os.path.exists(att_path):
                img_lbl = QLabel()
                pix = QPixmap(att_path)
                if not pix.isNull():
                    scaled = pix.scaled(
                        360, 250,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    img_lbl.setPixmap(scaled)
                    img_lbl.setStyleSheet(
                        "border: 1px solid rgba(0, 255, 231, 0.35); border-radius: 8px; padding: 2px; background: #040914;"
                    )
                    b_lay.addWidget(img_lbl)

            file_badge = QLabel(f"📎 {att_name}" + (f"  ({att_size})" if att_size else ""))
            file_badge.setWordWrap(True)
            file_badge.setStyleSheet("""
                color: #9be8ff;
                background: rgba(0, 255, 231, 0.10);
                border: 1px solid rgba(0, 255, 231, 0.28);
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: bold;
            """)
            b_lay.addWidget(file_badge)

        if self.raw_text:
            self.msg_lbl = QLabel(self.raw_text)
            self.msg_lbl.setWordWrap(True)
            self.msg_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.msg_lbl.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.msg_lbl.customContextMenuRequested.connect(
                lambda pos: self._show_text_context_menu(self.msg_lbl.mapToGlobal(pos))
            )
            self.msg_lbl.setStyleSheet(
                "color: #eaf6ff; font-family: 'Segoe UI'; font-size: 13px; border: none; background: transparent;"
            )
            b_lay.addWidget(self.msg_lbl)

        short_ts = timestamp[-8:-3] if len(timestamp) >= 8 else (timestamp or datetime.now().strftime("%H:%M"))
        tick = " ✓✓" if is_user else ""
        time_lbl = QLabel(f"{short_ts}{tick}")
        time_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        time_lbl.setStyleSheet(
            "color: rgba(160, 196, 224, 0.65); font-size: 10px; border: none; background: transparent;"
        )
        b_lay.addWidget(time_lbl)

        if is_user:
            self.bubble.setStyleSheet("""
                QFrame {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                                stop:0 #0c3b4e, stop:1 #092c3d);
                    border: 1px solid rgba(0, 255, 231, 0.45);
                    border-top-left-radius: 15px;
                    border-top-right-radius: 15px;
                    border-bottom-left-radius: 15px;
                    border-bottom-right-radius: 3px;
                }
            """)
            outer.addStretch(1)
            outer.addWidget(self.bubble)
        else:
            self.bubble.setStyleSheet("""
                QFrame {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                                stop:0 #0c1b33, stop:1 #081224);
                    border: 1px solid #1b3d63;
                    border-top-left-radius: 15px;
                    border-top-right-radius: 15px;
                    border-bottom-right-radius: 15px;
                    border-bottom-left-radius: 3px;
                }
            """)
            outer.addWidget(self.bubble)
            outer.addStretch(1)

    def _active_or_full_text(self) -> str:
        if self.msg_lbl and self.msg_lbl.hasSelectedText():
            return self.msg_lbl.selectedText().strip()
        return self.raw_text

    def _show_text_context_menu(self, global_pos: QPoint):
        target_text = self._active_or_full_text()
        if not target_text:
            return

        menu = QMenu(self)
        act_copy = menu.addAction("📋  Copy")
        act_speak = menu.addAction("🔊  Speak (Natural Voice)")
        act_stop_speak = None
        if SpeechEngine.is_speaking():
            menu.addSeparator()
            act_stop_speak = menu.addAction("⏹️  Stop Speaking")

        chosen = menu.exec(global_pos)
        if chosen == act_copy:
            if pyperclip:
                pyperclip.copy(target_text)
            else:
                QApplication.clipboard().setText(target_text)
        elif chosen == act_speak:
            SpeechEngine.speak(target_text)
        elif act_stop_speak and chosen == act_stop_speak:
            SpeechEngine.stop()


class CameraCaptureDialog(QDialog):
    """Modal dialog to snap a photo from the webcam and send to the bot."""
    def __init__(self, camera_tracker_widget=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Capture Photo from Camera")
        self.resize(660, 460)
        self.camera_tracker = camera_tracker_widget
        self.captured_path = None
        self.current_frame = None
        self.own_cap = None

        self.setStyleSheet("""
            QDialog { background-color: #050c18; color: #d0e8ff; font-family: 'Segoe UI'; }
            QPushButton {
                background: rgba(0, 255, 231, 0.16);
                border: 1px solid #00ffe7;
                color: #00ffe7;
                padding: 8px 18px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background: rgba(0, 255, 231, 0.32); }
        """)

        lay = QVBoxLayout(self)
        self.preview = QLabel("Initializing camera...")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background: #02060d; border: 1px solid #143a60; border-radius: 8px;")
        lay.addWidget(self.preview, 1)

        btn_row = QHBoxLayout()
        btn_snap = QPushButton("📸 Click & Send Photo")
        btn_snap.clicked.connect(self._snap_photo)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("border-color: #ff2d6b; color: #ff2d6b; background: rgba(255,45,107,0.15);")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_snap)
        lay.addLayout(btn_row)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_preview)
        self._start_feed()

    def _start_feed(self):
        if self.camera_tracker and self.camera_tracker.cap and self.camera_tracker.cap.isOpened():
            self.timer.start(40)
            return
        if cv2 is not None:
            self.own_cap = cv2.VideoCapture(0)
            self.timer.start(40)
        else:
            self.preview.setText("OpenCV (cv2) is not installed.")

    def _update_preview(self):
        cap = (
            self.camera_tracker.cap
            if (self.camera_tracker and self.camera_tracker.cap and self.camera_tracker.cap.isOpened())
            else self.own_cap
        )
        if cap is None or not cap.isOpened():
            self.preview.setText("Could not access webcam.")
            return
        ok, frame = cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        self.current_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview.setPixmap(pix)

    def _snap_photo(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Camera Not Ready", "No camera frame available yet.")
            return
        fname = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        out_path = os.path.join(UPLOADS_DIR, fname)
        cv2.imwrite(out_path, self.current_frame)
        self.captured_path = out_path
        self.accept()

    def closeEvent(self, event):
        self.timer.stop()
        if self.own_cap is not None and self.own_cap.isOpened():
            self.own_cap.release()
        super().closeEvent(event)


class DiagnosticModeDialog(QDialog):
    """
    Asks the user whether A.R.C should send diagnostic / error / screen context to the AI Brain
    as **Text** (terminal output / logs), **Image** (live screenshot + OCR), or **Both**.
    """
    def __init__(self, reason_title: str, terminal_snippet: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("A.R.C Brain Diagnostic — Choose Format (Text or Image)")
        self.resize(580, 310)
        self.chosen_mode = "text"

        self.setStyleSheet("""
            QDialog { background-color: #060d1a; color: #d0e6ff; font-family: 'Segoe UI'; }
            QTextEdit { background-color: #081224; color: #ff9e9e; border: 1px solid #164068;
                        border-radius: 6px; font-family: 'Consolas'; font-size: 12px; padding: 6px; }
            QPushButton {
                background: rgba(0, 255, 231, 0.15); border: 1px solid #00ffe7; color: #00ffe7;
                padding: 8px 14px; border-radius: 6px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { background: rgba(0, 255, 231, 0.32); }
        """)

        lay = QVBoxLayout(self)
        lbl = QLabel(
            f"<b>⚠️ {reason_title}</b><br><br>"
            "How should A.R.C send this state to the AI Brain so it can analyze & regenerate the next step?"
        )
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        if terminal_snippet:
            preview = QTextEdit()
            preview.setReadOnly(True)
            preview.setPlainText(terminal_snippet[-1000:])
            lay.addWidget(preview, 1)

        btn_row = QHBoxLayout()
        btn_txt = QPushButton("📄 Upload Text (Terminal / Logs)")
        btn_txt.clicked.connect(lambda: self._pick("text"))

        btn_img = QPushButton("🖼️ Upload Image (Screenshot)")
        btn_img.clicked.connect(lambda: self._pick("image"))

        btn_both = QPushButton("🔄 Upload Both (Text + Image)")
        btn_both.clicked.connect(lambda: self._pick("both"))

        btn_row.addWidget(btn_txt)
        btn_row.addWidget(btn_img)
        btn_row.addWidget(btn_both)
        lay.addLayout(btn_row)

    def _pick(self, mode: str):
        self.chosen_mode = mode
        self.accept()


class ChatStreamWidget(QWidget):
    """Multi-Session Chat History Panel with 3-lines (☰) fade toggle, '+ New Chat', Pin, Rename, Delete."""
    def __init__(self, history_store: ChatHistoryStore, parent=None):
        super().__init__(parent)
        self.history_store = history_store
        self.setMinimumHeight(240)
        self._in_list_view = False
        self._fade_anim = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self.header_frame = QFrame()
        self.header_frame.setStyleSheet("""
            QFrame {
                background-color: #071224;
                border: 1px solid #153860;
                border-radius: 10px;
            }
        """)
        h_lay = QHBoxLayout(self.header_frame)
        h_lay.setContentsMargins(10, 6, 10, 6)
        h_lay.setSpacing(10)

        self.btn_hamburger = QPushButton("☰")
        self.btn_hamburger.setToolTip("Show / Hide All Conversations")
        self.btn_hamburger.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_hamburger.setFixedSize(36, 32)
        self.btn_hamburger.setStyleSheet("""
            QPushButton {
                background: rgba(0, 255, 231, 0.12);
                border: 1px solid #00ffe7;
                border-radius: 7px;
                color: #00ffe7;
                font-size: 17px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(0, 255, 231, 0.28);
            }
        """)
        self.btn_hamburger.clicked.connect(self.toggle_conversations_view)

        self.lbl_chat_title = QLabel("New Chat")
        self.lbl_chat_title.setStyleSheet(
            "color: #dff2ff; font-size: 13px; font-weight: bold; border: none; background: transparent;"
        )

        self.btn_quick_new = QPushButton("+ New Chat")
        self.btn_quick_new.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_quick_new.setStyleSheet("""
            QPushButton {
                background: rgba(0, 255, 231, 0.14);
                border: 1px solid #00ffe7;
                border-radius: 7px;
                color: #00ffe7;
                font-size: 11px;
                font-weight: bold;
                padding: 5px 12px;
            }
            QPushButton:hover {
                background: rgba(0, 255, 231, 0.30);
            }
        """)
        self.btn_quick_new.clicked.connect(self.create_new_chat)

        h_lay.addWidget(self.btn_hamburger)
        h_lay.addWidget(self.lbl_chat_title, 1)
        h_lay.addWidget(self.btn_quick_new)
        root.addWidget(self.header_frame)

        self.stack = QStackedWidget()
        self.opacity_effect = QGraphicsOpacityEffect(self.stack)
        self.opacity_effect.setOpacity(1.0)
        self.stack.setGraphicsEffect(self.opacity_effect)

        self.page_chat = QWidget()
        p0_lay = QVBoxLayout(self.page_chat)
        p0_lay.setContentsMargins(0, 0, 0, 0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet("""
            QScrollArea {
                background-color: #050b16;
                border: 1px solid #122d4f;
                border-radius: 10px;
            }
            QScrollBar:vertical {
                background: #050b16;
                width: 8px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #19446e;
                border-radius: 4px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00ffe7;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self.container = QWidget()
        self.container.setStyleSheet("background-color: transparent;")
        self.chat_lay = QVBoxLayout(self.container)
        self.chat_lay.setContentsMargins(8, 10, 8, 10)
        self.chat_lay.setSpacing(6)
        self.chat_lay.addStretch(1)

        self.scroll.setWidget(self.container)
        p0_lay.addWidget(self.scroll)
        self.stack.addWidget(self.page_chat)

        self.page_list = QFrame()
        self.page_list.setStyleSheet("""
            QFrame {
                background-color: #050b16;
                border: 1px solid #122d4f;
                border-radius: 10px;
            }
        """)
        p1_lay = QVBoxLayout(self.page_list)
        p1_lay.setContentsMargins(12, 12, 12, 12)
        p1_lay.setSpacing(10)

        list_top = QHBoxLayout()
        lbl_all = QLabel("ALL CONVERSATIONS (Right-click any chat to Pin, Rename, or Delete)")
        lbl_all.setStyleSheet("color: #7fb2e0; font-size: 11px; font-weight: bold; border: none;")
        btn_new_in_list = QPushButton("＋ Create New Chat")
        btn_new_in_list.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_new_in_list.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00b4d8, stop:1 #00ffe7);
                color: #03101c;
                font-weight: bold;
                font-size: 12px;
                padding: 7px 16px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background: #56cfe1;
            }
        """)
        btn_new_in_list.clicked.connect(self.create_new_chat)
        list_top.addWidget(lbl_all, 1)
        list_top.addWidget(btn_new_in_list)
        p1_lay.addLayout(list_top)

        self.conv_list = QListWidget()
        self.conv_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.conv_list.customContextMenuRequested.connect(self._show_conv_context_menu)
        self.conv_list.itemClicked.connect(self._on_conv_item_clicked)
        self.conv_list.setStyleSheet("""
            QListWidget {
                background-color: #071120;
                border: 1px solid #16385e;
                border-radius: 8px;
                color: #eaf6ff;
                font-family: 'Segoe UI';
                font-size: 13px;
                padding: 6px;
            }
            QListWidget::item {
                padding: 11px 12px;
                border-bottom: 1px solid #0e223d;
                border-radius: 6px;
                margin-bottom: 3px;
            }
            QListWidget::item:hover {
                background-color: rgba(0, 255, 231, 0.10);
            }
            QListWidget::item:selected {
                background-color: rgba(0, 255, 231, 0.22);
                color: #00ffe7;
                border: 1px solid rgba(0, 255, 231, 0.55);
            }
        """)
        p1_lay.addWidget(self.conv_list, 1)
        self.stack.addWidget(self.page_list)

        root.addWidget(self.stack, 1)
        self._load_active_conversation()

    def _animate_to_page(self, target_idx: int):
        if self.stack.currentIndex() == target_idx:
            return
        self._fade_out = QPropertyAnimation(self.opacity_effect, b"opacity", self)
        self._fade_out.setDuration(150)
        self._fade_out.setStartValue(1.0)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)

        def _after_fade_out():
            self.stack.setCurrentIndex(target_idx)
            self._fade_in = QPropertyAnimation(self.opacity_effect, b"opacity", self)
            self._fade_in.setDuration(170)
            self._fade_in.setStartValue(0.0)
            self._fade_in.setEndValue(1.0)
            self._fade_in.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self._fade_in.start()

        self._fade_out.finished.connect(_after_fade_out)
        self._fade_out.start()

    def toggle_conversations_view(self):
        if not self._in_list_view:
            self._in_list_view = True
            self.btn_hamburger.setText("✕")
            self.refresh_conversations_list()
            self._animate_to_page(1)
        else:
            self._in_list_view = False
            self.btn_hamburger.setText("☰")
            self._animate_to_page(0)

    def refresh_conversations_list(self):
        self.conv_list.clear()
        active_id = self.history_store.get_active_chat().get("id")
        for c in self.history_store.get_chats():
            cid = c.get("id")
            title = c.get("title", "New Chat")
            pinned = c.get("pinned", False)
            msgs = c.get("messages", [])
            updated = c.get("updated_at", "")
            last_preview = ""
            if msgs:
                last_txt = msgs[-1].get("text", "").replace("\n", " ").strip()
                last_preview = f" — {last_txt[:45]}..." if len(last_txt) > 45 else f" — {last_txt}"

            pin_icon = "📌 " if pinned else "💬 "
            active_tag = "  [ACTIVE]" if cid == active_id else ""
            label = f"{pin_icon}{title}{active_tag}   ({len(msgs)} msgs • {updated[:16]}){last_preview}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, cid)
            self.conv_list.addItem(item)
            if cid == active_id:
                item.setSelected(True)

    def _on_conv_item_clicked(self, item: QListWidgetItem):
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        if chat_id:
            self.history_store.set_active_chat(chat_id)
            self._load_active_conversation()
            self._in_list_view = False
            self.btn_hamburger.setText("☰")
            self._animate_to_page(0)

    def create_new_chat(self):
        self.history_store.create_chat("New Chat")
        self._load_active_conversation()
        self.refresh_conversations_list()
        if self._in_list_view:
            self._in_list_view = False
            self.btn_hamburger.setText("☰")
            self._animate_to_page(0)

    def _show_conv_context_menu(self, pos: QPoint):
        item = self.conv_list.itemAt(pos)
        if not item:
            return
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        chats = {c["id"]: c for c in self.history_store.get_chats()}
        chat_obj = chats.get(chat_id)
        if not chat_obj:
            return

        menu = QMenu(self)
        is_pinned = bool(chat_obj.get("pinned", False))
        act_pin = menu.addAction("📌 Unpin Conversation" if is_pinned else "📌 Pin Conversation to Top")
        act_rename = menu.addAction("✏️ Rename Conversation...")
        act_clear = menu.addAction("🧹 Clear Messages in Chat")
        menu.addSeparator()
        act_delete = menu.addAction("🗑️ Delete Entire Conversation")

        chosen = menu.exec(self.conv_list.viewport().mapToGlobal(pos))
        if chosen == act_pin:
            self.history_store.toggle_pin_chat(chat_id)
            self.refresh_conversations_list()
            self._update_header_title()
        elif chosen == act_rename:
            new_name, ok = QInputDialog.getText(
                self, "Rename Conversation",
                "Enter new name for this conversation:",
                QLineEdit.EchoMode.Normal,
                chat_obj.get("title", "New Chat")
            )
            if ok and new_name.strip():
                self.history_store.rename_chat(chat_id, new_name.strip())
                self.refresh_conversations_list()
                self._update_header_title()
        elif chosen == act_clear:
            ans = QMessageBox.question(
                self, "Clear Conversation",
                f"Clear all messages inside '{chat_obj.get('title')}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if ans == QMessageBox.StandardButton.Yes:
                self.history_store.clear(chat_id)
                self._load_active_conversation()
                self.refresh_conversations_list()
        elif chosen == act_delete:
            ans = QMessageBox.question(
                self, "Delete Entire Conversation",
                f"Permanently delete the conversation '{chat_obj.get('title')}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if ans == QMessageBox.StandardButton.Yes:
                self.history_store.delete_chat(chat_id)
                self._load_active_conversation()
                self.refresh_conversations_list()

    def _clear_bubble_widgets(self):
        while self.chat_lay.count() > 1:
            item = self.chat_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _update_header_title(self):
        active = self.history_store.get_active_chat()
        pin_prefix = "📌 " if active.get("pinned") else "💬 "
        self.lbl_chat_title.setText(f"{pin_prefix}{active.get('title', 'New Chat')}")

    def _load_active_conversation(self):
        self._clear_bubble_widgets()
        self._update_header_title()
        items = self.history_store.get_all()
        for m in items:
            self._insert_bubble_widget(
                text=m.get("text", ""),
                is_user=(m.get("sender") == "user"),
                timestamp=m.get("timestamp", ""),
                attachment=m.get("attachment")
            )
        QTimer.singleShot(80, self.scroll_to_bottom)

    def _insert_bubble_widget(self, text: str, is_user: bool, timestamp: str, attachment: dict = None):
        bubble = ChatBubbleWidget(
            text, is_user=is_user, timestamp=timestamp, attachment=attachment, parent=self.container
        )
        self.chat_lay.insertWidget(self.chat_lay.count() - 1, bubble)
        QTimer.singleShot(30, self.scroll_to_bottom)

    def add_message(self, text: str, is_user: bool = True, save: bool = True, attachment: dict = None):
        if (not text or not text.strip()) and not attachment:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if save:
            self.history_store.add_message("user" if is_user else "ai", (text or "").strip(), ts, attachment=attachment)
            self._update_header_title()
        self._insert_bubble_widget((text or "").strip(), is_user=is_user, timestamp=ts, attachment=attachment)

    def clear_history(self):
        self.history_store.clear()
        self._clear_bubble_widgets()
        self._update_header_title()

    def scroll_to_bottom(self):
        sb = self.scroll.verticalScrollBar()
        sb.setValue(sb.maximum())


# ==========================================
# 3. VISIBLE BOTTOM TERMINAL WITH AI COMMAND & OUTPUT BRIDGE
# ==========================================
class DirectTerminalWidget(QTextEdit):
    """
    Interactive OS Terminal that also lets the AI paste & execute commands visibly
    and captures output + exit code back so the AI can monitor and debug results.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 180)
        self.cwd = os.getcwd()
        self.user = getpass.getuser() or "kali"
        self.host = socket.gethostname() or "Kali"
        self.cmd_history = []
        self.hist_idx = 0
        self.prompt_pos = 0

        self._ai_cmd_buffer = []
        self._ai_done_event = threading.Event()
        self._ai_exit_code = 0

        self.setAcceptRichText(False)
        self.setUndoRedoEnabled(False)
        self.setCursorWidth(8)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #050a14;
                color: #d8f0ff;
                border: 1px solid #123252;
                border-radius: 6px;
                font-family: 'Fira Code', 'JetBrains Mono', 'Consolas', 'Monospace';
                font-size: 13px;
                padding: 8px;
                selection-background-color: rgba(0, 255, 231, 0.3);
            }
        """)

        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.finished.connect(self._on_proc_finished)

        self._print_prompt()

    def _format_cwd(self) -> str:
        home = os.path.expanduser("~")
        if CURRENT_OS != "Windows" and self.cwd.startswith(home):
            return "~" + self.cwd[len(home):]
        return self.cwd

    def _build_prompt_str(self) -> str:
        if CURRENT_OS == "Windows":
            return f"PS {self.cwd}> "
        return f"┌──({self.user}㉿{self.host})-[{self._format_cwd()}]\n└─$ "

    def _print_prompt(self):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if self.document().characterCount() > 1 and not self.toPlainText().endswith("\n"):
            cursor.insertText("\n")
        cursor.insertText(self._build_prompt_str())
        self.setTextCursor(cursor)
        self.prompt_pos = self.textCursor().position()
        self.ensureCursorVisible()

    def _insert_output(self, text: str):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.setTextCursor(cursor)
        self.prompt_pos = self.textCursor().position()
        self.ensureCursorVisible()

    def _current_command(self) -> str:
        full = self.toPlainText()
        return full[self.prompt_pos:]

    def _replace_current_command(self, new_cmd: str):
        cursor = self.textCursor()
        cursor.setPosition(self.prompt_pos)
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(new_cmd)
        self.setTextCursor(cursor)

    def get_recent_terminal_text(self, max_chars: int = 3000) -> str:
        txt = self.toPlainText()
        return txt[-max_chars:] if len(txt) > max_chars else txt

    def run_ai_command_on_ui(self, cmd: str):
        if self.proc.state() != QProcess.ProcessState.NotRunning:
            self.proc.kill()
            self.proc.waitForFinished(300)
            self._insert_output("\n")
            self._print_prompt()

        self._ai_cmd_buffer = []
        self._ai_exit_code = 0
        self._ai_done_event.clear()
        self._replace_current_command(cmd)
        self._insert_output("\n")
        self.execute_command(cmd)

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()

        if key == Qt.Key.Key_C and (mods & Qt.KeyboardModifier.ControlModifier):
            if self.textCursor().hasSelection():
                super().keyPressEvent(event)
                return
            if self.proc.state() != QProcess.ProcessState.NotRunning:
                self.proc.kill()
                self.proc.waitForFinished(200)
                self._insert_output("^C\n")
            else:
                self._insert_output("^C\n")
                self._print_prompt()
            return

        if key == Qt.Key.Key_L and (mods & Qt.KeyboardModifier.ControlModifier):
            self.clear()
            self._print_prompt()
            return

        if self.proc.state() != QProcess.ProcessState.NotRunning:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                line = self._current_command()
                self._insert_output("\n")
                self.proc.write((line + "\n").encode("utf-8"))
                return
            if key == Qt.Key.Key_Backspace and self.textCursor().position() <= self.prompt_pos:
                return
            super().keyPressEvent(event)
            return

        cursor = self.textCursor()
        if cursor.position() < self.prompt_pos or (cursor.hasSelection() and cursor.selectionStart() < self.prompt_pos):
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.setTextCursor(cursor)

        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Left):
            if self.textCursor().position() <= self.prompt_pos and not self.textCursor().hasSelection():
                return

        if key == Qt.Key.Key_Home:
            c = self.textCursor()
            c.setPosition(self.prompt_pos)
            self.setTextCursor(c)
            return

        if key == Qt.Key.Key_Up:
            if self.cmd_history and self.hist_idx > 0:
                self.hist_idx -= 1
                self._replace_current_command(self.cmd_history[self.hist_idx])
            return

        if key == Qt.Key.Key_Down:
            if self.cmd_history:
                if self.hist_idx < len(self.cmd_history) - 1:
                    self.hist_idx += 1
                    self._replace_current_command(self.cmd_history[self.hist_idx])
                else:
                    self.hist_idx = len(self.cmd_history)
                    self._replace_current_command("")
            return

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            cmd = self._current_command().strip()
            self._insert_output("\n")
            self.execute_command(cmd)
            return

        super().keyPressEvent(event)

    def execute_command(self, cmd: str):
        if not cmd:
            self._print_prompt()
            self._ai_done_event.set()
            return

        if not self.cmd_history or self.cmd_history[-1] != cmd:
            self.cmd_history.append(cmd)
        self.hist_idx = len(self.cmd_history)

        if cmd in ("clear", "cls"):
            self.clear()
            self._print_prompt()
            self._ai_done_event.set()
            return

        if cmd == "cd" or cmd.startswith("cd "):
            target = cmd[3:].strip().strip("\"'") if len(cmd) > 2 else "~"
            expanded = os.path.expanduser(target)
            new_dir = os.path.abspath(os.path.join(self.cwd, expanded))
            if os.path.isdir(new_dir):
                self.cwd = new_dir
                self._ai_cmd_buffer.append(f"Changed directory to {self.cwd}\n")
            else:
                msg = f"cd: no such file or directory: {target}\n"
                self._insert_output(msg)
                self._ai_cmd_buffer.append(msg)
                self._ai_exit_code = 1
            self._print_prompt()
            self._ai_done_event.set()
            return

        self.proc.setWorkingDirectory(self.cwd)
        if CURRENT_OS == "Windows":
            if shutil.which("powershell.exe"):
                self.proc.start("powershell.exe", ["-NoProfile", "-Command", cmd])
            else:
                self.proc.start("cmd.exe", ["/c", cmd])
        else:
            shell_bin = "/bin/zsh" if os.path.exists("/bin/zsh") else "/bin/bash"
            self.proc.start(shell_bin, ["-c", cmd])

    def _on_proc_output(self):
        raw = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        clean = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\].*?\x07", "", raw)
        if clean:
            self._ai_cmd_buffer.append(clean)
            self._insert_output(clean)

    def _on_proc_finished(self, code, _status):
        self._ai_exit_code = code
        self._print_prompt()
        self._ai_done_event.set()


# ==========================================
# 4. SMART FUZZY FILE FINDER, FILE WORKSPACE EDITOR & MULTI-DELETE DIALOG
# ==========================================
SYSTEM_PROTECTED_PREFIXES = (
    "/bin", "/sbin", "/usr", "/etc", "/boot", "/lib", "/lib64", "/sys", "/proc", "/dev", "/var/lib",
    "c:\\windows", "c:\\program files", "c:\\program files (x86)", "c:\\system volume information"
)


def is_system_protected_path(path: str) -> bool:
    if not path:
        return False
    ap = os.path.abspath(os.path.expanduser(path)).lower()
    if ap in ("/", "c:\\", os.path.expanduser("~").lower()):
        return True
    return any(ap == pref or ap.startswith(pref + os.sep) for pref in SYSTEM_PROTECTED_PREFIXES)


def normalize_filename(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolve_search_directory(dest_hint: str) -> str:
    if not dest_hint:
        return BASE_DIR
    d = dest_hint.strip().strip("\"'")
    low = d.lower()
    home = os.path.expanduser("~")

    shortcuts = {
        "here": BASE_DIR,
        "current": BASE_DIR,
        "current folder": BASE_DIR,
        "this folder": BASE_DIR,
        "beta": BASE_DIR,
        "desktop": os.path.join(home, "Desktop"),
        "my desktop": os.path.join(home, "Desktop"),
        "downloads": os.path.join(home, "Downloads"),
        "download": os.path.join(home, "Downloads"),
        "documents": os.path.join(home, "Documents"),
        "document": os.path.join(home, "Documents"),
        "home": home,
        "~": home,
    }
    if low in shortcuts:
        return shortcuts[low]

    expanded = os.path.expanduser(d)
    if os.path.isdir(expanded):
        return os.path.abspath(expanded)

    rel_home = os.path.join(home, d)
    if os.path.isdir(rel_home):
        return os.path.abspath(rel_home)

    rel_base = os.path.join(BASE_DIR, d)
    if os.path.isdir(rel_base):
        return os.path.abspath(rel_base)

    for parent in (home, os.path.join(home, "Desktop"), os.path.join(home, "Documents"), BASE_DIR):
        if os.path.isdir(parent):
            for entry in os.listdir(parent):
                full = os.path.join(parent, entry)
                if os.path.isdir(full) and normalize_filename(entry) == normalize_filename(d):
                    return full

    return BASE_DIR


def find_fuzzy_matching_files(query: str, search_dir: str, max_depth: int = 3, ext_filter: str = ""):
    q_norm = normalize_filename(query)
    if not q_norm:
        return []

    exact_or_prefix_matches = []
    fuzzy_candidates = []
    base_depth = search_dir.rstrip(os.sep).count(os.sep)

    for root, dirs, files in os.walk(search_dir):
        cur_depth = root.rstrip(os.sep).count(os.sep) - base_depth
        if cur_depth >= max_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for name in files + dirs:
            if name.startswith(".") or name == "__pycache__":
                continue
            full_path = os.path.join(root, name)
            stem, ext = os.path.splitext(name)
            if ext_filter and ext.lower() != ext_filter.lower() and not os.path.isdir(full_path):
                continue
            n_full = normalize_filename(name)
            n_stem = normalize_filename(stem)
            if not n_stem:
                continue

            if q_norm in n_stem or q_norm in n_full or n_stem in q_norm:
                ratio = difflib.SequenceMatcher(None, q_norm, n_stem).ratio()
                exact_or_prefix_matches.append((1.0 + ratio, full_path, name))
            else:
                ratio = difflib.SequenceMatcher(None, q_norm, n_stem).ratio()
                alpha_stem = re.sub(r"\d+$", "", n_stem)
                alpha_ratio = difflib.SequenceMatcher(None, q_norm, alpha_stem).ratio() if alpha_stem else 0.0
                best_r = max(ratio, alpha_ratio)
                if best_r >= 0.55:
                    fuzzy_candidates.append((best_r, full_path, name))

    if exact_or_prefix_matches:
        exact_or_prefix_matches.sort(key=lambda x: (-x[0], x[2]))
        return [p for _, p, _ in exact_or_prefix_matches]

    if fuzzy_candidates:
        fuzzy_candidates.sort(key=lambda x: (-x[0], x[2]))
        top_score = fuzzy_candidates[0][0]
        return [p for score, p, _ in fuzzy_candidates if score >= top_score - 0.15]

    return []


class FileWorkspaceDialog(QDialog):
    """Interactive dialog when the user asks to find/edit/manipulate a file or folder."""
    def __init__(self, query: str, search_dir: str, matches: list, ai_engine=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"File & Folder Workspace — Matches for '{query}'")
        self.resize(820, 620)
        self.matches = matches
        self.ai_engine = ai_engine
        self.current_path = matches[0] if matches else ""
        self.action_summary = "No changes made."

        self.setStyleSheet("""
            QDialog { background-color: #060d1a; color: #d0e6ff; font-family: 'Segoe UI'; }
            QListWidget {
                background-color: #091426; color: #00ffe7; border: 1px solid #164068;
                border-radius: 6px; font-family: 'Consolas'; font-size: 13px; padding: 4px;
            }
            QListWidget::item { padding: 6px 8px; border-bottom: 1px solid #102542; }
            QListWidget::item:selected {
                background-color: rgba(0, 255, 231, 0.22);
                color: #ffffff;
                border: 1px solid #00ffe7;
            }
            QTextEdit, QLineEdit {
                background-color: #081222; color: #eaf6ff; border: 1px solid #1a4977;
                border-radius: 6px; font-family: 'Consolas'; font-size: 13px; padding: 6px;
            }
            QPushButton {
                background: rgba(0, 255, 231, 0.14); border: 1px solid #00ffe7; color: #00ffe7;
                padding: 7px 14px; border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background: rgba(0, 255, 231, 0.30); }
        """)

        lay = QVBoxLayout(self)
        hdr = QLabel(
            f"Found <b>{len(matches)}</b> item(s) matching <b>\"{query}\"</b> in <span style='color:#00ffe7;'>{search_dir}</span>.<br>"
            "<b>Step 1:</b> Select the file/folder below.  <b>Step 2:</b> Edit manually, ask the AI Brain to rewrite it, or Rename:"
        )
        hdr.setWordWrap(True)
        lay.addWidget(hdr)

        self.lst = QListWidget()
        self.lst.setMaximumHeight(125)
        for p in matches:
            kind = "[FOLDER]" if os.path.isdir(p) else "[FILE]  "
            item = QListWidgetItem(f"{kind} {os.path.basename(p)}   —   {p}")
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.lst.addItem(item)
        self.lst.currentItemChanged.connect(self._on_selection_changed)
        lay.addWidget(self.lst)

        self.lbl_path = QLabel("")
        self.lbl_path.setStyleSheet("color: #ffc300; font-size: 12px; font-weight: bold;")
        lay.addWidget(self.lbl_path)

        self.editor = QTextEdit()
        self.editor.setPlaceholderText("File content will appear here for editing...")
        lay.addWidget(self.editor, 1)

        ai_box = QGroupBox("🧠 Ask AI Brain to Edit / Rewrite This File (HTML, Python, Text)")
        ai_lay = QHBoxLayout(ai_box)
        self.inp_ai_instruct = QLineEdit()
        self.inp_ai_instruct.setPlaceholderText("e.g., 'Add a modern neon navbar and dark theme to this HTML' or 'Fix syntax errors'...")
        btn_ai_rewrite = QPushButton("✨ AI Rewrite File")
        btn_ai_rewrite.clicked.connect(self._ai_rewrite_file)
        ai_lay.addWidget(self.inp_ai_instruct, 1)
        ai_lay.addWidget(btn_ai_rewrite)
        lay.addWidget(ai_box)

        rep_box = QGroupBox("Rewrite / Replace Specific Part")
        rep_lay = QHBoxLayout(rep_box)
        self.inp_find = QLineEdit()
        self.inp_find.setPlaceholderText("Text/part to find...")
        self.inp_replace = QLineEdit()
        self.inp_replace.setPlaceholderText("Rewrite / replace with...")
        btn_replace = QPushButton("Rewrite Part")
        btn_replace.clicked.connect(self._rewrite_part)
        rep_lay.addWidget(self.inp_find, 1)
        rep_lay.addWidget(self.inp_replace, 1)
        rep_lay.addWidget(btn_replace)
        lay.addWidget(rep_box)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("💾 Save Edited Content")
        btn_save.clicked.connect(self._save_content)

        btn_rename_file = QPushButton("✏️ Rename Selected File/Item")
        btn_rename_file.clicked.connect(self._rename_selected_item)

        btn_rename_folder = QPushButton("📁 Rename Parent Folder")
        btn_rename_folder.clicked.connect(self._rename_parent_folder)

        btn_close = QPushButton("Done / Close")
        btn_close.clicked.connect(self.accept)

        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_rename_file)
        btn_row.addWidget(btn_rename_folder)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

        if self.lst.count() > 0:
            self.lst.setCurrentRow(0)

    def _on_selection_changed(self, current: QListWidgetItem, _prev):
        if not current:
            return
        path = current.data(Qt.ItemDataRole.UserRole)
        self.current_path = path
        self.lbl_path.setText(f"Selected: {path}")
        if os.path.isdir(path):
            try:
                entries = os.listdir(path)
                self.editor.setPlainText(f"[Directory containing {len(entries)} items]:\n" + "\n".join(entries[:100]))
                self.editor.setReadOnly(True)
            except Exception as e:
                self.editor.setPlainText(f"Could not list directory: {e}")
        else:
            self.editor.setReadOnly(False)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    self.editor.setPlainText(f.read())
            except Exception as e:
                self.editor.setPlainText(f"Could not read file as text: {e}")

    def _ai_rewrite_file(self):
        if not self.current_path or os.path.isdir(self.current_path):
            QMessageBox.warning(self, "Select a File", "Please select a file to edit with AI.")
            return
        instruction = self.inp_ai_instruct.text().strip()
        if not instruction:
            QMessageBox.warning(self, "Instruction Needed", "Tell the AI Brain how you want to edit this file.")
            return
        if not self.ai_engine:
            return
        cur_content = self.editor.toPlainText()
        new_code, plan_text = self.ai_engine.rewrite_file_with_brain(
            filepath=self.current_path,
            original_content=cur_content,
            instruction=instruction
        )
        if new_code:
            self.editor.setPlainText(new_code)
            self._save_content(custom_note=f"AI Brain edited '{os.path.basename(self.current_path)}' ({instruction}).\nPlan:\n{plan_text}")

    def _rewrite_part(self):
        if not self.current_path or os.path.isdir(self.current_path):
            QMessageBox.warning(self, "Not a Text File", "Select a text file to rewrite a specific part.")
            return
        find_str = self.inp_find.text()
        rep_str = self.inp_replace.text()
        if not find_str:
            QMessageBox.warning(self, "Missing Target", "Enter the specific text/part you want to rewrite.")
            return
        cur_txt = self.editor.toPlainText()
        if find_str not in cur_txt:
            QMessageBox.information(self, "Not Found", f"'{find_str}' was not found in the file.")
            return
        updated = cur_txt.replace(find_str, rep_str)
        self.editor.setPlainText(updated)
        self._save_content(custom_note=f"Rewrote '{find_str}' -> '{rep_str}' in {os.path.basename(self.current_path)}")

    def _save_content(self, custom_note: str = ""):
        if not self.current_path or os.path.isdir(self.current_path):
            QMessageBox.warning(self, "Cannot Save", "Selected item is a folder, not a text file.")
            return
        try:
            with open(self.current_path, "w", encoding="utf-8") as f:
                f.write(self.editor.toPlainText())
            self.action_summary = custom_note or f"Saved updated content to {self.current_path}."
            QMessageBox.information(self, "Saved", self.action_summary)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _rename_selected_item(self):
        if not self.current_path or not os.path.exists(self.current_path):
            return
        old_name = os.path.basename(self.current_path)
        parent_dir = os.path.dirname(self.current_path)
        new_name, ok = QInputDialog.getText(
            self, "Rename File / Folder", "Enter new name:", QLineEdit.EchoMode.Normal, old_name
        )
        if ok and new_name.strip() and new_name.strip() != old_name:
            new_path = os.path.join(parent_dir, new_name.strip())
            try:
                os.rename(self.current_path, new_path)
                self.action_summary = f"Renamed '{old_name}' to '{new_name.strip()}' ({new_path})."
                self.current_path = new_path
                self.lbl_path.setText(f"Selected: {new_path}")
                cur_item = self.lst.currentItem()
                if cur_item:
                    cur_item.setData(Qt.ItemDataRole.UserRole, new_path)
                    cur_item.setText(f"[RENAMED] {new_name.strip()}   —   {new_path}")
                QMessageBox.information(self, "Renamed", self.action_summary)
            except Exception as e:
                QMessageBox.critical(self, "Rename Failed", str(e))

    def _rename_parent_folder(self):
        if not self.current_path:
            return
        folder_path = self.current_path if os.path.isdir(self.current_path) else os.path.dirname(self.current_path)
        old_folder_name = os.path.basename(folder_path)
        grand_parent = os.path.dirname(folder_path)
        new_folder_name, ok = QInputDialog.getText(
            self, "Rename Folder",
            f"Enter new name for folder '{old_folder_name}':",
            QLineEdit.EchoMode.Normal,
            old_folder_name
        )
        if ok and new_folder_name.strip() and new_folder_name.strip() != old_folder_name:
            new_folder_path = os.path.join(grand_parent, new_folder_name.strip())
            try:
                os.rename(folder_path, new_folder_path)
                self.action_summary = f"Renamed folder '{old_folder_name}' to '{new_folder_name.strip()}' ({new_folder_path})."
                QMessageBox.information(self, "Folder Renamed", self.action_summary)
                self.accept()
            except Exception as e:
                QMessageBox.critical(self, "Rename Folder Error", str(e))


class MultiFileDeleteDialog(QDialog):
    def __init__(self, query: str, search_dir: str, file_paths: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Multiple Similar Files Found for '{query}'")
        self.resize(620, 380)
        self.selected_paths = []
        self.setStyleSheet("""
            QDialog { background-color: #060d1a; color: #d0e6ff; font-family: 'Segoe UI'; }
            QListWidget {
                background-color: #091426; color: #00ffe7; border: 1px solid #164068;
                border-radius: 6px; font-family: 'Consolas'; font-size: 13px; padding: 6px;
            }
            QListWidget::item { padding: 6px 8px; border-bottom: 1px solid #102542; }
            QListWidget::item:selected {
                background-color: rgba(255, 45, 107, 0.28);
                color: #ffffff;
                border: 1px solid #ff2d6b;
            }
            QPushButton {
                background: rgba(0, 255, 231, 0.14); border: 1px solid #00ffe7; color: #00ffe7;
                padding: 7px 14px; border-radius: 5px; font-weight: bold;
            }
            QPushButton:hover { background: rgba(0, 255, 231, 0.28); }
        """)

        lay = QVBoxLayout(self)
        hdr = QLabel(
            f"Found <b>{len(file_paths)}</b> similar files matching <b>\"{query}\"</b> in:<br>"
            f"<span style='color:#00ffe7;'>{search_dir}</span><br><br>"
            f"Select one or multiple files to delete (Press <b>Ctrl + A</b> to Select All):"
        )
        hdr.setWordWrap(True)
        lay.addWidget(hdr)

        self.lst = QListWidget()
        self.lst.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for p in file_paths:
            rel = os.path.relpath(p, search_dir)
            kind = "[DIR] " if os.path.isdir(p) else "[FILE]"
            item = QListWidgetItem(f"{kind}  {os.path.basename(p)}    ({rel})")
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.lst.addItem(item)
        if self.lst.count() > 0:
            self.lst.item(0).setSelected(True)
        lay.addWidget(self.lst, 1)

        sc_all = QShortcut(QKeySequence("Ctrl+A"), self)
        sc_all.activated.connect(self.lst.selectAll)

        btn_row = QHBoxLayout()
        btn_sel_all = QPushButton("Select All (Ctrl+A)")
        btn_sel_all.clicked.connect(self.lst.selectAll)

        btn_delete = QPushButton("Delete Selected →")
        btn_delete.setStyleSheet("""
            QPushButton { background: rgba(255, 45, 107, 0.2); border: 1px solid #ff2d6b; color: #ff2d6b;
                          padding: 7px 16px; border-radius: 5px; font-weight: bold; }
            QPushButton:hover { background: rgba(255, 45, 107, 0.38); }
        """)
        btn_delete.clicked.connect(self._on_proceed)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_row.addWidget(btn_sel_all)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_delete)
        lay.addLayout(btn_row)

    def _on_proceed(self):
        items = self.lst.selectedItems()
        if not items:
            QMessageBox.warning(self, "No File Selected", "Please select at least one file (or press Ctrl+A to select all).")
            return
        self.selected_paths = [i.data(Qt.ItemDataRole.UserRole) for i in items]
        self.accept()


# ==========================================
# 5. AI ENGINE (STEP-BY-STEP PLANNER, CLARIFICATION DIALOGS, DIAGNOSTICS)
# ==========================================
SYSTEM_PROMPT = """You are A.R.C, an autonomous self-coding OS, terminal, window, and vision agent running on {OS}.
You control:
1. The user's visible bottom OS Terminal (`run_command`, `read_terminal`)
2. An autonomous Python Script Generator & Executor (`write_and_run_python`) with auto-pip package installation (`install_package`)
3. Interactive User Clarification Dialogs (`prompt_dialog`, `ask_user`):
   - Whenever the user's request is missing a specific detail (for example, if the user says "check my browser" without saying which browser, or if you need a filename, port, or choice), DO NOT guess! First call `prompt_dialog` (e.g. `{"action":"prompt_dialog","question":"Which browser should I check, sir?","options":["Brave","Chrome","Firefox"]}`) with `"done": false`.
   - The user's answer will be returned to you in `Observation:`, and you will then proceed step-by-step!
4. Diagnostic Context Uploader (`send_context_to_brain`):
   - Whenever an error appears on the terminal or screen, or text cannot be copied directly, call `{"action":"send_context_to_brain","reason":"Inspect error or screen state"}` with `"done": false`. The bot will ask the user whether to upload **Text** or **Screenshot Image** and return the full analysis to you so you can regenerate the failed step.
5. Smart Window & Tab Management (`list_windows`, `switch_window`, `switch_tab`)
6. Step-by-Step HTML / Code File Inspector & Editor (`ai_edit_file`, `find_and_edit_file`)

MANDATORY STEP-BY-STEP PLANNING & SELF-REGENERATION PROTOCOL:
1. Break every task into numbered steps inside `"plan": ["Step 1: ...", "Step 2: ..."]`.
2. Execute 1 step (or a small group of steps) at a time with `"done": false` so you can verify the `Observation:` output.
3. If any step fails in `Observation:`, regenerate an alternative step or command in `"steps"` and continue until the task succeeds.
4. Set `"done": true` only when the entire task is finished.

Reply ONLY with one JSON object (no markdown fences):
{
  "thought": "<your reasoning, clarification check, or error recovery>",
  "plan": ["Step 1: ...", "Step 2: ..."],
  "steps": [ <actions> ],
  "done": true|false
}

AVAILABLE ACTIONS:
Clarification & Diagnostic Actions:
 {"action":"prompt_dialog","title":"Input Needed","question":"Which browser would you like me to check, sir?","options":["Brave","Google Chrome","Firefox"]}
 {"action":"ask_user","title":"Info needed","fields":[{"name":"browser","label":"Which browser, sir?","type":"text"}]}
 {"action":"send_context_to_brain","reason":"Check terminal/screen output and choose Text or Image format"}
Python & Package Automation:
 {"action":"write_and_run_python","filename":"create_pdf.py","description":"Create PDF report","packages":["reportlab"],"code":"from reportlab.pdfgen import canvas\\n..."}
 {"action":"install_package","package":"reportlab"}
File & HTML Brain Editing:
 {"action":"ai_edit_file","path":"~/Desktop/index.html","instruction":"Add a neon cyber button and dark mode styling"}
 {"action":"find_and_edit_file","query":"data","destination":"Desktop","ext":".txt"}
Window & Tab Switching:
 {"action":"list_windows"}
 {"action":"switch_window","app":"brave","maximize":true}
 {"action":"switch_tab","direction":"next|prev|index","index":1,"count":1}
Terminal & OS Basics:
 {"action":"reply","text":"..."}
 {"action":"run_command","command":"ls -la ~/Desktop"}
 {"action":"read_terminal"}
 {"action":"remember","key":"candy","value":"I have a red candy today"}
 {"action":"delete_fuzzy","query":"probotauto","destination":"Documents/BETA"}
 {"action":"delete_path","path":"/x/y"}
 {"action":"generate_image","prompt":"a futuristic cyberpunk city at night"}
 {"action":"analyze_screen","question":"What is currently on my screen?"}
 {"action":"navigate_to_target","phrase":"Hello world","click":"none|left|right|double"}
 {"action":"timer","seconds":180,"label":"3 minute timer"}
 {"action":"alarm","time":"15:00","label":"3 PM alarm"}
 {"action":"open_app","app":"brave"}
 {"action":"mouse_move","x":500,"y":500}
 {"action":"mouse_click","button":"left|right|middle","clicks":1,"x":500,"y":500}
 {"action":"keyboard_type","text":"hello"}
 {"action":"keyboard_press","key":"enter"}
 {"action":"keyboard_hotkey","keys":["ctrl","s"]}
 {"action":"copy"}  {"action":"paste"}  {"action":"set_clipboard","text":"..."}
Browser (Selenium):
 {"action":"browser_open","url":"youtube.com"}
 {"action":"browser_search","query":"..."}
 {"action":"browser_click","text":"Create account"}
 {"action":"browser_type","field":"first name","text":"{{first_name}}","submit":false}
 {"action":"browser_press","key":"enter"}
 {"action":"browser_read"}
 {"action":"browser_list_tabs"}
 {"action":"browser_switch_tab","index":0}
 {"action":"browser_find_phones","deep":true}
 {"action":"browser_close"}
Screen (OCR):
 {"action":"screen_read"}
 {"action":"screen_find","phrase":"..."}
 {"action":"screen_click_text","phrase":"Sign in","button":"left|right","clicks":1,"index":0}
 {"action":"screen_select_text","phrase":"..."}
 {"action":"highlight_phrase","phrase":"..."}
Dialogs:
 {"action":"wait_for_user","message":"Please complete step, then press OK."}
 {"action":"confirm","message":"Proceed with X?"}
"""


class AIEngine:
    def __init__(self, config):
        self.config = config
        self.local_llm = None
        self.local_pipe = None
        self.system = SYSTEM_PROMPT.replace("{OS}", CURRENT_OS)

    def is_using_cloud_api(self) -> bool:
        return self.config.get("provider", "Groq (Official SDK)") in ENV_KEYS

    def fetch_available_models(self):
        provider = self.config.get("provider", "Groq (Official SDK)")
        api_key = self.config.get("api_key", "").strip()
        if provider == "Groq (Official SDK)" and api_key:
            skip = ["whisper", "guard", "safeguard", "tts", "orpheus", "playai"]
            if Groq is not None:
                res = Groq(api_key=api_key).models.list()
                return sorted(m.id for m in res.data if not any(s in m.id.lower() for s in skip))
            if requests:
                r = requests.get("https://api.groq.com/openai/v1/models",
                                 headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
                r.raise_for_status()
                return sorted(m["id"] for m in r.json().get("data", []) if not any(s in m["id"].lower() for s in skip))
        elif provider == "Local (Ollama)" and requests:
            r = requests.get(f"{self.config.get('ollama_url', '').rstrip('/')}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        return PROVIDER_MODELS.get(provider, [])

    def chat(self, messages, task=""):
        provider = self.config.get("provider", "Groq (Official SDK)")
        key = self.config.get("api_key", "").strip()
        model = self.config.get("model_name", "").strip()

        if provider in ENV_KEYS and not check_internet_available():
            return json.dumps({
                "action": "reply",
                "text": "⚠️ Internet not available! Kindly switch to Offline AI Models (Local Ollama or Local GGUF File) in Settings -> Configure AI Models."
            })

        if provider in ENV_KEYS and not key:
            return self.offline_fallback_parser(task)
        try:
            if provider == "Groq (Official SDK)":
                return self._groq(messages, key, model or "qwen/qwen3.8-27b")
            if provider == "Gemini":
                return self._gemini(messages, key, model or "gemini-2.5-flash")
            if provider == "OpenAI (ChatGPT)":
                return self._openai_compat("https://api.openai.com/v1/chat/completions", key, model or "gpt-4o-mini", messages)
            if provider == "Grok (xAI)":
                return self._openai_compat("https://api.x.ai/v1/chat/completions", key, model or "grok-2-latest", messages)
            if provider == "Local (Ollama)":
                return self._ollama(messages, model or "gemma4:e4b")
            if provider == "Local GGUF File (llama.cpp)":
                return self._gguf(messages)
        except Exception as e:
            if not check_internet_available() and provider in ENV_KEYS:
                return json.dumps({
                    "action": "reply",
                    "text": "⚠️ Internet connection lost! Kindly switch to Offline AI Models in Settings -> Configure AI Models."
                })
            return json.dumps({"action": "reply", "text": f"AI Error ({provider}): {e}"})
        return self.offline_fallback_parser(task)

    def rewrite_file_with_brain(self, filepath: str, original_content: str, instruction: str) -> tuple:
        prompt = (
            f"You are editing the file: {filepath}\n"
            f"User Instruction: {instruction}\n\n"
            f"--- CURRENT FILE CONTENT ---\n{original_content[:12000]}\n--- END FILE CONTENT ---\n\n"
            "Reply ONLY with a valid JSON object in this exact format:\n"
            "{\"plan\": [\"Step 1: ...\", \"Step 2: ...\"], \"updated_content\": \"<full updated file code>\"}"
        )
        msgs = [
            {"role": "system", "content": "You are an expert HTML, CSS, JavaScript, and Python code editor. Reply only with JSON."},
            {"role": "user", "content": prompt}
        ]
        raw = self.chat(msgs, task=instruction)
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        try:
            data = json.loads(m.group(0) if m else cleaned)
            plan_list = data.get("plan", [])
            plan_str = "\n".join(f"• {p}" for p in plan_list) if isinstance(plan_list, list) else str(plan_list)
            updated = data.get("updated_content", "")
            if updated:
                return updated, plan_str
        except Exception:
            pass
        return "", "Could not parse AI file rewrite response."

    def generate_python_solution_with_brain(self, task_description: str, last_error: str = "") -> dict:
        err_section = f"\nPrevious script failed with this error — debug and fix it:\n{last_error}\n" if last_error else ""
        prompt = (
            f"Write a complete, standalone Python 3 script to accomplish this task on {CURRENT_OS}:\n"
            f"Task: {task_description}\n"
            f"{err_section}\n"
            f"Note: You have full access to `pyautogui` (mouse/keyboard), `cv2`, `reportlab` / `fpdf2` (for PDFs), "
            f"and `from screen_vision import ScreenReader` (in `{BASE_DIR}`).\n"
            "Print clear status messages to stdout so the user and agent can verify the result.\n"
            "Reply ONLY with a JSON object:\n"
            "{\"plan\": [\"Step 1...\", \"Step 2...\"], \"filename\": \"task_script.py\", \"packages\": [\"...\"], \"code\": \"import os\\n...\"}"
        )
        msgs = [
            {"role": "system", "content": "You are an expert Python automation & systems engineer. Reply only with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        raw = self.chat(msgs, task=task_description)
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        try:
            return json.loads(m.group(0) if m else cleaned)
        except Exception:
            return {}

    def analyze_image_with_ai(self, image_path: str, question: str, ocr_summary: str = "") -> str:
        provider = self.config.get("provider", "Groq (Official SDK)")
        key = self.config.get("api_key", "").strip()
        model = self.config.get("model_name", "").strip()

        if provider in ENV_KEYS and not check_internet_available():
            return "⚠️ Internet not available! Kindly switch to Offline AI Models (Local Ollama or Local GGUF) in Settings."

        with open(image_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        prompt_with_ocr = (
            f"User Question / Context: {question}\n\n"
            f"OCR Detected Screen Text & Coordinates:\n{ocr_summary}\n\n"
            "Provide a clear, helpful analysis based on the screen image and OCR data."
        )

        try:
            if provider == "Gemini" and key:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model or 'gemini-2.5-flash'}:generateContent"
                payload = {
                    "contents": [{
                        "parts": [
                            {"text": prompt_with_ocr},
                            {"inline_data": {"mime_type": "image/png", "data": b64_img}}
                        ]
                    }]
                }
                r = requests.post(url, params={"key": key}, json=payload, timeout=60)
                r.raise_for_status()
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]

            if provider in ("OpenAI (ChatGPT)", "Grok (xAI)", "Groq (Official SDK)") and key:
                if provider == "Groq (Official SDK)":
                    vis_model = "meta-llama/llama-4-scout-17b-16e-instruct"
                    api_url = "https://api.groq.com/openai/v1/chat/completions"
                elif provider == "OpenAI (ChatGPT)":
                    vis_model = model if "4o" in model else "gpt-4o-mini"
                    api_url = "https://api.openai.com/v1/chat/completions"
                else:
                    vis_model = "grok-2-vision-latest"
                    api_url = "https://api.x.ai/v1/chat/completions"

                vision_msgs = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_with_ocr},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]
                }]
                r = requests.post(
                    api_url,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": vis_model, "messages": vision_msgs, "temperature": 0.2},
                    timeout=60
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]

            if provider == "Local (Ollama)" and requests:
                url = f"{self.config.get('ollama_url', 'http://localhost:11434').rstrip('/')}/api/chat"
                r = requests.post(
                    url,
                    json={
                        "model": model or "gemma4:e4b",
                        "messages": [{"role": "user", "content": prompt_with_ocr, "images": [b64_img]}],
                        "stream": False
                    },
                    timeout=120
                )
                if r.status_code == 200:
                    return r.json()["message"]["content"]
        except Exception:
            pass

        fallback_msgs = [
            {"role": "system", "content": "You are a helpful screen analyzer assistant."},
            {"role": "user", "content": prompt_with_ocr}
        ]
        raw = self.chat(fallback_msgs, task=question)
        try:
            d = json.loads(raw)
            if isinstance(d, dict) and "text" in d:
                return d["text"]
        except Exception:
            pass
        return raw if raw else f"Screen OCR Summary:\n{ocr_summary}"

    def generate_image(self, prompt: str) -> tuple:
        img_prov = self.config.get("image_provider", "Pollinations AI (Free Flux API)")
        out_path = os.path.join(UPLOADS_DIR, f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        try:
            if img_prov == "Pollinations AI (Free Flux API)":
                if not check_internet_available():
                    return None, "⚠️ Internet not available! Switch to a Local Image Model in Settings -> Configure AI Models."
                model_slug = self.config.get("image_model", "flux") or "flux"
                url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width=1024&height=1024&nologo=true&model={quote(model_slug)}"
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return out_path, f"Generated image for: \"{prompt}\" ({img_prov})"

            if img_prov == "OpenAI DALL-E 3 (API)":
                if not check_internet_available():
                    return None, "⚠️ Internet not available! Switch to a Local Image Model in Settings."
                key = self.config.get("image_api_key", "").strip() or self.config.get("api_keys_by_provider", {}).get("OpenAI (ChatGPT)", "")
                if not key:
                    return None, "OpenAI API Key is required for DALL-E 3. Set it in Settings -> Configure AI Models."
                r = requests.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": "dall-e-3", "prompt": prompt, "n": 1, "size": "1024x1024"},
                    timeout=90
                )
                r.raise_for_status()
                img_url = r.json()["data"][0]["url"]
                img_bytes = requests.get(img_url, timeout=60).content
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                return out_path, f"Generated image with DALL-E 3 for: \"{prompt}\""

            if img_prov == "Local Stable Diffusion WebUI API (127.0.0.1:7860)":
                sd_url = self.config.get("sd_webui_url", "http://127.0.0.1:7860").rstrip("/")
                r = requests.post(
                    f"{sd_url}/sdapi/v1/txt2img",
                    json={"prompt": prompt, "steps": 20, "width": 512, "height": 512},
                    timeout=120
                )
                r.raise_for_status()
                b64_data = r.json()["images"][0]
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(b64_data))
                return out_path, f"Generated image via Local SD WebUI for: \"{prompt}\""

            if img_prov == "Local Diffusers Model (.safetensors / Folder)":
                model_path = self.config.get("local_image_model_path", "").strip()
                if not model_path or not os.path.exists(model_path):
                    return None, "Local image model file/folder not found! Configure it in Settings -> Configure AI Models."
                import torch
                from diffusers import StableDiffusionPipeline
                if self.local_pipe is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    dtype = torch.float16 if device == "cuda" else torch.float32
                    if os.path.isfile(model_path):
                        self.local_pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=dtype).to(device)
                    else:
                        self.local_pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
                image = self.local_pipe(prompt, num_inference_steps=20).images[0]
                image.save(out_path)
                return out_path, f"Generated image locally via Diffusers for: \"{prompt}\""

        except Exception as e:
            return None, f"Image generation error ({img_prov}): {e}"

        return None, "Unknown image generation provider."

    def _groq(self, messages, key, model):
        if Groq is not None:
            r = Groq(api_key=key).chat.completions.create(messages=messages, model=model, temperature=0.1)
            return r.choices[0].message.content
        return self._openai_compat("https://api.groq.com/openai/v1/chat/completions", key, model, messages)

    def _gemini(self, messages, key, model):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        payload = {"systemInstruction": {"parts": [{"text": messages[0]["content"]}]}, "contents": contents}
        r = requests.post(url, params={"key": key}, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    def _openai_compat(self, url, key, model, messages):
        r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json={"model": model, "messages": messages, "temperature": 0.1}, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _ollama(self, messages, model):
        url = f"{self.config.get('ollama_url', 'http://localhost:11434').rstrip('/')}/api/chat"
        r = requests.post(url, json={"model": model, "messages": messages, "stream": False}, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _gguf(self, messages):
        path = self.config.get("local_model_path", "")
        if not path or not os.path.exists(path):
            return json.dumps({"action": "reply", "text": "Local GGUF model file not found! Locate it in Settings -> Configure AI Models."})
        try:
            from llama_cpp import Llama
            if self.local_llm is None:
                self.local_llm = Llama(model_path=path, n_ctx=4096, verbose=False)
            return self.local_llm.create_chat_completion(messages=messages, temperature=0.1)["choices"][0]["message"]["content"]
        except ImportError:
            return json.dumps({"action": "reply", "text": "Install llama-cpp-python to run local .gguf files."})

    def offline_fallback_parser(self, text):
        t = text.lower().strip()
        words = t.split()
        if "browser" in t and ("check" in t or "open" in t) and not any(b in t for b in ("brave", "chrome", "firefox", "edge")):
            return json.dumps({
                "thought": "User asked to check browser without specifying which one.",
                "plan": ["Step 1: Ask user which browser to check via dialog", "Step 2: Open or switch to the chosen browser"],
                "steps": [{"action": "prompt_dialog", "title": "Select Browser", "question": "Which browser would you like me to check, sir?", "options": ["Brave", "Google Chrome", "Firefox"]}],
                "done": False
            })
        if "timer for" in t:
            for i, w in enumerate(words):
                if w.isdigit():
                    unit = words[i + 1] if i + 1 < len(words) else "sec"
                    secs = int(w) * (60 if "min" in unit else 3600 if "hour" in unit else 1)
                    return json.dumps({"action": "timer", "seconds": secs, "label": text})
        if t.startswith("delete "):
            return json.dumps({"action": "delete_fuzzy", "query": text[7:].strip(), "destination": ""})
        if t.startswith("type "):
            return json.dumps({"action": "keyboard_type", "text": text[5:]})
        if t.startswith("open "):
            return json.dumps({"action": "open_app", "app": text[5:].strip()})
        if t.startswith(("go to ", "goto ")):
            return json.dumps({"action": "browser_open", "url": text.split(" ", 2)[-1].strip()})
        return json.dumps({
            "action": "reply",
            "text": "[Offline Mode] Open 'Settings -> Configure AI Models' to set your API key or select a Local Model!"
        })


# ==========================================
# 6. CUSTOM DOCK WIDGET & SCREEN MIRROR
# ==========================================
class SyncedDockWidget(QDockWidget):
    def __init__(self, title: str, menu_action: QAction, parent=None, on_close_cb=None):
        super().__init__(title, parent)
        self.menu_action = menu_action
        self.on_close_cb = on_close_cb
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

    def closeEvent(self, event):
        if self.menu_action.isChecked():
            self.menu_action.blockSignals(True)
            self.menu_action.setChecked(False)
            self.menu_action.blockSignals(False)
        if self.on_close_cb:
            self.on_close_cb()
        super().closeEvent(event)


class ScreenLiveWidget(QWidget):
    def __init__(self, screen_reader: ScreenReader, parent=None):
        super().__init__(parent)
        self.screen_reader = screen_reader
        self.setMinimumSize(460, 280)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        top = QHBoxLayout()
        self.lbl_status = QLabel("◈ LIVE SCREEN FEED (ACTIVE)")
        self.lbl_status.setStyleSheet("color: #00ffe7; font-weight: bold; font-size: 11px;")
        self.btn_ocr = QPushButton("Scan OCR Elements Now")
        self.btn_ocr.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_ocr.setStyleSheet("""
            QPushButton { background: rgba(0,255,231,0.14); border: 1px solid #00ffe7; color: #00ffe7;
                          padding: 4px 10px; border-radius: 5px; font-weight: bold; font-size: 11px; }
            QPushButton:hover { background: rgba(0,255,231,0.28); }
        """)
        top.addWidget(self.lbl_status)
        top.addStretch()
        top.addWidget(self.btn_ocr)
        lay.addLayout(top)

        self.view = QLabel("Starting Live Screen Capture...")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setStyleSheet("background: #02060d; border: 1px solid #0d3050; border-radius: 6px; color: #6c8ba8;")
        lay.addWidget(self.view, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_frame)

    def start_stream(self):
        if not self.timer.isActive():
            self.timer.start(120)
            self._refresh_frame()

    def stop_stream(self):
        self.timer.stop()

    def _refresh_frame(self):
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            pix = screen.grabWindow(0)
            if not pix.isNull():
                scaled = pix.scaled(
                    self.view.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.view.setPixmap(scaled)
                self.lbl_status.setText(f"◈ LIVE SCREEN FEED ({pix.width()}x{pix.height()})")
                return

        try:
            img = self.screen_reader.grab()
            if cv2 is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(
                    self.view.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.view.setPixmap(pix)
                return
        except Exception as e:
            self.view.setText(
                f"Screen capture blocked ({e}).\n"
                "Tip: Run `xhost +local:` in terminal or use an X11 session."
            )


# ==========================================
# 7. SETTINGS -> MEMORY & AI + IMAGE GENERATION CONFIG DIALOGS
# ==========================================
class MemoryManagerDialog(QDialog):
    def __init__(self, mem_store: MemoryStore, parent=None):
        super().__init__(parent)
        self.mem_store = mem_store
        self.setWindowTitle("Settings — Memory Manager (memory.json)")
        self.resize(720, 440)
        self.setStyleSheet("""
            QDialog { background-color: #060d1a; color: #d0e6ff; font-family: 'Segoe UI'; }
            QTableWidget { background-color: #091426; color: #d0e6ff; gridline-color: #163252;
                           border: 1px solid #164068; border-radius: 6px; }
            QHeaderView::section { background-color: #0c1d36; color: #00ffe7; font-weight: bold;
                                   border: 1px solid #163252; padding: 5px; }
            QLineEdit { background: #09162b; border: 1px solid #1c4b78; color: #fff; padding: 6px; border-radius: 5px; }
            QPushButton { background: rgba(0, 255, 231, 0.14); border: 1px solid #00ffe7; color: #00ffe7;
                          padding: 6px 14px; border-radius: 5px; font-weight: bold; }
            QPushButton:hover { background: rgba(0, 255, 231, 0.28); }
        """)
        self._loading = False
        self._build_ui()
        self.refresh_table()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        info = QLabel(f"Memory File Location: {self.mem_store.filepath}\n"
                      "Double-click any cell to manipulate/edit its Key, Content, Date, or Time directly.")
        info.setStyleSheet("color: #7fa6cc; font-size: 11px;")
        lay.addWidget(info)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Key / Category", "Remembered Info / Content", "Date", "Time"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemChanged.connect(self.on_cell_edited)
        lay.addWidget(self.table, 1)

        add_box = QHBoxLayout()
        self.inp_key = QLineEdit()
        self.inp_key.setPlaceholderText("Key (e.g., candy)")
        self.inp_key.setFixedWidth(150)
        self.inp_content = QLineEdit()
        self.inp_content.setPlaceholderText("Memory content (e.g., I have a red candy today)...")
        btn_add = QPushButton("+ Create Memory")
        btn_add.clicked.connect(self.create_memory)
        add_box.addWidget(self.inp_key)
        add_box.addWidget(self.inp_content, 1)
        add_box.addWidget(btn_add)
        lay.addLayout(add_box)

        bot = QHBoxLayout()
        btn_del = QPushButton("Delete Selected Row")
        btn_del.setStyleSheet("border-color: #ff9800; color: #ff9800; background: rgba(255,152,0,0.12);")
        btn_del.clicked.connect(self.delete_selected)

        btn_erase = QPushButton("Erase All Memory")
        btn_erase.setStyleSheet("border-color: #ff2d6b; color: #ff2d6b; background: rgba(255,45,107,0.14);")
        btn_erase.clicked.connect(self.erase_all_memories)

        btn_close = QPushButton("Done")
        btn_close.clicked.connect(self.accept)

        bot.addWidget(btn_del)
        bot.addWidget(btn_erase)
        bot.addStretch()
        bot.addWidget(btn_close)
        lay.addLayout(bot)

    def refresh_table(self):
        self._loading = True
        items = self.mem_store.get_all()
        self.table.setRowCount(len(items))
        for r, item in enumerate(items):
            self.table.setItem(r, 0, QTableWidgetItem(str(item.get("key", "note"))))
            self.table.setItem(r, 1, QTableWidgetItem(str(item.get("content", ""))))
            self.table.setItem(r, 2, QTableWidgetItem(str(item.get("date", ""))))
            self.table.setItem(r, 3, QTableWidgetItem(str(item.get("time", ""))))
        self._loading = False

    def create_memory(self):
        content = self.inp_content.text().strip()
        if not content:
            return
        key = self.inp_key.text().strip() or "user_memory"
        self.mem_store.add_memory(content=content, key=key)
        self.inp_key.clear()
        self.inp_content.clear()
        self.refresh_table()

    def on_cell_edited(self, item: QTableWidgetItem):
        if self._loading:
            return
        r = item.row()
        k = self.table.item(r, 0).text() if self.table.item(r, 0) else "note"
        c = self.table.item(r, 1).text() if self.table.item(r, 1) else ""
        d = self.table.item(r, 2).text() if self.table.item(r, 2) else ""
        t = self.table.item(r, 3).text() if self.table.item(r, 3) else ""
        self.mem_store.update_memory(r, k, c, d, t)

    def delete_selected(self):
        r = self.table.currentRow()
        if r < 0:
            return
        self.mem_store.delete_memory(r)
        self.refresh_table()

    def erase_all_memories(self):
        ans = QMessageBox.question(
            self, "Erase All Memory",
            "Are you sure you want to permanently erase all entries in memory.json?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if ans == QMessageBox.StandardButton.Yes:
            self.mem_store.erase_all()
            self.refresh_table()


class ConfigureAIModelsDialog(QDialog):
    hf_results_signal = pyqtSignal(list)

    def __init__(self, app_window, parent=None):
        super().__init__(parent)
        self.app_win = app_window
        self.config = app_window.config
        self.setWindowTitle("Settings — Configure AI & Image Generation Models")
        self.resize(680, 650)
        self.setStyleSheet("""
            QDialog { background-color: #060d1a; color: #d0e6ff; font-family: 'Segoe UI'; }
            QGroupBox { border: 1px solid #164068; border-radius: 8px; margin-top: 10px; padding-top: 12px;
                        color: #00ffe7; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
            QLineEdit, QComboBox { background: #09162b; border: 1px solid #1c4b78; color: #fff;
                                   padding: 5px 8px; border-radius: 5px; }
            QListWidget { background: #081224; border: 1px solid #164068; color: #a0c4e0; border-radius: 5px; }
            QPushButton { background: rgba(0, 255, 231, 0.14); border: 1px solid #00ffe7; color: #00ffe7;
                          padding: 6px 12px; border-radius: 5px; font-weight: bold; }
            QPushButton:hover { background: rgba(0, 255, 231, 0.28); }
        """)
        self.hf_results_signal.connect(self._populate_hf_list)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)

        grp = QGroupBox("1. Text & Vision AI Model + Superuser Configuration")
        form = QFormLayout(grp)

        self.cb_prov = QComboBox()
        self.cb_prov.addItems(list(PROVIDER_MODELS.keys()))
        self.cb_prov.setCurrentText(self.config.get("provider", "Groq (Official SDK)"))
        self.cb_prov.currentTextChanged.connect(self.on_provider_changed)
        form.addRow("AI Provider:", self.cb_prov)

        mod_row = QHBoxLayout()
        self.cb_model = QComboBox()
        self.cb_model.setEditable(True)
        self.cb_model.addItems(PROVIDER_MODELS.get(self.cb_prov.currentText(), []))
        self.cb_model.setCurrentText(self.config.get("model_name", "qwen/qwen3.8-27b"))
        btn_fetch = QPushButton("Fetch Live Models")
        btn_fetch.clicked.connect(self.fetch_live_models)
        mod_row.addWidget(self.cb_model, 1)
        mod_row.addWidget(btn_fetch)
        form.addRow("Model Name:", mod_row)

        key_row = QHBoxLayout()
        self.inp_key = QLineEdit(self.config.get("api_key", ""))
        self.inp_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.btn_show = QPushButton("Show Key")
        self.btn_show.clicked.connect(self.toggle_key_vis)
        key_row.addWidget(self.inp_key, 1)
        key_row.addWidget(self.btn_show)
        form.addRow("API Key:", key_row)

        gguf_row = QHBoxLayout()
        self.inp_gguf = QLineEdit(self.config.get("local_model_path", ""))
        btn_browse = QPushButton("Browse .gguf...")
        btn_browse.clicked.connect(self.browse_gguf)
        gguf_row.addWidget(self.inp_gguf, 1)
        gguf_row.addWidget(btn_browse)
        form.addRow("Local LLM (.gguf):", gguf_row)

        self.inp_ollama = QLineEdit(self.config.get("ollama_url", "http://localhost:11434"))
        form.addRow("Ollama URL:", self.inp_ollama)

        self.cb_voice = QComboBox()
        self.cb_voice.addItems([
            "en-US-AriaNeural",
            "en-US-GuyNeural",
            "en-US-JennyNeural",
            "en-GB-SoniaNeural",
            "en-IN-NeerjaNeural"
        ])
        self.cb_voice.setCurrentText(self.config.get("tts_voice", "en-US-AriaNeural"))
        form.addRow("Natural TTS Voice:", self.cb_voice)

        self.inp_pwd = QLineEdit()
        self.inp_pwd.setEchoMode(QLineEdit.EchoMode.Password)
        self.inp_pwd.setPlaceholderText("Leave blank to keep existing safety password")
        form.addRow("Safety Password:", self.inp_pwd)

        lay.addWidget(grp)

        img_grp = QGroupBox("2. Image Generation Model (Cloud API or Local Model)")
        img_form = QFormLayout(img_grp)

        self.cb_img_prov = QComboBox()
        self.cb_img_prov.addItems(IMAGE_PROVIDERS)
        self.cb_img_prov.setCurrentText(self.config.get("image_provider", IMAGE_PROVIDERS[0]))
        img_form.addRow("Image Provider:", self.cb_img_prov)

        self.inp_img_model = QLineEdit(self.config.get("image_model", "flux"))
        self.inp_img_model.setPlaceholderText("e.g., flux, turbo, dall-e-3")
        img_form.addRow("API Image Model:", self.inp_img_model)

        self.inp_img_key = QLineEdit(self.config.get("image_api_key", ""))
        self.inp_img_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.inp_img_key.setPlaceholderText("Optional API Key (for DALL-E 3, etc.)")
        img_form.addRow("Image API Key:", self.inp_img_key)

        loc_img_row = QHBoxLayout()
        self.inp_loc_img = QLineEdit(self.config.get("local_image_model_path", ""))
        self.inp_loc_img.setPlaceholderText("Path to local .safetensors or Diffusers model directory")
        btn_browse_img = QPushButton("Import Local Model...")
        btn_browse_img.clicked.connect(self.browse_local_image_model)
        loc_img_row.addWidget(self.inp_loc_img, 1)
        loc_img_row.addWidget(btn_browse_img)
        img_form.addRow("Local Image Model:", loc_img_row)

        lay.addWidget(img_grp)

        save_row = QHBoxLayout()
        btn_save = QPushButton("💾 Save & Apply All Settings")
        btn_save.clicked.connect(self.save_settings)
        btn_test = QPushButton("Test AI Model")
        btn_test.clicked.connect(self.test_model)
        save_row.addWidget(btn_save)
        save_row.addWidget(btn_test)
        lay.addLayout(save_row)

        hf_grp = QGroupBox("3. Search & Download Models (HuggingFace Hub)")
        hf_lay = QVBoxLayout(hf_grp)
        s_row = QHBoxLayout()
        self.inp_hf = QLineEdit("gemma 4b gguf")
        btn_hf = QPushButton("Search Hub")
        btn_hf.clicked.connect(self.search_hf)
        s_row.addWidget(self.inp_hf, 1)
        s_row.addWidget(btn_hf)
        hf_lay.addLayout(s_row)

        self.lst_hf = QListWidget()
        hf_lay.addWidget(self.lst_hf, 1)
        btn_open_hf = QPushButton("Open Selected Model in Browser")
        btn_open_hf.clicked.connect(self.open_hf_page)
        hf_lay.addWidget(btn_open_hf)

        lay.addWidget(hf_grp, 1)

    def on_provider_changed(self, prov: str):
        self.cb_model.clear()
        models = PROVIDER_MODELS.get(prov, [])
        self.cb_model.addItems(models)
        saved_key = self.config.get("api_keys_by_provider", {}).get(prov, "") or os.environ.get(ENV_KEYS.get(prov, ""), "")
        self.inp_key.setText(saved_key)

    def toggle_key_vis(self):
        if self.inp_key.echoMode() == QLineEdit.EchoMode.Password:
            self.inp_key.setEchoMode(QLineEdit.EchoMode.Normal)
            self.btn_show.setText("Hide Key")
        else:
            self.inp_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.btn_show.setText("Show Key")

    def browse_gguf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Local GGUF Model", "", "GGUF Files (*.gguf);;All Files (*.*)")
        if path:
            self.inp_gguf.setText(path)
            self.cb_prov.setCurrentText("Local GGUF File (llama.cpp)")

    def browse_local_image_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Local Image Generation Model (.safetensors / .ckpt)", "",
            "Stable Diffusion Checkpoints (*.safetensors *.ckpt *.pt);;All Files (*.*)"
        )
        if path:
            self.inp_loc_img.setText(path)
            self.cb_img_prov.setCurrentText("Local Diffusers Model (.safetensors / Folder)")

    def fetch_live_models(self):
        self.config["provider"] = self.cb_prov.currentText()
        self.config["api_key"] = self.inp_key.text().strip()
        try:
            models = self.app_win.ai.fetch_available_models()
            if models:
                self.cb_model.clear()
                self.cb_model.addItems(models)
                QMessageBox.information(self, "Models Loaded", f"Fetched {len(models)} live models.")
            else:
                QMessageBox.warning(self, "No Models", "Check your API key or connection.")
        except Exception as e:
            QMessageBox.critical(self, "Fetch Error", str(e))

    def save_settings(self):
        prov = self.cb_prov.currentText()
        key = self.inp_key.text().strip()
        model = self.cb_model.currentText().strip()
        voice = self.cb_voice.currentText().strip()
        self.config.update(
            provider=prov,
            model_name=model,
            api_key=key,
            local_model_path=self.inp_gguf.text().strip(),
            ollama_url=self.inp_ollama.text().strip(),
            tts_voice=voice,
            image_provider=self.cb_img_prov.currentText(),
            image_model=self.inp_img_model.text().strip() or "flux",
            image_api_key=self.inp_img_key.text().strip(),
            local_image_model_path=self.inp_loc_img.text().strip()
        )
        SpeechEngine.VOICE = voice
        self.config.setdefault("api_keys_by_provider", {})[prov] = key
        if self.inp_pwd.text():
            self.config["password_record"] = hash_password(self.inp_pwd.text())
            self.inp_pwd.clear()
        save_config(self.config)
        self.app_win.ai = AIEngine(self.config)
        self.app_win._update_top_badge()
        QMessageBox.information(self, "Saved", f"Active AI: {prov} ({model})\nImage Gen: {self.config['image_provider']}")

    def test_model(self):
        self.save_settings()
        self.accept()
        self.app_win.entry_cmd.setText("Say hello and confirm you are online.")
        self.app_win.handle_user_command()

    def search_hf(self):
        q = self.inp_hf.text().strip()
        if not q or not requests:
            return
        self.lst_hf.clear()
        self.lst_hf.addItem("Searching HuggingFace Hub...")

        def _worker():
            try:
                r = requests.get("https://huggingface.co/api/models",
                                 params={"search": q, "limit": 15, "sort": "downloads", "direction": -1}, timeout=10)
                items = [f"{m.get('id', '')}  (Downloads: {m.get('downloads', 0)})" for m in r.json()]
            except Exception as e:
                items = [f"Error: {e}"]
            self.hf_results_signal.emit(items or ["No models found."])

        threading.Thread(target=_worker, daemon=True).start()

    def _populate_hf_list(self, items: list):
        self.lst_hf.clear()
        self.lst_hf.addItems(items)

    def open_hf_page(self):
        item = self.lst_hf.currentItem()
        if item:
            mid = item.text().split("  (")[0].strip()
            if "/" in mid:
                webbrowser.open(f"https://huggingface.co/{mid}")


class DynamicFormDialogQt(QDialog):
    def __init__(self, parent, title, fields):
        super().__init__(parent)
        self.setWindowTitle(title or "Information Needed")
        self.fields = fields
        self.widgets = {}
        self.result = None
        self.setStyleSheet("""
            QDialog { background: #060d1a; color: #d0e6ff; font-family: 'Segoe UI'; }
            QLineEdit, QComboBox, QTextEdit { background: #09162b; border: 1px solid #00ffe7; color: #fff; padding: 5px; }
            QPushButton { background: rgba(0,255,231,0.18); border: 1px solid #00ffe7; color: #00ffe7; padding: 6px 14px; font-weight: bold; }
        """)
        form = QFormLayout(self)
        for f in fields:
            kind = f.get("type", "text")
            label = f.get("label") or f.get("name", "")
            if kind == "checkbox":
                w = QCheckBox(label)
                w.setChecked(bool(f.get("default", False)))
                form.addRow("", w)
            elif kind == "choice":
                w = QComboBox()
                w.addItems([str(x) for x in f.get("options", [])])
                form.addRow(label + ":", w)
            elif kind == "multiline":
                w = QTextEdit(str(f.get("default", "")))
                w.setFixedHeight(70)
                form.addRow(label + ":", w)
            else:
                w = QLineEdit(str(f.get("default", "")))
                if kind == "password":
                    w.setEchoMode(QLineEdit.EchoMode.Password)
                form.addRow(label + ":", w)
            self.widgets[f["name"]] = (kind, w, f.get("required", True))

        btns = QHBoxLayout()
        btn_ok = QPushButton("Submit")
        btn_ok.clicked.connect(self._submit)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        form.addRow("", btns)

    def _submit(self):
        out = {}
        for name, (kind, w, req) in self.widgets.items():
            if kind == "checkbox":
                val = w.isChecked()
            elif kind == "choice":
                val = w.currentText()
            elif kind == "multiline":
                val = w.toPlainText().strip()
            else:
                val = w.text().strip()
            if req and kind != "checkbox" and val == "":
                QMessageBox.warning(self, "Missing Field", f"Please fill in: {name}")
                return
            out[name] = val
        self.result = out
        self.accept()


# ==========================================
# 8. MAIN PYQT6 DESKTOP AGENT WINDOW
# ==========================================
SAFE_COMMANDS = re.compile(r"^\s*(ls|pwd|echo|date|whoami|uname|hostname|uptime|df|free|which|dir|ver|cd|find|cat|head|tail|wmctrl|xdotool|pip|python|python3)\b[^;&|`$><]*$", re.I)
DESTRUCTIVE = re.compile(r"\b(rm|rmdir|del|erase|format|mkfs|dd|shutdown|reboot|poweroff|kill|killall|chmod|chown|mv|truncate|shred)\b|>\s*/", re.I)
CRITICAL_UI = re.compile(r"\b(delete|remove|erase|pay|purchase|buy now|place order|confirm order|transfer|send money|close account|deactivate)\b", re.I)

REMEMBER_RE = re.compile(r"^\s*(?:please\s+|bro\s+|hey\s+)?remember\s+(?:that\s+)?(.+)$", re.I)
DELETE_IN_RE = re.compile(
    r"^\s*(?:please\s+|bro\s+|hey\s+)?(?:delete|remove|erase)\s+(?:file\s+|folder\s+)?(.+?)(?:\s+(?:in|from|at|inside)\s+(.+))?$",
    re.I
)
WHAT_ON_SCREEN_RE = re.compile(
    r"(?:what(?:'s|\s+is)\s+on\s+my\s+screen|analyze\s+(?:my\s+)?screen|read\s+my\s+screen|look\s+at\s+my\s+screen)",
    re.I
)
NAVIGATE_TO_RE = re.compile(
    r"(?:navigate|move(?:\s+cursor)?|go|point)\s+to\s+(?:a\s+|the\s+)?(?:button|text|element|link|icon|image)?\s*(?:with\s+text\s+|named\s+|called\s+)?[\"']?([^\"'\n]+?)[\"']?(?:\s+and\s+(right\s*click|double\s*click|left\s*click|click))?$",
    re.I
)
IMAGE_GEN_RE = re.compile(
    r"^\s*(?:please\s+|bro\s+|hey\s+)?(?:generate|create|draw|make)\s+(?:an?\s+)?(?:image|picture|photo|art|illustration)\s+(?:of\s+|showing\s+)?(.+)$",
    re.I
)
SWITCH_APP_RE = re.compile(
    r"^\s*(?:please\s+|bro\s+|hey\s+)?(?:switch\s+(?:app|window)?\s*(?:from\s+\w+\s+)?to|maximize|focus|bring\s+up)\s+([a-zA-Z0-9_ -]+)$",
    re.I
)
FIND_EDIT_FILE_RE = re.compile(
    r"(?:look\s+for|search\s+for|find)\s+(?:a\s+)?(?:(txt|text|py|json|md|html|css|js)\s+)?(?:file|folder)s?\s+(?:in|on|inside)?\s*(?:my\s+)?([a-zA-Z0-9_~/\\.-]+)?\s*(?:with\s+name|named|called)?\s*[\"']?([a-zA-Z0-9_.-]+)[\"']?",
    re.I
)


class ThreadBridge(QObject):
    log_signal = pyqtSignal(str)
    chat_bubble_signal = pyqtSignal(str, bool, object)
    busy_signal = pyqtSignal(bool)
    exec_main_signal = pyqtSignal(object, object, object)
    voice_done_signal = pyqtSignal(str, str)


class DesktopAgentWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"A.R.C — Autonomous AI OS Agent ({CURRENT_OS})")
        self.resize(1600, 900)

        self.config = load_config()
        self.ai = AIEngine(self.config)
        self.mem_store = MemoryStore(MEMORY_FILE)
        self.chat_store = ChatHistoryStore(CHAT_HISTORY_FILE)
        self.session_memory = {}
        self.secret_keys = set()
        self.stop_event = threading.Event()
        self.busy = False
        self._last_overload_warn_ts = 0.0

        self.bridge = ThreadBridge()
        self.bridge.chat_bubble_signal.connect(self._append_chat_bubble_ui)
        self.bridge.busy_signal.connect(lambda b: self.btn_stop.setVisible(b))
        self.bridge.exec_main_signal.connect(self._exec_on_main_slot)
        self.bridge.voice_done_signal.connect(self._on_voice_transcribed)

        self.screen = ScreenReader(self.log)
        self.web = WebAgent(self.log, self.wait_for_user)

        self._apply_global_theme()
        self._build_menus_and_docks()
        self._build_central_chat_and_input()

        self.overload_timer = QTimer(self)
        self.overload_timer.timeout.connect(self._check_local_llm_resource_overload)
        self.overload_timer.start(8000)

        QTimer.singleShot(120, self._apply_default_dock_layout)

    def _apply_default_dock_layout(self):
        self.resizeDocks(
            [self.dock_terminal],
            [220],
            Qt.Orientation.Vertical
        )
        self.resizeDocks(
            [self.dock_sysmon, self.dock_camera],
            [380, 560],
            Qt.Orientation.Horizontal
        )

    def _apply_global_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #040914;
            }
            QMenuBar {
                background-color: #061020;
                color: #d0e8ff;
                border-bottom: 1px solid #143558;
                font-family: 'Segoe UI';
                font-size: 13px;
                font-weight: bold;
                padding: 4px 8px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 5px 14px;
                border-radius: 5px;
            }
            QMenuBar::item:selected {
                background: rgba(0, 255, 231, 0.18);
                color: #00ffe7;
            }
            QMenu {
                background-color: #071326;
                color: #c8e2ff;
                border: 1px solid #00ffe7;
                border-radius: 6px;
                padding: 6px;
                font-family: 'Segoe UI';
                font-size: 13px;
            }
            QMenu::item {
                padding: 7px 28px 7px 24px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: rgba(0, 255, 231, 0.22);
                color: #00ffe7;
            }
            QMenu::indicator {
                width: 15px;
                height: 15px;
                margin-left: 6px;
            }
            QDockWidget {
                color: #00ffe7;
                font-family: 'Segoe UI';
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #143a60;
            }
            QDockWidget::title {
                background: #08172e;
                padding: 6px 10px;
                border-bottom: 1px solid #16426b;
            }
        """)

    # ==========================================
    # TOP MENU BAR (WINDOW & SETTINGS) + DOCKS
    # ==========================================
    def _build_menus_and_docks(self):
        self.setDockNestingEnabled(True)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.BottomDockWidgetArea)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        self.win_menu = menubar.addMenu("Window")

        self.act_chat_history = QAction("Chat History", self, checkable=True)
        self.act_chat_history.setChecked(True)
        self.act_chat_history.toggled.connect(self.toggle_chat_history_dock)
        self.win_menu.addAction(self.act_chat_history)

        self.act_terminal = QAction("Terminal", self, checkable=True)
        self.act_terminal.setChecked(True)
        self.act_terminal.toggled.connect(self.toggle_terminal_dock)
        self.win_menu.addAction(self.act_terminal)

        self.act_console = QAction("Console (Backend Logs)", self, checkable=True)
        self.act_console.setChecked(False)
        self.act_console.toggled.connect(self.toggle_console_dock)
        self.win_menu.addAction(self.act_console)

        self.act_camera = QAction("Camera", self, checkable=True)
        self.act_camera.setChecked(True)
        self.act_camera.toggled.connect(self.toggle_camera_dock)
        self.win_menu.addAction(self.act_camera)

        self.act_screen = QAction("Screen", self, checkable=True)
        self.act_screen.toggled.connect(self.toggle_screen_dock)
        self.win_menu.addAction(self.act_screen)

        self.win_menu.addSeparator()

        self.sys_menu = self.win_menu.addMenu("System Monitor")
        self.act_sysmon = QAction("Show System Monitor", self, checkable=True)
        self.act_sysmon.setChecked(True)
        self.act_sysmon.toggled.connect(self.toggle_sysmon_dock)
        self.sys_menu.addAction(self.act_sysmon)
        self.sys_menu.addSeparator()

        self.act_sys_cpu = QAction("CPU", self, checkable=True)
        self.act_sys_cpu.setChecked(True)
        self.act_sys_cpu.toggled.connect(lambda v: self._on_sys_sub_toggled("cpu", v))
        self.sys_menu.addAction(self.act_sys_cpu)

        self.act_sys_ram = QAction("RAM", self, checkable=True)
        self.act_sys_ram.setChecked(True)
        self.act_sys_ram.toggled.connect(lambda v: self._on_sys_sub_toggled("ram", v))
        self.sys_menu.addAction(self.act_sys_ram)

        self.act_sys_storage = QAction("Storage", self, checkable=True)
        self.act_sys_storage.setChecked(True)
        self.act_sys_storage.toggled.connect(lambda v: self._on_sys_sub_toggled("storage", v))
        self.sys_menu.addAction(self.act_sys_storage)

        self.act_sys_network = QAction("Network", self, checkable=True)
        self.act_sys_network.setChecked(True)
        self.act_sys_network.toggled.connect(lambda v: self._on_sys_sub_toggled("network", v))
        self.sys_menu.addAction(self.act_sys_network)

        self.win_menu.addSeparator()
        act_clear_chat = QAction("Clear Active Chat History", self)
        act_clear_chat.triggered.connect(lambda: self.chat_stream.clear_history())
        self.win_menu.addAction(act_clear_chat)

        self.settings_menu = menubar.addMenu("Settings")

        self.act_superuser = QAction("⚡ Superuser Mode (Auto-Approve)", self, checkable=True)
        self.act_superuser.setChecked(bool(self.config.get("superuser_mode", False)))
        self.act_superuser.triggered.connect(self.on_toggle_superuser_mode)
        self.settings_menu.addAction(self.act_superuser)

        self.settings_menu.addSeparator()

        act_cfg_ai = QAction("Configure AI & Image Models...", self)
        act_cfg_ai.triggered.connect(self.open_configure_ai_dialog)
        self.settings_menu.addAction(act_cfg_ai)

        act_memory = QAction("Memory...", self)
        act_memory.triggered.connect(self.open_memory_dialog)
        self.settings_menu.addAction(act_memory)

        self.terminal_widget = DirectTerminalWidget(self)
        self.dock_terminal = SyncedDockWidget("TERMINAL", self.act_terminal, self)
        self.dock_terminal.setWidget(self.terminal_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_terminal)
        self.dock_terminal.show()

        self.console_widget = BackendConsoleWidget(CONSOLE_CAPTURE, self)
        self.dock_console = SyncedDockWidget("BACKEND CONSOLE (STDERR / STDOUT)", self.act_console, self)
        self.dock_console.setWidget(self.console_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_console)
        self.splitDockWidget(self.dock_terminal, self.dock_console, Qt.Orientation.Horizontal)
        self.dock_console.hide()

        self.sysmon_widget = SystemMonitorWidget(self)
        self.dock_sysmon = SyncedDockWidget("SYSTEM RESOURCE STATUS", self.act_sysmon, self)
        self.dock_sysmon.setWidget(self.sysmon_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_sysmon)
        self.dock_sysmon.show()

        self.camera_widget = CameraTrackerWidget(self)
        self.dock_camera = SyncedDockWidget(
            "CAMERA — HOLISTIC BODY & FACE-ID TRACKER",
            self.act_camera,
            self,
            on_close_cb=self.camera_widget.stop_camera
        )
        self.dock_camera.setWidget(self.camera_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_camera)

        self.splitDockWidget(self.dock_sysmon, self.dock_camera, Qt.Orientation.Horizontal)
        self.dock_camera.show()
        self.camera_widget.start_camera()

        self.screen_widget = ScreenLiveWidget(self.screen, self)
        self.screen_widget.btn_ocr.clicked.connect(self.scan_screen_ui)
        self.dock_screen = SyncedDockWidget(
            "REAL-TIME SCREEN MONITOR",
            self.act_screen,
            self,
            on_close_cb=self.screen_widget.stop_stream
        )
        self.dock_screen.setWidget(self.screen_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_screen)
        self.dock_screen.hide()

    # ==========================================
    # SUPERUSER MODE PASSWORD GATE & LOCAL LLM OVERLOAD GUARD
    # ==========================================
    def on_toggle_superuser_mode(self, checked: bool):
        if checked:
            if not self._verify_or_create_password_ui("Enable Superuser Mode (Auto-Approve Actions)"):
                self.act_superuser.blockSignals(True)
                self.act_superuser.setChecked(False)
                self.act_superuser.blockSignals(False)
                return
            self.config["superuser_mode"] = True
            save_config(self.config)
            self._update_top_badge()
            self.show_reply(
                "⚡ Superuser Mode ENABLED.\n"
                "All commands, Python scripts, package installations, app launches, and file deletions will be auto-approved — "
                "EXCEPT system files, deleting an entire folder, or deleting a .txt file with > 500 characters."
            )
        else:
            self.config["superuser_mode"] = False
            save_config(self.config)
            self._update_top_badge()
            self.show_reply("🔒 Superuser Mode DISABLED. Standard security confirmations restored.")

    def _check_local_llm_resource_overload(self):
        if not psutil:
            return
        prov = self.config.get("provider", "")
        is_local_llm = prov in ("Local (Ollama)", "Local GGUF File (llama.cpp)")
        if not is_local_llm:
            return

        try:
            vm = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)
            now = time.time()
            if (vm.percent >= 85.0 or (vm.percent >= 78.0 and cpu >= 92.0)) and (now - self._last_overload_warn_ts > 90):
                self._last_overload_warn_ts = now
                self.show_reply(
                    f"⚠️ SYSTEM MEMORY / CPU OVERLOAD ALERT:\n"
                    f"Your PC usage is critically high (RAM: {vm.percent:.1f}%, CPU: {cpu:.0f}%) while running a Local LLM ({prov}).\n"
                    f"To stay safe from memory shortage and system freeze, kindly switch to an API AI Model (e.g., Groq / Gemini / OpenAI) in Settings -> Configure AI Models."
                )
        except Exception:
            pass

    def _update_top_badge(self):
        su_tag = "  •  🔥 SUPERUSER ON" if self.config.get("superuser_mode") else ""
        self.lbl_active_model.setText(f"⚡ {self.config['provider']} • {self.config['model_name']}{su_tag}")
        self.lbl_active_model.setStyleSheet(
            f"color: {'#00ffe7' if self.config.get('superuser_mode') else '#4a78a8'}; font-size: 11px; font-weight: bold;"
        )

    # ==========================================
    # CENTRAL MULTI-CHAT + PLUG + VOICE MIC BUTTON + INPUT BAR
    # ==========================================
    def _build_central_chat_and_input(self):
        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(8)

        top_status = QHBoxLayout()
        top_status.addStretch()
        self.lbl_active_model = QLabel("")
        self._update_top_badge()
        top_status.addWidget(self.lbl_active_model)
        lay.addLayout(top_status)

        self.chat_stream = ChatStreamWidget(self.chat_store, self)
        lay.addWidget(self.chat_stream, 1)

        inp_row = QHBoxLayout()
        inp_row.setSpacing(8)

        self.btn_plug = QPushButton("➕")
        self.btn_plug.setToolTip(
            f"Attach / Plug Media (Max {MAX_FILE_SIZE_MB} MB):\n"
            "• Upload Image\n"
            "• Click Photo from Camera\n"
            "• Take Quick Screenshot (Auto-Send)\n"
            "• Upload Any File\n"
            "• Send Diagnostic Context (Text or Image)"
        )
        self.btn_plug.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_plug.setFixedSize(42, 42)
        self.btn_plug.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 255, 231, 0.14);
                border: 1.5px solid #00ffe7;
                border-radius: 21px;
                color: #00ffe7;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(0, 255, 231, 0.30);
            }
        """)
        self.btn_plug.clicked.connect(self._show_plug_attachment_menu)

        self.btn_mic = QPushButton("🎙️")
        self.btn_mic.setToolTip("Speak a voice command to A.R.C (Speech-to-Text)")
        self.btn_mic.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mic.setFixedSize(42, 42)
        self.btn_mic.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 195, 0, 0.14);
                border: 1.5px solid #ffc300;
                border-radius: 21px;
                color: #ffc300;
                font-size: 17px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 195, 0, 0.32);
            }
        """)
        self.btn_mic.clicked.connect(self.start_voice_command_listening)

        self.entry_cmd = QLineEdit()
        self.entry_cmd.setPlaceholderText(
            "Type or click 🎙️ to speak: 'check my browser', 'create a pdf on Desktop', 'switch app to brave', 'edit index.html'..."
        )
        self.entry_cmd.setStyleSheet("""
            QLineEdit {
                background-color: rgba(10, 22, 42, 0.95);
                border: 1.5px solid #1b4d7e;
                border-radius: 18px;
                color: #ffffff;
                font-family: 'Segoe UI';
                font-size: 14px;
                padding: 10px 18px;
            }
            QLineEdit:focus {
                border: 1.5px solid #00ffe7;
                background-color: rgba(12, 28, 52, 1.0);
            }
        """)
        self.entry_cmd.returnPressed.connect(self.handle_user_command)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setVisible(False)
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background: rgba(255, 45, 107, 0.2);
                border: 1px solid #ff2d6b;
                color: #ff2d6b;
                font-weight: bold;
                border-radius: 14px;
                padding: 10px 16px;
            }
        """)
        self.btn_stop.clicked.connect(self.stop_task)

        inp_row.addWidget(self.btn_plug)
        inp_row.addWidget(self.btn_mic)
        inp_row.addWidget(self.entry_cmd, 1)
        inp_row.addWidget(self.btn_stop)
        lay.addLayout(inp_row)

    # ==========================================
    # VOICE COMMAND (SPEECH-TO-TEXT) HANDLER
    # ==========================================
    def start_voice_command_listening(self):
        if self.busy:
            self.show_reply("A task is currently running — press STOP first or wait.")
            return
        self.btn_mic.setEnabled(False)
        self.entry_cmd.setPlaceholderText("🎙️ Listening to your voice command (speak now)...")

        def _worker():
            txt, status = VoiceInputEngine.record_and_transcribe(self.config, duration_sec=5)
            self.bridge.voice_done_signal.emit(txt, status)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_voice_transcribed(self, text: str, status: str):
        self.btn_mic.setEnabled(True)
        self.entry_cmd.setPlaceholderText(
            "Type or click 🎙️ to speak: 'check my browser', 'create a pdf on Desktop', 'switch app to brave', 'edit index.html'..."
        )
        if not text:
            self.show_reply(f"🎙️ {status}")
            return
        self.entry_cmd.setText(text)
        self.handle_user_command()

    # ==========================================
    # PLUG / ATTACHMENT MENU & HANDLERS
    # ==========================================
    def _show_plug_attachment_menu(self):
        menu = QMenu(self)
        act_img = menu.addAction("🖼️  Upload Image...")
        act_cam = menu.addAction("📷  Click Image from Camera...")
        act_shot = menu.addAction("🖥️  Take Quick Screenshot (Auto-Send)")
        menu.addSeparator()
        act_diag = menu.addAction("🧠  Send Terminal / Screen Diagnostic (Ask Text or Image)...")
        menu.addSeparator()
        act_file = menu.addAction(f"📎  Upload Any File (Max {MAX_FILE_SIZE_MB} MB)...")

        pos = self.btn_plug.mapToGlobal(QPoint(0, -menu.sizeHint().height() - 6))
        chosen = menu.exec(pos)
        if chosen == act_img:
            self._handle_upload_image()
        elif chosen == act_cam:
            self._handle_click_camera_image()
        elif chosen == act_shot:
            self._handle_quick_screenshot()
        elif chosen == act_diag:
            threading.Thread(
                target=self._ask_and_send_diagnostic_context,
                args=("Manual Brain Diagnostic Check",),
                daemon=True
            ).start()
        elif chosen == act_file:
            self._handle_upload_any_file()

    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.2f} MB"

    def _validate_file_size(self, filepath: str) -> bool:
        try:
            sz = os.path.getsize(filepath)
        except OSError as e:
            QMessageBox.critical(self, "File Error", f"Could not read file:\n{e}")
            return False

        if sz > MAX_FILE_SIZE_BYTES:
            QMessageBox.warning(
                self,
                "File Too Large — Limit Exceeded",
                f"The selected file '{os.path.basename(filepath)}' is {self._format_bytes(sz)}.\n\n"
                f"Maximum allowed file size is {MAX_FILE_SIZE_MB} MB.\n"
                f"Please choose a smaller or compressed file."
            )
            return False
        return True

    def _send_attachment_to_bot(self, filepath: str, att_type: str, default_caption: str):
        if not self._validate_file_size(filepath):
            return

        if self.chat_stream._in_list_view:
            self.chat_stream.toggle_conversations_view()

        sz_str = self._format_bytes(os.path.getsize(filepath))
        fname = os.path.basename(filepath)

        dest_path = filepath
        if os.path.abspath(os.path.dirname(filepath)) != os.path.abspath(UPLOADS_DIR):
            dest_path = os.path.join(UPLOADS_DIR, f"{int(time.time())}_{fname}")
            try:
                shutil.copy2(filepath, dest_path)
            except Exception:
                dest_path = filepath

        user_note = self.entry_cmd.text().strip()
        if user_note:
            self.entry_cmd.clear()
        display_caption = user_note or default_caption

        attachment_meta = {
            "type": att_type,
            "path": dest_path,
            "name": fname,
            "size_str": sz_str
        }

        self.bridge.chat_bubble_signal.emit(display_caption, True, attachment_meta)

        if att_type in ("image", "screenshot", "camera"):
            def _vis_worker():
                self.busy = True
                self.bridge.busy_signal.emit(True)
                try:
                    ocr_txt = ""
                    try:
                        self.screen.scan()
                        ocr_txt = self.screen.summary(35)
                    except Exception:
                        pass
                    ans = self.ai.analyze_image_with_ai(dest_path, display_caption, ocr_txt)
                    self.show_reply(ans)
                finally:
                    self.busy = False
                    self.bridge.busy_signal.emit(False)

            threading.Thread(target=_vis_worker, daemon=True).start()
            return

        ai_prompt = f"[User attached {att_type.upper()}: '{fname}' ({sz_str}) saved at '{dest_path}'.] {display_caption}"
        if att_type == "file":
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".txt", ".py", ".json", ".md", ".csv", ".log", ".html", ".css", ".js", ".sh", ".yaml", ".yml", ".xml"):
                try:
                    with open(dest_path, "r", encoding="utf-8", errors="ignore") as f:
                        snippet = f.read(4000)
                    ai_prompt += f"\n\nFile Content Preview:\n{snippet}"
                except Exception:
                    pass

        if not self.busy:
            self.bridge.busy_signal.emit(True)
            threading.Thread(target=self._run_task, args=(ai_prompt,), daemon=True).start()

    def _handle_upload_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image to Upload", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif);;All Files (*.*)"
        )
        if path:
            self._send_attachment_to_bot(path, "image", f"Analyze this uploaded image: {os.path.basename(path)}")

    def _handle_click_camera_image(self):
        dlg = CameraCaptureDialog(camera_tracker_widget=self.camera_widget, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.captured_path:
            self._send_attachment_to_bot(dlg.captured_path, "camera", "Captured photo from camera. What do you see?")

    def _capture_clean_screenshot(self) -> str:
        fname = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        out_path = os.path.join(UPLOADS_DIR, fname)

        def _grab_on_ui():
            prev_opacity = self.windowOpacity()
            self.setWindowOpacity(0.0)
            QApplication.processEvents()
            time.sleep(0.18)
            saved_ok = False
            try:
                screen = QGuiApplication.primaryScreen()
                if screen is not None:
                    pix = screen.grabWindow(0)
                    if not pix.isNull():
                        saved_ok = pix.save(out_path, "PNG")
                if not saved_ok and cv2 is not None:
                    img = self.screen.grab()
                    saved_ok = cv2.imwrite(out_path, img)
            finally:
                self.setWindowOpacity(prev_opacity)
                QApplication.processEvents()
            return out_path if saved_ok and os.path.exists(out_path) else ""

        return self.run_on_main(_grab_on_ui)

    def _handle_quick_screenshot(self):
        out_path = self._capture_clean_screenshot()
        if out_path:
            self._send_attachment_to_bot(
                out_path,
                "screenshot",
                "Quick screenshot captured and auto-sent to A.R.C AI. What is on my screen?"
            )
        else:
            QMessageBox.warning(self, "Screenshot Error", "Failed to save screenshot image.")

    def _handle_upload_any_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Upload Any File (Max {MAX_FILE_SIZE_MB} MB)", "", "All Files (*.*)"
        )
        if path:
            self._send_attachment_to_bot(path, "file", f"Uploaded file: {os.path.basename(path)}")

    # ---------- Window Menu Dock Toggles ----------
    def _ensure_expanded_size(self):
        if self.height() < 560:
            self.resize(max(self.width(), 1080), 680)

    def toggle_chat_history_dock(self, checked: bool):
        self.chat_stream.setVisible(checked)
        if not checked and not any([
            self.act_terminal.isChecked(),
            self.act_console.isChecked(),
            self.act_camera.isChecked(),
            self.act_screen.isChecked(),
            self.act_sysmon.isChecked()
        ]):
            self.resize(self.width(), 130)
        elif checked and self.height() < 420:
            self.resize(self.width(), 520)

    def toggle_terminal_dock(self, checked: bool):
        if checked:
            self._ensure_expanded_size()
            self.dock_terminal.show()
            self.dock_terminal.raise_()
            self.terminal_widget.setFocus()
        else:
            self.dock_terminal.hide()

    def toggle_console_dock(self, checked: bool):
        if checked:
            self._ensure_expanded_size()
            self.dock_console.show()
            self.dock_console.raise_()
        else:
            self.dock_console.hide()

    def toggle_camera_dock(self, checked: bool):
        if checked:
            self._ensure_expanded_size()
            self.dock_camera.show()
            self.dock_camera.raise_()
            self.camera_widget.start_camera()
        else:
            self.camera_widget.stop_camera()
            self.dock_camera.hide()

    def toggle_screen_dock(self, checked: bool):
        if checked:
            self._ensure_expanded_size()
            self.dock_screen.show()
            self.dock_screen.raise_()
            self.screen_widget.start_stream()
        else:
            self.screen_widget.stop_stream()
            self.dock_screen.hide()

    def toggle_sysmon_dock(self, checked: bool):
        if checked:
            self._ensure_expanded_size()
            self.dock_sysmon.show()
            self.dock_sysmon.raise_()
        else:
            self.dock_sysmon.hide()

    def _on_sys_sub_toggled(self, section: str, checked: bool):
        self.sysmon_widget.set_section_visible(section, checked)
        if checked and not self.act_sysmon.isChecked():
            self.act_sysmon.setChecked(True)

    def open_configure_ai_dialog(self):
        dlg = ConfigureAIModelsDialog(self, self)
        dlg.exec()

    def open_memory_dialog(self):
        dlg = MemoryManagerDialog(self.mem_store, self)
        dlg.exec()

    # ==========================================
    # THREAD-SAFE CHAT BUBBLE & DIALOG HELPERS
    # ==========================================
    def log(self, msg: str):
        pass

    def show_reply(self, text: str, attachment: dict = None):
        self.bridge.chat_bubble_signal.emit(text, False, attachment)

    def _append_chat_bubble_ui(self, text: str, is_user: bool, attachment: object = None):
        att_dict = attachment if isinstance(attachment, dict) else None
        self.chat_stream.add_message(text, is_user=is_user, save=True, attachment=att_dict)

    def run_on_main(self, fn):
        if threading.current_thread() is threading.main_thread():
            return fn()
        ev = threading.Event()
        box = {}
        self.bridge.exec_main_signal.emit(fn, ev, box)
        ev.wait()
        if "e" in box:
            raise box["e"]
        return box.get("v")

    def _exec_on_main_slot(self, fn, ev, box):
        try:
            box["v"] = fn()
        except Exception as e:
            box["e"] = e
        finally:
            ev.set()

    def ask_form(self, title, fields):
        def _show():
            d = DynamicFormDialogQt(self, title, fields)
            return d.result if d.exec() == QDialog.DialogCode.Accepted else None
        return self.run_on_main(_show)

    def ask_prompt_dialog(self, title: str, question: str, options: list = None) -> str:
        """
        Opens a dialog box asking the user for clarification (e.g., 'Which browser would you like me to check, sir?').
        Supports either a dropdown/editable choice list or free-text input, and returns the user's answer to the AI Brain.
        """
        def _ask():
            if options and isinstance(options, list) and len(options) > 0:
                val, ok = QInputDialog.getItem(
                    self, title or "Clarification Needed", question, [str(x) for x in options], 0, True
                )
            else:
                val, ok = QInputDialog.getText(
                    self, title or "Clarification Needed", question, QLineEdit.EchoMode.Normal
                )
            return val.strip() if ok and val else ""

        ans = self.run_on_main(_ask)
        if ans:
            self.bridge.chat_bubble_signal.emit(f"[Dialog Answer] {ans}", True, None)
        return ans

    def _ask_and_send_diagnostic_context(self, reason: str, terminal_snippet: str = "") -> str:
        """
        Asks the user whether to upload diagnostic context to the AI Brain as Text, Image (Screenshot), or Both.
        """
        term_buf = terminal_snippet or self.run_on_main(lambda: self.terminal_widget.get_recent_terminal_text(2000))

        def _choose_mode():
            dlg = DiagnosticModeDialog(reason, term_buf, self)
            return dlg.chosen_mode if dlg.exec() == QDialog.DialogCode.Accepted else "text"

        mode = self.run_on_main(_choose_mode)
        out_parts = []

        if mode in ("text", "both"):
            out_parts.append(f"[Uploaded Terminal / Log Text]:\n{term_buf}")

        if mode in ("image", "both"):
            shot_path = self._capture_clean_screenshot()
            if shot_path:
                ocr_txt = ""
                try:
                    self.screen.scan()
                    ocr_txt = self.screen.summary(35)
                except Exception:
                    pass
                vis_analysis = self.ai.analyze_image_with_ai(
                    shot_path,
                    f"Analyze any error, dialog, or state visible on the screen for: {reason}",
                    ocr_txt
                )
                att_meta = {
                    "type": "screenshot",
                    "path": shot_path,
                    "name": os.path.basename(shot_path),
                    "size_str": self._format_bytes(os.path.getsize(shot_path))
                }
                self.show_reply(f"📸 **Screen Diagnostic Sent to Brain:**\n{vis_analysis}", attachment=att_meta)
                out_parts.append(f"[Uploaded Screen Image Analysis]:\n{vis_analysis}\n[Screen OCR]:\n{ocr_txt}")

        return "\n\n".join(out_parts) if out_parts else "No diagnostic context captured."

    def wait_for_user(self, message):
        def _ask():
            ans = QMessageBox.question(
                self, "Human Assistance Needed",
                f"{message}\n\nPress Yes/OK when done, or No to stop the task.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            return ans == QMessageBox.StandardButton.Yes
        return self.run_on_main(_ask)

    def confirm(self, message):
        if self.config.get("superuser_mode"):
            return True

        def _ask():
            ans = QMessageBox.question(
                self, "Please Confirm", message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            return ans == QMessageBox.StandardButton.Yes
        return self.run_on_main(_ask)

    def set_clipboard(self, text):
        if pyperclip:
            pyperclip.copy(text)
        else:
            self.run_on_main(lambda: QApplication.clipboard().setText(text))

    # ==========================================
    # VISIBLE BOTTOM TERMINAL EXECUTION HELPER
    # ==========================================
    def execute_in_bottom_terminal(self, cmd: str, timeout: float = 45.0, ask_on_error: bool = True) -> str:
        """
        Executes `cmd` visibly inside the bottom terminal.
        If the command fails (non-zero exit code) and `ask_on_error` is True, asks the user
        whether to send the error to the AI Brain as Text or Image (Screenshot) so it can regenerate the step!
        """
        def _trigger():
            if not self.dock_terminal.isVisible():
                self.act_terminal.setChecked(True)
                self.dock_terminal.show()
            self.terminal_widget.run_ai_command_on_ui(cmd)

        self.run_on_main(_trigger)
        self.terminal_widget._ai_done_event.wait(timeout=timeout)
        output = "".join(self.terminal_widget._ai_cmd_buffer).strip()
        code = self.terminal_widget._ai_exit_code
        status_hdr = f"[Exit Code: {code}]\n"
        base_res = (status_hdr + output)[:3800] if output else f"{status_hdr}(command finished with no stdout)"

        if code != 0 and ask_on_error:
            diag_extra = self._ask_and_send_diagnostic_context(
                f"Command `{cmd[:60]}` failed with Exit Code {code}",
                terminal_snippet=base_res
            )
            return f"{base_res}\n\n{diag_extra}"

        return base_res

    # ==========================================
    # AUTONOMOUS PYTHON CODE RUNNER & AUTO-PIP MODULE INSTALLER
    # ==========================================
    @staticmethod
    def _extract_missing_module(error_output: str) -> str:
        m = re.search(r"ModuleNotFoundError:\s*No module named ['\"]([^'\"]+)['\"]", error_output)
        if not m:
            m = re.search(r"ImportError:\s*No module named ['\"]?([a-zA-Z0-9_\-.]+)['\"]?", error_output)
        if m:
            top_mod = m.group(1).split(".")[0]
            return PIP_PACKAGE_MAP.get(top_mod, top_mod)
        return ""

    @staticmethod
    def _scan_imports_from_code(code: str) -> list:
        pkgs = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        pkgs.append(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    pkgs.append(node.module.split(".")[0])
        except Exception:
            pass
        return pkgs

    def ensure_pip_package_installed(self, pkg_name: str) -> str:
        clean_pkg = PIP_PACKAGE_MAP.get(pkg_name.strip(), pkg_name.strip())
        if not clean_pkg:
            return ""
        self.show_reply(f"📦 Auto-installing missing Python module `{clean_pkg}` via bottom terminal...")
        pip_cmd = f"\"{sys.executable}\" -m pip install {clean_pkg}"
        return self.execute_in_bottom_terminal(pip_cmd, timeout=120.0, ask_on_error=False)

    def execute_python_script_with_auto_install(
        self,
        code: str,
        filename: str = "arc_task.py",
        packages: list = None,
        description: str = ""
    ) -> str:
        if not filename.endswith(".py"):
            filename += ".py"
        script_path = os.path.join(SCRIPTS_DIR, os.path.basename(filename))

        bootstrap_header = (
            f"# Auto-generated by A.R.C AI Brain — {description or filename}\n"
            f"import sys, os\n"
            f"if {BASE_DIR!r} not in sys.path:\n"
            f"    sys.path.insert(0, {BASE_DIR!r})\n\n"
        )
        full_code = bootstrap_header + code

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(full_code)

        candidate_mods = list(packages or [])
        for mod_name in self._scan_imports_from_code(code):
            if mod_name in PIP_PACKAGE_MAP:
                candidate_mods.append(mod_name)

        for mod in dict.fromkeys(candidate_mods):
            check = subprocess.run([sys.executable, "-c", f"import {mod.split('.')[0]}"], capture_output=True)
            if check.returncode != 0:
                self.ensure_pip_package_installed(mod)

        run_cmd = f"\"{sys.executable}\" \"{script_path}\""
        for attempt in range(3):
            out = self.execute_in_bottom_terminal(run_cmd, timeout=90.0, ask_on_error=False)
            missing_pkg = self._extract_missing_module(out)
            if missing_pkg:
                self.ensure_pip_package_installed(missing_pkg)
                continue
            if "[Exit Code: 0]" not in out:
                diag_extra = self._ask_and_send_diagnostic_context(
                    f"Python script `{filename}` encountered an error",
                    terminal_snippet=out
                )
                return f"Script `{script_path}` failed:\n{out}\n\n{diag_extra}"
            return f"Script `{script_path}` executed.\n{out}"

        return f"Script `{script_path}` output after auto-install retries:\n{out}"

    # ==========================================
    # SMART BACKEND WINDOW & TAB MANAGER
    # ==========================================
    def get_open_windows_list(self) -> str:
        try:
            if CURRENT_OS == "Linux":
                if shutil.which("wmctrl"):
                    out = subprocess.check_output(["wmctrl", "-l"], text=True, timeout=5)
                    return out.strip() or "No windows reported by wmctrl."
                if shutil.which("xdotool"):
                    ids = subprocess.check_output(["xdotool", "search", "--onlyvisible", "--name", ""], text=True, timeout=5).splitlines()
                    names = []
                    for wid in ids[:25]:
                        try:
                            wname = subprocess.check_output(["xdotool", "getwindowname", wid], text=True, timeout=2).strip()
                            if wname:
                                names.append(f"{wid}: {wname}")
                        except Exception:
                            pass
                    return "\n".join(names) or "No visible X11 windows found."
            elif CURRENT_OS == "Windows":
                ps = "Get-Process | Where-Object {$_.MainWindowTitle -ne ''} | Select-Object Id, ProcessName, MainWindowTitle | Format-Table -AutoSize"
                out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps], text=True, timeout=8)
                return out.strip()
            elif CURRENT_OS == "macOS":
                osa = 'tell application "System Events" to get name of every process whose background only is false'
                out = subprocess.check_output(["osascript", "-e", osa], text=True, timeout=5)
                return out.strip()
        except Exception as e:
            return f"Window listing note: {e}"
        return "Could not query window manager."

    def switch_or_maximize_window(self, app_hint: str, maximize: bool = True) -> str:
        target = app_hint.strip().lower()
        if not target:
            return "No application name provided to switch_window."

        if CURRENT_OS == "Linux":
            if not shutil.which("wmctrl") and not shutil.which("xdotool"):
                self.execute_in_bottom_terminal("sudo -n apt-get install -y wmctrl xdotool 2>/dev/null || true", timeout=15, ask_on_error=False)

            if shutil.which("wmctrl"):
                try:
                    lines = subprocess.check_output(["wmctrl", "-l"], text=True, timeout=4).splitlines()
                    matched_wid = None
                    matched_title = ""
                    for line in lines:
                        parts = line.split(None, 3)
                        if len(parts) >= 4:
                            wid, _desk, _host, title = parts
                            if target in title.lower():
                                matched_wid = wid
                                matched_title = title
                                break
                    if matched_wid:
                        cmd = f"wmctrl -i -a {matched_wid} && wmctrl -i -r {matched_wid} -b add,maximized_vert,maximized_horz"
                        self.execute_in_bottom_terminal(cmd, timeout=5.0, ask_on_error=False)
                        msg = f"Switched to and maximized open window: '{matched_title}' ({matched_wid})."
                        self.show_reply(f"🪟 {msg}")
                        return msg
                except Exception:
                    pass

            if shutil.which("xdotool"):
                try:
                    wids = subprocess.check_output(["xdotool", "search", "--onlyvisible", "--name", target], text=True, timeout=4).splitlines()
                    if wids:
                        wid = wids[0]
                        cmd = f"xdotool windowactivate {wid}"
                        self.execute_in_bottom_terminal(cmd, timeout=5.0, ask_on_error=False)
                        msg = f"Activated open window matching '{app_hint}' (ID: {wid})."
                        self.show_reply(f"🪟 {msg}")
                        return msg
                except Exception:
                    pass

        elif CURRENT_OS == "Windows":
            ps_script = (
                f"$p = Get-Process | Where-Object {{ $_.MainWindowTitle -match '{target}' -or $_.ProcessName -match '{target}' }} | Select-Object -First 1; "
                "if ($p) { "
                "$ws = New-Object -ComObject WScript.Shell; $ws.AppActivate($p.Id); "
                "Write-Output ('Switched to window: ' + $p.MainWindowTitle) "
                "} else { Write-Output 'NOT_FOUND' }"
            )
            out = self.execute_in_bottom_terminal(ps_script, timeout=8.0, ask_on_error=False)
            if "Switched to window:" in out:
                self.show_reply(f"🪟 {out}")
                return out

        elif CURRENT_OS == "macOS":
            osa = f'tell application "{app_hint}" to activate'
            out = self.execute_in_bottom_terminal(f"osascript -e '{osa}'", timeout=6.0, ask_on_error=False)
            self.show_reply(f"🪟 Activated {app_hint}.")
            return out

        return self.act_open_app({"app": app_hint})

    # ==========================================
    # SECURITY GATE, SUPERUSER EXCEPTIONS & FUZZY MULTI-DELETE
    # ==========================================
    def _verify_or_create_password_ui(self, prompt_label: str) -> bool:
        if not self.config.get("password_record"):
            p1, ok1 = QInputDialog.getText(
                self, "Create Safety Password",
                f"No safety password is set yet.\nCreate one now to authorize ({prompt_label}):",
                QLineEdit.EchoMode.Password
            )
            if not ok1 or not p1:
                return False
            p2, ok2 = QInputDialog.getText(
                self, "Confirm Safety Password",
                "Type your password again:",
                QLineEdit.EchoMode.Password
            )
            if not ok2 or p1 != p2:
                QMessageBox.critical(self, "Mismatch", "Passwords did not match. Action cancelled.")
                return False
            self.config["password_record"] = hash_password(p1)
            save_config(self.config)
            return True

        for attempt in range(3):
            pwd, ok = QInputDialog.getText(
                self, "Password Required",
                f"{prompt_label}\nEnter your safety password ({3 - attempt} tries left):",
                QLineEdit.EchoMode.Password
            )
            if not ok:
                return False
            if verify_password(pwd, self.config["password_record"]):
                return True
        QMessageBox.critical(self, "Access Denied", "Incorrect password. Action cancelled.")
        return False

    def request_security_approval(self, description: str, force_even_in_superuser: bool = False):
        if self.config.get("superuser_mode") and not force_even_in_superuser:
            return True

        def _flow():
            ans = QMessageBox.warning(
                self, "Security Alert — Confirm Action",
                f"Are you sure you want to proceed with this action?\n\n{description}\n\nAllow it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if ans != QMessageBox.StandardButton.Yes:
                return False
            return self._verify_or_create_password_ui(description[:120])

        return self.run_on_main(_flow)

    def _check_deletion_requires_manual_approval_in_superuser(self, targets: list) -> tuple:
        reasons = []
        for t in targets:
            if is_system_protected_path(t):
                reasons.append(f"• Protected System Path: {t}")
            elif os.path.isdir(t):
                reasons.append(f"• Entire Folder / Directory: {t}")
            elif os.path.isfile(t) and t.lower().endswith(".txt"):
                try:
                    with open(t, "r", encoding="utf-8", errors="ignore") as f:
                        char_count = len(f.read())
                    if char_count > 500:
                        reasons.append(f"• Text file > 500 characters ({char_count} chars): {os.path.basename(t)}")
                except Exception:
                    pass
        return (len(reasons) > 0), "\n".join(reasons)

    def execute_fuzzy_delete(self, query: str, dest_hint: str = ""):
        search_dir = resolve_search_directory(dest_hint)

        if CURRENT_OS != "Windows":
            self.execute_in_bottom_terminal(f"find \"{search_dir}\" -maxdepth 3 -iname \"*{query}*\" 2>/dev/null", ask_on_error=False)

        direct_candidate = os.path.expanduser(query.strip("\"'"))
        if os.path.exists(direct_candidate):
            matches = [os.path.abspath(direct_candidate)]
        else:
            matches = find_fuzzy_matching_files(query, search_dir)

        if not matches:
            msg = f"No files similar to '{query}' were found in {search_dir}."
            self.show_reply(msg)
            return msg

        if len(matches) > 1:
            def _pick_files():
                dlg = MultiFileDeleteDialog(query, search_dir, matches, self)
                return dlg.selected_paths if dlg.exec() == QDialog.DialogCode.Accepted else None

            targets = self.run_on_main(_pick_files)
            if not targets:
                msg = "File deletion cancelled by user."
                self.show_reply(msg)
                return msg
        else:
            targets = matches

        how = "moved to Trash" if send2trash else "PERMANENTLY deleted"
        names_preview = "\n".join(f"• {os.path.basename(t)}  ({t})" for t in targets[:12])
        if len(targets) > 12:
            names_preview += f"\n... and {len(targets) - 12} more."

        force_confirm, su_reason = self._check_deletion_requires_manual_approval_in_superuser(targets)
        desc = f"DELETE {len(targets)} ITEM(S) ({how}):\n\n{names_preview}"
        if force_confirm and self.config.get("superuser_mode"):
            desc = f"[SUPERUSER SAFETY EXCEPTION TRIGGERED]\n{su_reason}\n\n{desc}"

        if not self.request_security_approval(desc, force_even_in_superuser=force_confirm):
            msg = "Deletion aborted at security password check."
            self.show_reply(msg)
            return msg

        deleted_names = []
        errors = []
        for target in targets:
            try:
                if send2trash:
                    send2trash(target)
                elif os.path.isdir(target):
                    shutil.rmtree(target)
                else:
                    os.remove(target)
                deleted_names.append(os.path.basename(target))
            except Exception as e:
                errors.append(f"{os.path.basename(target)}: {e}")

        summary = f"Deleted {len(deleted_names)} item(s): {', '.join(deleted_names)}"
        if errors:
            summary += f"\nErrors: {'; '.join(errors)}"
        self.show_reply(summary)
        return summary

    def execute_find_and_edit_workflow(self, query: str, dest_hint: str = "Desktop", ext_filter: str = ""):
        search_dir = resolve_search_directory(dest_hint or "Desktop")
        clean_q = re.sub(r"\.(txt|py|json|md|html|css|js)$", "", query.strip(), flags=re.I)

        if CURRENT_OS == "Windows":
            term_cmd = f"Get-ChildItem -Path \"{search_dir}\" -Recurse -ErrorAction SilentlyContinue | Where-Object {{ $_.Name -match \"{clean_q}\" }} | Select-Object -ExpandProperty FullName"
        else:
            term_cmd = f"find \"{search_dir}\" -maxdepth 3 -iname \"*{clean_q}*\" 2>/dev/null"

        term_out = self.execute_in_bottom_terminal(term_cmd, ask_on_error=False)

        matches = find_fuzzy_matching_files(clean_q, search_dir, ext_filter=ext_filter)
        if not matches and os.path.exists(os.path.join(search_dir, query)):
            matches = [os.path.join(search_dir, query)]

        if not matches:
            msg = f"Searched `{search_dir}` via bottom terminal, but no file or folder matching '{query}' was found."
            self.show_reply(msg)
            return f"{msg}\nTerminal Output:\n{term_out}"

        def _open_workspace():
            dlg = FileWorkspaceDialog(query, search_dir, matches, ai_engine=self.ai, parent=self)
            dlg.exec()
            return dlg.action_summary

        summary = self.run_on_main(_open_workspace)
        self.show_reply(f"📂 File Workspace Complete: {summary}")
        return f"Terminal Search Output:\n{term_out}\nUser Workspace Action: {summary}"

    def fill(self, text):
        used_secret = False

        def sub(m):
            nonlocal used_secret
            k = m.group(1).strip()
            if k in self.secret_keys:
                used_secret = True
            return str(self.session_memory.get(k, m.group(0)))

        return re.sub(r"\{\{\s*([\w\- ]+?)\s*\}\}", sub, text), used_secret

    def scan_screen_ui(self):
        def _go():
            try:
                self.screen.scan()
                path = self.screen.annotate("screen_debug.png")
                summary = self.screen.summary(15)
                if path:
                    summary += f"\n\nAnnotated screenshot saved to {os.path.abspath(path)}"
                self.show_reply(summary)
            except Exception as e:
                self.show_reply(f"OCR Screen scan note: {e}")
        threading.Thread(target=_go, daemon=True).start()

    # ==========================================
    # COMMAND EXECUTION & STEP-BY-STEP PLANNER LOOP
    # ==========================================
    def stop_task(self):
        self.stop_event.set()
        self.show_reply("STOP requested — halting after current step.")

    def handle_user_command(self):
        cmd = self.entry_cmd.text().strip()
        if not cmd:
            return
        if self.busy:
            self.show_reply("A task is currently running — press STOP first or wait.")
            return
        self.entry_cmd.clear()

        if self.chat_stream._in_list_view:
            self.chat_stream.toggle_conversations_view()

        self.bridge.chat_bubble_signal.emit(cmd, True, None)

        m_rem = REMEMBER_RE.match(cmd)
        if m_rem:
            fact = m_rem.group(1).strip()
            entry = self.mem_store.add_memory(content=fact, key="remembered_item")
            msg = f"Remembered: \"{entry['content']}\"\nSaved to memory.json on {entry['date']} at {entry['time']}."
            self.show_reply(msg)
            return

        m_del = DELETE_IN_RE.match(cmd)
        if m_del:
            file_query = m_del.group(1).strip()
            dest_folder = (m_del.group(2) or "").strip()
            threading.Thread(
                target=self.execute_fuzzy_delete,
                args=(file_query, dest_folder),
                daemon=True
            ).start()
            return

        if WHAT_ON_SCREEN_RE.search(cmd):
            threading.Thread(
                target=self._run_screen_vision_analysis,
                args=(cmd,),
                daemon=True
            ).start()
            return

        m_nav = NAVIGATE_TO_RE.search(cmd)
        if m_nav:
            target_phrase = m_nav.group(1).strip()
            click_raw = (m_nav.group(2) or "").lower()
            click_mode = "right" if "right" in click_raw else "double" if "double" in click_raw else "left" if "click" in click_raw else "none"
            threading.Thread(
                target=self.act_navigate_to_target,
                args=({"phrase": target_phrase, "click": click_mode},),
                daemon=True
            ).start()
            return

        m_img = IMAGE_GEN_RE.match(cmd)
        if m_img:
            img_prompt = m_img.group(1).strip()
            threading.Thread(
                target=self.act_generate_image,
                args=({"prompt": img_prompt},),
                daemon=True
            ).start()
            return

        m_sw = SWITCH_APP_RE.match(cmd)
        if m_sw:
            target_app = m_sw.group(1).strip()
            threading.Thread(
                target=self.switch_or_maximize_window,
                args=(target_app, True),
                daemon=True
            ).start()
            return

        m_find = FIND_EDIT_FILE_RE.search(cmd)
        if m_find and any(k in cmd.lower() for k in ("look for", "find", "search for")):
            ext_hint = f".{m_find.group(1).lower()}" if m_find.group(1) else ""
            folder_hint = m_find.group(2) or "Desktop"
            name_query = m_find.group(3)
            threading.Thread(
                target=self.execute_find_and_edit_workflow,
                args=(name_query, folder_hint, ext_hint),
                daemon=True
            ).start()
            return

        self.bridge.busy_signal.emit(True)
        threading.Thread(target=self._run_task, args=(cmd,), daemon=True).start()

    def _run_screen_vision_analysis(self, question: str):
        self.busy = True
        self.bridge.busy_signal.emit(True)
        try:
            shot_path = self._capture_clean_screenshot()
            if not shot_path:
                self.show_reply("Could not capture screenshot for screen analysis.")
                return

            ocr_summary = ""
            try:
                self.screen.scan()
                ocr_summary = self.screen.summary(40)
            except Exception:
                pass

            sz_str = self._format_bytes(os.path.getsize(shot_path))
            att_meta = {
                "type": "screenshot",
                "path": shot_path,
                "name": os.path.basename(shot_path),
                "size_str": sz_str
            }
            ans = self.ai.analyze_image_with_ai(shot_path, question, ocr_summary)
            self.show_reply(ans, attachment=att_meta)
        finally:
            self.busy = False
            self.bridge.busy_signal.emit(False)

    @staticmethod
    def _parse(raw):
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        try:
            return json.loads(m.group(0) if m else cleaned), cleaned
        except Exception:
            return None, cleaned

    def _offline_memory_lookup(self, query: str):
        memories = self.mem_store.get_all()
        if not memories:
            return None
        q_words = [w for w in re.findall(r"\w+", query.lower())
                   if w not in ("what", "which", "when", "where", "did", "i", "have", "had", "the", "a", "an", "my", "was", "is")]
        best_m, best_score = None, 0
        for m in reversed(memories):
            hay = f"{m.get('key', '')} {m.get('content', '')}".lower()
            score = sum(1 for w in q_words if w in hay)
            if score > best_score:
                best_score = score
                best_m = m
        if best_m and best_score > 0:
            return (f"From your memory.json: \"{best_m['content']}\"\n"
                    f"Recorded on {best_m['date']} at {best_m['time']}.")
        return None

    def _run_task(self, cmd):
        """
        Step-by-step autonomous agent loop:
        - Breaks tasks into steps (`plan`).
        - If clarification is needed (e.g. "check my browser"), opens `prompt_dialog`, sends the user's answer
          back to the AI Brain, and continues.
        - If a step fails, asks the user whether to send diagnostic context as Text or Image,
          feeds the result back to the AI Brain, regenerates the failed step, and proceeds.
        """
        self.busy = True
        self.stop_event.clear()
        shown_plan = False
        try:
            saved_memories = self.mem_store.get_all()

            prov = self.config.get("provider", "Groq (Official SDK)")
            if prov in ENV_KEYS and not check_internet_available():
                self.show_reply(
                    "⚠️ Internet not available! Kindly switch to Offline AI Models (Local Ollama or Local GGUF File) in Settings -> Configure AI Models."
                )
                return

            if prov in ENV_KEYS and not self.config.get("api_key", "").strip():
                mem_ans = self._offline_memory_lookup(cmd)
                if mem_ans:
                    self.show_reply(mem_ans)
                    return

            active_history = self.chat_store.get_all()[-8:]
            conv_context = "\n".join(
                f"{m.get('sender', 'user').upper()}: {m.get('text', '')}" for m in active_history
            )
            term_snapshot = self.run_on_main(lambda: self.terminal_widget.get_recent_terminal_text(1500))
            open_wins = self.get_open_windows_list()[:1000]

            visible_session = {k: v for k, v in self.session_memory.items() if k not in self.secret_keys}
            messages = [
                {"role": "system", "content": self.ai.system},
                {"role": "user", "content":
                    f"Saved Memories (from memory.json): {json.dumps(saved_memories, ensure_ascii=False)}\n"
                    f"Current Chat Recent Context:\n{conv_context}\n\n"
                    f"Open Desktop Windows:\n{open_wins}\n\n"
                    f"Bottom Terminal Recent Buffer:\n{term_snapshot}\n\n"
                    f"Session Form Info: {json.dumps(visible_session)}\n"
                    f"Secret placeholders available: {sorted(self.secret_keys)}\n\n"
                    f"User Task / Message: {cmd}"},
            ]
            for iteration in range(MAX_ITERATIONS):
                if self.stop_event.is_set():
                    return
                raw = self.ai.chat(messages, task=cmd)
                data, cleaned = self._parse(raw)
                if not isinstance(data, dict):
                    self.show_reply(cleaned)
                    return

                plan = data.get("plan")
                steps = data.get("steps") or ([data] if "action" in data else [])
                if not shown_plan and isinstance(plan, list) and len(plan) > 1 and not (len(steps) == 1 and steps[0].get("action") == "reply"):
                    shown_plan = True
                    plan_formatted = "🧠 **A.R.C Step-by-Step Plan:**\n" + "\n".join(f"  {p}" for p in plan)
                    self.show_reply(plan_formatted)

                messages.append({"role": "assistant", "content": cleaned})
                observations = []
                for step in steps:
                    if self.stop_event.is_set():
                        return
                    obs = self._safe_execute(step, original_user_task=cmd)
                    if obs:
                        observations.append(f"[{step.get('action')}] {obs}")
                if data.get("done", True) is not False:
                    return
                feedback = "\n".join(observations) or "(steps ran with no output)"
                messages.append({
                    "role": "user",
                    "content": (
                        f"Observation (Step {iteration + 1} Result):\n{feedback[:4000]}\n\n"
                        "If the user answered a dialog above, use their answer to proceed with the next steps. "
                        "If any step failed above, regenerate a corrected step and continue."
                    )
                })
                if len(messages) > 16:
                    messages = messages[:2] + messages[-12:]
            self.show_reply("Reached the step limit for this task. Tell me to continue if needed.")
        except Exception as e:
            self.show_reply(f"Task error: {e}")
        finally:
            self.busy = False
            self.bridge.busy_signal.emit(False)

    def _safe_execute(self, step, original_user_task: str = ""):
        try:
            action_name = step.get("action")
            handler = getattr(self, f"act_{action_name}", None)
            if handler is None:
                self.show_reply(f"⚙️ Action `{action_name}` is not built-in — asking AI Brain to generate & execute a Python tool for it...")
                py_spec = self.ai.generate_python_solution_with_brain(
                    f"User task: {original_user_task}. Requested step: {json.dumps(step)}"
                )
                if py_spec.get("code"):
                    return self.execute_python_script_with_auto_install(
                        code=py_spec["code"],
                        filename=py_spec.get("filename", "auto_fallback.py"),
                        packages=py_spec.get("packages", []),
                        description=f"Fallback for {action_name}"
                    )
                return f"Unknown action {action_name!r}"
            return handler(step)
        except Exception as e:
            return f"ERROR: {e}"

    def _need_gui(self):
        return None if pyautogui else "pyautogui is not installed (pip install pyautogui)."

    # ==========================================
    # AGENT ACTION HANDLERS
    # ==========================================
    def act_reply(self, s):
        txt = s.get("text", "")
        self.show_reply(txt)

    def act_prompt_dialog(self, s):
        """
        Opens a clarification dialog (e.g., 'Which browser should I check, sir?')
        and returns the user's input to the AI Brain so it can continue step-by-step.
        """
        title = s.get("title") or "Clarification Needed"
        question = s.get("question") or s.get("message") or "Please provide input:"
        options = s.get("options") or []
        ans = self.ask_prompt_dialog(title, question, options)
        if not ans:
            self.stop_event.set()
            return "User cancelled clarification dialog."
        self.session_memory["last_dialog_answer"] = ans
        return f"User answered in dialog: {ans!r}"

    def act_send_context_to_brain(self, s):
        """Asks the user whether to send diagnostic context as Text or Image (Screenshot) and returns it to the AI."""
        reason = s.get("reason") or "AI requested terminal / screen diagnostic check"
        return self._ask_and_send_diagnostic_context(reason)

    def act_write_and_run_python(self, s):
        code = s.get("code", "")
        filename = s.get("filename", "arc_generated.py")
        packages = s.get("packages") or []
        desc = s.get("description", "Custom Python automation script")
        if not code.strip():
            return "No Python code provided."

        if not self.confirm(f"Allow A.R.C to create and execute Python script `{filename}`?\n\nPurpose: {desc}"):
            return "User denied Python script execution."

        self.show_reply(f"🐍 Creating & executing `{filename}` in bottom terminal ({desc})...")
        result = self.execute_python_script_with_auto_install(
            code=code,
            filename=filename,
            packages=packages,
            description=desc
        )
        return result

    def act_install_package(self, s):
        pkg = s.get("package") or s.get("name") or ""
        if not pkg:
            return "No package name specified."
        return self.ensure_pip_package_installed(pkg)

    def act_ai_edit_file(self, s):
        raw_path = s.get("path") or s.get("query") or ""
        instruction = s.get("instruction") or s.get("text") or "Improve and update this file."
        dest = s.get("destination") or "Desktop"

        expanded = os.path.expanduser(raw_path)
        if os.path.isfile(expanded):
            target_path = os.path.abspath(expanded)
        else:
            search_dir = resolve_search_directory(dest)
            matches = find_fuzzy_matching_files(os.path.basename(raw_path), search_dir)
            if not matches:
                return f"File '{raw_path}' not found in {search_dir}."
            target_path = matches[0]

        try:
            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                orig_content = f.read()
        except Exception as e:
            return f"Could not read {target_path}: {e}"

        self.show_reply(f"🧠 Sending `{os.path.basename(target_path)}` to AI Brain for step-by-step editing...")
        updated_content, plan_str = self.ai.rewrite_file_with_brain(target_path, orig_content, instruction)
        if not updated_content:
            return f"AI Brain could not generate updated content for {target_path}."

        try:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(updated_content)
        except Exception as e:
            return f"Failed to write updated content to {target_path}: {e}"

        msg = f"✅ Updated `{target_path}`.\n\n**Steps Taken:**\n{plan_str}"
        self.show_reply(msg)
        return f"Successfully edited {target_path}.\nSteps:\n{plan_str}"

    def act_list_windows(self, s):
        wins = self.get_open_windows_list()
        return f"Open Desktop Windows:\n{wins}"

    def act_switch_window(self, s):
        app = s.get("app") or s.get("title") or ""
        maximize = bool(s.get("maximize", True))
        return self.switch_or_maximize_window(app, maximize=maximize)

    def act_switch_tab(self, s):
        if err := self._need_gui():
            return err
        direction = (s.get("direction") or "next").lower()
        count = max(1, int(s.get("count", 1)))
        idx = s.get("index")
        if idx is not None and 1 <= int(idx) <= 9:
            pyautogui.hotkey(MOD, str(int(idx)))
            return f"Switched to tab #{idx} ({MOD}+{idx})."
        for _ in range(count):
            if direction == "prev":
                pyautogui.hotkey("ctrl", "shift", "tab")
            else:
                pyautogui.hotkey("ctrl", "tab")
            time.sleep(0.15)
        return f"Switched tab ({direction}, {count}x)."

    def act_keyboard_hotkey(self, s):
        if err := self._need_gui():
            return err
        keys = [str(k).lower() for k in (s.get("keys") or [])]
        if not keys:
            return "No keys provided for hotkey."
        pyautogui.hotkey(*keys)
        return f"Pressed hotkey: {'+'.join(keys)}"

    def act_remember(self, s):
        key = s.get("key", "memory")
        val = s.get("value") or s.get("text") or ""
        if val:
            self.session_memory[key] = val
            entry = self.mem_store.add_memory(content=str(val), key=str(key))
            return f"Saved to memory.json at {entry['timestamp']}"
        return None

    def act_delete_fuzzy(self, s):
        query = s.get("query") or s.get("path") or ""
        dest = s.get("destination") or ""
        return self.execute_fuzzy_delete(query, dest)

    def act_delete_path(self, s):
        path = s.get("path", "")
        if os.path.exists(os.path.expanduser(path)):
            return self.execute_fuzzy_delete(path, "")
        folder, name = os.path.split(path)
        return self.execute_fuzzy_delete(name or path, folder)

    def act_find_and_edit_file(self, s):
        query = s.get("query") or s.get("name") or ""
        dest = s.get("destination") or "Desktop"
        ext = s.get("ext") or ""
        return self.execute_find_and_edit_workflow(query, dest, ext)

    def act_generate_image(self, s):
        prompt = s.get("prompt") or s.get("text") or ""
        if not prompt:
            return "No image prompt provided."
        self.show_reply(f"🎨 Generating image for: \"{prompt}\"...")
        img_path, status = self.ai.generate_image(prompt)
        if img_path and os.path.exists(img_path):
            sz_str = self._format_bytes(os.path.getsize(img_path))
            att = {
                "type": "generated_image",
                "path": img_path,
                "name": os.path.basename(img_path),
                "size_str": sz_str
            }
            self.show_reply(status, attachment=att)
            return f"Image generated and saved to {img_path}"
        self.show_reply(status)
        return status

    def act_analyze_screen(self, s):
        q = s.get("question") or "What is currently on my screen?"
        self._run_screen_vision_analysis(q)
        return "Screen analyzed and sent to chat."

    def act_navigate_to_target(self, s):
        if err := self._vision_ready() or self._need_gui():
            self.show_reply(err)
            return err
        phrase = s.get("phrase") or s.get("text") or ""
        click_type = (s.get("click") or "none").lower()

        shot_path = self._capture_clean_screenshot()
        hits = self.screen.find_text(phrase, rescan=True)
        if not hits:
            msg = f"Could not find text or button '{phrase}' on the screen."
            self.show_reply(msg)
            return msg

        target = hits[0]
        pyautogui.moveTo(target.cx, target.cy, duration=0.45)
        action_desc = f"Moved cursor to '{target.text}' at coordinates ({target.cx}, {target.cy})."

        if click_type == "right":
            pyautogui.click(target.cx, target.cy, button="right")
            action_desc += " Performed Right-Click."
        elif click_type == "double":
            pyautogui.doubleClick(target.cx, target.cy)
            action_desc += " Performed Double-Click."
        elif click_type == "left":
            pyautogui.click(target.cx, target.cy, button="left")
            action_desc += " Performed Left-Click."

        att = None
        if shot_path and os.path.exists(shot_path):
            att = {
                "type": "screenshot",
                "path": shot_path,
                "name": os.path.basename(shot_path),
                "size_str": self._format_bytes(os.path.getsize(shot_path))
            }
        self.show_reply(f"🎯 {action_desc}", attachment=att)
        return action_desc

    def act_read_terminal(self, s):
        buf = self.run_on_main(lambda: self.terminal_widget.get_recent_terminal_text(3000))
        return f"Current Bottom Terminal Buffer:\n{buf}"

    def act_timer(self, s):
        secs, label = int(s.get("seconds", 60)), s.get("label", "Timer")
        self.show_reply(f"⏱ Timer '{label}' started for {secs} seconds.")
        threading.Thread(target=lambda: (time.sleep(secs), self.notify("Timer Finished!", f"Your timer ({label}) is up!")), daemon=True).start()
        return f"Timer started for {secs}s."

    def act_alarm(self, s):
        t, label = s.get("time", "15:00"), s.get("label", "Alarm")

        def _run():
            now = datetime.now()
            h, m = map(int, t.split(":"))
            target = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            time.sleep((target - now).total_seconds())
            self.notify("Alarm Ringing!", f"It is now {t}! ({label})")

        self.show_reply(f"⏰ Alarm set for {t} ({label}).")
        threading.Thread(target=_run, daemon=True).start()
        return f"Alarm set for {t}."

    def notify(self, title, message):
        self.show_reply(f"🔔 {title}: {message}")
        try:
            if CURRENT_OS == "Windows":
                import winsound
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            elif CURRENT_OS == "Linux":
                subprocess.Popen(["notify-send", title, message])
        except Exception:
            pass

    def act_mouse_move(self, s):
        if err := self._need_gui():
            return err
        pyautogui.moveTo(int(s.get("x", 0)), int(s.get("y", 0)), duration=0.5)
        return "Moved."

    def act_mouse_click(self, s):
        if err := self._need_gui():
            return err
        if "x" in s and "y" in s:
            pyautogui.moveTo(int(s["x"]), int(s["y"]), duration=0.4)
        clicks = int(s.get("clicks", 1))
        pyautogui.click(button=s.get("button", "left"), clicks=clicks)
        return f"Clicked ({s.get('button', 'left')}, clicks={clicks})."

    def type_text(self, text):
        if text.isascii():
            pyautogui.write(text, interval=0.04)
        else:
            self.set_clipboard(text)
            pyautogui.hotkey(MOD, "v")

    def act_keyboard_type(self, s):
        if err := self._need_gui():
            return err
        text, _secret = self.fill(s.get("text", ""))
        self.type_text(text)
        return "Typed."

    def act_keyboard_press(self, s):
        if err := self._need_gui():
            return err
        pyautogui.press(s.get("key", "enter"))
        return "Pressed."

    def act_copy(self, s):
        if err := self._need_gui():
            return err
        pyautogui.hotkey(MOD, "c")
        return "Copied selection."

    def act_paste(self, s):
        if err := self._need_gui():
            return err
        pyautogui.hotkey(MOD, "v")
        return "Pasted."

    def act_set_clipboard(self, s):
        text, _ = self.fill(s.get("text", ""))
        self.set_clipboard(text)
        return "Clipboard set."

    def act_run_command(self, s):
        cmd = s.get("command", "").strip()
        if not cmd:
            return "Empty command."

        force_su_check = any(pref in cmd.lower() for pref in SYSTEM_PROTECTED_PREFIXES) and bool(DESTRUCTIVE.search(cmd))
        if DESTRUCTIVE.search(cmd):
            if not self.request_security_approval(
                f"RUN DESTRUCTIVE TERMINAL COMMAND:\n{cmd}",
                force_even_in_superuser=force_su_check
            ):
                return "User denied the command."
        elif not SAFE_COMMANDS.match(cmd):
            if not self.confirm(f"Allow the agent to run this command in the bottom terminal?\n\n{cmd}"):
                return "User denied the command."

        out = self.execute_in_bottom_terminal(cmd, ask_on_error=True)
        return out

    def act_open_app(self, s):
        app = s.get("app", "").strip()
        if not app:
            return "No app name specified."

        if CURRENT_OS == "Linux":
            low = app.lower()
            candidates = [app, low, low.replace(" ", "-")]
            if low in ("brave", "brave browser"):
                candidates = ["brave-browser", "brave", "flatpak run com.brave.Browser"]
            elif low in ("chrome", "google chrome"):
                candidates = ["google-chrome", "chromium", "chromium-browser"]

            for cand in candidates:
                bin_name = cand.split()[0]
                if shutil.which(bin_name):
                    term_cmd = f"nohup {cand} >/dev/null 2>&1 &"
                    self.execute_in_bottom_terminal(term_cmd, timeout=4.0, ask_on_error=False)
                    self.show_reply(f"Launched `{cand}` via the bottom terminal.")
                    return f"Executed `{term_cmd}` in bottom terminal."

        if CURRENT_OS == "Windows":
            term_cmd = f"Start-Process \"{app}\""
            self.execute_in_bottom_terminal(term_cmd, timeout=5.0, ask_on_error=False)
            self.show_reply(f"Launched `{app}` via the bottom terminal.")
            return f"Executed `{term_cmd}` in bottom terminal."

        if CURRENT_OS == "macOS":
            term_cmd = f"open -a \"{app}\""
            self.execute_in_bottom_terminal(term_cmd, timeout=5.0, ask_on_error=False)
            self.show_reply(f"Launched `{app}` via the bottom terminal.")
            return f"Executed `{term_cmd}` in bottom terminal."

        if err := self._need_gui():
            return f"Could not launch {app}. {err}"
        pyautogui.press("win")
        time.sleep(1.0)
        self.type_text(app)
        time.sleep(1.2)
        pyautogui.press("enter")
        self.show_reply(f"Launched {app} via system launcher.")
        return f"Launched {app} via system launcher."

    def act_browser_open(self, s):
        return self.web.open(s.get("url", ""))

    def act_browser_search(self, s):
        return self.web.search(s.get("query", ""))

    def act_browser_click(self, s):
        text = s.get("text", "")
        if CRITICAL_UI.search(text) and not self.request_security_approval(f"CLICK sensitive browser element: '{text}'"):
            return "User denied click."
        return self.web.click_text(text)

    def act_browser_type(self, s):
        text, _secret = self.fill(s.get("text", ""))
        return self.web.type_into(s.get("field", ""), text, bool(s.get("submit", False)))

    def act_browser_press(self, s):
        return self.web.press(s.get("key", "enter"))

    def act_browser_read(self, s):
        return self.web.page_state()

    def act_browser_list_tabs(self, s):
        if not SELENIUM_OK or not self.web.alive():
            return "Selenium browser is not currently open."
        handles = self.web.driver.window_handles
        cur = self.web.driver.current_window_handle
        info = []
        for idx, h in enumerate(handles):
            self.web.driver.switch_to.window(h)
            marker = " [ACTIVE]" if h == cur else ""
            info.append(f"Tab {idx}{marker}: {self.web.driver.title} ({self.web.driver.current_url})")
        self.web.driver.switch_to.window(cur)
        res = f"Open Browser Tabs ({len(handles)} total):\n" + "\n".join(info)
        self.show_reply(res)
        return res

    def act_browser_switch_tab(self, s):
        if not SELENIUM_OK or not self.web.alive():
            return "Selenium browser is not currently open."
        handles = self.web.driver.window_handles
        idx = int(s.get("index", 0))
        if 0 <= idx < len(handles):
            self.web.driver.switch_to.window(handles[idx])
            return f"Switched to Browser Tab {idx}: {self.web.driver.title}"
        return f"Tab index {idx} out of range (0..{len(handles)-1})."

    def act_browser_find_phones(self, s):
        res = self.web.find_phones(deep=bool(s.get("deep", True)))
        self.show_reply(res)
        return res

    def act_browser_close(self, s):
        self.web.close()
        return "Browser closed."

    def _vision_ready(self):
        return None if VISION_OK else f"Screen decoder unavailable: {VISION_ERR}"

    def act_screen_read(self, s):
        if err := self._vision_ready():
            return err
        self.screen.scan()
        return self.screen.summary(60)

    def act_screen_find(self, s):
        if err := self._vision_ready():
            return err
        hits = self.screen.find_text(s.get("phrase", ""))
        if not hits:
            return f"'{s.get('phrase')}' not found on screen."
        return "Found: " + "; ".join(h.brief() for h in hits[:8])

    def act_screen_click_text(self, s):
        if err := self._vision_ready() or self._need_gui():
            return err
        phrase = s.get("phrase", "")
        if CRITICAL_UI.search(phrase) and not self.request_security_approval(f"CLICK sensitive screen text: '{phrase}'"):
            return "User denied click."
        hits = self.screen.find_text(phrase)
        idx = int(s.get("index", 0))
        if not hits or idx >= len(hits):
            return f"'{phrase}' not found on screen."
        btn = s.get("button", "left")
        clicks = int(s.get("clicks", 1))
        self.screen.click(hits[idx], clicks=clicks, button=btn)
        return f"Clicked ({btn}, {clicks}x) at ({hits[idx].cx},{hits[idx].cy})."

    def act_screen_select_text(self, s):
        if err := self._vision_ready() or self._need_gui():
            return err
        hits = self.screen.find_text(s.get("phrase", ""))
        if not hits:
            return f"'{s.get('phrase')}' not found on screen."
        self.screen.select(hits[0])
        return f"Selected text at ({hits[0].cx},{hits[0].cy})."

    def act_highlight_phrase(self, s):
        phrase = s.get("phrase", "")
        if SELENIUM_OK and self.web.alive():
            n = self.web.highlight(phrase)
            return f"Highlighted {n} occurrence(s) in browser." if n else f"'{phrase}' not found."
        return self.act_screen_find(s)

    def act_ask_user(self, s):
        fields = [f for f in s.get("fields", []) if f.get("name") and f["name"] not in self.session_memory]
        if not fields:
            return "All requested values are already known."
        res = self.ask_form(s.get("title", "Information needed"), fields)
        if res is None:
            self.stop_event.set()
            return "User cancelled form."
        secret_names = []
        for f in fields:
            name = f["name"]
            self.session_memory[name] = res[name]
            if f.get("type") == "password" or f.get("secret"):
                self.secret_keys.add(name)
                secret_names.append(name)
        shown = {k: v for k, v in res.items() if k not in secret_names}
        return f"User provided: {json.dumps(shown, ensure_ascii=False)}; secrets: {secret_names}"

    def act_wait_for_user(self, s):
        if not self.wait_for_user(s.get("message", "Please complete the step, then press OK.")):
            self.stop_event.set()
            return "User cancelled."
        return "User completed the step."

    def act_confirm(self, s):
        return "User said YES." if self.confirm(s.get("message", "Proceed?")) else "User said NO."

    def closeEvent(self, event):
        SpeechEngine.stop()
        self.camera_widget.stop_camera()
        self.screen_widget.stop_stream()
        super().closeEvent(event)


if __name__ == "__main__":
    CONSOLE_CAPTURE.install()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    win = DesktopAgentWindow()
    win.showMaximized()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())