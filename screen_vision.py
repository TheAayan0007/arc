"""
screen_vision.py - Screen capture & OCR decoder.
Decouples screen grabbing from pytesseract so live screen streaming works
even if Tesseract OCR is not installed.
"""
import re
import time
import difflib
import threading
from dataclasses import dataclass

try:
    import numpy as np
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import mss
    MSS_OK = True
except ImportError:
    mss = None
    MSS_OK = False

try:
    import pytesseract
    OCR_OK = True
    VISION_ERR = ""
except ImportError as _e:
    pytesseract = None
    OCR_OK = False
    VISION_ERR = f"{_e}. Run: pip install pytesseract (and sudo apt install tesseract-ocr)."

VISION_OK = CV2_OK and (MSS_OK or OCR_OK)

try:
    import pyautogui
except ImportError:
    pyautogui = None

OCR_CONFIG = "--psm 3"
MIN_WORD_CONF = 35
INVERTED_PASS = True


@dataclass
class Element:
    kind: str
    x: int
    y: int
    w: int
    h: int
    text: str = ""
    conf: float = 0.0
    words: tuple = ()

    @property
    def cx(self):
        return self.x + self.w // 2

    @property
    def cy(self):
        return self.y + self.h // 2

    def brief(self):
        label = f' "{self.text}"' if self.text else ""
        return f"{self.kind}{label} @({self.cx},{self.cy}) {self.w}x{self.h}"


def _norm(t):
    return re.sub(r"[^\w\s]", "", t.lower()).strip()


def _union(elems):
    x1 = min(e.x for e in elems)
    y1 = min(e.y for e in elems)
    x2 = max(e.x + e.w for e in elems)
    y2 = max(e.y + e.h for e in elems)
    return x1, y1, x2 - x1, y2 - y1


def iou(a, b):
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    return inter / float(a.w * a.h + b.w * b.h - inter or 1)


def group_words(words):
    words = sorted(words, key=lambda w: (w.cy, w.x))
    rows = []
    for w in words:
        for row in rows:
            ref = row[0]
            if abs(w.cy - ref.cy) <= max(ref.h, w.h) * 0.6:
                row.append(w)
                break
        else:
            rows.append([w])
    chunks = []
    for row in rows:
        row.sort(key=lambda w: w.x)
        cur = [row[0]]
        for w in row[1:]:
            prev = cur[-1]
            gap = w.x - (prev.x + prev.w)
            if gap > max(prev.h, w.h) * 1.6:
                chunks.append(cur)
                cur = [w]
            else:
                cur.append(w)
        chunks.append(cur)
    out = []
    for c in chunks:
        x, y, w, h = _union(c)
        out.append(Element("text", x, y, w, h,
                           " ".join(i.text for i in c),
                           sum(i.conf for i in c) / len(c), tuple(c)))
    out.sort(key=lambda e: (e.y // 12, e.x))
    return out


def match_phrase(chunks, phrase, min_ratio=0.82):
    p = _norm(phrase)
    if not p:
        return []
    ptoks = p.split()
    n = len(ptoks)
    hits = []
    for chunk in chunks:
        toks = [_norm(w.text) for w in chunk.words]
        for i in range(0, max(len(toks) - n + 1, 0)):
            window = " ".join(toks[i:i + n])
            if window == p:
                ratio = 1.0
            elif p in window or (len(window) >= 3 and window in p and n == 1):
                ratio = 0.95
            else:
                ratio = difflib.SequenceMatcher(None, window, p).ratio()
            if ratio >= min_ratio:
                ws = chunk.words[i:i + n]
                x, y, w, h = _union(ws)
                hits.append(Element("match", x, y, w, h,
                                    " ".join(k.text for k in ws), ratio))
    hits.sort(key=lambda e: (-e.conf, e.y, e.x))
    unique = []
    for h in hits:
        if all(iou(h, u) < 0.4 for u in unique):
            unique.append(h)
    return unique


class ScreenReader:
    def __init__(self, log=print):
        self.log = log
        self.scale = 1.0
        self.words, self.chunks, self.regions = [], [], []
        self.stamp = 0.0
        self._img = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def grab(self):
        # 1. Try MSS fast capture
        if MSS_OK and CV2_OK:
            try:
                with mss.mss() as sct:
                    mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                    shot = sct.grab(mon)
                    img = np.array(shot)[:, :, :3].copy()
                    if img.mean() > 0.5:  # Ensure not an all-black Wayland frame
                        return img
            except Exception:
                pass

        # 2. Fallback to PIL ImageGrab
        try:
            from PIL import ImageGrab
            pil_img = ImageGrab.grab()
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            raise RuntimeError(f"Screen grab failed: {e}")

    def scan(self, detect_regions=True):
        if not OCR_OK:
            raise RuntimeError(VISION_ERR or "pytesseract is not installed.")
        img = self.grab()
        s = self.scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        words = self._ocr_words(gray)
        if INVERTED_PASS:
            for w in self._ocr_words(255 - gray):
                if all(iou(w, k) < 0.3 for k in words):
                    words.append(w)
        chunks = group_words(words) if words else []
        regions = self._regions(img, gray, chunks) if detect_regions else []

        def to_logical(e):
            e.x, e.y = int(e.x / s), int(e.y / s)
            e.w, e.h = max(1, int(e.w / s)), max(1, int(e.h / s))
            return e

        for e in words + chunks + regions:
            to_logical(e)
        with self._lock:
            self.words, self.chunks, self.regions = words, chunks, regions
            self._img, self.stamp = img, time.time()
        return chunks + regions

    @staticmethod
    def _ocr_words(gray):
        data = pytesseract.image_to_data(gray, config=OCR_CONFIG, output_type=pytesseract.Output.DICT)
        words = []
        for i in range(len(data["text"])):
            txt = (data["text"][i] or "").strip()
            try:
                conf = float(data["conf"][i])
            except ValueError:
                conf = -1
            if txt and conf >= MIN_WORD_CONF:
                words.append(Element("word", data["left"][i], data["top"][i],
                                     data["width"][i], data["height"][i], txt, conf))
        return words

    def _regions(self, img, gray, chunks):
        H, W = gray.shape
        edges = cv2.dilate(cv2.Canny(gray, 60, 160), np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 0.6 * W * H:
                continue
            inside = [k for k in chunks if x <= k.cx <= x + w and y <= k.cy <= y + h]
            aspect = w / float(h)
            if 24 <= h <= 90 and 40 <= w <= 420 and 1.4 <= aspect <= 10 and len(inside) == 1:
                found.append(Element("button", x, y, w, h, inside[0].text, 0.6))
        return found[:60]

    def find_text(self, phrase, rescan=True):
        if rescan or not self.chunks:
            self.scan(detect_regions=False)
        return match_phrase(self.chunks, phrase)

    def summary(self, limit=60):
        items = sorted(self.chunks + self.regions, key=lambda e: (e.cy // 15, e.cx))[:limit]
        if not items:
            return "Screen scan found nothing readable."
        return f"{len(self.chunks)} text chunks, {len(self.regions)} button regions. Top items:\n" + \
               "\n".join(f"- {e.brief()}" for e in items)

    @staticmethod
    def click(el, clicks=1, button="left"):
        if pyautogui:
            pyautogui.moveTo(el.cx, el.cy, duration=0.35)
            pyautogui.click(clicks=clicks, button=button)

    @staticmethod
    def select(el):
        if not pyautogui:
            return
        if " " not in el.text.strip():
            pyautogui.moveTo(el.cx, el.cy, duration=0.3)
            pyautogui.doubleClick()
        else:
            pyautogui.moveTo(el.x + 1, el.cy, duration=0.3)
            pyautogui.dragTo(el.x + el.w - 1, el.cy, duration=0.4, button="left")

    def annotate(self, path="screen_debug.png"):
        with self._lock:
            img = None if self._img is None else self._img.copy()
        if img is None:
            return None
        s = self.scale
        colors = {"text": (0, 200, 0), "button": (255, 120, 0), "image": (0, 0, 255)}
        for e in self.chunks + self.regions:
            p1 = (int(e.x * s), int(e.y * s))
            p2 = (int((e.x + e.w) * s), int((e.y + e.h) * s))
            cv2.rectangle(img, p1, p2, colors.get(e.kind, (255, 0, 255)), 2)
        cv2.imwrite(path, img)
        return path