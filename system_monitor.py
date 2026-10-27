"""
system_monitor.py - Futuristic Cyber-HUD System Resource Monitor (PyQt6).
Matches the exact visual layout of the reference HUD:
- CPU Usage: Smooth area spline chart (left) + vertical frequency/core bars (right)
- RAM Usage: Dual glowing cyan gradient bars (Physical RAM + Swap/Commit)
- Storage: Drive C: / Root & Drive D: / Home free space with gradient bar
- Network: Live Download / Upload speed in Mbps with signal icon
"""
import os
import time
import psutil
from collections import deque
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QLinearGradient, QFont, QPainterPath
)
from PyQt6.QtWidgets import QWidget


class SystemMonitorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(340, 390)

        # Sub-feature visibility flags controlled by Window -> System Monitor menu
        self.show_cpu = True
        self.show_ram = True
        self.show_storage = True
        self.show_network = True

        # Telemetry buffers
        self.cpu_total = 0.0
        self.cpu_history = deque([12.0] * 30, maxlen=30)
        self.bar_values = [15.0] * 28

        self.ram_used_gb = 0.0
        self.ram_total_gb = 16.0
        self.ram_pct = 0.0
        self.swap_pct = 0.0

        self.drives = []  # List of (label, free_gb, pct_used)

        self.net_mbps = 0.0
        self.net_up_mbps = 0.0
        self._last_net = psutil.net_io_counters()
        self._last_net_time = time.time()

        # Update timer (1000ms)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(1000)
        self.update_stats()

    def set_section_visible(self, section: str, visible: bool):
        setattr(self, f"show_{section}", visible)
        self.update()

    def update_stats(self):
        # 1. CPU
        try:
            self.cpu_total = psutil.cpu_percent(interval=None)
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        except Exception:
            self.cpu_total = 0.0
            per_cpu = [0.0]

        self.cpu_history.append(self.cpu_total)

        # Build 28 spectrum bars from per-core usage + rolling history
        bars = []
        hist_list = list(self.cpu_history)
        for i in range(28):
            core_val = per_cpu[i % len(per_cpu)] if per_cpu else self.cpu_total
            hist_val = hist_list[-(i % len(hist_list)) - 1]
            val = max(6.0, min(100.0, (core_val * 0.65) + (hist_val * 0.35)))
            bars.append(val)
        self.bar_values = bars

        # 2. RAM
        try:
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            self.ram_used_gb = vm.used / (1024 ** 3)
            self.ram_total_gb = max(1.0, vm.total / (1024 ** 3))
            self.ram_pct = vm.percent
            # Second bar shows swap or cache/active ratio so both bars look alive
            self.swap_pct = sw.percent if sw.total > 0 else min(100.0, vm.percent * 0.82)
        except Exception:
            pass

        # 3. Storage
        drives_info = []
        try:
            if os.name == "nt":
                for part in psutil.disk_partitions(all=False):
                    if "cdrom" in part.opts or not part.fstype:
                        continue
                    u = psutil.disk_usage(part.mountpoint)
                    letter = part.device.rstrip("\\")
                    drives_info.append((f"DRIVE {letter}", u.free / (1024 ** 3), u.percent))
            else:
                u_root = psutil.disk_usage("/")
                drives_info.append(("DRIVE C:", u_root.free / (1024 ** 3), u_root.percent))
                home_path = os.path.expanduser("~")
                u_home = psutil.disk_usage(home_path)
                drives_info.append(("DRIVE D:", u_home.free / (1024 ** 3), u_home.percent))
        except Exception:
            pass
        self.drives = drives_info[:2]

        # 4. Network
        try:
            now = time.time()
            cur = psutil.net_io_counters()
            dt = max(0.1, now - self._last_net_time)
            dl_bytes = max(0, cur.bytes_recv - self._last_net.bytes_recv)
            ul_bytes = max(0, cur.bytes_sent - self._last_net.bytes_sent)
            self.net_mbps = (dl_bytes * 8.0) / (dt * 1_000_000)
            self.net_up_mbps = (ul_bytes * 8.0) / (dt * 1_000_000)
            self._last_net = cur
            self._last_net_time = now
        except Exception:
            pass

        self.update()

    # ==========================================
    # CUSTOM CYBER-HUD QPAINTER RENDERING
    # ==========================================
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # Deep space navy background
        bg_grad = QLinearGradient(0, 0, 0, h)
        bg_grad.setColorAt(0.0, QColor(7, 15, 28))
        bg_grad.setColorAt(1.0, QColor(4, 9, 18))
        p.fillRect(0, 0, w, h, bg_grad)

        # Header title
        p.setPen(QColor(200, 222, 245))
        p.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        p.drawText(QRectF(16, 10, w - 32, 24), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   "SYSTEM RESOURCE STATUS")

        # Outer HUD card frame
        card_rect = QRectF(12, 38, w - 24, h - 50)
        p.setPen(QPen(QColor(22, 65, 105, 200), 1.3))
        p.setBrush(QColor(9, 20, 36, 210))
        p.drawRoundedRect(card_rect, 6, 6)

        y = 52.0
        left_icon_x = 24.0
        content_x = 68.0
        content_w = w - content_x - 26.0

        # ---------------- 1. CPU SECTION ----------------
        if self.show_cpu:
            self._draw_cpu_icon(p, left_icon_x, y + 2)
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            p.setPen(QColor(175, 205, 235))
            p.drawText(QRectF(content_x, y, content_w, 18),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f"CPU USAGE ({self.cpu_total:.0f}%)")

            graph_y = y + 22
            graph_h = 46.0
            half_w = (content_w - 14) / 2.0

            # Left: Smooth Spline Area Graph
            self._draw_spline_graph(p, QRectF(content_x, graph_y, half_w, graph_h))

            # Center Vertical Divider
            div_x = content_x + half_w + 7
            p.setPen(QPen(QColor(30, 80, 125, 160), 1))
            p.drawLine(QPointF(div_x, graph_y), QPointF(div_x, graph_y + graph_h))

            # Right: Vertical Spectrum Bars
            self._draw_spectrum_bars(p, QRectF(div_x + 7, graph_y, half_w, graph_h))

            y += 80
            self._draw_divider(p, 22, w - 22, y)
            y += 12

        # ---------------- 2. RAM SECTION ----------------
        if self.show_ram:
            self._draw_ram_icon(p, left_icon_x, y + 2)
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            p.setPen(QColor(175, 205, 235))
            p.drawText(QRectF(content_x, y, content_w, 18),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f"RAM USAGE ({self.ram_used_gb:.1f} GB / {self.ram_total_gb:.0f} GB)")

            bar1_y = y + 24
            self._draw_glow_bar(p, QRectF(content_x, bar1_y, content_w, 10), self.ram_pct)
            bar2_y = bar1_y + 18
            self._draw_glow_bar(p, QRectF(content_x, bar2_y, content_w, 10), self.swap_pct)

            y += 64
            self._draw_divider(p, 22, w - 22, y)
            y += 12

        # ---------------- 3. STORAGE SECTION ----------------
        if self.show_storage:
            self._draw_folder_icon(p, left_icon_x, y + 4)
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            p.setPen(QColor(175, 205, 235))
            p.drawText(QRectF(content_x, y, content_w, 18),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       "STORAGE:")
            y += 20

            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Normal))
            if self.drives:
                d0_label, d0_free, d0_pct = self.drives[0]
                p.setPen(QColor(160, 192, 220))
                p.drawText(QRectF(content_x, y, content_w, 18),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                           f"{d0_label} ({d0_free:.0f} GB FREE)")
                y += 20
                self._draw_glow_bar(p, QRectF(content_x, y, content_w, 7), d0_pct)
                y += 13

                if len(self.drives) > 1:
                    d1_label, d1_free, _ = self.drives[1]
                    p.setPen(QColor(160, 192, 220))
                    p.drawText(QRectF(content_x, y, content_w, 18),
                               Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                               f"{d1_label} ({d1_free:.0f} GB FREE)")
                    y += 22
            y += 4
            self._draw_divider(p, 22, w - 22, y)
            y += 12

        # ---------------- 4. NETWORK SECTION ----------------
        if self.show_network:
            self._draw_net_icon(p, left_icon_x, y + 4)
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            p.setPen(QColor(175, 205, 235))
            p.drawText(QRectF(content_x, y, content_w, 18),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       "NETWORK:")
            y += 20
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Normal))
            p.setPen(QColor(160, 192, 220))
            disp_speed = f"{self.net_mbps:.1f}" if self.net_mbps < 10 else f"{self.net_mbps:.0f}"
            p.drawText(QRectF(content_x, y, content_w, 18),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f"DOWNLOAD ({disp_speed} Mbps)   •   UP ({self.net_up_mbps:.1f} Mbps)")

    # ==========================================
    # VECTOR ICON & GRAPH HELPERS
    # ==========================================
    def _draw_divider(self, p: QPainter, x1: float, x2: float, y: float):
        p.setPen(QPen(QColor(22, 60, 95, 180), 1))
        p.drawLine(QPointF(x1, y), QPointF(x2, y))

    def _draw_glow_bar(self, p: QPainter, rect: QRectF, pct: float):
        # Track background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(16, 38, 64))
        p.drawRoundedRect(rect, 3, 3)

        # Active fill
        fill_w = max(4.0, rect.width() * (max(0.0, min(100.0, pct)) / 100.0))
        fill_rect = QRectF(rect.x(), rect.y(), fill_w, rect.height())
        grad = QLinearGradient(fill_rect.topLeft(), fill_rect.topRight())
        grad.setColorAt(0.0, QColor(38, 128, 215))
        grad.setColorAt(1.0, QColor(115, 235, 255))
        p.setBrush(grad)
        p.drawRoundedRect(fill_rect, 3, 3)

    def _draw_spline_graph(self, p: QPainter, rect: QRectF):
        pts = list(self.cpu_history)
        if len(pts) < 2:
            return
        step_x = rect.width() / (len(pts) - 1)
        coords = []
        for i, val in enumerate(pts):
            px = rect.x() + i * step_x
            py = rect.bottom() - (max(5.0, min(100.0, val)) / 100.0) * (rect.height() - 6)
            coords.append(QPointF(px, py))

        path = QPainterPath()
        path.moveTo(coords[0])
        for i in range(len(coords) - 1):
            c1 = QPointF((coords[i].x() + coords[i + 1].x()) / 2, coords[i].y())
            c2 = QPointF((coords[i].x() + coords[i + 1].x()) / 2, coords[i + 1].y())
            path.cubicTo(c1, c2, coords[i + 1])

        fill_path = QPainterPath(path)
        fill_path.lineTo(rect.right(), rect.bottom())
        fill_path.lineTo(rect.left(), rect.bottom())
        fill_path.closeSubpath()

        grad = QLinearGradient(0, rect.top(), 0, rect.bottom())
        grad.setColorAt(0.0, QColor(65, 195, 255, 135))
        grad.setColorAt(1.0, QColor(15, 55, 95, 15))
        p.fillPath(fill_path, grad)

        p.setPen(QPen(QColor(95, 220, 255), 1.8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    def _draw_spectrum_bars(self, p: QPainter, rect: QRectF):
        n = len(self.bar_values)
        if n == 0:
            return
        gap = 2.0
        bar_w = max(1.5, (rect.width() - gap * (n - 1)) / n)
        for i, val in enumerate(self.bar_values):
            bh = (max(8.0, min(100.0, val)) / 100.0) * rect.height()
            bx = rect.x() + i * (bar_w + gap)
            by = rect.bottom() - bh
            grad = QLinearGradient(0, by, 0, rect.bottom())
            grad.setColorAt(0.0, QColor(115, 235, 255))
            grad.setColorAt(1.0, QColor(30, 105, 175))
            p.fillRect(QRectF(bx, by, bar_w, bh), grad)

    def _draw_cpu_icon(self, p: QPainter, x: float, y: float):
        pen = QPen(QColor(75, 185, 240), 1.4)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(x + 4, y + 4, 20, 20), 2, 2)
        p.drawRect(QRectF(x + 9, y + 9, 10, 10))
        for i in range(4):
            off = 7 + i * 4.5
            p.drawLine(QPointF(x + off, y), QPointF(x + off, y + 4))
            p.drawLine(QPointF(x + off, y + 24), QPointF(x + off, y + 28))
            p.drawLine(QPointF(x, y + off), QPointF(x + 4, y + off))
            p.drawLine(QPointF(x + 24, y + off), QPointF(x + 28, y + off))

    def _draw_ram_icon(self, p: QPainter, x: float, y: float):
        pen = QPen(QColor(75, 185, 240), 1.4)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(x, y + 6, 30, 16), 2, 2)
        for i in range(6):
            px = x + 4 + i * 4.2
            p.drawLine(QPointF(px, y + 22), QPointF(px, y + 25))

    def _draw_folder_icon(self, p: QPainter, x: float, y: float):
        pen = QPen(QColor(75, 185, 240), 1.4)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        path.moveTo(x, y + 4)
        path.lineTo(x + 10, y + 4)
        path.lineTo(x + 13, y + 8)
        path.lineTo(x + 28, y + 8)
        path.lineTo(x + 28, y + 24)
        path.lineTo(x, y + 24)
        path.closeSubpath()
        p.drawPath(path)

    def _draw_net_icon(self, p: QPainter, x: float, y: float):
        pen = QPen(QColor(75, 185, 240), 1.4)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        cx, cy = x + 14, y + 12
        p.drawEllipse(QPointF(cx, cy), 2, 2)
        # Antenna tower
        p.drawLine(QPointF(cx, cy + 2), QPointF(cx - 5, y + 26))
        p.drawLine(QPointF(cx, cy + 2), QPointF(cx + 5, y + 26))
        p.drawLine(QPointF(cx - 3, y + 20), QPointF(cx + 3, y + 20))
        # Waves
        p.drawArc(QRectF(cx - 8, cy - 8, 16, 16), -45 * 16, 90 * 16)
        p.drawArc(QRectF(cx - 8, cy - 8, 16, 16), 135 * 16, 90 * 16)
        p.drawArc(QRectF(cx - 13, cy - 13, 26, 26), -45 * 16, 90 * 16)
        p.drawArc(QRectF(cx - 13, cy - 13, 26, 26), 135 * 16, 90 * 16)