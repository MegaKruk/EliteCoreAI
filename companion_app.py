"""
Elite Dangerous Core Mining AI Companion

Captures the Elite Dangerous game window, runs segmentation inference locally
(CPU by default so the GPU stays free for the game), and displays detected
core asteroid polygons with confidence scores.

Display modes:
  monitor2  - window on your second monitor showing the game feed with
              detections drawn on it (always works, use this first)
  overlay   - transparent click-through window drawn directly over the game
              (requires the game to run in BORDERLESS WINDOWED mode)

Runtimes:
  torch     - loads exports/<ring>_best.pt via ultralytics (default)
  onnxrt    - loads exports/<ring>_best.onnx via ONNX Runtime with this app's
              own post-processing (conf filter, NMS, mask decode). Ultralytics
              cannot run its own ONNX seg exports because the exported graph
              contains no NMS - this app decodes the raw output itself.
              Typically 1.5-2x faster than torch on CPU.
  openvino  - onnxrt with Intel's OpenVINO execution provider, if
              onnxruntime-openvino is installed.

Usage:
    python companion_app.py --ring-type ice
    python companion_app.py --ring-type ice --display overlay --conf 0.4
    python companion_app.py --ring-type ice --runtime onnxrt --threads 4
    python companion_app.py --model-path exports/metal_rich_yolo11n_best.pt
    python companion_app.py --ring-type ice --debug

Controls:
    monitor2: press Q or ESC in the display window to quit
    overlay:  Ctrl+C in the terminal (the overlay is click-through, so it
              never has keyboard focus)

Architecture (3 threads):
    capture thread   - grabs frames continuously (dxcam, mss fallback) and
                       re-detects the game window every 2s so the overlay
                       follows if the window is moved or resized
    inference thread - stretch-resizes the latest frame to the model input
                       size (matches Roboflow training preprocessing), runs
                       the engine, maps polygons back to frame space, feeds
                       the smoother
    main thread      - renders at a fixed 20 fps independent of inference
                       speed, so the video/overlay stays smooth even when
                       inference runs at 5 fps
"""

import argparse
import ctypes
import os
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("YOLO_VERBOSE", "False")

RING_TYPES = ["ice", "rocky", "metallic", "metal_rich"]

TRANSPARENT_HEX = "#000001"   # overlay key color, made invisible by Windows
UI_FPS  = 20                  # display refresh rate, independent of inference
IOU_NMS = 0.5

COLOR_RED    = (40, 40, 230)   # BGR
COLOR_ORANGE = (0, 165, 255)
COLOR_GREEN  = (60, 220, 60)


def conf_to_color(conf):
    """
    Map a confidence score to a red-orange-green gradient.
    <=0.70 solid red, 0.85 orange, 1.00 green. Returns (bgr_tuple, hex_str).
    """
    def lerp(a, b, t):
        return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))

    c = max(0.0, min(1.0, conf))
    if c <= 0.70:
        bgr = COLOR_RED
    elif c <= 0.85:
        bgr = lerp(COLOR_RED, COLOR_ORANGE, (c - 0.70) / 0.15)
    else:
        bgr = lerp(COLOR_ORANGE, COLOR_GREEN, (c - 0.85) / 0.15)
    return bgr, "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0])


def find_game_window():
    """
    Return (left, top, width, height) of the Elite Dangerous window, or None.
    The title must contain BOTH "elite" and "dangerous" - matching "elite"
    alone also hits e.g. a PyCharm window with this project open.
    """
    try:
        import pygetwindow as gw
    except ImportError:
        print("pygetwindow not installed: pip install pygetwindow")
        return None
    try:
        for wnd in gw.getAllWindows():
            title = wnd.title.lower()
            if "elite" in title and "dangerous" in title and wnd.width > 200:
                return (wnd.left, wnd.top, wnd.width, wnd.height)
    except Exception as e:
        print(f"Window lookup error: {e}")
    return None


def get_primary_monitor_rect():
    """Return (left, top, width, height) of the primary monitor."""
    import mss
    with mss.mss() as sct:
        m = sct.monitors[1]
        return (m["left"], m["top"], m["width"], m["height"])


def compute_capture(rect, crop, y_offset=0):
    """
    Turn the game window rect into the actual capture rect.

    crop is either ("center", w, h) for a centered region of that size, or
    ("margins", l, t, r, b) for pixel margins trimmed from each edge.
    y_offset shifts the region vertically (negative = up), clamped so the
    region always stays inside the window.

    Returns (capture_rect, rel_x, rel_y) where rel_* is the region's offset
    inside the window - the overlay uses it to keep polygons aligned.
    """
    wl, wt, ww, wh = rect
    if crop[0] == "center":
        cw = min(crop[1], ww)
        ch = min(crop[2], wh)
        rel_x = (ww - cw) // 2
        rel_y = (wh - ch) // 2 + y_offset
    else:
        cl, ct, cr, cb = crop[1:]
        cw = max(16, ww - cl - cr)
        ch = max(16, wh - ct - cb)
        rel_x = cl
        rel_y = ct + y_offset
    rel_x = max(0, min(rel_x, ww - cw))
    rel_y = max(0, min(rel_y, wh - ch))
    return (wl + rel_x, wt + rel_y, cw, ch), rel_x, rel_y


# ---------------------------------------------------------------------------
# Screen capture backends
# ---------------------------------------------------------------------------

class DxcamSource:
    """
    DXGI Desktop Duplication capture via dxcam: 5-10ms per grab vs 30-40ms
    for mss (GDI). dxcam returns None when the desktop has no new frame,
    so we keep and return the last good one.
    """
    name = "dxcam"

    def __init__(self, rect):
        import dxcam
        self.cam  = dxcam.create(output_color="BGR")
        self.rect = rect
        self.last = None

    def grab(self):
        l, t, w, h = self.rect
        frame = self.cam.grab(region=(l, t, l + w, t + h))
        if frame is not None:
            self.last = frame
        return self.last

    def set_rect(self, rect):
        self.rect = rect

    def close(self):
        try:
            self.cam.release()
        except Exception:
            pass


class MssSource:
    """GDI capture fallback. The mss handle is created lazily so it lives on
    the capture thread (mss handles are not safe to share across threads)."""
    name = "mss"

    def __init__(self, rect):
        self.rect = rect
        self._sct = None

    def grab(self):
        import mss
        if self._sct is None:
            self._sct = mss.mss()
        l, t, w, h = self.rect
        raw = self._sct.grab({"left": l, "top": t, "width": w, "height": h})
        return np.array(raw)[:, :, :3]

    def set_rect(self, rect):
        self.rect = rect

    def close(self):
        if self._sct is not None:
            self._sct.close()


def make_source(rect):
    """Prefer dxcam, fall back to mss with a hint."""
    try:
        src = DxcamSource(rect)
        if src.grab() is not None:
            print("Capture backend: dxcam (DXGI, ~5-10ms/frame)")
            return src
        src.close()
        print("dxcam produced no frame, falling back to mss")
    except Exception as e:
        print(f"dxcam unavailable ({type(e).__name__}: {e}), falling back to mss")
        print("For 3-5x faster capture: pip install dxcam")
    print("Capture backend: mss (GDI, ~30-40ms/frame)")
    return MssSource(rect)


# ---------------------------------------------------------------------------
# Inference engines
# ---------------------------------------------------------------------------

class TorchEngine:
    """Ultralytics .pt weights on CPU (or CUDA if explicitly requested)."""
    name = "torch"

    def __init__(self, model_path, device, size, threads):
        if threads > 0:
            try:
                import torch
                torch.set_num_threads(threads)
                print(f"Torch inference capped to {threads} CPU threads")
            except Exception:
                pass
        from ultralytics import YOLO
        self.model  = YOLO(str(model_path))
        self.device = device
        self.size   = size

        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        r = self.model.predict(dummy, imgsz=size, conf=0.25,
                               device=device, verbose=False)[0]
        n = 0 if r.boxes is None else len(r.boxes)
        print(f"Warmup ({self.name}): {n} detections on blank frame (should be 0)")
        if n > 10:
            print("WARNING: many detections on a blank frame - wrong weights?")

    def infer(self, img, conf, max_det):
        # imgsz must be passed explicitly: without it ultralytics resizes
        # internally to its own default and silently ignores our preprocessing
        r = self.model.predict(img, imgsz=self.size, conf=conf, iou=IOU_NMS,
                               max_det=max_det, device=self.device,
                               retina_masks=False, verbose=False)[0]
        out = []
        if r.masks is not None and r.boxes is not None:
            for poly, box in zip(r.masks.xy, r.boxes):
                if len(poly) >= 3:
                    out.append((np.asarray(poly, dtype=np.float32),
                                float(box.conf)))
        return out


def decode_seg(pred, proto, conf_thres, iou_thres, max_det, size):
    """
    Full post-processing for a raw ultralytics ONNX segmentation export.

    The export contains NO NMS and NO mask decoding: pred is
    (1, 4+nc+32, anchors) raw predictions and proto is (1, 32, mh, mw) mask
    prototypes. Loading such a file back through ultralytics floods every
    frame with max_det garbage boxes - so we decode here ourselves:
    confidence filter, NMS, coeff x proto matmul, sigmoid, crop to box,
    threshold, contour extraction.
    """
    # Exports differ in output orientation depending on the
    # ultralytics/onnxslim version: the export log prints (1, ch, anchors)
    # but the slimmed graph can emit (1, anchors, ch). Reading the wrong
    # orientation turns pixel coordinates into "confidences" in the
    # hundreds and saturates the mask sigmoid. The anchors dim is always
    # the much larger one, and only the correct orientation yields class
    # scores inside [0, 1] - validate both ways.
    def _oriented(mat):
        ncc = mat.shape[1] - 36
        if ncc < 1:
            return None
        sc = mat[:, 4:4 + ncc].max(axis=1)
        if float(sc.max()) > 1.001 or float(sc.min()) < -0.001:
            return None
        return mat, ncc, sc

    pred0 = pred[0]
    first = pred0.T if pred0.shape[0] < pred0.shape[1] else pred0
    other = pred0 if first is not pred0 else pred0.T
    got = _oriented(first) or _oriented(other)
    if got is None:
        return []
    p, nc, scores = got

    keep = scores >= conf_thres
    if not keep.any():
        return []
    p, scores = p[keep], scores[keep]

    boxes  = p[:, :4]                    # xywh, center-based, input pixels
    coeffs = p[:, 4 + nc:]
    tlwh   = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, 2:]], 1)

    idx = cv2.dnn.NMSBoxes(tlwh.tolist(),
                           scores.astype(float).tolist(),
                           float(conf_thres), float(iou_thres))
    if len(idx) == 0:
        return []
    idx = np.array(idx).reshape(-1)
    idx = idx[np.argsort(-scores[idx])][:max_det]

    pr = proto[0]                        # (32, mh, mw)
    mh, mw = pr.shape[1], pr.shape[2]
    pr = pr.reshape(32, -1)

    dets = []
    for i in idx:
        m = coeffs[i] @ pr
        m = 1.0 / (1.0 + np.exp(-np.clip(m, -30.0, 30.0)))
        m = m.reshape(mh, mw)

        x, y, w, h = tlwh[i]
        x1 = int(max(x, 0));            y1 = int(max(y, 0))
        x2 = int(min(x + w, size));     y2 = int(min(y + h, size))
        if x2 <= x1 or y2 <= y1:
            continue

        gx1 = x1 * mw // size;          gy1 = y1 * mh // size
        gx2 = min(x2 * mw // size + 1, mw)
        gy2 = min(y2 * mh // size + 1, mh)
        crop = m[gy1:gy2, gx1:gx2]
        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (x2 - x1, y2 - y1),
                          interpolation=cv2.INTER_LINEAR)
        binm = (crop > 0.5).astype(np.uint8)
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).astype(np.float32)
        if len(c) < 3:
            continue
        c[:, 0] += x1
        c[:, 1] += y1
        dets.append((c, float(scores[i])))
    return dets


class OnnxEngine:
    """ONNX Runtime inference with the decode_seg post-processor above."""

    def __init__(self, model_path, size, threads, use_openvino=False):
        import onnxruntime as ort
        self.name = "openvino" if use_openvino else "onnxrt"

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            so.intra_op_num_threads = threads
            print(f"ONNX Runtime capped to {threads} CPU threads")

        providers = ["CPUExecutionProvider"]
        if use_openvino:
            avail = ort.get_available_providers()
            if "OpenVINOExecutionProvider" not in avail:
                print("OpenVINO provider not available in this onnxruntime build.")
                print("Install it with: pip install onnxruntime-openvino")
                print("Continuing on the plain CPU provider.")
                self.name = "onnxrt"
            else:
                providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

        self.sess = ort.InferenceSession(str(model_path), so, providers=providers)
        self.inp  = self.sess.get_inputs()[0].name

        ishape = self.sess.get_inputs()[0].shape   # [1, 3, H, W]
        model_h = ishape[2]
        if isinstance(model_h, int) and model_h > 0 and model_h != size:
            print(f"Model was exported at {model_h}x{model_h}, "
                  f"overriding --img-size {size}")
            size = model_h
        self.size = size

        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        n = len(self.infer(dummy, 0.25, 8))
        print(f"Warmup ({self.name}): {n} detections on blank frame (should be 0)")

    def infer(self, img, conf, max_det):
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        outs  = self.sess.run(None, {self.inp: x})
        pred  = next(o for o in outs if o.ndim == 3)
        proto = next(o for o in outs if o.ndim == 4)
        return decode_seg(pred, proto, conf, IOU_NMS, max_det, self.size)


# ---------------------------------------------------------------------------
# Detection smoothing
# ---------------------------------------------------------------------------

class Smoother:
    """
    Hard frame-count persistence. When the model misses for a frame or two
    (rotation angle, momentary occlusion) we keep drawing the last known
    polygons for up to persist_frames misses instead of flickering off.
    Returning the CACHED list on a miss (not an empty one) is what actually
    stops the flicker - an EMA over polygon points was tried earlier and
    produced ghost polygons, so a plain counter it is.
    """

    def __init__(self, persist_frames=8, enabled=True):
        self.persist = persist_frames
        self.enabled = enabled
        self.last    = []
        self.miss    = 0

    def update(self, dets):
        if not self.enabled:
            return dets
        if dets:
            self.last = dets
            self.miss = 0
            return dets
        self.miss += 1
        if self.last and self.miss <= self.persist:
            return self.last
        self.last = []
        return []


# ---------------------------------------------------------------------------
# Shared state between the three threads
# ---------------------------------------------------------------------------

class SharedState:
    def __init__(self, rect):
        self.lock      = threading.Lock()
        self.frame     = None
        self.seq       = 0
        self.new_frame = threading.Event()
        self.draw      = []      # [(poly Nx2 float32 in frame space, conf)]
        self.n_raw     = 0
        self.inf_ms    = 0.0
        self.inf_count = 0       # completed inferences, drives the heartbeat
        self.rect         = rect  # full game-window rect (for the overlay)
        self.rect_changed = False
        self.running   = True

    def set_frame(self, frame):
        with self.lock:
            self.frame = frame
            self.seq  += 1
        self.new_frame.set()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None, self.seq
            return self.frame.copy(), self.seq

    def set_result(self, draw, n_raw, inf_ms):
        with self.lock:
            self.draw       = draw
            self.n_raw      = n_raw
            self.inf_ms     = inf_ms
            self.inf_count += 1

    def get_result(self):
        with self.lock:
            return list(self.draw), self.n_raw, self.inf_ms, self.inf_count

    def set_rect(self, rect):
        with self.lock:
            if rect != self.rect:
                self.rect         = rect
                self.rect_changed = True

    def pop_rect_change(self):
        with self.lock:
            changed, self.rect_changed = self.rect_changed, False
            return changed, self.rect


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

def capture_thread_fn(state, source, args):
    """Grab frames continuously; re-detect the game window every 2 seconds so
    everything follows if the window is moved or resized."""
    last_check = 0.0
    while state.running:
        now = time.time()
        if args.capture == "game" and now - last_check > 2.0:
            last_check = now
            rect = find_game_window()
            if rect is not None and rect != state.rect:
                print(f"Game window moved/resized: {rect}")
                cap_rect, _, _ = compute_capture(rect, args.crop, args.y_offset)
                source.set_rect(cap_rect)
                state.set_rect(rect)
        frame = source.grab()
        if frame is not None:
            state.set_frame(frame)
        time.sleep(0.005)


def inference_thread_fn(state, engine, smoother, args):
    period    = 1.0 / max(1, args.fps)
    frames    = 0
    last_seq  = -1
    debug_dir = None
    if args.debug:
        debug_dir = Path("debug_frames")
        debug_dir.mkdir(exist_ok=True)
        print(f"Debug: saving annotated frames to {debug_dir}/")

    while state.running:
        t0 = time.perf_counter()
        state.new_frame.wait(timeout=0.5)
        state.new_frame.clear()

        frame, seq = state.get_frame()
        if frame is None or seq == last_seq:
            continue
        last_seq = seq

        H, W = frame.shape[:2]
        # stretch resize, NOT letterbox: the models were trained on Roboflow
        # exports that were stretch-resized to 640x640, and matching the
        # training preprocessing measurably improves detection quality
        inp = cv2.resize(frame, (engine.size, engine.size),
                         interpolation=cv2.INTER_LINEAR)

        dets640 = engine.infer(inp, args.conf, args.max_det)

        sx, sy = W / engine.size, H / engine.size
        dets = []
        for poly, cf in dets640:
            p = poly.copy()
            p[:, 0] *= sx
            p[:, 1] *= sy
            if args.min_area > 0:
                if cv2.contourArea(p.astype(np.int32)) < args.min_area:
                    continue
            dets.append((p, cf))

        draw   = smoother.update(dets)
        inf_ms = (time.perf_counter() - t0) * 1000
        state.set_result(draw, len(dets), inf_ms)
        frames += 1

        if args.debug and frames % 10 == 0:
            print(f"Frame {frames}: raw={len(dets)} drawn={len(draw)} "
                  f"inf={inf_ms:.0f}ms")
        if debug_dir is not None and frames % 30 == 0 and draw:
            img = draw_on_frame(frame, draw)
            p = debug_dir / f"frame_{frames:05d}_{len(draw)}cores.jpg"
            cv2.imwrite(str(p), img)
            print(f"\tSaved {p}")

        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def draw_on_frame(frame, dets, alpha=0.30):
    """Annotate a copy of frame with filled polygons, outlines and labels.
    The alpha blend writes to a fresh buffer - blending in place over one of
    the inputs is undefined behavior in OpenCV and produced blank output."""
    out = frame.copy()
    if not dets:
        return out

    overlay = frame.copy()
    for poly, cf in dets:
        pts = poly.astype(np.int32)
        bgr, _ = conf_to_color(cf)
        cv2.fillPoly(overlay, [pts], bgr)

    blended = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)

    for poly, cf in dets:
        pts = poly.astype(np.int32)
        bgr, _ = conf_to_color(cf)
        cv2.polylines(blended, [pts], True, bgr, 3)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        label = f"core {cf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(blended, (cx - 4, cy - th - 8), (cx + tw + 4, cy + 4),
                      (30, 30, 30), -1)
        cv2.putText(blended, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, bgr, 2, cv2.LINE_AA)
    return blended


def place_on_monitor2(win_name, cap_w, cap_h):
    """Place the OpenCV window on the second monitor at a sensible landscape
    size (never fullscreen - looks terrible on a portrait monitor)."""
    try:
        import screeninfo
        monitors = screeninfo.get_monitors()
        if len(monitors) >= 2:
            m2 = monitors[1]
            disp_w = min(m2.width - 40, 1280)
            disp_h = int(disp_w * cap_h / cap_w)
            cv2.moveWindow(win_name, m2.x + 10, max(m2.y, 0) + 10)
            cv2.resizeWindow(win_name, disp_w, disp_h)
            kind = "portrait" if m2.height > m2.width else "landscape"
            print(f"Window on monitor 2 ({m2.width}x{m2.height} {kind}) "
                  f"at {disp_w}x{disp_h}. Resize/move freely.")
            return
        print("Only one monitor detected - window opens on primary")
    except ImportError:
        cv2.moveWindow(win_name, 1920, 0)
        print("screeninfo not installed - guessed monitor 2 at x=1920")
        print("Install for auto placement: pip install screeninfo")


def run_monitor2(state, args, ring_label, engine_name, cap_w, cap_h):
    win = f"Core Mining AI [{ring_label}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    place_on_monitor2(win, cap_w, cap_h)

    period = 1.0 / UI_FPS
    print("Running. Press Q or ESC in the display window to quit.")
    while state.running:
        t0 = time.perf_counter()
        frame, _ = state.get_frame()
        draw, n, inf_ms, _ticks = state.get_result()

        if frame is None:
            canvas = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(canvas, "waiting for capture...", (30, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        else:
            canvas = draw_on_frame(frame, draw)

        best = max((cf for _, cf in draw), default=0.0)
        hud = (f"{ring_label}  {n} core(s)"
               + (f"  best {best:.2f}" if draw else "")
               + f"  {inf_ms:.0f}ms  {engine_name}  conf>={args.conf:.2f}")
        cv2.putText(canvas, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)

        wait_ms = max(1, int((period - (time.perf_counter() - t0)) * 1000))
        k = cv2.waitKey(wait_ms) & 0xFF
        if k in (ord("q"), 27):
            state.running = False
    cv2.destroyAllWindows()


def make_clickthrough(root):
    """Windows only: mark the overlay so mouse clicks pass through to the
    game underneath instead of hitting the (invisible) tk window."""
    try:
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        if not hwnd:
            hwnd = root.winfo_id()
        GWL_EXSTYLE       = -20
        WS_EX_LAYERED     = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(
            hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
        return True
    except Exception as e:
        print(f"Click-through setup failed ({e}) - overlay will catch the mouse")
        return False


def run_overlay(state, args, ring_label, engine_name):
    import tkinter as tk

    l, t, w, h = state.rect
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-transparentcolor", TRANSPARENT_HEX)
    if args.opacity < 1.0:
        # global window alpha: only the drawn polygons and text are visible,
        # so this dims the overlay graphics without touching the game
        root.attributes("-alpha", args.opacity)
    root.configure(bg=TRANSPARENT_HEX)
    root.geometry(f"{w}x{h}+{l}+{t}")

    canvas = tk.Canvas(root, width=w, height=h, bg=TRANSPARENT_HEX,
                       highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    if not args.no_clickthrough:
        if make_clickthrough(root):
            print("Overlay is click-through: mouse and keys go to the game.")

    # capture region inside the window: polygons arrive in region coordinates
    # while the canvas spans the full window, so drawing shifts by the region
    # offset. region = [rel_x, rel_y, width, height], refreshed if the game
    # window moves or resizes.
    cap0, off_x, off_y = compute_capture(state.rect, args.crop, args.y_offset)
    region = [off_x, off_y, cap0[2], cap0[3]]

    def draw_region_box(draw_list, ticks):
        """Mark the model's field of view: dashed outline plus corner
        brackets. Bracket color follows the best current detection (dim
        gray-green while the region is empty), and a small heartbeat square
        alternates on every completed inference - if it freezes, the
        inference pipeline has stalled."""
        rx, ry, rw, rh = region
        if draw_list:
            _, color = conf_to_color(max(cf for _, cf in draw_list))
        else:
            color = "#3f523f"
        canvas.create_rectangle(rx, ry, rx + rw, ry + rh,
                                outline="#2c3a2c", width=1, dash=(4, 8))
        arm = max(16, min(34, rw // 12))
        for cx, cy, dx, dy in ((rx, ry, 1, 1), (rx + rw, ry, -1, 1),
                               (rx, ry + rh, 1, -1), (rx + rw, ry + rh, -1, -1)):
            canvas.create_line(cx, cy, cx + dx * arm, cy, fill=color, width=3)
            canvas.create_line(cx, cy, cx, cy + dy * arm, fill=color, width=3)
        beat = "#30ff70" if ticks % 2 == 0 else "#1a5c33"
        canvas.create_rectangle(rx + 8, ry + 8, rx + 16, ry + 16,
                                fill=beat, outline="")

    def tick():
        if not state.running:
            root.destroy()
            return
        changed, rect2 = state.pop_rect_change()
        if changed:
            l2, t2, w2, h2 = rect2
            root.geometry(f"{w2}x{h2}+{l2}+{t2}")
            canvas.config(width=w2, height=h2)
            cap2, ox, oy = compute_capture(rect2, args.crop, args.y_offset)
            region[:] = [ox, oy, cap2[2], cap2[3]]

        canvas.delete("all")
        draw, n, inf_ms, ticks = state.get_result()

        if not args.no_region_box:
            draw_region_box(draw, ticks)

        for poly, cf in draw:
            _, hexc = conf_to_color(cf)
            pts = poly.copy()
            pts[:, 0] += region[0]
            pts[:, 1] += region[1]
            flat = pts.astype(int).flatten().tolist()
            # stipple fill fakes translucency - tkinter has no real alpha,
            # gray25 paints every 4th pixel for a ~25% wash
            canvas.create_polygon(flat, outline=hexc, fill=hexc,
                                  stipple="gray25", width=3)
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            label = f"core {cf:.2f}"
            canvas.create_text(cx + 1, cy + 1, text=label, fill="#101010",
                               font=("Consolas", 12, "bold"))
            canvas.create_text(cx, cy, text=label, fill=hexc,
                               font=("Consolas", 12, "bold"))

        best = max((cf for _, cf in draw), default=0.0)
        hud = (f"{ring_label}  {n} core(s)"
               + (f"  best {best:.2f}" if draw else "")
               + f"  |  {inf_ms:.0f}ms {engine_name}  conf>={args.conf:.2f}")
        hx, hy = region[0] + 22, region[1] + 12
        canvas.create_text(hx + 1, hy + 1, text=hud, anchor="w", fill="#102010",
                           font=("Consolas", 10, "bold"))
        canvas.create_text(hx, hy, text=hud, anchor="w", fill="#30ff70",
                           font=("Consolas", 10, "bold"))
        root.after(int(1000 / UI_FPS), tick)

    def reassert_topmost():
        # some fullscreen-ish games steal z-order; re-assert every 2s
        if not state.running:
            return
        try:
            root.attributes("-topmost", True)
        except Exception:
            return
        root.after(2000, reassert_topmost)

    def on_close(*_):
        state.running = False

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Escape>", on_close)
    signal.signal(signal.SIGINT, lambda *_: on_close())

    print(f"Overlay active: {w}x{h} at ({l},{t})")
    print("Elite Dangerous must be in BORDERLESS WINDOWED mode.")
    print("Ctrl+C in this terminal to quit.")

    root.after(100, tick)
    root.after(2000, reassert_topmost)
    root.mainloop()
    state.running = False


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def find_model(ring_type, models_dir, want_onnx):
    """
    Locate weights for a ring type. Torch runtime prefers .pt (exports/ first,
    then the newest matching best.pt under runs/). ONNX runtimes require the
    .onnx export.
    """
    models_dir = Path(models_dir)

    if want_onnx:
        cand = models_dir / f"{ring_type}_best.onnx"
        if cand.exists():
            return cand
        return None

    cand = models_dir / f"{ring_type}_best.pt"
    if cand.exists():
        return cand
    if Path("runs").exists():
        hits = sorted(Path("runs").rglob(f"*{ring_type}*/weights/best.pt"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if hits:
            return hits[0]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_crop(text):
    s = text.lower().strip()
    if "x" in s and "," not in s:
        try:
            w, h = (int(v) for v in s.split("x"))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "centered crop must be WxH, e.g. 1440x810")
        if w < 64 or h < 64:
            raise argparse.ArgumentTypeError("centered crop must be >= 64x64")
        return ("center", w, h)
    try:
        parts = [int(v) for v in s.split(",")]
    except ValueError:
        parts = []
    if len(parts) != 4 or any(v < 0 for v in parts):
        raise argparse.ArgumentTypeError(
            "crop-area is either WxH (centered region, e.g. 1440x810) or "
            "L,T,R,B edge margins (e.g. 0,0,0,300)")
    return ("margins", *parts)


def main():
    parser = argparse.ArgumentParser(
        description="Elite Dangerous Core Mining AI Companion",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ring-type", choices=RING_TYPES,
        help="Which ring model to load: exports/<ring>_best.pt")
    parser.add_argument("--model-path",
        help="Explicit weights path, overrides --ring-type lookup "
             "(e.g. exports/metal_rich_yolo11n_best.pt)")
    parser.add_argument("--models-dir", default="exports",
        help="Folder with exported models (default: exports)")
    parser.add_argument("--conf", type=float, default=0.35,
        help="Confidence threshold (default 0.35)")
    parser.add_argument("--fps", type=int, default=6,
        help="Target inference fps (default 6). Display always runs at 20.")
    parser.add_argument("--display", choices=["monitor2", "overlay"],
        default="monitor2")
    parser.add_argument("--capture", choices=["game", "primary"],
        default="game")
    parser.add_argument("--runtime", choices=["torch", "onnxrt", "openvino"],
        default="torch",
        help="torch=.pt via ultralytics (default). onnxrt=.onnx via ONNX "
             "Runtime with built-in NMS/mask decode, usually 1.5-2x faster "
             "on CPU. openvino=onnxrt with the Intel OpenVINO provider.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
        help="torch runtime device. cuda is NOT recommended while the game "
             "runs: torch.cuda.synchronize contends with the game's render "
             "pipeline and causes 500-1500ms stalls.")
    parser.add_argument("--img-size", type=int, default=640,
        help="Model input size (default 640, must match training)")
    parser.add_argument("--threads", type=int, default=0,
        help="Cap inference CPU threads so the game keeps headroom "
             "(0=library default; 4 is a good value on the 12600KF)")
    parser.add_argument("--max-det", type=int, default=8,
        help="Max detections per frame (default 8 - cores are rare and "
             "almost never appear more than one at a time)")
    parser.add_argument("--min-area", type=int, default=0,
        help="Drop polygons smaller than this many pixels (frame space). "
             "Kills distant speck false positives. 0=off.")
    parser.add_argument("--crop-area", type=parse_crop, dest="crop",
        default=("margins", 0, 0, 0, 0), metavar="WxH|L,T,R,B",
        help="Capture region. WxH grabs a centered region of that size "
             "(e.g. 1440x810 - prospected rocks sit near screen center, so "
             "the core fills more of the model input: faster AND more "
             "accurate). L,T,R,B trims pixel margins from the window edges "
             "(e.g. 0,0,0,300 ignores the cockpit dash). Overlay polygons "
             "stay aligned automatically in both formats.")
    parser.add_argument("--y-offset", type=int, default=0,
        help="Shift the capture region vertically by N px, negative = up "
             "(e.g. -120 centers on the space above the cockpit dash). "
             "Works with both crop formats.")
    parser.add_argument("--opacity", type=float, default=1.0,
        help="Overlay graphics opacity 0.1-1.0 (default 1.0)")
    parser.add_argument("--no-smoothing", action="store_true",
        help="Disable detection persistence (raw per-frame results)")
    parser.add_argument("--persist-frames", type=int, default=8,
        help="Miss frames to keep drawing the last polygons for (default 8)")
    parser.add_argument("--no-clickthrough", action="store_true",
        help="Overlay catches the mouse instead of passing it to the game")
    parser.add_argument("--no-region-box", action="store_true",
        help="Hide the capture-region marker (corner brackets, dashed "
             "outline, status line, heartbeat) in overlay mode")
    parser.add_argument("--debug", action="store_true",
        help="Print detection stats every 10 frames and save annotated "
             "frames to debug_frames/ (these make great hard negatives "
             "for retraining when they are false positives)")
    args = parser.parse_args()

    if not args.ring_type and not args.model_path:
        parser.error("provide --ring-type or --model-path")
    ring_label = args.ring_type or Path(args.model_path).stem

    # --- resolve model ---
    want_onnx = args.runtime in ("onnxrt", "openvino")
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            sys.exit(1)
        if want_onnx and model_path.suffix != ".onnx":
            print(f"--runtime {args.runtime} needs a .onnx file, got {model_path}")
            sys.exit(1)
    else:
        model_path = find_model(args.ring_type, args.models_dir, want_onnx)
        if model_path is None:
            if want_onnx:
                print(f"No {args.ring_type}_best.onnx in {args.models_dir}/")
                print("Export one in the notebook: set EXPORT_ONNX = True and "
                      "run export_best_models(), or run "
                      "export_specific(ring, model, export_onnx=True)")
            else:
                print(f"No {args.ring_type}_best.pt in {args.models_dir}/ "
                      f"and nothing under runs/")
                print("Run export_best_models() in the notebook first.")
            sys.exit(1)

    print(f"Loading: {model_path}  (runtime: {args.runtime})")
    if args.runtime == "torch":
        if args.device == "cuda":
            print("WARNING: CUDA inference alongside the game is known to "
                  "stall 500-1500ms per frame (render pipeline contention). "
                  "CPU is the recommended device.")
        engine = TorchEngine(model_path, args.device, args.img_size,
                             args.threads)
    else:
        engine = OnnxEngine(model_path, args.img_size, args.threads,
                            use_openvino=(args.runtime == "openvino"))

    # --- capture region ---
    game_rect = None
    if args.capture == "game":
        game_rect = find_game_window()
        if game_rect is None:
            print("Elite Dangerous window not found "
                  "(title must contain 'elite' and 'dangerous').")
            print("Falling back to primary monitor capture.")
    if game_rect is None:
        game_rect = get_primary_monitor_rect()
        if args.display == "overlay":
            print("Overlay needs the game window position - "
                  "switching to monitor2 display.")
            args.display = "monitor2"

    cap_rect, rel_x, rel_y = compute_capture(game_rect, args.crop,
                                             args.y_offset)
    print(f"Window: {game_rect[2]}x{game_rect[3]} at "
          f"({game_rect[0]},{game_rect[1]})")
    if args.crop[0] == "center":
        print(f"Capture: centered {cap_rect[2]}x{cap_rect[3]} at "
              f"window offset (+{rel_x},+{rel_y})")
    elif args.crop != ("margins", 0, 0, 0, 0) or args.y_offset:
        print(f"Capture after crop: {cap_rect[2]}x{cap_rect[3]} at "
              f"window offset (+{rel_x},+{rel_y})")

    state    = SharedState(game_rect)
    source   = make_source(cap_rect)
    smoother = Smoother(args.persist_frames, enabled=not args.no_smoothing)

    print(f"Ring:       {ring_label}")
    print(f"Confidence: {args.conf}")
    print(f"Inference:  {args.fps} fps target, display fixed at {UI_FPS} fps")
    print(f"Smoothing:  "
          f"{'off' if args.no_smoothing else f'persist {args.persist_frames} frames'}")

    t_cap = threading.Thread(target=capture_thread_fn,
                             args=(state, source, args), daemon=True)
    t_inf = threading.Thread(target=inference_thread_fn,
                             args=(state, engine, smoother, args), daemon=True)
    t_cap.start()
    t_inf.start()

    try:
        if args.display == "overlay":
            run_overlay(state, args, ring_label, engine.name)
        else:
            run_monitor2(state, args, ring_label, engine.name,
                         cap_rect[2], cap_rect[3])
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        time.sleep(0.1)
        source.close()
        print("Stopped.")


if __name__ == "__main__":
    main()
