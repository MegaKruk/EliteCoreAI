"""
Elite Dangerous Core Mining AI Companion

Captures the Elite Dangerous game window, runs segmentation inference, and
displays detected core asteroid polygon masks with confidence scores.

Uses CPU for inference by default. GPU (CUDA) is available via --device cuda
but causes massive stalls (500-1500ms) when the game is rendering because
torch.cuda.synchronize() blocks on ALL pending GPU work including the game's
rendering pipeline. CPU inference is ~150-200ms per frame on an i5-12600KF,
which is consistent and stall-free.

Screen capture uses dxcam (DXGI Desktop Duplication API) if installed, which
is ~3x faster than mss. Falls back to mss if dxcam is not available.
Install: pip install dxcam

Architecture: 3 threads for overlay mode, single-threaded for monitor2.
  Overlay: capture thread -> latest_frame -> inference thread -> latest_polys
           main thread reads latest_polys and redraws canvas at 20fps.

Two display modes:
  monitor2  - OpenCV window on your second monitor (always works, use this first)
  overlay   - transparent window drawn over the game (requires borderless mode)

Usage:
    python companion_app.py --ring-type ice
    python companion_app.py --ring-type ice --conf 0.35
    python companion_app.py --ring-type ice --display overlay --conf 0.35
    python companion_app.py --ring-type ice --debug
    python companion_app.py --ring-type metal_rich --model-path exports/metal_rich_yolo11_best.pt
    python companion_app.py --ring-type ice --no-smoothing
    python companion_app.py --ring-type ice --device cuda
    python companion_app.py --ring-type ice --runtime openvino
    python companion_app.py --ring-type ice --runtime onnxrt

Controls:
    monitor2 mode: press Q in the display window to quit
    overlay mode:  Ctrl+C in terminal to quit

NOTE on model format:
    The app loads .pt weights directly (not .onnx). ONNX exports of segmentation
    models do not apply NMS properly when loaded via ultralytics, causing every
    frame to return exactly max_det=300 garbage detections. The .pt file works
    correctly and runs fine on CPU since ultralytics is already installed.
"""

import argparse
import os
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np

os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO


RING_TYPES        = ["ice", "rocky", "metallic", "metal_rich"]
GAME_WINDOW_TITLE = "EliteDangerous64"

TRANSPARENT_HEX = "#000001"
POLY_COLOR_BGR  = (0, 255, 80)
POLY_COLOR_HEX  = "#00FF50"
LABEL_BG_BGR    = (0, 140, 40)
TEXT_BGR        = (255, 255, 255)
MASK_ALPHA      = 0.30

# max_det cap - lower than YOLO default 300 to reduce noise on cluttered frames
MAX_DET = 50


def conf_to_color_bgr(conf):
    """
    Map a confidence value to a BGR color.
    <= 0.7 = red, 0.85 = orange, 1.0 = green.
    """
    conf = max(0.0, min(1.0, conf))
    if conf <= 0.7:
        # pure red
        return (0, 0, 255)
    elif conf < 0.85:
        # red (0,0,255) -> orange (0,165,255)
        t = (conf - 0.7) / 0.15
        return (0, int(165 * t), 255)
    else:
        # orange (0,165,255) -> green (0,255,0)
        t = (conf - 0.85) / 0.15
        return (0, 165 + int(90 * t), 255 - int(255 * t))


def conf_to_color_hex(conf):
    """Same as conf_to_color_bgr but returns a hex string for tkinter."""
    b, g, r = conf_to_color_bgr(conf)
    return f"#{r:02X}{g:02X}{b:02X}"


def conf_to_label_bg_bgr(conf):
    """Darker version of the confidence color for label backgrounds."""
    b, g, r = conf_to_color_bgr(conf)
    return (b // 2, g // 2, r // 2)


def find_model(ring_type, models_dir):
    """
    Look for the best trained weights for a ring type.
    Prefers .pt over .onnx because ONNX segmentation models loaded via
    ultralytics do not apply NMS, returning max_det garbage detections.
    Returns the Path to the model file, or None if not found.
    """
    models_dir = Path(models_dir)

    # prefer .pt - works correctly with NMS
    candidates = [models_dir / f"{ring_type}_best.pt"]

    # also check inside runs/ in case export_best_models() wasn't run yet
    if Path("runs").exists():
        candidates += sorted(Path("runs").rglob(f"*{ring_type}*best.pt"))

    # .onnx as last resort - works but has an NMS bug, see module docstring
    candidates.append(models_dir / f"{ring_type}_best.onnx")

    for path in candidates:
        if path.exists():
            return path

    return None


def find_game_window():
    """Return (left, top, width, height) of the Elite Dangerous window, or None."""
    try:
        import pygetwindow as gw
        wins = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)
        if not wins:
            # fallback: match "elite" AND "dangerous" to avoid matching our own
            # project windows like "EliteCoreAI - companion_app.py" in PyCharm
            wins = [w for w in gw.getAllWindows()
                    if "elite" in w.title.lower() and "dangerous" in w.title.lower()]
        if wins:
            w = wins[0]
            print(f"Found window: '{w.title}' at ({w.left},{w.top}) size {w.width}x{w.height}")
            return (w.left, w.top, w.width, w.height)
    except ImportError:
        print("pygetwindow not installed: pip install pygetwindow")
    except Exception as e:
        print(f"Window lookup error: {e}")
    return None


def get_primary_monitor_rect():
    """Return (left, top, width, height) of the primary monitor."""
    import mss
    with mss.mss() as sct:
        m = sct.monitors[1]
        return (m["left"], m["top"], m["width"], m["height"])


def get_monitor2_rect():
    """
    Return (x, y, width, height) of the second monitor for window placement.
    Returns None if there is only one monitor.
    """
    try:
        import screeninfo
        monitors = screeninfo.get_monitors()
        if len(monitors) >= 2:
            m = monitors[1]
            return (m.x, m.y, m.width, m.height)
    except ImportError:
        pass
    return None


def capture_frame(sct, left, top, width, height):
    """Capture a screen region and return a BGR numpy array. (mss backend)"""
    raw = sct.grab({"left": left, "top": top, "width": width, "height": height})
    return np.array(raw)[:, :, :3]


def create_dxcam_camera(capture_rect):
    """
    Try to create a dxcam camera for fast screen capture via DXGI.
    Returns the camera object or None if dxcam is not installed.
    dxcam is ~3x faster than mss on Windows (5-10ms vs 30-40ms at 2560x1440).
    """
    try:
        import dxcam
        left, top, w, h = capture_rect
        camera = dxcam.create(output_color="BGR")
        # start continuous capture so grab() reads from an in-memory ring buffer
        # instead of polling DXGI each time. region is (left, top, right, bottom).
        region = (left, top, left + w, top + h)
        camera.start(target_fps=60, video_mode=True, region=region)
        print(f"Screen capture: dxcam (DXGI, ~5-10ms per frame)")
        return camera
    except ImportError:
        print("Screen capture: mss (GDI, ~30-40ms per frame)")
        print("For faster capture: pip install dxcam")
        return None
    except Exception as e:
        print(f"dxcam init failed ({e}), falling back to mss")
        return None


def prepare_for_inference(frame, imgsz=640):
    """
    Resize frame to imgsz x imgsz using stretch (no letterbox, no padding).
    This matches the Roboflow preprocessing used during training.
    Ultralytics predict() on a numpy array uses letterbox by default, which
    adds grey bars and mismatches training, hurting accuracy on wide frames.
    Returns (resized, scale_x, scale_y) so coordinates can be mapped back.
    """
    H, W = frame.shape[:2]
    resized = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    return resized, W / float(imgsz), H / float(imgsz)


def extract_detections(result, conf_threshold, scale_xy):
    """
    Pull all detections from a YOLO result into a flat list.
    Each entry is (pts, conf, cx, cy) where pts is a scaled numpy polygon,
    conf is the raw confidence, and (cx, cy) is the polygon centroid.
    """
    boxes = result.boxes
    masks = result.masks
    out = []

    if boxes is None or masks is None or len(boxes) == 0:
        return out

    sx, sy = scale_xy

    for box, mask_xy in zip(boxes, masks.xy):
        conf = float(box.conf)
        if conf < conf_threshold:
            continue

        pts = mask_xy.astype(np.int32)
        if len(pts) < 3:
            continue

        pts = (pts * np.array([[sx, sy]])).astype(np.int32)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        out.append((pts, conf, cx, cy))

    return out


class DetectionTracker:
    """
    Prevents bounding box flickering using the same approach as production
    trackers (ByteTrack, SORT): frame counting with a persistence buffer.

    Two counters control visibility:
    - min_hits: how many strong detections (above --conf) at roughly the same
      screen position before we start drawing. Prevents single-frame noise
      from triggering display. Default 2 (~0.5s at 4fps).
    - persist_frames: after the last strong detection, keep drawing the last
      known polygon for this many frames. Bridges the detection gaps that
      happen when the model misses every other frame. Default 8 (~2s at 4fps).
      When the ship moves away and detections stop, the box disappears after
      persist_frames with no ghost lingering.

    The tracker also uses a lower internal_conf (conf * 0.4) for position
    tracking: weak detections below --conf update the tracked centroid so
    spatial matching stays accurate, but only strong detections (>= --conf)
    count toward min_hits and reset the persistence timer.
    """

    def __init__(self, conf_threshold, min_hits=2, persist_frames=8,
                 max_match_dist=400):
        self.conf_threshold = conf_threshold
        self.min_hits = min_hits
        self.persist_frames = persist_frames
        self.max_match_dist = max_match_dist
        self.internal_conf = max(0.10, conf_threshold * 0.4)

        self.hit_count = 0
        self.frames_since_hit = 999
        self.active = False
        self.last_centroid = None
        self.last_draw_list = []
        self.last_best_conf = 0.0

    def update(self, detections, debug=False):
        """
        Feed one frame's detections into the tracker.

        detections: list of (pts, conf, cx, cy) from extract_detections().
        Returns: (visible, best_conf, detections_to_draw)
        """
        # match detections near the tracked position
        matched = []
        for pts, conf, cx, cy in detections:
            if self.last_centroid is not None:
                dx = cx - self.last_centroid[0]
                dy = cy - self.last_centroid[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > self.max_match_dist:
                    continue
            matched.append((pts, conf, cx, cy))

        # separate strong (above --conf) from weak detections
        strong = [(p, c, x, y) for p, c, x, y in matched
                  if c >= self.conf_threshold]
        best_weak = max(matched, key=lambda d: d[1]) if matched else None

        if strong:
            strong.sort(key=lambda d: d[1], reverse=True)
            best = strong[0]
            self.last_centroid = (best[2], best[3])
            self.last_draw_list = [(p, c) for p, c, x, y in strong]
            self.last_best_conf = best[1]
            self.hit_count += 1
            self.frames_since_hit = 0

            if not self.active and self.hit_count >= self.min_hits:
                self.active = True
                if debug:
                    print(f"  Tracker: ACTIVATED after {self.hit_count} hits")

            if debug:
                print(f"  Tracker: strong hit conf={best[1]:.3f}, "
                      f"hits={self.hit_count}, age=0, "
                      f"fragments={len(strong)}")

        elif best_weak is not None:
            # weak detection: update centroid for spatial tracking,
            # but don't count as a hit and don't reset persistence timer
            self.last_centroid = (best_weak[2], best_weak[3])
            self.frames_since_hit += 1

            if debug:
                print(f"  Tracker: weak hit conf={best_weak[1]:.3f}, "
                      f"age={self.frames_since_hit}")
        else:
            self.frames_since_hit += 1

            if debug and self.active:
                print(f"  Tracker: miss, age={self.frames_since_hit}")

        # deactivate if persistence window expired
        if self.frames_since_hit > self.persist_frames and self.active:
            self.active = False
            if debug:
                print(f"  Tracker: DEACTIVATED (no strong hit for "
                      f"{self.frames_since_hit} frames)")

        # reset tracking state after a longer absence so a new core
        # elsewhere on screen can be picked up fresh
        if self.frames_since_hit > self.persist_frames + 30:
            self.last_centroid = None
            self.last_draw_list = []
            self.hit_count = 0
            self.last_best_conf = 0.0

        if self.active and self.last_draw_list:
            return True, self.last_best_conf, self.last_draw_list
        else:
            return False, self.last_best_conf, []


def draw_from_tracked(frame, tracked_detections, best_conf, opacity=0.30, debug=False):
    """
    Draw detections selected by the tracker onto a frame.
    tracked_detections: list of (pts, conf) from DetectionTracker.update().
    best_conf: strongest raw confidence from the last strong detection.
    opacity: fill opacity (0.0-1.0). Outline is always fully visible.
    Returns (annotated_frame, n_drawn).
    """
    if not tracked_detections:
        return frame.copy(), 0

    H, W = frame.shape[:2]
    out = frame.copy()
    overlay = frame.copy()
    n_drawn = 0

    for pts, raw_conf in tracked_detections:
        if len(pts) < 3:
            continue

        color = conf_to_color_bgr(raw_conf)
        label_bg = conf_to_label_bg_bgr(raw_conf)

        if debug:
            print(f"  Drawing: conf={raw_conf:.3f}, "
                  f"pts={len(pts)}, "
                  f"x=[{pts[:,0].min()}-{pts[:,0].max()}], "
                  f"y=[{pts[:,1].min()}-{pts[:,1].max()}]")

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=3)

        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        label = f"core {raw_conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        pad = 4
        cv2.rectangle(out,
                      (cx - pad, cy - th - pad * 2),
                      (cx + tw + pad, cy + pad),
                      label_bg, -1)
        cv2.putText(out, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_BGR, 2, cv2.LINE_AA)
        n_drawn += 1

    blended = np.empty_like(out)
    cv2.addWeighted(overlay, opacity, out, 1.0 - opacity, 0, blended)
    return blended, n_drawn


def draw_detections(frame, result, conf_threshold, scale_xy=(1.0, 1.0),
                    opacity=0.30, debug=False):
    """
    Draw segmentation polygon masks and confidence labels on a copy of frame.

    scale_xy: (scale_x, scale_y) to map polygon coordinates from inference
              space (640x640) back to the original frame dimensions. Pass the
              values returned by prepare_for_inference().
    opacity: fill opacity (0.0-1.0). Outline is always fully visible.

    Returns (annotated_frame, n_drawn).
    """
    boxes = result.boxes
    masks = result.masks

    if boxes is None or len(boxes) == 0:
        return frame.copy(), 0

    keep = [i for i, b in enumerate(boxes) if float(b.conf) >= conf_threshold]
    if not keep:
        return frame.copy(), 0

    H, W    = frame.shape[:2]
    out     = frame.copy()
    overlay = frame.copy()
    n_drawn = 0
    sx, sy  = scale_xy

    if masks is not None:
        for i in keep:
            pts  = masks.xy[i].astype(np.int32)
            conf = float(boxes[i].conf)

            if len(pts) < 3:
                continue

            # scale from 640x640 inference space back to frame space
            pts = (pts * np.array([[sx, sy]])).astype(np.int32)

            color = conf_to_color_bgr(conf)
            label_bg = conf_to_label_bg_bgr(conf)

            if debug:
                print(f"  Detection {i}: conf={conf:.3f}, pts={len(pts)}, "
                      f"x=[{pts[:,0].min()}-{pts[:,0].max()}], "
                      f"y=[{pts[:,1].min()}-{pts[:,1].max()}], "
                      f"frame={W}x{H}")

            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=3)

            cx    = int(pts[:, 0].mean())
            cy    = int(pts[:, 1].mean())
            label = f"core {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            pad = 4
            cv2.rectangle(out,
                          (cx - pad, cy - th - pad * 2),
                          (cx + tw + pad, cy + pad),
                          label_bg, -1)
            cv2.putText(out, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_BGR, 2, cv2.LINE_AA)
            n_drawn += 1

    # blend mask fill with outlines - write to a separate buffer (not in-place)
    blended = np.empty_like(out)
    cv2.addWeighted(overlay, opacity, out, 1.0 - opacity, 0, blended)
    return blended, n_drawn


def get_overlay_polygons(result, conf_threshold, scale_xy):
    """
    Extract polygon data for tkinter canvas drawing.
    scale_xy: (scale_x, scale_y) from prepare_for_inference().
    """
    boxes = result.boxes
    masks = result.masks
    out   = []

    if boxes is None or masks is None or len(boxes) == 0:
        return out

    sx, sy = scale_xy

    for box, mask_xy in zip(boxes, masks.xy):
        conf = float(box.conf)
        if conf < conf_threshold:
            continue

        pts = mask_xy.astype(np.int32)
        if len(pts) < 3:
            continue

        # scale from 640x640 inference space to frame/overlay space
        pts  = (pts * np.array([[sx, sy]])).astype(np.int32)
        flat = pts.flatten().tolist()
        cx   = int(pts[:, 0].mean())
        cy   = int(pts[:, 1].mean())
        out.append((flat, conf, cx, cy))

    return out


def run_monitor2(model, capture_rect, crop_offset, imgsz,
                 conf, target_fps, ring_type, debug,
                 smoothing=True, device="cpu", use_half=False, opacity=0.30):
    """
    Second-monitor display mode using an OpenCV window.
    The window is always shown at a sensible landscape size regardless of
    whether the second monitor is portrait or landscape.
    Press Q to quit.
    """

    left, top, cap_w, cap_h = capture_rect
    win_name  = f"Core Mining AI  [{ring_type}]"
    disp_w, disp_h = 1080, 608

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, disp_w, disp_h)

    m2 = get_monitor2_rect()
    if m2:
        mx, my, mw, mh = m2
        # place window in top-left of second monitor at a sensible size
        # do NOT try to fullscreen on portrait monitors - it looks terrible
        cv2.moveWindow(win_name, mx, max(my, 0))
        print(f"Window placed on monitor 2 at ({mx},{max(my,0)}), size {disp_w}x{disp_h}")
        print(f"Monitor 2 resolution: {mw}x{mh} "
              f"({'portrait' if mh > mw else 'landscape'})")
        print(f"You can resize/move the window freely.")
    else:
        cv2.moveWindow(win_name, 1920, 0)
        print("Could not detect monitor 2 - placed at x=1920. "
              "Drag the window to your second monitor if needed.")
        print("Install screeninfo for auto-detection: pip install screeninfo")

    debug_dir = None
    if debug:
        debug_dir = Path("debug_frames")
        debug_dir.mkdir(exist_ok=True)
        print(f"Debug: saving frames to {debug_dir}/")

    tracker = DetectionTracker(conf_threshold=conf) if smoothing else None

    # when smoothing, use the tracker's lower internal threshold for inference
    # so the model returns more candidates for spatial tracking.
    # the tracker's hit counting handles the display decision.
    inf_conf = tracker.internal_conf if tracker is not None else conf

    # try dxcam for fast capture, fall back to mss
    dxcam_camera = create_dxcam_camera(capture_rect)

    frame_count  = 0
    total_inf_ms = 0.0
    frame_time   = 1.0 / target_fps

    print(f"\nRunning. Press Q in the display window to quit.")
    print(f"Conf={conf}  Device={device}  Capture {cap_w}x{cap_h}")
    if tracker is not None:
        print(f"Smoothing: ON  (internal_conf={inf_conf:.2f}, "
              f"min_hits={tracker.min_hits}, persist={tracker.persist_frames} frames)")
    else:
        print(f"Smoothing: OFF")

    # open mss as fallback (used if dxcam is not available)
    import mss as _mss
    sct = _mss.mss() if dxcam_camera is None else None

    try:
        while True:
            t0 = time.perf_counter()

            if dxcam_camera is not None:
                frame = dxcam_camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
            else:
                frame = capture_frame(sct, left, top, cap_w, cap_h)

            t_cap             = time.perf_counter()
            inf_input, sx, sy = prepare_for_inference(frame, imgsz=imgsz)
            result            = model.predict(inf_input, imgsz=imgsz, conf=inf_conf,
                                              max_det=MAX_DET, device=device,
                                              half=use_half, verbose=False)[0]
            t_inf             = time.perf_counter()

            cap_ms = (t_cap - t0) * 1000
            inf_ms = (t_inf - t_cap) * 1000
            total_ms = (t_inf - t0) * 1000
            total_inf_ms += total_ms
            frame_count  += 1

            n_raw = len(result.boxes) if result.boxes is not None else 0

            is_debug_frame = debug and frame_count % 10 == 0

            if tracker is not None:
                detections = extract_detections(result, inf_conf, scale_xy=(sx, sy))
                visible, sm_conf, to_draw = tracker.update(detections,
                                                           debug=is_debug_frame)
                annotated, n_drawn = draw_from_tracked(frame, to_draw, sm_conf,
                                                       opacity=opacity,
                                                       debug=is_debug_frame)
            else:
                # no smoothing - original behavior
                annotated, n_drawn = draw_detections(frame, result, conf,
                                                     scale_xy=(sx, sy),
                                                     opacity=opacity,
                                                     debug=is_debug_frame)

            if is_debug_frame:
                extra = ""
                if tracker is not None:
                    extra = (f", hits={tracker.hit_count}, "
                             f"age={tracker.frames_since_hit}, "
                             f"active={'Y' if tracker.active else 'N'}")
                print(f"Frame {frame_count}: raw={n_raw}, drawn={n_drawn}, "
                      f"cap={cap_ms:.0f}ms, inf={inf_ms:.0f}ms, "
                      f"avg={total_inf_ms/frame_count:.0f}ms"
                      f"{extra}")
                if n_raw == MAX_DET:
                    print(f"  WARNING: hit max_det cap ({MAX_DET}). "
                          f"Raise MAX_DET in code if this happens every frame.")

            if debug and debug_dir and frame_count % 30 == 0 and n_drawn > 0:
                p = debug_dir / f"frame_{frame_count:05d}_{n_drawn}cores.jpg"
                cv2.imwrite(str(p), annotated)
                print(f"  Saved: {p}")

            cv2.setWindowTitle(win_name,
                f"Core Mining AI  [{ring_type}]  |  "
                f"{n_drawn} core(s)  |  {total_ms:.0f}ms")
            cv2.imshow(win_name, annotated)

            elapsed = time.perf_counter() - t0
            wait_ms = max(1, int((frame_time - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    finally:
        if dxcam_camera is not None:
            try:
                dxcam_camera.stop()
            except Exception:
                pass
        if sct is not None:
            sct.close()

    cv2.destroyAllWindows()
    if frame_count > 0:
        print(f"\nStopped. Average: {total_inf_ms/frame_count:.0f}ms/frame")


def run_overlay(model, window_rect, capture_rect, crop_offset, imgsz,
                conf, target_fps, ring_type, debug,
                smoothing=True, device="cpu", use_half=False, opacity=0.30):
    """
    Transparent overlay mode using tkinter with a 3-thread pipeline:
      Thread 1 (capture):   grabs screen frames as fast as possible
      Thread 2 (inference): runs YOLO on the latest frame, stores results
      Main thread (display): redraws tkinter canvas at 20fps from latest results

    The overlay covers the full game window. If --crop-area is set, only the
    center rectangle is captured and analyzed. Detections are offset from crop
    space to window space for correct overlay positioning. A rectangle is drawn
    on the overlay showing the active capture area.

    Elite Dangerous MUST be in BORDERLESS WINDOWED mode (game graphics settings).
    Ctrl+C in the terminal to quit.
    """
    import tkinter as tk

    # overlay covers the full game window
    wl, wt, ww, wh = window_rect
    # capture area may be cropped (centered)
    cl, ct, cw, ch = capture_rect
    ox, oy = crop_offset
    has_crop = (ox != 0 or oy != 0)

    root = tk.Tk()
    root.title("Core Mining AI Overlay")
    root.geometry(f"{ww}x{wh}+{wl}+{wt}")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-transparentcolor", TRANSPARENT_HEX)
    root.configure(bg=TRANSPARENT_HEX)

    canvas = tk.Canvas(root, width=ww, height=wh,
                       bg=TRANSPARENT_HEX, highlightthickness=0)
    canvas.pack()

    # shared state between threads (GIL makes reference assignment atomic)
    latest_frame = [None]       # set by capture thread, read by inference thread
    latest_scale = [(1.0, 1.0)] # (sx, sy) from prepare_for_inference
    latest_polys = [None]       # set by inference thread, read by display
    running = [True]
    cap_count = [0]
    inf_count = [0]

    tracker = DetectionTracker(conf_threshold=conf) if smoothing else None
    inf_conf = tracker.internal_conf if tracker is not None else conf

    debug_dir = None
    if debug:
        debug_dir = Path("debug_frames")
        debug_dir.mkdir(exist_ok=True)

    # try dxcam for fast capture, fall back to mss
    dxcam_camera = create_dxcam_camera(capture_rect)

    def capture_loop():
        """Grabs screen frames from the capture area and resizes for inference."""
        if dxcam_camera is not None:
            # dxcam path: read from ring buffer (very fast, ~2-5ms)
            while running[0]:
                frame = dxcam_camera.get_latest_frame()
                if frame is not None:
                    resized, sx, sy = prepare_for_inference(frame, imgsz=imgsz)
                    latest_frame[0] = resized
                    latest_scale[0] = (sx, sy)
                    cap_count[0] += 1
                else:
                    time.sleep(0.001)
        else:
            # mss fallback: grab screen directly (~30-40ms)
            import mss
            with mss.mss() as sct:
                while running[0]:
                    frame = capture_frame(sct, cl, ct, cw, ch)
                    resized, sx, sy = prepare_for_inference(frame, imgsz=imgsz)
                    latest_frame[0] = resized
                    latest_scale[0] = (sx, sy)
                    cap_count[0] += 1

    def inference_loop():
        """Runs YOLO inference on whatever frame is latest."""
        last_processed_cap = -1

        while running[0]:
            frame = latest_frame[0]
            current_cap = cap_count[0]

            if frame is None or current_cap == last_processed_cap:
                time.sleep(0.001)
                continue

            last_processed_cap = current_cap
            sx, sy = latest_scale[0]

            t0 = time.perf_counter()
            result = model.predict(frame, imgsz=imgsz, conf=inf_conf, max_det=MAX_DET,
                                   device=device, half=use_half,
                                   verbose=False)[0]
            inf_ms = (time.perf_counter() - t0) * 1000

            n_raw = len(result.boxes) if result.boxes is not None else 0
            inf_count[0] += 1

            is_debug_frame = debug and inf_count[0] % 10 == 0

            if tracker is not None:
                detections = extract_detections(result, inf_conf, scale_xy=(sx, sy))
                visible, best_conf, to_draw = tracker.update(detections,
                                                              debug=is_debug_frame)
                polys = []
                if visible:
                    for pts, raw_conf in to_draw:
                        # offset from crop space to window space for overlay
                        offset_pts = pts + np.array([[ox, oy]])
                        flat = offset_pts.flatten().tolist()
                        cx = int(offset_pts[:, 0].mean())
                        cy = int(offset_pts[:, 1].mean())
                        polys.append((flat, best_conf, cx, cy))
            else:
                raw_polys = get_overlay_polygons(result, conf, scale_xy=(sx, sy))
                polys = []
                for flat, conf_val, cx, cy in raw_polys:
                    # offset flat coords (x0,y0,x1,y1,...) by crop offset
                    coords = list(flat)
                    for i in range(0, len(coords), 2):
                        coords[i] += ox
                        coords[i + 1] += oy
                    polys.append((coords, conf_val, cx + ox, cy + oy))

            latest_polys[0] = polys

            if is_debug_frame:
                extra = ""
                if tracker is not None:
                    extra = (f", hits={tracker.hit_count}, "
                             f"age={tracker.frames_since_hit}, "
                             f"active={'Y' if tracker.active else 'N'}")
                print(f"Inf {inf_count[0]}: raw={n_raw}, drawn={len(polys)}, "
                      f"inf={inf_ms:.0f}ms, caps={cap_count[0]}{extra}")

            if debug and debug_dir and inf_count[0] % 30 == 0 and len(polys) > 0:
                # for debug frame saving, we need a full-size frame.
                # grab one from dxcam or skip if not available.
                if dxcam_camera is not None:
                    full_frame = dxcam_camera.get_latest_frame()
                    if full_frame is not None:
                        annotated, _ = draw_detections(full_frame, result, conf,
                                                       scale_xy=(sx, sy))
                        p = debug_dir / f"frame_{inf_count[0]:05d}_{len(polys)}cores.jpg"
                        cv2.imwrite(str(p), annotated)
                        print(f"  Saved: {p}")

    # canvas updates at fixed 20fps. this is the display refresh rate.
    # inference results appear on the next canvas update after they're ready.
    CANVAS_INTERVAL_MS = 50  # 20fps

    # tkinter doesn't support real alpha. map opacity to a stipple pattern.
    if opacity < 0.15:
        stipple = ""
    elif opacity < 0.35:
        stipple = "gray12"
    elif opacity < 0.60:
        stipple = "gray25"
    elif opacity < 0.85:
        stipple = "gray50"
    else:
        stipple = "gray75"

    def update_canvas():
        canvas.delete("all")

        # draw crop area rectangle if cropping is active
        if has_crop:
            canvas.create_rectangle(ox, oy, ox + cw, oy + ch,
                                    outline="#FFAA00", width=2, dash=(6, 4))

        polys = latest_polys[0]

        if polys:
            for flat, conf_val, cx, cy in polys:
                color_hex = conf_to_color_hex(conf_val)
                canvas.create_polygon(flat, outline=color_hex,
                                      fill=color_hex,
                                      stipple=stipple, width=3)
                label = f"core {conf_val:.2f}"
                canvas.create_text(cx + 1, cy + 1, text=label,
                                   fill="#002200", font=("Consolas", 12, "bold"))
                canvas.create_text(cx, cy, text=label,
                                   fill=color_hex, font=("Consolas", 12, "bold"))

        if running[0]:
            root.after(CANVAS_INTERVAL_MS, update_canvas)

    def on_close():
        running[0] = False
        if dxcam_camera is not None:
            try:
                dxcam_camera.stop()
            except Exception:
                pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # start capture thread first, then inference thread
    t_cap = threading.Thread(target=capture_loop, daemon=True)
    t_cap.start()

    t_inf = threading.Thread(target=inference_loop, daemon=True)
    t_inf.start()

    print(f"\nOverlay: {ww}x{wh} window, capturing {cw}x{ch}")
    print(f"Img size: {imgsz}x{imgsz}  Device: {device}")
    if tracker is not None:
        print(f"Smoothing: ON  (internal_conf={inf_conf:.2f}, "
              f"min_hits={tracker.min_hits}, persist={tracker.persist_frames} frames)")
    else:
        print(f"Smoothing: OFF")
    print(f"Elite Dangerous must be in BORDERLESS WINDOWED mode.")
    print(f"Ctrl+C to quit.")

    root.after(200, update_canvas)
    root.mainloop()
    running[0] = False


def main():
    parser = argparse.ArgumentParser(
        description="Elite Dangerous Core Mining AI Companion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ring-type", required=True, choices=RING_TYPES,
    )
    parser.add_argument(
        "--models-dir", default="exports",
        help="Folder containing exported model files (default: exports)",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Explicit path to a .pt model file. Overrides automatic lookup from "
             "--models-dir. Example: exports/metal_rich_yolo11_best.pt",
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Confidence threshold (default: 0.35). Start here and tune up/down.",
    )
    parser.add_argument(
        "--img-size", type=int, default=640,
        help="Inference resolution (default: 640). Lower = faster but less detail. "
             "Try 384 or 448 for ~2-3x speedup. Combine with --crop-area to "
             "preserve detail in the center of the screen.",
    )
    parser.add_argument(
        "--crop-area", default=None,
        help="Capture only a centered rectangle instead of the full game window. "
             "Format: WxH, e.g. --crop-area 1440x810. Reduces wasted pixels "
             "at edges and preserves more detail when combined with --img-size. "
             "A rectangle is drawn on the overlay/monitor2 showing the active area.",
    )
    parser.add_argument(
        "--y-offset", type=int, default=0,
        help="Shift the crop area center up (negative) or down (positive) by this "
             "many pixels. Use with --crop-area to avoid the cockpit at the bottom. "
             "Example: --y-offset -150 moves the crop 150px up.",
    )
    parser.add_argument(
        "--opacity", type=float, default=0.30,
        help="Opacity of the detection polygon fill (0.0 = fully transparent, "
             "1.0 = fully opaque). Default: 0.30. The outline is always fully visible.",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Target FPS (default: 30). GPU can sustain 30+, CPU tops out at 4-6.",
    )
    parser.add_argument(
        "--display", choices=["monitor2", "overlay"], default="monitor2",
    )
    parser.add_argument(
        "--capture", choices=["game", "primary"], default="game",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="cpu",
        help="Inference device (default: cpu). CPU is recommended because GPU "
             "inference stalls when the game is rendering. Use 'cuda' to try GPU.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print detection details every 10 frames and save annotated frames.",
    )
    parser.add_argument(
        "--no-smoothing", action="store_true",
        help="Disable detection smoothing. Shows raw detections "
             "like the old behavior (bounding boxes blink on/off each frame).",
    )
    parser.add_argument(
        "--runtime", choices=["pytorch", "openvino", "onnxrt"], default="pytorch",
        help="Inference runtime (default: pytorch). 'openvino' uses Intel's "
             "optimized engine (~2-3x faster on Intel CPUs, requires: pip install "
             "openvino). 'onnxrt' uses ONNX Runtime (requires: pip install "
             "onnxruntime). Both auto-export from the .pt file if needed.",
    )
    args = parser.parse_args()

    # --- find model ---
    if args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            sys.exit(1)
    else:
        model_path = find_model(args.ring_type, args.models_dir)
        if model_path is None:
            print(f"No model found for ring type '{args.ring_type}' in '{args.models_dir}'")
            print("Expected: exports/<ring_type>_best.pt")
            print("Run the export cell in the notebook, or check your models-dir path.")
            print("Or use --model-path to specify the exact .pt file.")
            sys.exit(1)

    if model_path.suffix == ".onnx":
        print(f"WARNING: loading ONNX model {model_path}")
        print("ONNX segmentation models do not apply NMS correctly via ultralytics,")
        print("causing 300 garbage detections on every frame.")
        print("Prefer .pt weights. Place <ring_type>_best.pt in the exports/ folder.")

    print(f"Loading: {model_path}")
    model = YOLO(str(model_path))

    # --- select runtime and export if needed ---
    if args.runtime == "openvino":
        try:
            import openvino
        except ImportError:
            print("OpenVINO not installed. Install with: pip install openvino")
            sys.exit(1)

        # ultralytics exports to a directory named <stem>_openvino_model/
        ov_dir = model_path.parent / (model_path.stem + "_openvino_model")
        if ov_dir.exists():
            print(f"Found existing OpenVINO model: {ov_dir}")
        else:
            print(f"Exporting to OpenVINO format (one-time)...")
            model.export(format="openvino", half=False)
            # export puts the directory next to the .pt file
            if not ov_dir.exists():
                # ultralytics might put it in the current directory instead
                alt = Path(model_path.stem + "_openvino_model")
                if alt.exists():
                    ov_dir = alt
                else:
                    print(f"Export failed: expected {ov_dir} not found")
                    sys.exit(1)
            print(f"Exported: {ov_dir}")

        model = YOLO(str(ov_dir))
        # OpenVINO manages its own device/threading, ignore --device
        device = "cpu"
        use_half = False
        print(f"Runtime: OpenVINO (Intel optimized CPU inference)")

    elif args.runtime == "onnxrt":
        try:
            import onnxruntime
        except ImportError:
            print("ONNX Runtime not installed. Install with: pip install onnxruntime")
            sys.exit(1)

        # look for existing .onnx file
        onnx_path = model_path.with_suffix(".onnx")
        if not onnx_path.exists():
            # try same stem in same directory
            onnx_path = model_path.parent / (model_path.stem + ".onnx")
        if not onnx_path.exists():
            print(f"Exporting to ONNX format (one-time)...")
            model.export(format="onnx", half=False)
            onnx_path = model_path.with_suffix(".onnx")
            if not onnx_path.exists():
                print(f"Export failed: expected {onnx_path} not found")
                sys.exit(1)
            print(f"Exported: {onnx_path}")
        else:
            print(f"Found existing ONNX model: {onnx_path}")

        model = YOLO(str(onnx_path))
        device = "cpu"
        use_half = False
        print(f"Runtime: ONNX Runtime")
        print(f"WARNING: ONNX segmentation via ultralytics may skip NMS.")
        print(f"If you get 300 garbage detections, this runtime won't work")
        print(f"for segmentation. Use --runtime openvino instead.")

    else:
        # pytorch (default)
        # --- select inference device ---
        import torch
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device

        use_half = False
        if device == "cuda":
            if not torch.cuda.is_available():
                print("WARNING: CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
            else:
                gpu_name = torch.cuda.get_device_name(0)
                vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                print(f"GPU: {gpu_name} ({vram_mb}MB VRAM)")
                use_half = True
                print(f"Device: {device} (FP16={'ON' if use_half else 'OFF'})")
        else:
            print(f"Runtime: PyTorch, Device: {device}")

    imgsz = args.img_size

    # warmup - first few inferences are slow due to JIT compilation.
    # run several iterations at the target imgsz so the main loop doesn't stutter.
    print("Warmup inference...", end=" ", flush=True)
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    n_warmup = 5 if device == "cuda" else 3
    for i in range(n_warmup):
        warmup = model.predict(dummy, imgsz=imgsz, conf=args.conf, max_det=MAX_DET,
                               device=device, half=use_half, verbose=False)[0]
    n_warm = len(warmup.boxes) if warmup.boxes is not None else 0
    print(f"done ({n_warmup} iterations). "
          f"{n_warm} detections on blank frame (should be 0)")
    if n_warm > 10:
        print("WARNING: too many detections on a blank frame.")
        print("If using .onnx, switch to .pt weights. If using .pt, retrain the model.")

    # --- capture region ---
    window_rect = None
    if args.capture == "game":
        window_rect = find_game_window()
        if window_rect is None:
            print("Game window not found - falling back to primary monitor.")

    if window_rect is None:
        window_rect = get_primary_monitor_rect()
        print(f"Capturing primary monitor: {window_rect[2]}x{window_rect[3]}")

    # --- crop area ---
    crop_offset = (0, 0)

    if args.crop_area is not None:
        try:
            crop_w, crop_h = [int(x) for x in args.crop_area.lower().split("x")]
        except ValueError:
            print(f"Invalid --crop-area format: '{args.crop_area}'. Use WxH, e.g. 1440x810")
            sys.exit(1)

        wl, wt, ww, wh = window_rect
        if crop_w > ww or crop_h > wh:
            print(f"Crop area {crop_w}x{crop_h} is larger than window {ww}x{wh}")
            sys.exit(1)

        # center the crop rectangle in the window, then apply y-offset
        cl = wl + (ww - crop_w) // 2
        ct = wt + (wh - crop_h) // 2 + args.y_offset

        # clamp so the crop doesn't go outside the window
        ct = max(wt, min(ct, wt + wh - crop_h))

        capture_rect = (cl, ct, crop_w, crop_h)
        # offset from window top-left to crop top-left (for overlay polygon positioning)
        crop_offset = (cl - wl, ct - wt)
        print(f"Crop area: {crop_w}x{crop_h} at offset ({crop_offset[0]},{crop_offset[1]})"
              + (f" (y-offset={args.y_offset})" if args.y_offset != 0 else ""))
    else:
        capture_rect = window_rect

    # --- test inference ---
    print("\nRunning test capture...")
    import mss as _mss
    with _mss.mss() as sct:
        l, t, fw, fh = capture_rect
        test_frame = capture_frame(sct, l, t, fw, fh)

    test_input, tsx, tsy = prepare_for_inference(test_frame, imgsz=imgsz)
    test_result = model.predict(test_input, imgsz=imgsz, conf=args.conf, max_det=MAX_DET,
                                device=device, half=use_half, verbose=False)[0]
    n_test = len(test_result.boxes) if test_result.boxes else 0
    print(f"Test frame: {fw}x{fh} -> {imgsz}x{imgsz} stretch, "
          f"{n_test} detection(s) at conf>={args.conf}")

    if test_result.masks is not None and n_test > 0:
        pts = (test_result.masks.xy[0] * np.array([[tsx, tsy]])).astype(np.int32)
        print(f"First detection: {len(pts)} polygon points, "
              f"x=[{pts[:,0].min()}-{pts[:,0].max()}], "
              f"y=[{pts[:,1].min()}-{pts[:,1].max()}]")
        print("Coordinates scaled back to frame space.")

    smoothing = not args.no_smoothing

    print(f"\nRing type:  {args.ring_type}")
    print(f"Model:      {model_path}")
    print(f"Runtime:    {args.runtime}" + (f", device={device}" if args.runtime == "pytorch" else ""))
    print(f"Img size:   {imgsz}x{imgsz}")
    if args.crop_area:
        yo = f", y-offset={args.y_offset}" if args.y_offset != 0 else ""
        print(f"Crop area:  {args.crop_area}{yo}")
    print(f"Confidence: {args.conf}")
    print(f"Opacity:    {args.opacity}")
    print(f"Display:    {args.display}")
    print(f"Smoothing:  {'ON (persist + min_hits)' if smoothing else 'OFF (raw detections)'}")
    if smoothing:
        int_conf = max(0.10, args.conf * 0.4)
        print(f"            internal_conf={int_conf:.2f}, "
              f"min_hits=2, persist=8 frames")
    print()

    if args.display == "overlay":
        if find_game_window() is None:
            print("Overlay needs the game window position. Falling back to monitor2.")
            args.display = "monitor2"

    if args.display == "overlay":
        run_overlay(model, window_rect, capture_rect, crop_offset, imgsz,
                    args.conf, args.fps, args.ring_type, args.debug,
                    smoothing=smoothing, device=device, use_half=use_half,
                    opacity=args.opacity)
    else:
        run_monitor2(model, capture_rect, crop_offset, imgsz,
                     args.conf, args.fps, args.ring_type, args.debug,
                     smoothing=smoothing, device=device, use_half=use_half,
                     opacity=args.opacity)


if __name__ == "__main__":
    main()
