"""
Elite Dangerous Core Mining AI Companion

Captures the Elite Dangerous game window, runs segmentation inference on a
locally hosted model (CPU only, GPU stays free for the game), and displays
detected core asteroid polygon masks with confidence scores.

Two display modes:
  monitor2  - OpenCV window on your second monitor (always works, use this first)
  overlay   - transparent window drawn over the game (requires borderless mode)

Usage:
    python companion_app.py --ring-type ice
    python companion_app.py --ring-type ice --conf 0.35
    python companion_app.py --ring-type ice --display overlay --conf 0.35
    python companion_app.py --ring-type ice --debug

Controls:
    monitor2 mode: press Q in the display window to quit
    overlay mode:  Ctrl+C in terminal to quit

NOTE on model format:
    The app loads .pt weights directly (not .onnx). ONNX exports of segmentation
    models do not apply NMS properly when loaded via ultralytics, causing every
    frame to return exactly max_det=300 garbage detections. The .pt file works
    correctly and runs fine on CPU since ultralytics is already installed.
    Inference speed on i5-12600KF: ~150ms per frame for yolov8s-seg (about 6fps).
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
            wins = [w for w in gw.getAllWindows() if "elite" in w.title.lower()]
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
    """Capture a screen region and return a BGR numpy array."""
    raw = sct.grab({"left": left, "top": top, "width": width, "height": height})
    return np.array(raw)[:, :, :3]


def draw_detections(frame, result, conf_threshold, debug=False):
    """
    Draw segmentation polygon masks and confidence labels on a copy of frame.
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

    if masks is not None:
        for i in keep:
            pts  = masks.xy[i].astype(np.int32)
            conf = float(boxes[i].conf)

            if len(pts) < 3:
                continue

            if debug:
                print(f"  Detection {i}: conf={conf:.3f}, pts={len(pts)}, "
                      f"x=[{pts[:,0].min()}-{pts[:,0].max()}], "
                      f"y=[{pts[:,1].min()}-{pts[:,1].max()}], "
                      f"frame={W}x{H}")

            # safety check: if coords are in inference space (640px) instead of
            # frame space, scale them up. this should not happen with .pt models
            # but is a known issue with some ONNX exports.
            if pts[:,0].max() < 700 and W > 700:
                sx = W / 640.0
                sy = H / 640.0
                pts = (pts * np.array([[sx, sy]])).astype(np.int32)
                if debug:
                    print(f"    Scaled coords by ({sx:.2f},{sy:.2f})")

            cv2.fillPoly(overlay, [pts], POLY_COLOR_BGR)
            cv2.polylines(out, [pts], isClosed=True, color=POLY_COLOR_BGR, thickness=3)

            cx    = int(pts[:, 0].mean())
            cy    = int(pts[:, 1].mean())
            label = f"core {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            pad = 4
            cv2.rectangle(out,
                          (cx - pad, cy - th - pad * 2),
                          (cx + tw + pad, cy + pad),
                          LABEL_BG_BGR, -1)
            cv2.putText(out, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_BGR, 2, cv2.LINE_AA)
            n_drawn += 1

    # blend mask fill with outlines - write to a separate buffer (not in-place)
    blended = np.empty_like(out)
    cv2.addWeighted(overlay, MASK_ALPHA, out, 1.0 - MASK_ALPHA, 0, blended)
    return blended, n_drawn


def get_overlay_polygons(result, frame_shape, conf_threshold):
    """Extract polygon data for tkinter canvas drawing."""
    boxes = result.boxes
    masks = result.masks
    out   = []

    if boxes is None or masks is None or len(boxes) == 0:
        return out

    H, W = frame_shape[:2]

    for box, mask_xy in zip(boxes, masks.xy):
        conf = float(box.conf)
        if conf < conf_threshold:
            continue

        pts = mask_xy.astype(np.int32)
        if len(pts) < 3:
            continue

        if pts[:,0].max() < 700 and W > 700:
            pts = (pts * np.array([[W / 640.0, H / 640.0]])).astype(np.int32)

        flat = pts.flatten().tolist()
        cx   = int(pts[:, 0].mean())
        cy   = int(pts[:, 1].mean())
        out.append((flat, conf, cx, cy))

    return out


def run_monitor2(model, capture_rect, conf, target_fps, ring_type, debug):
    """
    Second-monitor display mode using an OpenCV window.
    The window is always shown at a sensible landscape size regardless of
    whether the second monitor is portrait or landscape.
    Press Q to quit.
    """
    import mss

    left, top, cap_w, cap_h = capture_rect
    win_name  = f"Core Mining AI  [{ring_type}]"
    # display at 960x540 by default - easy to resize manually
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

    frame_count  = 0
    total_inf_ms = 0.0
    frame_time   = 1.0 / target_fps

    print(f"\nRunning. Press Q in the display window to quit.")
    print(f"Conf={conf}  FPS target={target_fps}  Capture {cap_w}x{cap_h}")

    with mss.mss() as sct:
        while True:
            t0 = time.perf_counter()

            frame  = capture_frame(sct, left, top, cap_w, cap_h)
            result = model.predict(frame, conf=conf, max_det=MAX_DET,
                                   device="cpu", verbose=False)[0]
            inf_ms = (time.perf_counter() - t0) * 1000

            total_inf_ms += inf_ms
            frame_count  += 1

            n_raw = len(result.boxes) if result.boxes is not None else 0

            is_debug_frame = debug and frame_count % 10 == 0
            annotated, n_drawn = draw_detections(frame, result, conf,
                                                 debug=is_debug_frame)

            if is_debug_frame:
                print(f"Frame {frame_count}: raw={n_raw}, drawn={n_drawn}, "
                      f"inf={inf_ms:.0f}ms, avg={total_inf_ms/frame_count:.0f}ms")
                if n_raw == MAX_DET:
                    print(f"  WARNING: hit max_det cap ({MAX_DET}). "
                          f"Raise MAX_DET in code if this happens every frame.")

            if debug and debug_dir and frame_count % 30 == 0 and n_drawn > 0:
                p = debug_dir / f"frame_{frame_count:05d}_{n_drawn}cores.jpg"
                cv2.imwrite(str(p), annotated)
                print(f"  Saved: {p}")

            cv2.setWindowTitle(win_name,
                f"Core Mining AI  [{ring_type}]  |  "
                f"{n_drawn} core(s)  |  {inf_ms:.0f}ms")
            cv2.imshow(win_name, annotated)

            elapsed = time.perf_counter() - t0
            wait_ms = max(1, int((frame_time - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    if frame_count > 0:
        print(f"\nStopped. Average inference: {total_inf_ms/frame_count:.0f}ms/frame")


def run_overlay(model, capture_rect, conf, target_fps, ring_type, debug):
    """
    Transparent overlay mode using tkinter.
    Background color TRANSPARENT_HEX is made invisible by Windows compositor.
    Only polygon outlines and labels float over the game.

    Elite Dangerous MUST be in BORDERLESS WINDOWED mode (game graphics settings).
    Ctrl+C in the terminal to quit.
    """
    import tkinter as tk

    left, top, w, h = capture_rect

    root = tk.Tk()
    root.title("Core Mining AI Overlay")
    root.geometry(f"{w}x{h}+{left}+{top}")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-transparentcolor", TRANSPARENT_HEX)
    root.configure(bg=TRANSPARENT_HEX)

    canvas = tk.Canvas(root, width=w, height=h,
                       bg=TRANSPARENT_HEX, highlightthickness=0)
    canvas.pack()

    latest_polys = [None]
    latest_n_raw = [0]
    running      = [True]
    frame_count  = [0]

    debug_dir = None
    if debug:
        debug_dir = Path("debug_frames")
        debug_dir.mkdir(exist_ok=True)

    def inference_loop():
        import mss
        frame_time = 1.0 / target_fps
        with mss.mss() as sct:
            while running[0]:
                t0     = time.perf_counter()
                frame  = capture_frame(sct, left, top, w, h)
                result = model.predict(frame, conf=conf, max_det=MAX_DET,
                                       device="cpu", verbose=False)[0]

                n_raw  = len(result.boxes) if result.boxes is not None else 0
                polys  = get_overlay_polygons(result, frame.shape, conf)
                latest_polys[0]  = polys
                latest_n_raw[0]  = n_raw
                frame_count[0]  += 1
                inf_ms = (time.perf_counter() - t0) * 1000

                if debug and frame_count[0] % 10 == 0:
                    print(f"Frame {frame_count[0]}: raw={n_raw}, drawn={len(polys)}, "
                          f"inf={inf_ms:.0f}ms")

                if debug and debug_dir and frame_count[0] % 30 == 0 and len(polys) > 0:
                    annotated, _ = draw_detections(frame, result, conf)
                    p = debug_dir / f"frame_{frame_count[0]:05d}_{len(polys)}cores.jpg"
                    cv2.imwrite(str(p), annotated)
                    print(f"  Saved: {p}")

                time.sleep(max(0.001, frame_time - inf_ms / 1000))

    def update_canvas():
        canvas.delete("all")
        polys = latest_polys[0]

        if polys:
            for flat, conf_val, cx, cy in polys:
                canvas.create_polygon(flat, outline=POLY_COLOR_HEX,
                                      fill="", width=3)
                label = f"core {conf_val:.2f}"
                canvas.create_text(cx + 1, cy + 1, text=label,
                                   fill="#002200", font=("Consolas", 12, "bold"))
                canvas.create_text(cx, cy, text=label,
                                   fill=POLY_COLOR_HEX, font=("Consolas", 12, "bold"))

        if running[0]:
            root.after(int(1000 / target_fps), update_canvas)

    def on_close():
        running[0] = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    print(f"\nOverlay active: {w}x{h} at ({left},{top})")
    print(f"Conf={conf}  FPS={target_fps}")
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
        "--conf", type=float, default=0.3,
        help="Confidence threshold (default: 0.3). Start here and tune up/down.",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Target inference FPS (default: 30). The more, the bigger train on CPU.",
    )
    parser.add_argument(
        "--display", choices=["monitor2", "overlay"], default="monitor2",
    )
    parser.add_argument(
        "--capture", choices=["game", "primary"], default="game",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print detection details every 10 frames and save annotated frames.",
    )
    args = parser.parse_args()

    # --- find model ---
    model_path = find_model(args.ring_type, args.models_dir)
    if model_path is None:
        print(f"No model found for ring type '{args.ring_type}' in '{args.models_dir}'")
        print("Expected: exports/<ring_type>_best.pt")
        print("Run the export cell in the notebook, or check your models-dir path.")
        sys.exit(1)

    if model_path.suffix == ".onnx":
        print(f"WARNING: loading ONNX model {model_path}")
        print("ONNX segmentation models do not apply NMS correctly via ultralytics,")
        print("causing 300 garbage detections on every frame.")
        print("Prefer .pt weights. Place <ring_type>_best.pt in the exports/ folder.")

    print(f"Loading: {model_path}")
    model = YOLO(str(model_path))

    # warmup
    dummy   = np.zeros((640, 640, 3), dtype=np.uint8)
    warmup  = model.predict(dummy, conf=args.conf, max_det=MAX_DET,
                             device="cpu", verbose=False)[0]
    n_warm  = len(warmup.boxes) if warmup.boxes is not None else 0
    print(f"Warmup: {n_warm} detections on blank frame (should be 0)")
    if n_warm > 10:
        print("WARNING: too many detections on a blank frame.")
        print("If using .onnx, switch to .pt weights. If using .pt, retrain the model.")

    # --- capture region ---
    capture_rect = None
    if args.capture == "game":
        capture_rect = find_game_window()
        if capture_rect is None:
            print("Game window not found - falling back to primary monitor.")

    if capture_rect is None:
        capture_rect = get_primary_monitor_rect()
        print(f"Capturing primary monitor: {capture_rect[2]}x{capture_rect[3]}")

    # --- test inference ---
    print("\nRunning test capture...")
    import mss as _mss
    with _mss.mss() as sct:
        l, t, fw, fh = capture_rect
        test_frame  = capture_frame(sct, l, t, fw, fh)

    test_result = model.predict(test_frame, conf=args.conf, max_det=MAX_DET,
                                device="cpu", verbose=False)[0]
    n_test = len(test_result.boxes) if test_result.boxes else 0
    print(f"Test frame: {fw}x{fh}, {n_test} detection(s) at conf>={args.conf}")

    if test_result.masks is not None and n_test > 0:
        pts = test_result.masks.xy[0].astype(np.int32)
        print(f"First detection: {len(pts)} polygon points, "
              f"x=[{pts[:,0].min()}-{pts[:,0].max()}], "
              f"y=[{pts[:,1].min()}-{pts[:,1].max()}]")
        if pts[:,0].max() < 700 and fw > 700:
            print("Coord scaling will be applied (640->frame space).")
        else:
            print("Coordinates look correct.")

    print(f"\nRing type:  {args.ring_type}")
    print(f"Model:      {model_path}")
    print(f"Confidence: {args.conf}")
    print(f"Target FPS: {args.fps}")
    print(f"Display:    {args.display}")
    print()

    if args.display == "overlay":
        if find_game_window() is None:
            print("Overlay needs the game window position. Falling back to monitor2.")
            args.display = "monitor2"

    if args.display == "overlay":
        run_overlay(model, capture_rect, args.conf, args.fps, args.ring_type, args.debug)
    else:
        run_monitor2(model, capture_rect, args.conf, args.fps, args.ring_type, args.debug)


if __name__ == "__main__":
    main()
