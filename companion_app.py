"""
Elite Dangerous Core Mining AI Companion

Captures the Elite Dangerous game window, runs segmentation inference on a
locally hosted ONNX model (CPU only, GPU stays free for the game), and
displays detected core asteroid polygon masks with confidence scores.

Two display modes:
  monitor2  - OpenCV window on your second monitor (always works)
  overlay   - transparent window drawn over the game (requires borderless mode)

Usage:
    python companion_app.py --ring-type ice
    python companion_app.py --ring-type ice --display overlay --conf 0.5
    python companion_app.py --ring-type metallic --fps 8 --models-dir exports

Controls:
    monitor2 mode: press Q in the display window to quit
    overlay mode:  press ESC to quit
"""

import argparse
import os
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np

# suppress ultralytics banner and per-inference prints
os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO


RING_TYPES        = ["ice", "rocky", "metallic", "metal_rich"]
GAME_WINDOW_TITLE = "EliteDangerous64"

# color used as the transparent key in overlay mode
# must be a color that never appears naturally in the game (near-black with B=1)
TRANSPARENT_HEX = "#000001"
TRANSPARENT_BGR = (1, 0, 0)

# drawing colors (BGR for OpenCV, hex for tkinter)
POLY_COLOR_BGR = (0, 255, 80)
POLY_COLOR_HEX = "#00FF50"
LABEL_BG_BGR   = (0, 160, 50)
TEXT_BGR        = (255, 255, 255)
MASK_ALPHA      = 0.30


def find_game_window():
    """
    Find the Elite Dangerous window and return (left, top, width, height).
    Returns None if pygetwindow is not installed or the window is not found.
    """
    try:
        import pygetwindow as gw
        wins = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)
        if not wins:
            # partial match fallback in case the title changed
            wins = [w for w in gw.getAllWindows() if "elite" in w.title.lower()]
        if wins:
            w = wins[0]
            return (w.left, w.top, w.width, w.height)
    except ImportError:
        print("pygetwindow not installed - run: pip install pygetwindow")
    except Exception as e:
        print(f"Window lookup failed: {e}")
    return None


def get_primary_monitor_rect():
    """Return (left, top, width, height) of the primary monitor via mss."""
    import mss
    with mss.mss() as sct:
        m = sct.monitors[1]  # index 0 is the all-monitors virtual screen
        return (m["left"], m["top"], m["width"], m["height"])


def capture_frame(sct, left, top, width, height):
    """
    Capture a screen region using mss and return a BGR numpy array.
    mss returns BGRA so we drop the alpha channel.
    """
    raw = sct.grab({"left": left, "top": top, "width": width, "height": height})
    return np.array(raw)[:, :, :3]


def draw_detections(frame, result, conf_threshold):
    """
    Draw segmentation masks and confidence labels onto frame.
    Returns a new annotated image - the input frame is not modified.
    """
    out     = frame.copy()
    overlay = frame.copy()

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return out

    for mask_xy, box in zip(result.masks.xy, result.boxes):
        conf = float(box.conf)
        if conf < conf_threshold:
            continue

        pts = mask_xy.astype(np.int32)
        if len(pts) < 3:
            continue

        # semi-transparent filled polygon drawn on the overlay layer
        cv2.fillPoly(overlay, [pts], POLY_COLOR_BGR)

        # solid outline on the output layer (drawn after blending so it stays sharp)
        cv2.polylines(out, [pts], isClosed=True, color=POLY_COLOR_BGR, thickness=2)

        # confidence label at polygon centroid
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        label = f"core {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        pad = 4
        cv2.rectangle(out,
                      (cx - pad, cy - th - pad * 2),
                      (cx + tw + pad, cy + pad),
                      LABEL_BG_BGR, -1)
        cv2.putText(out, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_BGR, 2, cv2.LINE_AA)

    # blend the filled overlay with the outline layer
    cv2.addWeighted(overlay, MASK_ALPHA, out, 1.0 - MASK_ALPHA, 0, out)
    return out


def run_monitor2(model, capture_rect, conf, target_fps, ring_type):
    """
    Second-monitor display mode.
    Opens an OpenCV window and tries to move it to the second monitor.
    Press Q to quit.
    """
    import mss

    left, top, w, h = capture_rect
    win_name = f"Core Mining AI  [{ring_type}]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # move to second monitor if available
    try:
        import screeninfo
        monitors = screeninfo.get_monitors()
        if len(monitors) >= 2:
            m2 = monitors[1]
            cv2.moveWindow(win_name, m2.x, m2.y)
            cv2.resizeWindow(win_name, m2.width, m2.height)
            print(f"Display window moved to monitor 2 ({m2.width}x{m2.height} at {m2.x},{m2.y})")
        else:
            print("Only one monitor detected - window will appear on primary")
    except ImportError:
        # guess: second monitor is 1920px to the right of primary
        cv2.moveWindow(win_name, 1920, 0)
        print("screeninfo not installed - guessing monitor 2 offset is x=1920.")
        print("Run: pip install screeninfo  for auto-detection")
        print("Or drag the window to your second monitor manually.")

    frame_time  = 1.0 / target_fps
    last_result = [None]

    print(f"Running. Press Q in the display window to quit.")
    print(f"Conf threshold: {conf}  |  Target FPS: {target_fps}")

    with mss.mss() as sct:
        while True:
            t0 = time.perf_counter()

            frame  = capture_frame(sct, left, top, w, h)
            result = model.predict(frame, conf=conf, device="cpu", verbose=False)[0]
            n      = len(result.boxes) if result.boxes is not None else 0

            annotated = draw_detections(frame, result, conf)
            cv2.setWindowTitle(win_name, f"Core Mining AI  [{ring_type}]  |  {n} core(s) detected")
            cv2.imshow(win_name, annotated)

            elapsed  = time.perf_counter() - t0
            wait_ms  = max(1, int((frame_time - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


def run_overlay(model, capture_rect, conf, target_fps, ring_type):
    """
    Transparent overlay mode.
    Draws a borderless, always-on-top tkinter window directly over the game.
    The background color TRANSPARENT_HEX is made invisible by Windows compositor,
    so only the polygon outlines and labels are visible.

    Requirements:
    - Elite Dangerous must be in BORDERLESS WINDOWED mode (not fullscreen).
      Set this in the game's graphics options.
    - Press ESC to quit.
    """
    import tkinter as tk

    left, top, w, h = capture_rect

    root = tk.Tk()
    root.title("Core Mining AI Overlay")
    root.geometry(f"{w}x{h}+{left}+{top}")
    root.overrideredirect(True)                                  # no title bar
    root.attributes("-topmost", True)                            # always on top
    root.attributes("-transparentcolor", TRANSPARENT_HEX)       # key color = invisible
    root.configure(bg=TRANSPARENT_HEX)
    root.attributes("-alpha", 1.0)

    canvas = tk.Canvas(root, width=w, height=h,
                       bg=TRANSPARENT_HEX, highlightthickness=0)
    canvas.pack()

    # shared state between inference thread and canvas update
    latest_result = [None]
    running       = [True]

    def inference_loop():
        import mss
        frame_time = 1.0 / target_fps
        with mss.mss() as sct:
            while running[0]:
                t0     = time.perf_counter()
                frame  = capture_frame(sct, left, top, w, h)
                result = model.predict(frame, conf=conf, device="cpu", verbose=False)[0]
                latest_result[0] = result
                elapsed = time.perf_counter() - t0
                time.sleep(max(0.001, frame_time - elapsed))

    def update_canvas():
        canvas.delete("all")
        result = latest_result[0]

        if result is not None and result.masks is not None and result.boxes is not None:
            for mask_xy, box in zip(result.masks.xy, result.boxes):
                conf_val = float(box.conf)
                if conf_val < conf:
                    continue
                pts = mask_xy.astype(np.int32)
                if len(pts) < 3:
                    continue

                # tkinter polygon needs flat x0,y0,x1,y1,... list
                flat = pts.flatten().tolist()
                canvas.create_polygon(flat, outline=POLY_COLOR_HEX,
                                      fill="", width=2)

                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())
                label = f"core {conf_val:.2f}"
                # draw a dark background rectangle behind the label
                canvas.create_text(cx + 1, cy + 1, text=label,
                                   fill="#004400", font=("Consolas", 11, "bold"))
                canvas.create_text(cx, cy, text=label,
                                   fill=POLY_COLOR_HEX, font=("Consolas", 11, "bold"))

        if running[0]:
            # schedule next canvas refresh slightly ahead of inference interval
            root.after(int(1000 / target_fps), update_canvas)

    def on_quit(event=None):
        running[0] = False
        root.destroy()

    root.bind("<Escape>", on_quit)

    inf_thread = threading.Thread(target=inference_loop, daemon=True)
    inf_thread.start()

    print(f"Overlay active over game window at {left},{top} ({w}x{h})")
    print(f"Press ESC to quit.")
    print(f"If the overlay is not visible, make sure Elite Dangerous is in BORDERLESS mode.")

    # start the canvas refresh loop
    root.after(100, update_canvas)
    root.mainloop()
    running[0] = False


def main():
    parser = argparse.ArgumentParser(
        description="Elite Dangerous Core Mining AI Companion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ring-type", required=True, choices=RING_TYPES,
        help="Ring type to detect. Loads <models-dir>/<ring_type>_best.onnx",
    )
    parser.add_argument(
        "--models-dir", default="exports",
        help="Folder with exported .onnx models (default: exports)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Minimum confidence score to show a detection (default: 0.4)",
    )
    parser.add_argument(
        "--fps", type=int, default=5,
        help="Target inference frames per second (default: 5). "
             "Higher uses more CPU. 3-8 is the useful range.",
    )
    parser.add_argument(
        "--display", choices=["monitor2", "overlay"], default="overlay",
        help="monitor2: show on second monitor (default). "
             "overlay: transparent window over game (needs borderless mode).",
    )
    parser.add_argument(
        "--capture", choices=["game", "primary"], default="game",
        help="game: capture Elite Dangerous window (default). "
             "primary: capture entire primary monitor.",
    )
    args = parser.parse_args()

    # --- find or confirm the model file ---
    model_path = Path(args.models_dir) / f"{args.ring_type}_best.onnx"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run export_best_models() in the notebook first, then try again.")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # warm-up pass so the first real inference is not slow
    _warmup = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(_warmup, device="cpu", verbose=False)
    print("Model ready.")

    # --- determine capture rectangle ---
    capture_rect = None

    if args.capture == "game":
        capture_rect = find_game_window()
        if capture_rect is None:
            print(f"Could not find '{GAME_WINDOW_TITLE}' window.")
            print("Make sure Elite Dangerous is running and try again.")
            print("Falling back to primary monitor capture.")

    if capture_rect is None:
        capture_rect = get_primary_monitor_rect()
        print(f"Capturing primary monitor: {capture_rect[2]}x{capture_rect[3]}")
    else:
        print(f"Capturing game window: "
              f"{capture_rect[2]}x{capture_rect[3]} at "
              f"({capture_rect[0]}, {capture_rect[1]})")

    # overlay mode requires a known game window position
    if args.display == "overlay" and args.capture != "game":
        print("Overlay mode requires --capture game so the window position is known.")
        sys.exit(1)

    if args.display == "overlay" and find_game_window() is None:
        print("Overlay mode requires the Elite Dangerous window to be found.")
        sys.exit(1)

    print(f"\nRing type:  {args.ring_type}")
    print(f"Confidence: {args.conf}")
    print(f"Target FPS: {args.fps}")
    print(f"Display:    {args.display}\n")

    if args.display == "overlay":
        run_overlay(model, capture_rect, args.conf, args.fps, args.ring_type)
    else:
        run_monitor2(model, capture_rect, args.conf, args.fps, args.ring_type)


if __name__ == "__main__":
    main()
