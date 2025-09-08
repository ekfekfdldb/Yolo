import os, glob, cv2
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from ultralytics import YOLO
import torch
from utils import find_latest_best


env_path = find_dotenv(usecwd=True)
load_dotenv(env_path, override=True)


DEVICE_ENV = os.getenv("DEVICE", "auto")
if DEVICE_ENV.lower() == "auto":
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
else:
    try:
        DEVICE = int(DEVICE_ENV)
    except ValueError:
        DEVICE = DEVICE_ENV
print(f"[INFO] device={DEVICE}")


best_path_env = os.getenv("BEST_PATH", "").strip()
if best_path_env and os.path.exists(best_path_env):
    best = best_path_env
else:
    best = find_latest_best()
    assert best, "best.pt를 찾지 못했습니다. .env의 BEST_PATH를 지정하거나 train을 완료하세요."
print("[INFO] Using best:", best)

model = YOLO(best)


CONF_THRESH      = float(os.getenv("CONF", "0.25"))
IOU_THRESH       = float(os.getenv("IOU",  "0.45"))
VID_STRIDE       = int(os.getenv("VID_STRIDE", "1"))

LINE_THICKNESS   = int(os.getenv("LINE_THICKNESS", "3"))
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = float(os.getenv("FONT_SCALE", "0.7"))
TEXT_THICKNESS   = int(os.getenv("TEXT_THICKNESS", "2"))
PAD_X            = int(os.getenv("TEXT_PAD_X", "6"))
PAD_Y            = int(os.getenv("TEXT_PAD_Y", "4"))
TEXT_BG_ALPHA    = float(os.getenv("TEXT_BG_ALPHA", "0.6"))
DEFAULT_COLOR    = (0, 255, 0)


LABEL_COLORS = {
    "crow": (0, 200, 255),
}

def draw_detections(frame, r):
    """Ultralytics Results를 커스텀 스타일로 렌더링."""
    if r is None or getattr(r, "boxes", None) is None or len(r.boxes) == 0:
        return frame

    overlay = frame.copy() if TEXT_BG_ALPHA > 0 else None
    names = r.names 

    for b in r.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cls_id = int(b.cls[0])
        conf   = float(b.conf[0])
        label  = names.get(cls_id, str(cls_id))
        color  = LABEL_COLORS.get(label, DEFAULT_COLOR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, LINE_THICKNESS)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, TEXT_THICKNESS)

        tx1, ty1 = x1, max(0, y1 - th - 2 * PAD_Y)
        tx2, ty2 = x1 + tw + 2 * PAD_X, y1

        if TEXT_BG_ALPHA > 0:
            cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), color, -1)
            cv2.addWeighted(overlay, TEXT_BG_ALPHA, frame, 1 - TEXT_BG_ALPHA, 0, frame)

        cv2.putText(frame, text, (tx1 + PAD_X, ty2 - PAD_Y),
                    FONT, FONT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA)

    return frame

in_dir   = Path("images")
out_root = Path("runs") / "detect" / "custom"
out_root.mkdir(parents=True, exist_ok=True)

video_files = sorted(
    set(glob.glob(str(in_dir / "*.mp4"))) |
    set(glob.glob(str(in_dir / "*.MP4")))
)

image_files = sorted(
    set(glob.glob(str(in_dir / "*.jpg")))  |
    set(glob.glob(str(in_dir / "*.jpeg"))) |
    set(glob.glob(str(in_dir / "*.png")))  |
    set(glob.glob(str(in_dir / "*.JPG")))  |
    set(glob.glob(str(in_dir / "*.JPEG"))) |
    set(glob.glob(str(in_dir / "*.PNG")))
)

processed_any = False

for video_in in video_files:
    processed_any = True
    src = Path(video_in)
    video_out = str(out_root / f"{src.stem}_pred.mp4")

    print(f"[INFO] Predict video: {video_in} -> {video_out}")
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_in}")
        continue

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out, fourcc, fps, (w, h))

    frame_idx, last_r = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if last_r is None or (VID_STRIDE <= 1) or (frame_idx % VID_STRIDE == 0):
            try:
                results = model.predict(
                    frame,
                    conf=CONF_THRESH,
                    iou=IOU_THRESH,
                    device=DEVICE,
                    verbose=False
                )
                last_r = results[0]
            except Exception as e:
                print(f"[ERR] predict failed on frame {frame_idx}: {e}")
                last_r = None

        out_frame = draw_detections(frame, last_r)
        writer.write(out_frame)

    cap.release()
    writer.release()
    print(f"[INFO] Saved video -> {video_out}")

for image_in in image_files:
    processed_any = True
    src = Path(image_in)
    image_out = str(out_root / f"{src.stem}_pred.jpg")

    print(f"[INFO] Predict image: {image_in} -> {image_out}")
    img = cv2.imread(image_in)
    if img is None:
        print(f"[WARN] Cannot read image: {image_in}")
        continue

    try:
        results = model.predict(
            img,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            device=DEVICE,
            verbose=False
        )
        out = draw_detections(img, results[0])
        cv2.imwrite(image_out, out)
    except Exception as e:
        print(f"[ERR] predict failed on image {image_in}: {e}")

if not processed_any:
    print(f"[WARN] 입력 파일을 찾지 못했습니다. videos: {in_dir}/*.mp4, images: {in_dir}/*.jpg|*.png 등")
else:
    print(f"[INFO] Saved to: {out_root}")