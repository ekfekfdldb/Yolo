import os
from dotenv import load_dotenv, find_dotenv
from roboflow import Roboflow
from ultralytics import YOLO
import torch

from utils import find_data_yaml, find_latest_best


def require(key: str) -> str:
    v = os.getenv(key)
    assert v not in (None, ""), f"[ENV] {key}가 비었습니다. .env를 채워주세요."
    return v


# ---- .env 로드 ----
env_path = find_dotenv(usecwd=True)
loaded = load_dotenv(env_path, override=True)
print(f"[dotenv] loaded={loaded} path={env_path}")

# ---- ENV 값 ----
RF_API_KEY   = require("RF_API_KEY")
WORKSPACE    = require("WORKSPACE")
PROJECT      = require("PROJECT")
VERSION      = int(require("VERSION"))

EPOCHS       = int(require("EPOCHS"))
IMG_SIZE     = int(require("IMG_SIZE"))
BATCH        = int(require("BATCH"))
MODEL_START  = require("MODEL_START")   
DEVICE_ENV   = require("DEVICE")        

PATIENCE     = int(os.getenv("PATIENCE", "18"))
WORKERS      = int(os.getenv("WORKERS", "2"))
CACHE        = os.getenv("CACHE", "true").lower() in ("1", "true", "yes")

# ---- DEVICE 해석 ----
if DEVICE_ENV.lower() == "auto":
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
elif DEVICE_ENV.lower() == "cpu":
    DEVICE = "cpu"
else:
    try:
        DEVICE = int(DEVICE_ENV)
    except ValueError:
        DEVICE = DEVICE_ENV

print(f"[INFO] device={DEVICE}  start_weights={MODEL_START}  "
      f"epochs={EPOCHS} img={IMG_SIZE} batch={BATCH} patience={PATIENCE} "
      f"workers={WORKERS} cache={CACHE}")
print(f"[INFO] rf: workspace={WORKSPACE} project={PROJECT} version={VERSION}")

# ---- Roboflow에서 데이터셋 다운로드 ----
print("[INFO] Downloading dataset from Roboflow...")
try:
    rf = Roboflow(api_key=RF_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8")
    print(f"[INFO] Dataset path: {dataset.location}")
except Exception as e:
    raise RuntimeError(f"[Roboflow] 다운로드/인증 실패: {e}")

# ---- data.yaml 탐색 ----
data_yaml = find_data_yaml(preferred_base=dataset.location)
assert data_yaml and os.path.exists(data_yaml), \
    "data.yaml을 찾지 못했습니다. Roboflow 경로나 data 폴더를 확인하세요."
print(f"[INFO] Using data.yaml: {data_yaml}")

# ---- 학습 실행 ----
if __name__ == "__main__":
    print(f"[INFO] Training start with weights: {MODEL_START}")
    model = YOLO(MODEL_START)
    r = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        workers=WORKERS,
        cache=CACHE,
    )
    best = find_latest_best()
    print(f"[INFO] Training finished. best: {best or 'not found'}")
