import os, shutil, random
import numpy as np
import torch
from dotenv import load_dotenv, find_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

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

PATIENCE     = int(os.getenv("PATIENCE", "20"))
WORKERS      = int(os.getenv("WORKERS", "2"))
CACHE        = os.getenv("CACHE", "true").lower() in ("1", "true", "yes")

# (옵션) 재현성
SEED = int(os.getenv("SEED", "42"))
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

# ---- 데이터셋 경로 & (재)다운로드 정책 ----
# DATASET_DIR이 비어있으면 ./<PROJECT>-<VERSION> 사용
dataset_dir_env = os.getenv("DATASET_DIR", "").strip()
dataset_dir = dataset_dir_env if dataset_dir_env else f"./{PROJECT}-{VERSION}"
force_download = os.getenv("FORCE_DOWNLOAD", "false").lower() in ("1", "true", "yes")

print(f"[INFO] dataset_dir={dataset_dir}  force_download={force_download}")

# 필요 시 강제 초기화
if force_download and os.path.exists(dataset_dir):
    print("[INFO] FORCE_DOWNLOAD=true → 기존 데이터셋 폴더 삭제")
    shutil.rmtree(dataset_dir, ignore_errors=True)

# 다운로드 or 캐시 사용
if os.path.exists(dataset_dir):
    print(f"[INFO] Using cached dataset at {dataset_dir}")
    dataset_location = dataset_dir
else:
    print("[INFO] Downloading dataset from Roboflow...")
    try:
        rf = Roboflow(api_key=RF_API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT)
        dataset = project.version(VERSION).download("yolov8")
        # 그냥 Roboflow SDK가 내려준 경로 사용
        dataset_location = dataset.location
        print(f"[INFO] Dataset ready at: {dataset_location}")
    except Exception as e:
        raise RuntimeError(f"[Roboflow] 다운로드/인증 실패: {e}")


# ---- data.yaml 탐색 ----
data_yaml = find_data_yaml(preferred_base=dataset_location)
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
        # (필요 시 증강값을 여기서 조정 가능 - 기본값 유지)
        # mosaic=0.2, mixup=0.0, perspective=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
    )
    best = find_latest_best()
    print(f"[INFO] Training finished. best: {best or 'not found'}")
