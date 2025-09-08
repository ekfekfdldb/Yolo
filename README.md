# YOLO Crow Detector (Roboflow + Ultralytics + OpenCV)

Roboflow에서 라벨링한 데이터를 내려받아 로컬에서 YOLO(v8/v11)로 학습하고, 이미지·동영상 일괄 추론(커스텀 테두리/라벨 스타일)까지 수행하는 프로젝트입니다.
Windows + VS Code 환경을 기준으로 설명합니다.

---

## 주요 기능
- Roboflow API로 데이터셋 버전 다운로드(라벨/분할/전처리는 Roboflow에서 설정)
- Ultralytics YOLO(v8 또는 v11) 로컬 학습
- 커스텀 오버레이(테두리·라벨 글자·투명도·두께 등)로 이미지/동영상 결과 저장
- images/*.mp4, images/*.jpg|*.png 일괄 처리
- .env 하나로 학습·추론·표시 옵션 제어
- Windows 멀티프로세싱/경로 이슈 대비

---

## 요구 사항
- Python 3.10 이상(권장 3.11)
- NVIDIA GPU(선택) + 최신 드라이버
- VS Code + PowerShell(또는 CMD)
- Roboflow 계정 및 프로젝트(Workspace/Project 슬러그, API Key)

---

## 프로젝트 구조

```
yolo/
├─ train.py
├─ predict.py
├─ utils.py
├─ requirements.txt
├─ .env
├─ images/
│  ├─ test.jpg
│  └─ test.mp4
├─ models/
│  └─ crow.pt                    # (선택) 학습된 best를 복사해 고정 경로로 사용
├─ Crow-2/                       # Roboflow 데이터셋(자동 생성/선택적으로 유지)
└─ runs/                         # 학습/추론 결과(자동 생성)
   └─ detect/
      ├─ train*/weights/best.pt
      └─ custom/                 # predict 결과 저장
```

> runs/와 Crow-2/는 실행 중 자동 생성됩니다.
> 원하면 학습된 best.pt를 models/crow.pt로 복사하고 .env의 BEST_PATH로 고정해 사용할 수 있습니다.

---

## 설치

### 1) 가상환경 생성 및 활성화

```powershell
cd D:\VsCode\yolo
py -3.11 -m venv .venv
.\.venv\Scripts\Activate
```
실행 정책 에러가 뜰 경우
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate
```

확인
```powershell
python -V
pip -V
```
### 2) 의존성 설치

```powershell
pip install -U pip
pip install -r requirements.txt
```

> GPU 사용 시 설치된 Torch가 CUDA 지원 빌드인지 확인하세요. 드라이버/런타임이 맞지 않으면 .env에서 DEVICE=cpu로 추론을 CPU로 전환할 수 있습니다.

---

## .env 작성

프로젝트 루트에 .env 파일을 생성하고 아래 내용을 채웁니다.
(슬러그는 Roboflow 프로젝트 페이지 주소창 그대로 사용)

```env
# ===== (필수) Roboflow에서 라벨링 후 버전 생성 =====
RF_API_KEY=YOUR_API_KEY
WORKSPACE=xxxxxxxxxxxx           # 예: 주소창 /xxxxxxxxxxxx/
PROJECT=xxxx-xxxxx               # 예: 주소창 /xxxxxxxxxxxx/xxxx-xxxxx/
VERSION=2                        # 생성한 데이터셋 버전 번호

# ===== (필수) 학습/추론 공통 =====
EPOCHS=50
IMG_SIZE=640                     # 메모리가 빡빡하면 512
BATCH=-1                         # 자동. 부족하면 8→4→2→1
MODEL_START=yolov8n.pt           # 또는 yolo11n.pt
DEVICE=auto                      # 0(첫 GPU)/cpu 등

# ===== (선택) 학습 가속/안정성 =====
PATIENCE=18                      # 개선 없으면 조기 종료
WORKERS=2                        # Win 2~4 권장, 문제 시 0
CACHE=true                       # true(램)/disk/false

# ===== (선택) 추론 가중치 경로 =====
# 비워두면 runs/detect/train*/weights/best.pt을 자동 탐색
BEST_PATH=models/crow.pt

# ===== (선택) 추론/표시 옵션 =====
CONF=0.25
IOU=0.45
LINE_THICKNESS=3
FONT_SCALE=0.7
TEXT_THICKNESS=2
TEXT_PAD_X=0                     # 라벨 배경 없애려면 0
TEXT_PAD_Y=0
TEXT_BG_ALPHA=0                  # 0=배경 없음(테두리+글자만)
```

---

## 사용법

### 1) 학습(Train)

```powershell
.\.venv\Scripts\Activate
python train.py
```
정상 완료 시:
- runs/detect/train*/weights/best.pt 생성
- 검증 지표(mAP 등), results.png 저장

참고
- Roboflow에서 데이터/라벨을 수정했다면 **새 버전(Generate)**을 만들고 .env의 VERSION을 갱신하세요.
- Crow-2/ 폴더가 이미 있으면 재다운로드 없이 data.yaml을 바로 사용합니다.
- 저장 루트를 프로젝트 내부로 고정하려면 한 번만 아래를 실행합니다.
```powershell
yolo settings runs_dir="D:\VsCode\yolo\runs"
```

### 2) 추론(Predict)

```powershell
python predict.py
```

- images/ 폴더의 모든 .mp4와 .jpg|.png를 자동 처리
- 결과는 runs/detect/custom/원본이름_pred.*로 저장
- .env의 CONF, IOU, VID_STRIDE, TEXT_BG_ALPHA 등으로 스타일/속도 조절

--- 

## 운영 팁

- 데이터셋/버전이 바뀔 때마다 .env의 VERSION을 갱신하세요.
- 장시간 학습은 PATIENCE로 조기 종료를 켜 두면 시간을 절약할 수 있습니다.
- 학습이 끝나면 best.pt를 models/프로젝트명_v버전.pt로 복사해 버전별로 보관하면 관리가 쉽습니다.
- predict.py는 대·소문자 확장자를 모두 스캔하고 FPS=0 반환 등도 안전하게 보정합니다

---

