# YOLO_EXE – 학습 & 추론 GUI

- [학습 (Training)](#학습-training)
- [추론 (Prediction)](#추론-prediction)

---

## 학습 (Training)

### 의존성 설치 (PowerShell)


# 1) 가상환경
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

# 2) 필수 패키지
```powershell
pip install --upgrade pip
pip install ultralytics torch torchvision opencv-python ruamel.yaml certifi
```

# 3) (옵션) Roboflow 연동
```powershell
pip install roboflow
```

> GPU 사용 시: CUDA 빌드의 torch/torchvision을 원하면 PyTorch 인덱스를 지정 (예: CUDA 12.1)

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 실행 (스크립트)

```powershell
# GUI 실행
python .\train.py
```

### 패키징 (PyInstaller, Windows .exe)

```powershell
python -m PyInstaller -F -w .\train.py `
  --name YOLO_Training `
  --collect-all ultralytics `
  --collect-all cv2 `
  --collect-submodules torch `
  --collect-submodules torchvision `
  --collect-all ruamel.yaml `
  --collect-submodules roboflow `
  --collect-data certifi
```

**메모**
- 동결(.exe) 환경에서는 코드에서 **workers=0**으로 고정됩니다.
- 학습 산출물 경로: `~/yolo_runs/detect/train*`
- v11 가중치 파일명은 **`yolo11*.pt`**, v8은 **`yolov8*.pt`** 입니다.

---

## 추론 (Prediction)

### 의존성 설치 (PowerShell)


# 1) 가상환경 (없으면 위와 동일)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

# 2) 필수 패키지
```powershell
pip install --upgrade pip
pip install ultralytics torch torchvision opencv-python certifi
```

# (옵션) 서버/헤드리스 환경이면
```powershell
pip install opencv-python-headless
```

### 실행 (스크립트)

# GUI 실행
```powershell
python .\predict.py
```

- 가중치(.pt)는 UI에서 선택 (예: `~/yolo_runs/detect/train*/weights/best.pt`)

### 패키징 (PyInstaller, Windows .exe)

```powershell
python -m PyInstaller -F -w .\predict.py `
  --name YOLO_Predict `
  --collect-all ultralytics `
  --collect-all cv2 `
  --collect-submodules torch `
  --collect-submodules torchvision `
  --collect-data certifi
```

**메모**
- 기본 출력 경로: `runs/detect/custom`
- 비디오 인코딩은 기본 `mp4v`, 실패 시 `XVID`로 자동 대체합니다.

---

## 프로그램 소개

- **개요**: Python · Ultralytics YOLO · OpenCV 기반의 **객체 탐지 모델 학습/추론 GUI**
- **데이터셋**: Roboflow 연동(옵션) 또는 로컬 `data.yaml` 자동 탐색
- **모델**: YOLOv8 / YOLOv11, 크기 n/s/m/l/x
- **설정**: Epochs, Image size, Batch, Device(auto/cpu/cuda), Patience, Workers, Cache
- **모니터링**: 진행률, 상태 로그, 지표(mAP@0.5, mAP@0.5:0.95, Precision, Recall)
- **동결 대응**: Windows .exe에서 멀티프로세싱 Worker=0, 출력은 사용자 홈(`~/yolo_runs/detect`)에 저장

---

## 환경 준비 / 공통 의존성


# 1) 가상환경
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

# 2) 공통 패키지
```powershell
pip install --upgrade pip
pip install ultralytics torch torchvision opencv-python ruamel.yaml certifi
```

# 3) (옵션) Roboflow
```powershell
pip install roboflow
```

**추가 팁**
- GPU가 없다면 torch CPU 빌드:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

- 방화벽/프록시 환경에서 인증서 이슈가 있으면 `certifi` 필요(이미 의존성에 포함).

---

## 프로젝트 구조

~~~text
YOLO_EXE/
├─ .venv/                 # 가상환경 (커밋 X)
├─ build/                 # PyInstaller 중간 산출물 (커밋 X)
├─ dist/                  # 빌드 결과 .exe (커밋 X)
├─ .gitignore
├─ predict.py             # 추론 GUI
├─ train.py               # 학습 GUI
├─ YOLO_Predict.spec      # (선택) PyInstaller 스펙 파일
├─ YOLO_Training.spec     # (선택) PyInstaller 스펙 파일
├─ yolo11n.pt             # (예시) v11 가중치 (커밋 X 권장)
└─ yolov8n.pt             # (예시) v8  가중치 (커밋 X 권장)
~~~

> `.pt` 가중치와 `runs/` 산출물은 저장소에 커밋하지 않는 것을 권장합니다.  
> 스펙 파일(`*.spec`)은 **재현 가능한 빌드**를 위해 커밋 권장(팀 작업 시 유용).

---

## 라이선스

이 저장소의 코드는 아래 라이선스를 따릅니다.  
Ultralytics, PyTorch, OpenCV, Roboflow 등 **제3자 라이브러리의 라이선스는 각 프로젝트**를 따릅니다.

### MIT License

```text
MIT License

Copyright (c) 2025 [ekfekfdldb]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


