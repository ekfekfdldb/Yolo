import os
import sys
import glob
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import torch
from ultralytics import YOLO
from datetime import datetime


BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)).resolve()
os.chdir(BASE_DIR)


def find_latest_best(runs_dir="runs/detect"):
    paths = glob.glob(os.path.join(runs_dir, "train*", "weights", "best.pt"))
    if not paths:
        return None
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths[0]


def list_devices():
    items = ["auto", "cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            items.append(f"cuda:{i}")
    return items

def resolve_device(device_choice: str):
    """Ultralytics predict에 넘길 device 파라미터로 변환"""
    dc = (device_choice or "auto").lower()
    if dc == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if dc == "cpu":
        return "cpu"
    if dc.startswith("cuda:"):
        try:
            idx = int(dc.split(":")[1])
            return idx
        except Exception:
            return 0
    try:
        return int(dc)
    except Exception:
        return "cpu"


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")
VID_EXTS = (".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV", ".mkv", ".MKV")

def collect_media(paths_or_dirs, recursive=True):
    files = []
    for p in paths_or_dirs:
        p = Path(p)
        if p.is_file() and p.suffix in IMG_EXTS + VID_EXTS:
            files.append(str(p))
        elif p.is_dir():
            if recursive:
                for ext in IMG_EXTS + VID_EXTS:
                    files.extend([str(x) for x in p.rglob(f"*{ext}")])
            else:
                for ext in IMG_EXTS + VID_EXTS:
                    files.extend([str(x) for x in p.glob(f"*{ext}")])
    files = sorted(set(files), key=lambda x: x.lower())
    img_files = [f for f in files if Path(f).suffix in IMG_EXTS]
    vid_files = [f for f in files if Path(f).suffix in VID_EXTS]
    return img_files, vid_files


def process_all(progress_q, status_q, done_q, error_q, config):
    """
    config: {
        'weights': str,
        'inputs': [str, ...],   # 파일 또는 폴더
        'recursive': bool,
        'conf': float,
        'iou': float,
        'vid_stride': int,
        'imgsz': int | None,
        'device': 'auto'|'cpu'|'cuda:0'|int,
        'out_dir': str
    }
    """
    try:
        weights = config["weights"]
        inputs  = config["inputs"]
        recursive = config.get("recursive", True)
        conf = float(config.get("conf", 0.25))
        iou  = float(config.get("iou", 0.45))
        vid_stride = int(config.get("vid_stride", 1))
        imgsz = config.get("imgsz", None)
        device = resolve_device(config.get("device", "auto"))

        out_dir = Path(config.get("out_dir") or (Path("runs") / "detect" / "custom"))
        out_dir.mkdir(parents=True, exist_ok=True)

        img_files, vid_files = collect_media(inputs, recursive=recursive)
        total_jobs = len(img_files) + len(vid_files)
        if total_jobs == 0:
            error_q.put("입력 이미지/비디오가 없습니다. 파일 또는 폴더를 선택하세요.")
            done_q.put(True)
            return

        status_q.put(f"모델 로딩: {Path(weights).name} (device={device})")
        model = YOLO(weights)

        done_units = 0

        for video_in in vid_files:
            name = Path(video_in).name
            status_q.put(f"비디오 처리 중: {name}")

            cap = cv2.VideoCapture(video_in)
            if not cap.isOpened():
                status_q.put(f"열 수 없음: {name}, 건너뜁니다.")
                done_units += 1
                progress_q.put(int(done_units / total_jobs * 100))
                continue

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_out = str(out_dir / f"{Path(video_in).stem}_pred.mp4")
            writer = cv2.VideoWriter(video_out, fourcc, fps, (w, h))
            if not writer.isOpened():
                fourcc2 = cv2.VideoWriter_fourcc(*"XVID")
                video_out = str(out_dir / f"{Path(video_in).stem}_pred.avi")
                writer = cv2.VideoWriter(video_out, fourcc2, fps, (w, h))

            processed = 0
            step_every = max(total_frames // 100, 1) if total_frames > 0 else 5

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    results = model.predict(
                        frame,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        device=device,
                        verbose=False,
                        workers=0,
                        vid_stride=vid_stride,
                        max_det=300,
                    )
                    out_frame = results[0].plot()
                    writer.write(out_frame)
                    processed += 1

                    if total_frames > 0 and processed % step_every == 0:
                        percent = int(((done_units + processed / total_frames) / max(1, total_jobs)) * 100)
                        progress_q.put(min(99, max(0, percent)))
            finally:
                cap.release()
                writer.release()

            done_units += 1
            progress_q.put(int(done_units / total_jobs * 100))

        for image_in in img_files:
            name = Path(image_in).name
            status_q.put(f"이미지 처리 중: {name}")
            img = cv2.imread(image_in)
            if img is None:
                status_q.put(f"읽기 실패: {name}, 건너뜁니다.")
                done_units += 1
                progress_q.put(int(done_units / total_jobs * 100))
                continue

            results = model.predict(
                img,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False,
                workers=0,
                max_det=300,
            )
            out = results[0].plot()
            image_out = str(out_dir / f"{Path(image_in).stem}_pred.jpg")
            cv2.imwrite(image_out, out)

            done_units += 1
            progress_q.put(int(done_units / total_jobs * 100))

        status_q.put(f"완료! 결과: {out_dir.resolve()}")
    except Exception as e:
        error_q.put(str(e))
    finally:
        done_q.put(True)

def run_ui():
    root = tk.Tk()
    root.title("YOLO Object Detection - Predict")
    root.geometry("900x450")
    root.resizable(False, False)

    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TButton", padding=(8, 6))
        style.configure("TLabel", padding=(2, 2))
        style.configure("TEntry", padding=(2, 2))
        style.configure("Horizontal.TProgressbar", thickness=16)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(
            "Green.Horizontal.TProgressbar",
            troughcolor="white",
            background="#4CAF50",
            lightcolor="#4CAF50",
            darkcolor="#4CAF50"
        )
    except Exception:
        pass

    container = ttk.Frame(root, padding=12)
    container.pack(fill="both", expand=True)

    status_var = tk.StringVar(value="대기 중")
    lbl_status = ttk.Label(container, textvariable=status_var)
    lbl_status.pack(anchor="w", pady=(0, 6))

    pb = ttk.Progressbar(container, style="Green.Horizontal.TProgressbar", orient="horizontal", mode="determinate", maximum=100)
    pb.pack(fill="x", pady=(0, 10))

    frm_weights = ttk.LabelFrame(container, text="가중치(Weights)")
    frm_weights.pack(fill="x", padx=0, pady=(0, 8))
    weight_path_var = tk.StringVar()

    ent_w = ttk.Entry(frm_weights, textvariable=weight_path_var, width=70)
    ent_w.pack(side="left", fill="x", expand=True, padx=(10, 6), pady=8)

    def browse_weights():
        path = filedialog.askopenfilename(
            title="가중치 파일 선택",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")]
        )
        if path:
            weight_path_var.set(path)

    def pick_latest_best():
        p = find_latest_best()
        if p:
            weight_path_var.set(p)
        else:
            messagebox.showinfo("알림", "runs/detect/train*/weights/best.pt 를 찾지 못했습니다.")

    ttk.Button(frm_weights, text="찾아보기", command=browse_weights, width=10).pack(side="left", padx=(0, 6))

    frm_inputs = ttk.LabelFrame(container, text="입력(이미지/비디오)")
    frm_inputs.pack(fill="x", padx=0, pady=(0, 8))

    input_listbox = tk.Listbox(frm_inputs, height=5)
    input_listbox.pack(side="left", fill="both", expand=True, padx=(10, 6), pady=8)
    sb = ttk.Scrollbar(frm_inputs, orient="vertical", command=input_listbox.yview)
    sb.pack(side="left", fill="y", pady=8)
    input_listbox.config(yscrollcommand=sb.set)

    inputs_paths = []

    def add_files():
        files = filedialog.askopenfilenames(
            title="파일 추가",
            filetypes=[
                ("이미지/비디오", "*.jpg;*.jpeg;*.png;*.bmp;*.mp4;*.avi;*.mov;*.mkv"),
                ("이미지", "*.jpg;*.jpeg;*.png;*.bmp"),
                ("비디오", "*.mp4;*.avi;*.mov;*.mkv"),
                ("모든 파일", "*.*"),
            ]
        )
        if files:
            for f in files:
                if f not in inputs_paths:
                    inputs_paths.append(f)
            refresh_input_list()

    def add_folder():
        d = filedialog.askdirectory(title="폴더 선택")
        if d:
            if d not in inputs_paths:
                inputs_paths.append(d)
            refresh_input_list()

    def clear_inputs():
        inputs_paths.clear()
        refresh_input_list()

    def refresh_input_list():
        input_listbox.delete(0, tk.END)
        for p in inputs_paths:
            input_listbox.insert(tk.END, p)

    btn_box = ttk.Frame(frm_inputs)
    btn_box.pack(side="left", fill="y", padx=(6, 10), pady=8)
    ttk.Button(btn_box, text="파일 추가", command=add_files, width=12).pack(pady=(0, 6), fill="x")
    ttk.Button(btn_box, text="폴더 선택", command=add_folder, width=12).pack(pady=(0, 6), fill="x")
    ttk.Button(btn_box, text="목록 비우기", command=clear_inputs, width=12).pack(pady=(0, 0), fill="x")

    recursive_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(btn_box, text="하위 폴더 포함", variable=recursive_var).pack(pady=(8, 0), anchor="w")

    frm_params = ttk.LabelFrame(container, text="파라미터")
    frm_params.pack(fill="x", padx=0, pady=(0, 8))

    def add_labeled_spin(parent, label, from_, to, inc, init, width=8):
        f = ttk.Frame(parent)
        ttk.Label(f, text=label, width=10).pack(side="left")
        var = tk.DoubleVar(value=init)
        sp = ttk.Spinbox(f, textvariable=var, from_=from_, to=to, increment=inc, width=width, format="%.2f")
        sp.pack(side="left", padx=(4, 12))
        return var, sp, f

    conf_var, _, f1 = add_labeled_spin(frm_params, "conf", 0.00, 1.00, 0.01, 0.25)
    iou_var,  _, f2 = add_labeled_spin(frm_params, "iou",  0.00, 1.00, 0.01, 0.45)
    f1.pack(side="left", padx=(10, 0), pady=8)
    f2.pack(side="left", padx=(0, 0), pady=8)

    f3 = ttk.Frame(frm_params)
    ttk.Label(f3, text="frame", width=10).pack(side="left")
    vid_stride_var = tk.IntVar(value=1)
    sp_vs = ttk.Spinbox(f3, textvariable=vid_stride_var, from_=1, to=30, increment=1, width=8)
    sp_vs.pack(side="left", padx=(4, 12))
    f3.pack(side="left", pady=8)

    f4 = ttk.Frame(frm_params)
    ttk.Label(f4, text="image size", width=10).pack(side="left")
    imgsz_var = tk.IntVar(value=640)
    sp_imgsz = ttk.Spinbox(f4, textvariable=imgsz_var, from_=256, to=1920, increment=32, width=8)
    sp_imgsz.pack(side="left", padx=(4, 12))
    f4.pack(side="left", pady=8)

    f5 = ttk.Frame(frm_params)
    ttk.Label(f5, text="device", width=10).pack(side="left")
    device_var = tk.StringVar(value="auto")
    cmb = ttk.Combobox(f5, textvariable=device_var, values=list_devices(), state="readonly", width=12)
    cmb.pack(side="left", padx=(4, 12))
    f5.pack(side="left", pady=8)

    frm_btn = ttk.Frame(container)
    frm_btn.pack(fill="x", pady=(6, 0))

    btn_start = ttk.Button(frm_btn, text="시작", width=14)
    btn_open  = ttk.Button(frm_btn, text="결과 폴더 열기", width=16, state="disabled")
    btn_exit  = ttk.Button(frm_btn, text="닫기", width=10, command=root.destroy)

    btn_start.pack(side="left", padx=(0, 6))
    btn_open.pack(side="left", padx=(0, 6))
    btn_exit.pack(side="right")

    progress_q = queue.Queue()
    status_q   = queue.Queue()
    done_q     = queue.Queue()
    error_q    = queue.Queue()

    def open_out():
        out_dir = Path("runs") / "detect" / "custom"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(out_dir)
            elif sys.platform.startswith("darwin"):
                os.system(f'open "{out_dir}"')
            else:
                os.system(f'xdg-open "{out_dir}"')
        except Exception:
            messagebox.showinfo("정보", str(out_dir.resolve()))

    def set_controls_enabled(enabled: bool):
        state = "normal" if enabled else "disabled"
        for w in (ent_w, btn_start, btn_open):
            try:
                w.config(state=state)
            except Exception:
                pass

    def start():
        weights = weight_path_var.get().strip()
        if not weights or not os.path.exists(weights):
            messagebox.showerror("오류", "가중치(.pt) 파일을 선택하세요.")
            return
        if not inputs_paths:
            messagebox.showerror("오류", "입력 파일 또는 폴더를 추가하세요.")
            return

        pb["value"] = 0
        status_var.set("초기화 중…")
        btn_open.config(state="disabled")
        set_controls_enabled(False)

        config = {
            "weights": weights,
            "inputs": inputs_paths.copy(),
            "recursive": bool(recursive_var.get()),
            "conf": float(conf_var.get()),
            "iou": float(iou_var.get()),
            "vid_stride": int(vid_stride_var.get()),
            "imgsz": int(imgsz_var.get()) if int(imgsz_var.get()) > 0 else None,
            "device": device_var.get(),
            "out_dir": str(Path("runs") / "detect" / "custom"),
        }

        t = threading.Thread(target=process_all, args=(progress_q, status_q, done_q, error_q, config), daemon=True)
        t.start()
        root.after(120, poll_queues)

    def poll_queues():
        while not progress_q.empty():
            v = progress_q.get_nowait()
            try:
                v = int(v)
            except Exception:
                v = 0
            pb["value"] = max(0, min(100, v))

        while not status_q.empty():
            s = status_q.get_nowait()
            status_var.set(s)

        if not error_q.empty():
            err = error_q.get_nowait()
            messagebox.showerror("오류", err)

        if not done_q.empty():
            pb["value"] = 100
            btn_open.config(state="normal")
            set_controls_enabled(True)
            status_var.set(f"{status_var.get()}  (완료)")
            return

        root.after(120, poll_queues)

    btn_start.config(command=start)
    btn_open.config(command=open_out)

    root.mainloop()

if __name__ == "__main__":
    run_ui()
