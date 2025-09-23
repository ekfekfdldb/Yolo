import os
import sys
import glob
import csv
import shutil
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import io

class _SilentIO(io.TextIOBase):
    def write(self, _): return 0
    def flush(self): pass
if getattr(sys, "stdout", None) is None:
    sys.stdout = _SilentIO()
if getattr(sys, "stderr", None) is None:
    sys.stderr = _SilentIO()

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except Exception:
    ROBOFLOW_AVAILABLE = False

import torch
from ultralytics import YOLO

IS_FROZEN = getattr(sys, "frozen", False)

BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)).resolve()
os.chdir(BASE_DIR)

RUNS_DIR = (Path.home() / "yolo_runs" / "detect").resolve()
RUNS_DIR.mkdir(parents=True, exist_ok=True)

def find_data_yaml(preferred_base=None):
    """data.yaml / data.yml 자동 탐색"""
    candidates = []
    search_roots = []
    if preferred_base and os.path.isdir(preferred_base):
        search_roots.append(preferred_base)
    search_roots += [".", "./data"]
    patterns = ["**/data.yaml", "**/data.yml"]
    for root in search_roots:
        for pat in patterns:
            candidates += glob.glob(os.path.join(root, pat), recursive=True)
    candidates = sorted(set(candidates), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None

def list_devices():
    items = ["auto", "cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            items.append(f"cuda:{i}")
    return items

def resolve_device(device_choice: str):
    dc = (device_choice or "auto").lower()
    if dc == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if dc == "cpu":
        return "cpu"
    if dc.startswith("cuda:"):
        try:
            return int(dc.split(":")[1])
        except Exception:
            return 0
    try:
        return int(dc)
    except Exception:
        return "cpu"

def compose_weight_name(ver: str, size: str) -> str:
    """
    ver: '8' or '11' (또는 'YOLOv8'/'YOLOv11')
    size: 'n','s','m','l','x'
    returns:
      - v8  -> 'yolov8{size}.pt'
      - v11 -> 'yolo11{size}.pt'   # v 없음
    """
    v = str(ver).strip()
    up = v.upper()
    if up.startswith("YOLOV"):
        v = v[5:] 
    elif up.startswith("YOLO"):
        v = v[4:]

    v = "11" if v in ("11", "11n", "11s", "11m", "11l", "11x") else "8"

    s = str(size).strip().lower()
    assert v in ("8", "11"), "지원 모델 버전은 v8, v11 입니다."
    assert s in ("n","s","m","l","x"), "모델 크기는 n/s/m/l/x 중 하나여야 합니다."

    return f"yolo11{s}.pt" if v == "11" else f"yolov8{s}.pt"

def train_worker(cfg, progress_q, status_q, done_q, error_q, metrics_q, runpath_q):
    """
    cfg keys:
      mode: "roboflow" | "local"
      # Roboflow
      rf_api_key, rf_workspace, rf_project, rf_version (int), force_download(bool)
      # Local
      data_yaml, data_root
      # Common
      model_version('8'|'11' or label), model_size('n'|'s'|'m'|'l'|'x'),
      epochs(int), imgsz(int), batch(int), device(str),
      patience(int), workers(int), cache(bool),
      delete_dataset(bool)
    """
    dataset_location = None
    try:
        if cfg["mode"] == "roboflow":
            if not ROBOFLOW_AVAILABLE:
                raise RuntimeError("Roboflow SDK가 설치되어 있지 않습니다. 'Roboflow' 설치 또는 Local 모드를 사용하세요.")
            status_q.put("데이터셋 다운로드 중(Roboflow)…")
            if cfg.get("force_download", False):
                target = f"./{cfg['rf_project']}-{cfg['rf_version']}"
                if os.path.exists(target):
                    shutil.rmtree(target, ignore_errors=True)
            rf = Roboflow(api_key=cfg["rf_api_key"])
            project = rf.workspace(cfg["rf_workspace"]).project(cfg["rf_project"])
            dataset = project.version(int(cfg["rf_version"])).download("yolov8")
            dataset_location = dataset.location
        else:
            status_q.put("로컬 데이터셋 사용…")
            dataset_location = None
            if cfg.get("data_yaml"):
                dataset_location = Path(cfg["data_yaml"]).parent.as_posix()
            elif cfg.get("data_root"):
                dataset_location = cfg["data_root"]

        data_yaml = cfg.get("data_yaml", None)
        if not data_yaml:
            data_yaml = find_data_yaml(preferred_base=dataset_location)
        if not data_yaml or not os.path.exists(data_yaml):
            raise RuntimeError("data.yaml을 찾지 못했습니다. 경로를 확인하세요.")

        weights_name = compose_weight_name(cfg["model_version"], cfg["model_size"])
        dev = resolve_device(cfg["device"])
        status_q.put(f"모델 로딩: {weights_name} (device={dev})")
        model = YOLO(weights_name) 

        total_epochs = int(cfg["epochs"])
        trainer_state = {"save_dir": None}

        def on_train_start(trainer):
            trainer_state["save_dir"] = str(trainer.save_dir)
            runpath_q.put(trainer_state["save_dir"])
            status_q.put(f"학습 폴더: {trainer_state['save_dir']}")

        def on_train_epoch_end(trainer):
            epoch = int(trainer.epoch) + 1
            pr = int(epoch / max(1, total_epochs) * 100)
            progress_q.put(min(99, max(0, pr)))
            try:
                loss_box = trainer.label_loss_items(trainer.tloss, prefix="train")
                loss_str = ", ".join([f"{k}={v:.4f}" for k, v in loss_box.items()])
            except Exception:
                loss_str = ""
            status_q.put(f"에폭 {epoch}/{total_epochs} 완료 {loss_str}")

        def on_train_end(trainer):
            progress_q.put(100)
            status_q.put("학습 종료.")

        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_train_end", on_train_end)

        safe_workers = int(cfg.get("workers", 0 if IS_FROZEN else 2))
        if IS_FROZEN:
            safe_workers = 0

        train_kwargs = dict(
            data=data_yaml,
            epochs=int(cfg["epochs"]),
            imgsz=int(cfg["imgsz"]),
            batch=int(cfg["batch"]),
            device=dev,
            patience=int(cfg.get("patience", 20)),
            workers=safe_workers,
            cache=bool(cfg.get("cache", True)),
            verbose=True,
            project=str(RUNS_DIR), 
            name=None,
        )

        status_q.put("학습 시작…")
        _ = model.train(**train_kwargs)

        run_dir = trainer_state["save_dir"]
        if not run_dir or not os.path.isdir(run_dir):
            cand = sorted([p for p in glob.glob(str(RUNS_DIR / "train*")) if os.path.isdir(p)],
                          key=os.path.getmtime, reverse=True)
            run_dir = cand[0] if cand else None

        summary = {"mAP50": None, "mAP5095": None, "precision": None, "recall": None}
        if run_dir:
            csv_path = os.path.join(run_dir, "results.csv")
            if os.path.exists(csv_path):
                with open(csv_path, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    last = rows[-1]
                    def find_key(d, cands):
                        for key in d.keys():
                            for c in cands:
                                if c.lower() in key.lower():
                                    return key
                        return None
                    k_p   = find_key(last, ["metrics/precision", "precision"])
                    k_r   = find_key(last, ["metrics/recall", "recall"])
                    k_m   = find_key(last, ["metrics/mAP50", "map50", "mAP@0.5"])
                    k_m95 = find_key(last, ["metrics/mAP50-95", "map50-95", "mAP@0.5:0.95", "mAP50-95", "metrics/mAP50-95(B)"])
                    def to_f(x):
                        try: return float(x)
                        except Exception: return None
                    summary["precision"] = to_f(last.get(k_p))   if k_p   else None
                    summary["recall"]    = to_f(last.get(k_r))   if k_r   else None
                    summary["mAP50"]     = to_f(last.get(k_m))   if k_m   else None
                    summary["mAP5095"]   = to_f(last.get(k_m95)) if k_m95 else None

        metrics_q.put({"run_dir": run_dir, "summary": summary})

        try:
            if cfg.get("delete_dataset", False) and dataset_location and os.path.isdir(dataset_location):
                shutil.rmtree(dataset_location, ignore_errors=True)
                status_q.put("데이터셋 폴더 삭제 완료(용량 절약).")
        except Exception as e:
            status_q.put(f"데이터셋 삭제 실패: {e}")

        status_q.put("학습 완료.")
    except Exception as e:
        error_q.put(str(e))
    finally:
        done_q.put(True)

def run_ui():
    root = tk.Tk()
    root.title("YOLO Training")
    root.geometry("800x700")
    root.minsize(800, 700)
    root.resizable(True, True)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(
        "Green.Horizontal.TProgressbar",
        troughcolor="white",
        background="#4CAF50",
        lightcolor="#4CAF50",
        darkcolor="#4CAF50",
        thickness=16,
    )

    main = ttk.Frame(root, padding=12)
    main.pack(fill="both", expand=True)

    status_var = tk.StringVar(value="대기 중")
    ttk.Label(main, textvariable=status_var).pack(anchor="w", pady=(0, 6))
    pb = ttk.Progressbar(main, style="Green.Horizontal.TProgressbar",
                         orient="horizontal", mode="determinate", maximum=100)
    pb.pack(fill="x", pady=(0, 10))

    source_nb = ttk.Notebook(main); source_nb.pack(fill="x", pady=(0, 10))

    tab_rf = ttk.Frame(source_nb, padding=8); source_nb.add(tab_rf, text="Roboflow")
    rf_api_var = tk.StringVar(); rf_ws_var = tk.StringVar()
    rf_proj_var = tk.StringVar(); rf_ver_var = tk.IntVar(value=1)
    rf_force_var = tk.BooleanVar(value=False)

    def row(parent, label_text, var):
        frm = ttk.Frame(parent)
        ttk.Label(frm, text=label_text, width=18).pack(side="left")
        ent = ttk.Entry(frm, textvariable=var); ent.pack(side="left", fill="x", expand=True)
        return frm

    r1 = row(tab_rf, "API Key", rf_api_var); r1.pack(fill="x", pady=4)
    r2 = row(tab_rf, "Workspace", rf_ws_var); r2.pack(fill="x", pady=4)
    r3 = row(tab_rf, "Project", rf_proj_var); r3.pack(fill="x", pady=4)

    frm_rf_ver = ttk.Frame(tab_rf)
    ttk.Label(frm_rf_ver, text="Version", width=18).pack(side="left")
    ttk.Spinbox(frm_rf_ver, textvariable=rf_ver_var, from_=1, to=9999, width=8).pack(side="left", padx=(0, 10))
    ttk.Checkbutton(frm_rf_ver, text="강제 재다운로드", variable=rf_force_var).pack(side="left")  # 한글화
    frm_rf_ver.pack(fill="x", pady=4)

    tab_local = ttk.Frame(source_nb, padding=8); source_nb.add(tab_local, text="Local data.yaml")
    data_yaml_var = tk.StringVar(); data_root_var = tk.StringVar()

    def browse_yaml():
        p = filedialog.askopenfilename(title="data.yaml 선택", filetypes=[("YAML", "*.yaml;*.yml"), ("All files", "*.*")])
        if p: data_yaml_var.set(p)

    def browse_data_root():
        d = filedialog.askdirectory(title="데이터셋 폴더 선택")
        if d: data_root_var.set(d)

    frm_yaml = ttk.Frame(tab_local)
    ttk.Label(frm_yaml, text="data.yaml", width=18).pack(side="left")
    ttk.Entry(frm_yaml, textvariable=data_yaml_var).pack(side="left", fill="x", expand=True)
    ttk.Button(frm_yaml, text="찾아보기", command=browse_yaml, width=10).pack(side="left", padx=(6,0))
    frm_yaml.pack(fill="x", pady=4)

    frm_root = ttk.Frame(tab_local)
    ttk.Label(frm_root, text="검색 폴더", width=18).pack(side="left")
    ttk.Entry(frm_root, textvariable=data_root_var).pack(side="left", fill="x", expand=True)
    ttk.Button(frm_root, text="찾아보기", command=browse_data_root, width=10).pack(side="left", padx=(6,0))
    frm_root.pack(fill="x", pady=4)

    params = ttk.LabelFrame(main, text="Training Parameters")
    params.pack(fill="x", pady=(0, 10))

    epochs_var  = tk.IntVar(value=50)
    imgsz_var   = tk.IntVar(value=640)
    batch_var   = tk.IntVar(value=16)
    device_var  = tk.StringVar(value="auto")
    patience_var= tk.IntVar(value=20)
    workers_var = tk.IntVar(value=0 if IS_FROZEN else 3)
    cache_var   = tk.BooleanVar(value=True)
    delete_ds_var = tk.BooleanVar(value=False)

    model_ver_var = tk.StringVar(value="")
    model_sz_var  = tk.StringVar(value="")

    ttk.Label(params, text="Model Version").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Combobox(params, textvariable=model_ver_var, values=["YOLOv8","YOLOv11"], state="readonly", width=9)\
        .grid(row=0, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(params, text="Model Size").grid(row=0, column=2, sticky="w", padx=6, pady=4)
    ttk.Combobox(params, textvariable=model_sz_var, values=["n","s","m","l","x"], state="readonly", width=9)\
        .grid(row=0, column=3, sticky="w", padx=6, pady=4)

    ttk.Label(params, text="Epochs").grid(row=1, column=0, sticky="w", padx=6, pady=4)
    ttk.Spinbox(params, textvariable=epochs_var, from_=1, to=2000, width=8)\
        .grid(row=1, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(params, text="Image Size").grid(row=1, column=2, sticky="w", padx=6, pady=4)
    ttk.Spinbox(params, textvariable=imgsz_var, from_=256, to=1920, increment=32, width=8)\
        .grid(row=1, column=3, sticky="w", padx=6, pady=4)
    ttk.Label(params, text="Batch").grid(row=1, column=4, sticky="w", padx=6, pady=4)
    ttk.Spinbox(params, textvariable=batch_var, from_=1, to=512, width=8)\
        .grid(row=1, column=5, sticky="w", padx=6, pady=4)
    ttk.Label(params, text="Device").grid(row=1, column=6, sticky="w", padx=6, pady=4)
    ttk.Combobox(params, textvariable=device_var, values=list_devices(), state="readonly", width=10)\
        .grid(row=1, column=7, sticky="w", padx=6, pady=4)

    ttk.Label(params, text="Patience").grid(row=2, column=0, sticky="w", padx=6, pady=4)
    ttk.Spinbox(params, textvariable=patience_var, from_=0, to=200, width=8)\
        .grid(row=2, column=1, sticky="w", padx=6, pady=4)
    ttk.Label(params, text="Workers").grid(row=2, column=2, sticky="w", padx=6, pady=4)
    ttk.Spinbox(params, textvariable=workers_var, from_=0, to=16, width=8)\
        .grid(row=2, column=3, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(params, text="데이터 로더 캐시", variable=cache_var)\
        .grid(row=2, column=4, columnspan=2, sticky="w", padx=6, pady=4)
    ttk.Checkbutton(params, text="학습 후 데이터셋 삭제", variable=delete_ds_var)\
        .grid(row=2, column=6, columnspan=2, sticky="w", padx=6, pady=4)

    for col in range(8):
        params.grid_columnconfigure(col, weight=1, minsize=90)

    logfrm = ttk.LabelFrame(main, text="Training Log")
    logfrm.pack(fill="both", expand=True)
    txt = tk.Text(logfrm, height=12)
    txt.pack(side="left", fill="both", expand=True)
    sb = ttk.Scrollbar(logfrm, orient="vertical", command=txt.yview)
    sb.pack(side="left", fill="y")
    txt.config(yscrollcommand=sb.set)

    def log(msg):
        txt.insert(tk.END, msg + "\n")
        txt.see(tk.END)

    metfrm = ttk.LabelFrame(main, text="결과 지표 (after training)")
    metfrm.pack(fill="x", pady=(8, 4))
    mAP_var   = tk.StringVar(value="—")
    mAP95_var = tk.StringVar(value="—")
    P_var     = tk.StringVar(value="—")
    R_var     = tk.StringVar(value="—")

    ttk.Label(metfrm, text="mAP@0.5").grid(row=0, column=0, sticky="w", padx=6, pady=2)
    ttk.Label(metfrm, textvariable=mAP_var).grid(row=0, column=1, sticky="w", padx=6, pady=2)
    ttk.Label(metfrm, text="mAP@0.5:95").grid(row=0, column=2, sticky="w", padx=20, pady=2)
    ttk.Label(metfrm, textvariable=mAP95_var).grid(row=0, column=3, sticky="w", padx=6, pady=2)
    ttk.Label(metfrm, text="Precision").grid(row=0, column=4, sticky="w", padx=20, pady=2)
    ttk.Label(metfrm, textvariable=P_var).grid(row=0, column=5, sticky="w", padx=6, pady=2)
    ttk.Label(metfrm, text="Recall").grid(row=0, column=6, sticky="w", padx=20, pady=2)
    ttk.Label(metfrm, textvariable=R_var).grid(row=0, column=7, sticky="w", padx=6, pady=2)

    btns = ttk.Frame(main); btns.pack(fill="x", pady=(8, 0))
    btn_start = ttk.Button(btns, text="학습 시작", width=16)
    btn_open  = ttk.Button(btns, text="결과 폴더 열기", width=16, state="disabled")
    btn_close = ttk.Button(btns, text="닫기", width=10, command=root.destroy)
    btn_start.pack(side="left")
    btn_open.pack(side="left", padx=(10, 0))
    btn_close.pack(side="right")

    progress_q = queue.Queue()
    status_q   = queue.Queue()
    done_q     = queue.Queue()
    error_q    = queue.Queue()
    metrics_q  = queue.Queue()
    runpath_q  = queue.Queue()
    last_run_dir = {"path": None}

    def find_latest_run_dir():
        cands = sorted([p for p in glob.glob(str(RUNS_DIR / "train*")) if os.path.isdir(p)],
                       key=os.path.getmtime, reverse=True)
        return cands[0] if cands else str(RUNS_DIR)

    def actually_open(path_str: str):
        p = Path(path_str).resolve()
        if not p.exists():
            p = Path(find_latest_run_dir()).resolve()
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))
            elif sys.platform.startswith("darwin"):
                os.system(f'open "{p}"')
            else:
                os.system(f'xdg-open "{p}"')
        except Exception as e:
            messagebox.showinfo("정보", f"{p}\n{e}")

    def open_runs():
        target = last_run_dir["path"] or find_latest_run_dir()
        Path(target).mkdir(parents=True, exist_ok=True)
        actually_open(target)

    def set_enabled(enabled: bool):
        state = "normal" if enabled else "disabled"
        for w in (btn_start,):
            try: w.config(state=state)
            except Exception: pass

    def start_training():
        tabidx = source_nb.index(source_nb.select())
        mode = "roboflow" if tabidx == 0 else "local"

        ver_val = model_ver_var.get().strip() or "11"
        size_val = model_sz_var.get().strip() or "n"

        requested_workers = int(workers_var.get())
        workers_for_cfg = 0 if IS_FROZEN else requested_workers

        cfg = {
            "mode": mode,
            "rf_api_key": rf_api_var.get().strip(),
            "rf_workspace": rf_ws_var.get().strip(),
            "rf_project": rf_proj_var.get().strip(),
            "rf_version": rf_ver_var.get(),
            "force_download": bool(rf_force_var.get()),
            "data_yaml": data_yaml_var.get().strip(),
            "data_root": data_root_var.get().strip(),
            "model_version": ver_val,
            "model_size": size_val,
            "epochs": int(epochs_var.get()),
            "imgsz": int(imgsz_var.get()),
            "batch": int(batch_var.get()),
            "device": device_var.get(),
            "patience": int(patience_var.get()),
            "workers": workers_for_cfg,
            "cache": bool(cache_var.get()),
            "delete_dataset": bool(delete_ds_var.get()),
        }

        if mode == "roboflow":
            if not (cfg["rf_api_key"] and cfg["rf_workspace"] and cfg["rf_project"] and int(cfg["rf_version"]) >= 1):
                messagebox.showerror("오류", "Roboflow: API Key / Workspace / Project / Version을 입력하세요.")
                return
        else:
            if not (cfg["data_yaml"] or cfg["data_root"]):
                messagebox.showerror("오류", "로컬 모드: data.yaml 또는 검색 폴더를 지정하세요.")
                return

        btn_open.config(state="disabled")
        pb["value"] = 0
        status_var.set("초기화 중…")
        mAP_var.set("—"); mAP95_var.set("—"); P_var.set("—"); R_var.set("—")
        txt.delete("1.0", tk.END)
        log("== 학습 시작 ==")
        set_enabled(False)

        t = threading.Thread(target=train_worker,
                             args=(cfg, progress_q, status_q, done_q, error_q, metrics_q, runpath_q),
                             daemon=True)
        t.start()
        root.after(120, poll)

    def poll():
        while not progress_q.empty():
            v = progress_q.get_nowait()
            try: v = int(v)
            except Exception: v = 0
            pb["value"] = max(0, min(100, v))

        while not status_q.empty():
            s = status_q.get_nowait()
            status_var.set(s)
            txt.insert(tk.END, s + "\n"); txt.see(tk.END)

        if not error_q.empty():
            err = error_q.get_nowait()
            messagebox.showerror("오류", err)
            txt.insert(tk.END, f"[ERROR] {err}\n"); txt.see(tk.END)

        while not runpath_q.empty():
            rp = runpath_q.get_nowait()
            if rp and os.path.isdir(rp):
                last_run_dir["path"] = rp
                btn_open.config(state="normal")

        if not done_q.empty():
            btn_open.config(state="normal")
            set_enabled(True)
            status_var.set(f"{status_var.get()}  (완료)")
            if not metrics_q.empty():
                info = metrics_q.get_nowait()
                summ = info.get("summary", {})
                m50   = summ.get("mAP50"); m5095 = summ.get("mAP5095"); p = summ.get("precision"); r = summ.get("recall")
                mAP_var.set(f"{m50:.4f}" if isinstance(m50, float) else "N/A")
                mAP95_var.set(f"{m5095:.4f}" if isinstance(m5095, float) else "N/A")
                P_var.set(f"{p:.4f}" if isinstance(p, float) else "N/A")
                R_var.set(f"{r:.4f}" if isinstance(r, float) else "N/A")
            return

        root.after(120, poll)

    btn_start.config(command=start_training)
    btn_open.config(command=open_runs)
    root.mainloop()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    run_ui()
