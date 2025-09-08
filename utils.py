import os
import glob

def find_latest_best(runs_dir="runs/detect"):
    """
    (3) 가장 최근 학습의 best.pt 경로 반환
    """
    paths = glob.glob(os.path.join(runs_dir, "train*", "weights", "best.pt"))
    if not paths:
        return None
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths[0]

def find_data_yaml(preferred_base=None):
    """
    (4) data.yaml / data.yml 자동 탐색
    - preferred_base: Roboflow가 내려준 dataset.location 등 우선 검색 루트
    - 그 외에도 프로젝트 내 일반적인 위치를 폭넓게 스캔
    """
    candidates = []

    search_roots = []
    if preferred_base and os.path.isdir(preferred_base):
        search_roots.append(preferred_base)

    
    search_roots += [
        ".",             
        "./data",        
    ]

    patterns = ["**/data.yaml", "**/data.yml"]
    for root in search_roots:
        for pat in patterns:
            candidates += glob.glob(os.path.join(root, pat), recursive=True)

    
    candidates = sorted(set(candidates), key=os.path.getmtime, reverse=True)

    if candidates:
        print("Found data files:")
        for i, p in enumerate(candidates[:10], 1):
            print(f"{i:2d}. {p}")
        return candidates[0]

    return None
