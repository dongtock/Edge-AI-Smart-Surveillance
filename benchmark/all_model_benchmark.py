import time
import json
import glob
import os
import gc
import torch
import numpy as np
from ultralytics import YOLO

# --------------------------
# 모델 정보 파싱 함수 (파일명 기반)
# --------------------------
def parse_model_info(filename):
    name_lower = filename.lower()
    
    # 1. 사이즈 파악
    size = "Unknown"
    if 'yolov8n' in name_lower: size = "Nano"
    elif 'yolov8s' in name_lower: size = "Small"
    elif 'yolov8m' in name_lower: size = "Medium"
    elif 'yolov8l' in name_lower: size = "Large"
    elif 'yolov8x' in name_lower: size = "XLarge"
    
    # 2. 정밀도(Precision) 파악
    precision = "FP32" # 기본값
    if 'int8' in name_lower: precision = "INT8"
    elif 'fp16' in name_lower: precision = "FP16"
    
    return size, precision

# --------------------------
# 단일 모델 벤치마크 함수
# --------------------------
def run_single_benchmark(model_path, data_yaml, num_workers, img_size=640, device='cuda:0'):
    filename = os.path.basename(model_path)
    model_size, model_precision = parse_model_info(filename)
    
    print(f"\n>>> Processing: {filename}")
    print(f"    [Type: {model_size} | Precision: {model_precision}]")

    # 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()

    try:
        model = YOLO(model_path, task='detect')
    except Exception as e:
        print(f"Error loading model {filename}: {e}")
        return None

    # 웜업
    print("   -> Warming up...")
    try:
        model(np.zeros((img_size, img_size, 3), dtype=np.uint8), verbose=False)
    except:
        pass # 웜업 실패해도 진행 시도

    # 검증 수행
    print("   -> Running Validation...")
    try:
        metrics = model.val(
            data=data_yaml, 
            imgsz=img_size, 
            device=device, 
            verbose=False, 
            batch=1, 
            half=True, # TensorRT 엔진은 로드될 때 이미 정밀도가 정해져 있으나, 파이프라인 최적화를 위해 유지
            workers=num_workers
        )
    except Exception as e:
        print(f"Error validating {filename}: {e}")
        return None

    # 수치 추출
    precision_val = metrics.box.mp
    recall_val = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    
    f1_score = 0.0
    if (precision_val + recall_val) > 0:
        f1_score = 2 * (precision_val * recall_val) / (precision_val + recall_val)

    # 속도 (ms -> fps)
    inference_time_ms = metrics.speed['inference']
    total_time_ms = sum(metrics.speed.values())
    fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0

    # 결과 정리
    results = {
        "filename": filename,
        "info": {
            "size": model_size,       # Nano, Small, Medium, Large
            "precision": model_precision # INT8, FP16
        },
        "metrics": {
            "f1_score": round(f1_score, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "precision": round(precision_val, 4),
            "recall": round(recall_val, 4)
        },
        "performance": {
            "fps": round(fps, 2),
            "inference_ms": round(inference_time_ms, 2),
            "total_latency_ms": round(total_time_ms, 2)
        }
    }
    
    del model
    return results

# --------------------------
# 메인 실행
# --------------------------
if __name__ == "__main__":
    MODELS_DIR = "/home/riseagx01/works/models"   
    DATA_YAML = "/home/riseagx01/works/datasets/data.yaml"
    
    detected_cores = os.cpu_count() or 4
    
    # 파일 검색
    engine_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.engine")))
    
    if not engine_files:
        print(f"No .engine files found in {MODELS_DIR}")
        exit()

    print(f"Found {len(engine_files)} engines. Sorting by size...")
    
    # (선택사항) 보기 좋게 정렬: N -> S -> M -> L 순서, 그다음 FP16 -> INT8
    # 파일명에 n, s, m, l이 포함되어 있다고 가정
    def sort_key(name):
        order = {'n': 0, 's': 1, 'm': 2, 'l': 3, 'x': 4}
        p_order = {'fp16': 0, 'int8': 1}
        
        name_lower = name.lower()
        s_score = 5
        for k, v in order.items():
            if f"yolov8{k}" in name_lower:
                s_score = v
                break
        
        p_score = 1 if 'int8' in name_lower else 0
        return (s_score, p_score)

    engine_files.sort(key=lambda x: sort_key(os.path.basename(x)))

    all_results = []
    
    for i, engine_path in enumerate(engine_files):
        if i > 0:
            time.sleep(3) # 모델 간 간섭 방지 쿨다운
            
        res = run_single_benchmark(engine_path, DATA_YAML, num_workers=detected_cores)
        
        if res:
            all_results.append(res)
            # 중간 저장
            with open("benchmark_results.json", "w") as f:
                json.dump(all_results, f, indent=4)

    print("\nBenchmark Completed. Check 'benchmark_results.json'.")
