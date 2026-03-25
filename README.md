# Edge-AI Smart Surveillance

Jetson AGX Orin 기반의 Edge AI 스마트 감시 시스템 프로젝트입니다.  
본 프로젝트는 먼저 YOLOv8 TensorRT 엔진의 성능을 비교·분석하기 위한 벤치마크를 수행하고,  
그 결과를 바탕으로 실시간 객체 탐지와 시스템 모니터링이 가능한 웹 기반 관제 시스템을 구현하는 것을 목표로 하였습니다.

이후 확장 버전에서는 Multi-ROI 기반 관심영역 필터링과 구역별 시각화 기능을 추가하여  
실제 관제 환경에서 활용할 수 있는 형태로 발전시켰습니다.

## Project Flow

### 1. Benchmark for Model Selection
실시간 감시 시스템에 적합한 추론 모델을 선정하기 위해 Jetson 환경에서 YOLOv8 TensorRT 엔진의 성능을 비교하였습니다.

- YOLOv8n / s / m / l 모델 비교
- FP16 / INT8 정밀도 비교
- FPS, Inference Time, Latency, Precision, Recall, mAP, F1 Score 측정
- OpenCV-CUDA 및 DeepStream 환경 성능 비교

### 2. Web-based Surveillance System
벤치마크를 통해 확인한 추론 성능을 기반으로, 실시간 객체 탐지 결과를 확인할 수 있는 웹 관제 시스템을 구현하였습니다.

- 실시간 비디오 스트리밍
- 객체 수 / FPS / 탐지 결과 표시
- 모델 엔진 동적 교체
- Confidence / IoU Threshold 조절
- CPU / RAM / GPU 상태 모니터링

### 3. System Extension (v2)
기본 관제 시스템을 확장하여, 특정 관심 영역만 탐지할 수 있는 Multi-ROI 기능과 구역별 통계 UI를 추가하였습니다.

- 다중 ROI 설정 및 저장
- 관심영역 기반 객체 필터링
- 구역별 탐지 현황 시각화
- 실사용 중심 대시보드 확장
