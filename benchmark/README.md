##  YOLOv8(*.engine) OpenCV-CUDA Benchmark Results
**Tested on NVIDIA Jetson AGX Orin Developer Kit**

### - Single Core Performance -

#### **FP16 Precision**
| Model | Quantization | Input Size | mAP(50-95) | mAP(50) | F1 Score | Precision | Recall | FPS | Avg Inference (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** | FP16 | 640x640 | 0.6014 | 0.8802 | 0.8137 | 0.8183 | 0.8091 | 34.35 | 6.88 |
| **YOLOv8s** | FP16 | 640x640 | 0.6379 | 0.8929 | 0.8277 | 0.8335 | 0.8220 | 27.64 | 11.27 |
| **YOLOv8m** | FP16 | 640x640 | 0.6525 | 0.8982 | 0.8323 | 0.8336 | 0.8310 | 16.58 | 24.72 |
| **YOLOv8l** | FP16 | 640x640 | 0.6594 | 0.8981 | 0.8319 | 0.8371 | 0.8268 | 15.32 | 30.28 |

#### **INT8 Precision**
| Model | Quantization | Input Size | mAP(50-95) | mAP(50) | F1 Score | Precision | Recall | FPS | Avg Inference (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** | INT8 | 640x640 | 0.5582 | 0.8730 | 0.8057 | 0.8091 | 0.8023 | 35.54 | 5.75 |
| **YOLOv8s** | INT8 | 640x640 | 0.4864 | 0.8462 | 0.7945 | 0.7878 | 0.8012 | 32.68 | 7.97 |
| **YOLOv8m** | INT8 | 640x640 | 0.3949 | 0.7865 | 0.7534 | 0.7235 | 0.7858 | 20.85 | 16.85 |
| **YOLOv8l** | INT8 | 640x640 | 0.3904 | 0.7608 | 0.7360 | 0.6877 | 0.7917 | 16.87 | 22.64 |

---
### - Multi Core [12] Performance -

#### **FP16 Precision**
| Model | Quantization | Input Size | mAP(50-95) | mAP(50) | F1 Score | Precision | Recall | FPS | Avg Inference (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** | FP16 | 640x640 | 0.6014 | 0.8802 | 0.8137 | 0.8183 | 0.8091 | 168.95 | 2.35 |
| **YOLOv8s** | FP16 | 640x640 | 0.6379 | 0.8929 | 0.8277 | 0.8335 | 0.8220 | 147.14 | 3.3 |
| **YOLOv8m** | FP16 | 640x640 | 0.6525 | 0.8982 | 0.8323 | 0.8336 | 0.8310 | 92.38 | 7.07 |
| **YOLOv8l** | FP16 | 640x640 | 0.6594 | 0.8981 | 0.8319 | 0.8371 | 0.8268 | 67.75 | 10.85 |

#### **INT8 Precision**
| Model | Quantization | Input Size | mAP(50-95) | mAP(50) | F1 Score | Precision | Recall | FPS | Avg Inference (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** | INT8 | 640x640 | 0.5582 | 0.8730 | 0.8057 | 0.8091 | 0.8023 | 180.7 | 1.84 |
| **YOLOv8s** | INT8 | 640x640 | 0.4864 | 0.8462 | 0.7945 | 0.7878 | 0.8012 | 163.88 | 2.48 |
| **YOLOv8m** | INT8 | 640x640 | 0.3949 | 0.7865 | 0.7534 | 0.7235 | 0.7858 | 114.94 | 4.93 |
| **YOLOv8l** | INT8 | 640x640 | 0.3904 | 0.7608 | 0.7360 | 0.6877 | 0.7917 | 96.88 | 6.53 |

---
### 🎥 YOLOv8(&.engine) DeepStream Benchmark Result
**Tested on Jetson Orin Nano Super (aarch64)**

| Model | FP32 | FP16 | INT8 |
| :--- | :---: | :---: | :---: |
| **YOLOv8n** | 120.9 | 211.24 | 232.16 |
| **YOLOv8s** | 67.5 | 123.87 | 184.18 |
| **YOLOv8m** | 31.61 | 63.45 | 96.5 |
| **YOLOv8l** | 20.53 | 40.1 | 70.01 |

> **Environment Details**
> - **Platform**: Jetson Orin Nano Super (aarch64)
> - **OS**: Ubuntu 22.04 Jetpack 6.1
> - **Date**: 2025-12-10
> - **Settings**: Conf = 0.2 / IoU = 0.5
