## 📊 YOLOv8(*.engine) OpenCV-CUDA Benchmark Results
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

> **Note**: Benchmark results may vary depending on system power mode (MAXN recommended) and background processes.
