import sys
import threading
import time
import math
import os
import shutil
import glob  
import configparser
import psutil
import uvicorn
from collections import OrderedDict
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import gi
import pyds

# --- jtop 엔진을 파이썬으로 불러오기 ---
try:
    from jtop import jtop
    jetson = jtop()
    jetson.start()
    HAS_JTOP = True
except ImportError:
    HAS_JTOP = False
    print("⚠️ jetson-stats 모듈이 없습니다. 'pip install jetson-stats'를 권장합니다.")

# 시스템 플러그인 경로 설정
os.environ["GST_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/gstreamer-1.0"

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Gst 초기화
Gst.init(None)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- 전역 변수 ---
frame_buffer = None
buffer_lock = threading.Lock()
detected_objects_data = [] 
count_lock = threading.Lock()
pipeline = None
loop = None
restart_event = threading.Event()

current_fps = 0.0
last_frame_time = 0.0

# [중요] 영상 재생 위치 기억
last_video_position = 0 

# --- 무한 추적 트래커 ---
class SmartTracker:
    def __init__(self, max_disappeared=600, max_distance=2000):
        self.nextObjectID = 1
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        inputCentroids = []
        rect_map = [] 

        for (x, y, w, h, obj_meta) in rects:
            cX = int((x + x + w) / 2.0)
            cY = int((y + y + h) / 2.0)
            inputCentroids.append((cX, cY))
            rect_map.append((x, y, w, h, obj_meta))

        if len(inputCentroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return []

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = []
            for i in range(len(objectCentroids)):
                row = []
                for j in range(len(inputCentroids)):
                    dist = math.hypot(objectCentroids[i][0] - inputCentroids[j][0],
                                      objectCentroids[i][1] - inputCentroids[j][1])
                    row.append(dist)
                D.append(row)

            usedRows = set()
            usedCols = set()
            matches = []
            for i in range(len(D)):
                for j in range(len(D[i])):
                    matches.append((D[i][j], i, j))
            
            matches.sort(key=lambda x: x[0])

            for (dist, row, col) in matches:
                if row in usedRows or col in usedCols:
                    continue
                if dist > self.max_distance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            for row in range(len(objectIDs)):
                if row not in usedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.max_disappeared:
                        self.deregister(objectID)

            for col in range(len(inputCentroids)):
                if col not in usedCols:
                    self.register(inputCentroids[col])

        objects_bbs_ids = []
        for i, (cX, cY) in enumerate(inputCentroids):
            matched_id = -1
            for objID, centroid in self.objects.items():
                if centroid == (cX, cY):
                    matched_id = objID
                    break
            
            if matched_id != -1:
                (x, y, w, h, obj_meta) = rect_map[i]
                objects_bbs_ids.append((x, y, w, h, matched_id, obj_meta))

        return objects_bbs_ids

tracker = SmartTracker(max_disappeared=600, max_distance=2000)

# --- Config 관리 ---
CONFIG_FILE = "config_yolo.txt"

def read_config():
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(CONFIG_FILE)
    return config

def save_config_file(config):
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

# --- GStreamer 핵심 엔진 ---
def start_pipeline():
    global pipeline, loop, last_frame_time, tracker, last_video_position
    
    tracker = SmartTracker(max_disappeared=600, max_distance=2000)
    last_frame_time = time.time()

    pipeline_str = """
    filesrc location=./video/test1.mp4 ! 
    qtdemux ! h264parse ! nvv4l2decoder ! 
    m.sink_0 nvstreammux name=m batch-size=1 width=1280 height=720 ! 
    nvinfer config-file-path=./config_yolo.txt name=nvinfer ! 
    nvdsosd ! 
    nvvidconv ! video/x-raw(memory:NVMM), format=I420 ! 
    nvjpegenc ! 
    appsink name=mysink emit-signals=true sync=true max-buffers=1 drop=true
    """
    
    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        print(f"❌ 파이프라인 생성 에러: {e}")
        return

    appsink = pipeline.get_by_name("mysink")
    appsink.connect("new-sample", on_new_sample)

    nvinfer = pipeline.get_by_name("nvinfer")
    if nvinfer:
        src_pad = nvinfer.get_static_pad("src")
        src_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, None)

    pipeline.set_state(Gst.State.PAUSED)
    pipeline.get_state(Gst.CLOCK_TIME_NONE)
    
    if last_video_position > 0:
        pipeline.seek_simple(
            Gst.Format.TIME,
            Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
            last_video_position
        )
        print(f"⏩ 이전 시점({last_video_position / 1e9:.2f}초)으로 이동 완료!")

    pipeline.set_state(Gst.State.PLAYING)
    loop = GLib.MainLoop()
    try:
        loop.run()
    except:
        pass

def stop_pipeline():
    global pipeline, loop
    if pipeline:
        pipeline.set_state(Gst.State.NULL)
        pipeline = None
    if loop:
        loop.quit()

def gstreamer_manager():
    while True:
        t = threading.Thread(target=start_pipeline)
        t.start()
        restart_event.wait()
        stop_pipeline()
        t.join() 
        time.sleep(2.0)
        restart_event.clear()

def osd_sink_pad_buffer_probe(pad, info, u_data):
    global detected_objects_data, current_fps, last_frame_time, tracker
    
    now = time.time()
    dt = now - last_frame_time
    if dt > 0:
        current_fps = (0.9 * current_fps) + (0.1 * (1.0 / dt))
    last_frame_time = now

    gst_buffer = info.get_buffer()
    if not gst_buffer: return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    detections = []
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                    rect = obj.rect_params
                    detections.append((int(rect.left), int(rect.top), int(rect.width), int(rect.height), obj))
                except StopIteration: break
                l_obj = l_obj.next
        except StopIteration: break
        l_frame = l_frame.next

    tracked = tracker.update(detections)
    
    temp_data_list = []
    for (x, y, w, h, obj_id, obj_meta) in tracked:
        conf = obj_meta.confidence
        obj_meta.text_params.display_text = f"ID {obj_id} : {conf:.2f}"
        
        obj_meta.text_params.font_params.font_name = "Serif"
        obj_meta.text_params.font_params.font_size = 11
        obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        obj_meta.text_params.set_bg_clr = 1
        obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)

        temp_data_list.append({
            "id": obj_id,
            "conf": float(f"{conf:.2f}"),
            "bbox": [x, y, w, h]
        })
    
    with count_lock:
        detected_objects_data = temp_data_list
        
    return Gst.PadProbeReturn.OK

def on_new_sample(sink):
    global frame_buffer
    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    success, map_info = buf.map(Gst.MapFlags.READ)
    if not success: return Gst.FlowReturn.ERROR
    try:
        with buffer_lock:
            frame_buffer = bytes(map_info.data)
    finally:
        buf.unmap(map_info)
    return Gst.FlowReturn.OK

# --- FastAPI 엔드포인트 ---
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    config = read_config()
    conf = config.get('class-attrs-all', 'pre-cluster-threshold', fallback='0.25')
    iou = config.get('class-attrs-all', 'nms-iou-threshold', fallback='0.45')
    
    # [수정됨] .engine 확장자를 가진 파일만 골라서 리스트에 담기
    if os.path.exists("./models"):
        models = [f for f in os.listdir("./models") if f.endswith('.engine')]
    else:
        models = []
        
    current_model = config.get('property', 'model-engine-file', fallback='Unknown').split('/')[-1]
    return templates.TemplateResponse("index.html", {
        "request": request, "conf": conf, "iou": iou, "current_model": current_model, "models": models
    })

@app.get("/data")
def get_data():
    with count_lock:
        return JSONResponse(content={
            "count": len(detected_objects_data), 
            "fps": round(current_fps, 1),
            "objects": detected_objects_data
        })

@app.post("/api/save_settings")
async def save_settings(model_file: str = Form(...), conf_thresh: str = Form(...), iou_thresh: str = Form(...)):
    global pipeline, last_video_position
    
    if pipeline:
        try:
            success, pos = pipeline.query_position(Gst.Format.TIME)
            if success:
                last_video_position = pos
        except Exception:
            pass

    config = read_config()
    if not config.has_section('property'): config.add_section('property')
    config.set('property', 'model-engine-file', f"./models/{model_file}")
    if not config.has_section('class-attrs-all'): config.add_section('class-attrs-all')
    config.set('class-attrs-all', 'pre-cluster-threshold', conf_thresh)
    config.set('class-attrs-all', 'nms-iou-threshold', iou_thresh)
    save_config_file(config)
    
    restart_event.set()
    return JSONResponse(content={"status": "success", "message": "설정이 저장되었습니다."})

def get_jetson_gpu_info():
    gpu_usage = 0.0
    gpu_temp = 0.0

    if HAS_JTOP and jetson.ok():
        gpu_usage = float(jetson.stats.get('GPU', 0.0))
        try:
            temps = jetson.temperature
            if temps:
                if 'GPU' in temps:
                    val = temps['GPU']
                    gpu_temp = val['temp'] if isinstance(val, dict) else float(val)
                elif 'gpu' in temps:
                    val = temps['gpu']
                    gpu_temp = val['temp'] if isinstance(val, dict) else float(val)
        except Exception:
            pass
        return gpu_usage, gpu_temp
        
    try:
        search_paths = [
            "/sys/devices/platform/17000000.gpu/load",
            "/sys/devices/platform/17000000.ga10b/load",
            "/sys/class/devfreq/*/device/load",
            "/sys/devices/gpu.0/load"
        ]
        for pattern in search_paths:
            for path in glob.glob(pattern):
                try:
                    with open(path, "r") as f:
                        val = f.read().strip()
                        if val:
                            gpu_usage = round(float(val) / 10.0, 1)
                            break
                except:
                    continue
    except Exception:
        pass

    try:
        for path in glob.glob("/sys/class/thermal/thermal_zone*"):
            try:
                with open(os.path.join(path, "type"), "r") as f:
                    type_name = f.read().strip().lower()
                if "gpu" in type_name:
                    with open(os.path.join(path, "temp"), "r") as f:
                        temp_val = float(f.read().strip())
                        gpu_temp = round(temp_val / 1000.0, 1)
                        break
            except:
                continue
    except Exception:
        pass

    return gpu_usage, gpu_temp

@app.get("/api/system_stats")
def system_stats():
    mem = psutil.virtual_memory()
    gpu_usage, gpu_temp = get_jetson_gpu_info()
    return {
        "cpu": psutil.cpu_percent(),
        "ram_percent": mem.percent,
        "ram_used": round(mem.used / (1024**3), 2),
        "ram_total": round(mem.total / (1024**3), 2),
        "gpu": gpu_usage,
        "gpu_temp": gpu_temp 
    }

def generate_frames():
    while True:
        with buffer_lock:
            if frame_buffer:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_buffer + b'\r\n')
        time.sleep(0.01)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    t_manager = threading.Thread(target=gstreamer_manager, daemon=True)
    t_manager.start()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
