"""
FastAPI WebSocket 서버
- 클라이언트가 인코딩된 프레임 바이트(JPEG/PNG 등)를 보내면
  YOLO Pose 추론 + 6DoF 시각화를 입혀 JPEG 바이트로 즉시 반환.
- 가중치(best.pt)는 서버 시작 시 1회 로드.
- 카메라 매트릭스는 연결마다 첫 프레임 해상도로 자동 초기화.
"""

import asyncio
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

from structured_data import (
    ALL_CLASSES,
    CONF,
    DIST_COEFFS,
    IMG_SIZE,
    PNP_CLASSES,
    draw_hud,
    draw_legend,
    draw_predictions,
    init_camera_matrix,
)

WEIGHT_PATH = str(Path(__file__).parent / "best.pt")
JPEG_QUALITY = 85
HOST = "0.0.0.0"
PORT = 8000

MODEL: YOLO | None = None
MODEL_NAMES: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, MODEL_NAMES
    print(f"[startup] Loading weights: {WEIGHT_PATH}")
    MODEL = YOLO(WEIGHT_PATH)
    MODEL_NAMES = MODEL.names
    print(f"[startup] Model classes: {list(MODEL_NAMES.values())}")
    print(f"[startup] CONF>={CONF}  IMG_SIZE={IMG_SIZE}  PnP={PNP_CLASSES}")
    yield
    print("[shutdown] bye")


app = FastAPI(lifespan=lifespan, title="YOLO Pose + 6DoF Stream")


def _infer_and_render(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    frame_idx: int,
    fps_now: float,
) -> bytes | None:
    result = MODEL.predict(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]
    img_out, n_drawn, n_pnp = draw_predictions(
        frame, result, MODEL_NAMES, ALL_CLASSES, camera_matrix, DIST_COEFFS,
    )
    if n_drawn > 0:
        clss = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        cnt = Counter(MODEL_NAMES[int(c)] for c in clss)
        cls_str = " ".join(f"{k}={v}" for k, v in cnt.items())
        print(
            f"[detect] frame={frame_idx} n={n_drawn} pnp={n_pnp} "
            f"maxConf={float(confs.max()):.2f} | {cls_str}"
        )
    img_out = draw_legend(img_out)
    img_out = draw_hud(img_out, frame_idx, 0, fps_now, n_drawn, n_pnp, CONF)
    ok, buf = cv2.imencode(".jpg", img_out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


@app.websocket("/ws/infer")
async def ws_infer(ws: WebSocket):
    await ws.accept()
    camera_matrix: np.ndarray | None = None
    frame_idx = 0
    fps_now = 0.0
    t_prev = time.time()
    try:
        while True:
            data = await ws.receive_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_idx += 1
            if camera_matrix is None:
                h, w = frame.shape[:2]
                camera_matrix = init_camera_matrix(w, h)
                print(f"[ws] calibrated K from first frame: {w}x{h}")

            t_now = time.time()
            inst = 1.0 / max(t_now - t_prev, 1e-6)
            fps_now = inst if fps_now == 0 else (0.9 * fps_now + 0.1 * inst)
            t_prev = t_now

            jpeg = await asyncio.to_thread(
                _infer_and_render, frame.copy(), camera_matrix, frame_idx, fps_now,
            )
            if jpeg is None:
                continue
            await ws.send_bytes(jpeg)
    except WebSocketDisconnect:
        print(f"[ws] client disconnected after {frame_idx} frames")
    except Exception as e:
        print(f"[ws] error after {frame_idx} frames: {e!r}")
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<html><head><meta charset="utf-8"><title>YOLO Pose Stream</title></head>
<body style="font-family:sans-serif;background:#111;color:#eee;margin:0;padding:16px;">
<h3>WebSocket test client</h3>
<input id="file" type="file" accept="video/*">
<button id="start">Start</button>
<button id="stop">Stop</button>
<div style="margin-top:8px;">
  <img id="view" style="max-width:100%;border:1px solid #444;background:#000">
</div>
<script>
let ws=null, video=null, raf=null, canvas=document.createElement('canvas');
const view=document.getElementById('view');
const startBtn=document.getElementById('start');
const stopBtn=document.getElementById('stop');

function stop(){
  if(raf) cancelAnimationFrame(raf);
  if(ws){ ws.close(); ws=null; }
  if(video){ video.pause(); video.src=''; video=null; }
}
stopBtn.onclick=stop;

startBtn.onclick=async()=>{
  const f=document.getElementById('file').files[0];
  if(!f){ alert('select a video'); return; }
  stop();
  video=document.createElement('video');
  video.src=URL.createObjectURL(f);
  video.muted=true; video.playsInline=true;
  await video.play();
  canvas.width=video.videoWidth; canvas.height=video.videoHeight;
  const ctx=canvas.getContext('2d');

  ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws/infer');
  ws.binaryType='blob';
  ws.onmessage=(ev)=>{ view.src=URL.createObjectURL(ev.data); };
  ws.onopen=()=>{
    let sending=false;
    const tick=async()=>{
      if(!ws||ws.readyState!==1||video.ended) return;
      if(!sending){
        sending=true;
        ctx.drawImage(video,0,0,canvas.width,canvas.height);
        canvas.toBlob(async(b)=>{
          if(b && ws && ws.readyState===1){
            const buf=await b.arrayBuffer();
            ws.send(buf);
          }
          sending=false;
        },'image/jpeg',0.85);
      }
      raf=requestAnimationFrame(tick);
    };
    tick();
  };
};
</script>
</body></html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
