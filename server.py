"""
FastAPI WebSocket 서버
- 클라이언트가 인코딩된 프레임 바이트(JPEG/PNG 등)를 보내면
  YOLO Pose / Segmentation 추론 결과를 입혀 JPEG 바이트로 즉시 반환.
- 가중치(best.pt, best_seg.pt)는 서버 시작 시 1회 로드.
- 카메라 매트릭스/UI 스케일은 연결마다 첫 프레임 해상도로 자동 초기화.

엔드포인트:
  GET  /          HTML 테스트 클라이언트
  GET  /health    헬스체크
  WS   /ws/pose   YOLO Pose + 6DoF
  WS   /ws/mask   YOLO Segmentation
  WS   /ws/all    Pose + Segmentation 결합
"""

import asyncio
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable

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
from mask import (
    IOU as SEG_IOU,
    draw_predictions as draw_predictions_seg,
)

WEIGHT_PATH = str(Path(__file__).parent / "best.pt")
SEG_WEIGHT_PATH = str(Path(__file__).parent / "best_seg2.pt")
JPEG_QUALITY = 85
HOST = "0.0.0.0"
PORT = 8000

MODEL: YOLO | None = None
MODEL_NAMES: dict | None = None
SEG_MODEL: YOLO | None = None
SEG_MODEL_NAMES: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, MODEL_NAMES, SEG_MODEL, SEG_MODEL_NAMES
    print(f"[startup] Loading pose weights: {WEIGHT_PATH}")
    MODEL = YOLO(WEIGHT_PATH)
    MODEL_NAMES = MODEL.names
    print(f"[startup] Pose classes: {list(MODEL_NAMES.values())}")
    print(f"[startup] Loading seg weights: {SEG_WEIGHT_PATH}")
    SEG_MODEL = YOLO(SEG_WEIGHT_PATH)
    SEG_MODEL_NAMES = SEG_MODEL.names
    print(f"[startup] Seg classes: {list(SEG_MODEL_NAMES.values())}")
    print(f"[startup] CONF>={CONF}  IMG_SIZE={IMG_SIZE}  PnP={PNP_CLASSES}")
    yield
    print("[shutdown] bye")


app = FastAPI(lifespan=lifespan, title="YOLO Pose + Seg Stream")


# ----------------------------------------------------------------------------
# 공통 WebSocket 스트림 루프
# ----------------------------------------------------------------------------
InitState = Callable[[int, int], dict]
RenderFn = Callable[[np.ndarray, dict, int, float], Awaitable[bytes | None]]


async def _ws_stream(
    ws: WebSocket,
    tag: str,
    init_state: InitState,
    render: RenderFn,
) -> None:
    """프레임 수신 → 디코딩 → render() → 바이트 송신 루프."""
    await ws.accept()
    state: dict | None = None
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
            if state is None:
                h, w = frame.shape[:2]
                state = init_state(w, h)
                print(f"[{tag}] init from first frame {w}x{h}: {state}")

            t_now = time.time()
            inst = 1.0 / max(t_now - t_prev, 1e-6)
            fps_now = inst if fps_now == 0 else (0.9 * fps_now + 0.1 * inst)
            t_prev = t_now

            jpeg = await render(frame, state, frame_idx, fps_now)
            if jpeg is None:
                continue
            await ws.send_bytes(jpeg)
    except WebSocketDisconnect:
        print(f"[{tag}] client disconnected after {frame_idx} frames")
    except Exception as e:
        print(f"[{tag}] error after {frame_idx} frames: {e!r}")
        try:
            await ws.close()
        except Exception:
            pass


def _encode_jpeg(img: np.ndarray) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


def _log_detections(tag: str, names: dict, result: Any, frame_idx: int, extra: str = "") -> None:
    clss = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    cnt = Counter(names[int(c)] for c in clss)
    cls_str = " ".join(f"{k}={v}" for k, v in cnt.items())
    print(
        f"[{tag}] frame={frame_idx} n={len(clss)}{extra} "
        f"maxConf={float(confs.max()):.2f} | {cls_str}"
    )


# ----------------------------------------------------------------------------
# /ws/pose — YOLO Pose + 6DoF
# ----------------------------------------------------------------------------
def _render_pose(
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
        _log_detections("pose", MODEL_NAMES, result, frame_idx, extra=f" pnp={n_pnp}")
    img_out = draw_legend(img_out)
    img_out = draw_hud(img_out, frame_idx, 0, fps_now, n_drawn, n_pnp, CONF)
    return _encode_jpeg(img_out)


@app.websocket("/ws/pose")
async def ws_pose(ws: WebSocket):
    def _init(w: int, h: int) -> dict:
        return {"K": init_camera_matrix(w, h)}

    async def _render(frame, state, frame_idx, fps_now):
        return await asyncio.to_thread(
            _render_pose, frame.copy(), state["K"], frame_idx, fps_now,
        )

    await _ws_stream(ws, "ws/pose", _init, _render)


# ----------------------------------------------------------------------------
# /ws/mask — YOLO Segmentation
# ----------------------------------------------------------------------------
def _render_seg(
    frame: np.ndarray,
    frame_idx: int,
    ui_scale: float,
) -> bytes | None:
    result = SEG_MODEL.predict(
        frame, imgsz=IMG_SIZE, conf=CONF, iou=SEG_IOU, verbose=False,
    )[0]
    img_out, n_drawn = draw_predictions_seg(
        frame, result, SEG_MODEL_NAMES, ui_scale,
    )
    if n_drawn > 0:
        _log_detections("seg", SEG_MODEL_NAMES, result, frame_idx)
    return _encode_jpeg(img_out)


@app.websocket("/ws/mask")
async def ws_mask(ws: WebSocket):
    def _init(w: int, h: int) -> dict:
        return {"ui_scale": float(np.clip(w / 1280.0, 0.5, 1.0))}

    async def _render(frame, state, frame_idx, fps_now):
        return await asyncio.to_thread(
            _render_seg, frame.copy(), frame_idx, state["ui_scale"],
        )

    await _ws_stream(ws, "ws/mask", _init, _render)


# ----------------------------------------------------------------------------
# /ws/all — Pose + Seg 결합
# ----------------------------------------------------------------------------
def _render_combined(
    frame: np.ndarray,
    seg_result,
    pose_result,
    camera_matrix: np.ndarray,
    frame_idx: int,
    fps_now: float,
    ui_scale: float,
) -> bytes | None:
    # seg 마스크/박스 먼저 → 그 위에 pose 3D 박스/축/키포인트
    img_out, n_seg = draw_predictions_seg(
        frame, seg_result, SEG_MODEL_NAMES, ui_scale,
    )
    img_out, n_pose, n_pnp = draw_predictions(
        img_out, pose_result, MODEL_NAMES, ALL_CLASSES, camera_matrix, DIST_COEFFS,
    )
    if n_seg > 0:
        _log_detections("all/seg", SEG_MODEL_NAMES, seg_result, frame_idx)
    if n_pose > 0:
        _log_detections("all/pose", MODEL_NAMES, pose_result, frame_idx, extra=f" pnp={n_pnp}")
    img_out = draw_legend(img_out)
    img_out = draw_hud(img_out, frame_idx, 0, fps_now, n_seg + n_pose, n_pnp, CONF)
    return _encode_jpeg(img_out)


@app.websocket("/ws/all")
async def ws_all(ws: WebSocket):
    def _init(w: int, h: int) -> dict:
        return {
            "K": init_camera_matrix(w, h),
            "ui_scale": float(np.clip(w / 1280.0, 0.5, 1.0)),
        }

    async def _render(frame, state, frame_idx, fps_now):
        # 두 추론을 별도 스레드로 동시 실행 (PyTorch는 CUDA 호출 시 GIL 해제)
        seg_task = asyncio.to_thread(
            SEG_MODEL.predict, frame,
            imgsz=IMG_SIZE, conf=CONF, iou=SEG_IOU, verbose=False,
        )
        pose_task = asyncio.to_thread(
            MODEL.predict, frame,
            imgsz=IMG_SIZE, conf=CONF, verbose=False,
        )
        seg_results, pose_results = await asyncio.gather(seg_task, pose_task)
        return await asyncio.to_thread(
            _render_combined,
            frame.copy(), seg_results[0], pose_results[0],
            state["K"], frame_idx, fps_now, state["ui_scale"],
        )

    await _ws_stream(ws, "ws/all", _init, _render)


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pose_model_loaded": MODEL is not None,
        "seg_model_loaded": SEG_MODEL is not None,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<html><head><meta charset="utf-8"><title>YOLO Stream</title></head>
<body style="font-family:sans-serif;background:#111;color:#eee;margin:0;padding:16px;">
<h3>WebSocket test client</h3>
<input id="file" type="file" accept="video/*">
<select id="endpoint">
  <option value="/ws/pose">/ws/pose</option>
  <option value="/ws/mask">/ws/mask</option>
  <option value="/ws/all">/ws/all</option>
</select>
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
const endpointSel=document.getElementById('endpoint');

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

  const path=endpointSel.value;
  ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+path);
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
