import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time

# ============================================================
# 기본 환경 설정
# ============================================================
WEIGHT_PATH = 'C:/python_project/yolo_pose/pose_test_final_training/weights/best.pt'
VIDEO_PATH  = 'C:/python_project/yolo/test_video/test6.mp4'
SAVE_PATH   = 'C:/python_project/yolo_pose/YOLO_Pose_visual/output.mp4'

IMG_SIZE = 640 # 640 x 640 입력
CONF  = 0.7 # BBOX 신뢰성 설정
KPT_CONF_TH = 0.5 # Keypoint 신뢰성 설정

SHOW_WINDOW    = True # 결과 윈도우로 띄우기
SAVE_VIDEO     = False # 영상 저장 여부
FRAME_SKIP     = 1 # 프레임 스킵 -> 높일수록 배속
RESIZE_PREVIEW = 0.6 # 화면의 창 배율 (원래 해상도의 60%)
SHOW_KEYPOINTS = True # 특징점 보이게 설정

ALL_CLASSES = ['bollard', 'car', 'crosswalk', 'pole', 'scooter']
NO_KPT_CLASSES  = ['crosswalk']
PNP_CLASSES     = ['bollard', 'pole', 'scooter', 'car']

# ============================================================
# 카메라 캘리브레이션 - 임시값
# ============================================================
DEFAULT_FX = 1200.0 # x축 초점 거리
DEFAULT_FY = 1200.0 # y축 초점 거리
CAMERA_MATRIX = None
DIST_COEFFS   = np.zeros((5, 1), dtype=np.float32) # 렌즈 왜곡 계수 (현재 없다고 가정)

def init_camera_matrix(w, h): # 광학 중심을 정중앙(x/2, y/2)으로 설정
    return np.array([
        [DEFAULT_FX, 0,           w / 2.0],
        [0,          DEFAULT_FY,  h / 2.0],
        [0,          0,           1.0],
    ], dtype=np.float32)

# ============================================================
# 클래스별 3D 모델 하드코딩 (단위: meter)
# 좌표계: X=right(왼쪽은 음수), Y=up, Z=forward(뒷쪽은 음수)
# 전부 임의의 값이며 실제 측정 값 필요
# ============================================================

# Bollard 6점 - 기준점: 볼라드 정중앙 지면점에서 시작
BOLLARD_W = 0.10 # 직경 10cm
BOLLARD_H = 0.80 # 높이 80cm
BOLLARD_R = BOLLARD_W / 2 # 반지름

BOLLARD_3D = np.array([
    [-BOLLARD_W/2,  BOLLARD_H,  0        ],   # 0: TL - TOP LEFT
    [ BOLLARD_W/2,  BOLLARD_H,  0        ],   # 1: TR - TOP RIGHT
    [ BOLLARD_W/2,  0,          0        ],   # 2: BR - BOTTOM RIGHT
    [-BOLLARD_W/2,  0,          0        ],   # 3: BL - BOTTOM LEFT
    [ 0,            BOLLARD_H,  BOLLARD_R],   # 4: TC - TOP CENTER
    [ 0,            0,          BOLLARD_R],   # 5: BC - BOTTOM CENTER
], dtype=np.float32)

# Pole 6점 - 기준점: 기둥 정중앙 지면점에서 시작
POLE_W = 0.08
POLE_H = 2.50
POLE_R = POLE_W / 2

POLE_3D = np.array([
    [-POLE_W/2,  POLE_H,  0     ], # 0: TL
    [ POLE_W/2,  POLE_H,  0     ], # 1: TR
    [ POLE_W/2,  0,       0     ], # 2: BR
    [-POLE_W/2,  0,       0     ], # 3: BL
    [ 0,         POLE_H,  POLE_R], # 4: TC
    [ 0,         0,       POLE_R], # 5: BC
], dtype=np.float32)

# Scooter 4점 - 기준점: 앞바퀴 지면점에서 시작
SCOOTER_3D = np.array([
    [-0.275,  1.05,  0.00], # 핸들 좌
    [ 0.275,  1.05,  0.00], # 핸들 우
    [ 0.000,  0.18,  0.20], # 앞바퀴 센터
    [ 0.000,  0.18, -0.60], # 뒷바퀴 센터
], dtype=np.float32)

# ============================================================
# ✅ Car 6점 (Mean Vehicle Model: 세단 평균치)
#   라벨 순서:
#     0: 앞-좌 바퀴 접지점 (Front-Left  wheel ground contact)
#     1: 앞-우 바퀴 접지점 (Front-Right wheel ground contact)
#     2: 뒤-우 바퀴 접지점 (Rear-Right  wheel ground contact)
#     3: 뒤-좌 바퀴 접지점 (Rear-Left   wheel ground contact)
#     4: 전면 윈드실드 상단-좌 (A-pillar Top Left)
#     5: 전면 윈드실드 상단-우 (A-pillar Top Right)
#
#   평균 세단 치수: - AI 추천 임의값 사용
#     - 트레드(좌우 바퀴 간격): 1.55 m  →  ±0.775
#     - 휠베이스(앞뒤축 간격) : 2.70 m  →  앞축 Z=+1.35, 뒤축 Z=-1.35
#     - 윈드실드 상단 높이    : 1.35 m
#     - 윈드실드 상단 Z 위치  : +0.40 m (앞축보다 약간 뒤, 캐빈 시작점)
#     - 윈드실드 폭           : 1.40 m → ±0.70
# ============================================================
CAR_TREAD       = 1.55
CAR_WHEELBASE   = 2.70
CAR_WS_HEIGHT   = 1.35
CAR_WS_Z        = 0.40
CAR_WS_HALFW    = 0.70
CAR_BOX_HEIGHT = 1.45 # 자동차 3D 가상 박스 평균 높이

CAR_3D = np.array([ # 기준점 : 차량 정중앙 지면점
    [-CAR_TREAD/2,    0.0,           CAR_WHEELBASE/2],  # 0: FL wheel
    [ CAR_TREAD/2,    0.0,           CAR_WHEELBASE/2],  # 1: FR wheel
    [ CAR_TREAD/2,    0.0,          -CAR_WHEELBASE/2],  # 2: RR wheel
    [-CAR_TREAD/2,    0.0,          -CAR_WHEELBASE/2],  # 3: RL wheel
    [-CAR_WS_HALFW,   CAR_WS_HEIGHT, CAR_WS_Z       ],  # 4: A-pillar TL
    [ CAR_WS_HALFW,   CAR_WS_HEIGHT, CAR_WS_Z       ],  # 5: A-pillar TR
], dtype=np.float32)

CLASS_3D_MODEL = { # 모델 매핑
    'bollard': BOLLARD_3D,
    'pole'   : POLE_3D,
    'scooter': SCOOTER_3D,
    'car'    : CAR_3D,
}

# ============================================================
# 클래스별 시각화 및 색상 설정
# ============================================================
CLASS_CONFIG = {
    'bollard':   {'n_kpts': 6, 'labels': ['TL','TR','BR','BL','TC','BC'],
                  'box_color': (0,165,255),   'draw_mode': 'rounded6'}, # 위,아래 둥글게 그리기
    'pole':      {'n_kpts': 6, 'labels': ['TL','TR','BR','BL','TC','BC'],
                  'box_color': (255,100,200), 'draw_mode': 'rounded6'}, # 위,아래 둥글게 그리기
    'scooter':   {'n_kpts': 4, 'labels': ['H-L','H-R','Wh-F','Wh-B'],
                  'box_color': (255,200,0),   'draw_mode': 'scooter'},
    'car':       {'n_kpts': 6, 'labels': ['FL','FR','RR','RL','WS-L','WS-R'],
                  'box_color': (100,255,100), 'draw_mode': 'car'},
    'crosswalk': {'n_kpts': 0, 'labels': [], 'box_color': (255,255,255), 'draw_mode': 'none'},
}

# 키포인트 색상 설정(0 ~ 6)
KPT_COLORS = [(0,255,0), (0,200,255), (0,100,255), (0,255,255), (255,0,255), (255,255,0)]

# ============================================================
# 1. solvePnP - 6DoF 자세 추정 output: rotation, translation
# ============================================================
def solve_pnp_for_object(cls_name, kp_xy, kp_conf, camera_matrix, dist_coeffs):
    if cls_name not in CLASS_3D_MODEL: # 클래스가 아니면 아무런 결과도 도출하지 않음
        return False, None, None, None

    obj_3d_full = CLASS_3D_MODEL[cls_name]
    n_total = len(obj_3d_full)

    # 1. 신뢰도가 임계치 이상인 특징점만 필터링
    valid_idx = []
    for i in range(min(n_total, len(kp_xy))):
        x, y = kp_xy[i]
        c = kp_conf[i] if kp_conf is not None and i < len(kp_conf) else 1.0
        if x > 0 and y > 0 and c >= KPT_CONF_TH:
            valid_idx.append(i) # 높은 신뢰도 특징점을 배열에 추가

    # 2. PnP를 풀기 위한 최소 조건인 점 4개 미달 시 도출 포기
    if len(valid_idx) < 4:
        return False, None, None, valid_idx

    # 3. 필터링된 유효한 3D 정답점과 2D 예측점 매칭
    obj_3d = obj_3d_full[valid_idx].astype(np.float32)
    img_2d = np.array([kp_xy[i] for i in valid_idx], dtype=np.float32)

    # 차량은 SQPNP로 연산, 그 외는 SOLVEPNP 연산
    flag = cv2.SOLVEPNP_SQPNP if len(valid_idx) >= 4 else cv2.SOLVEPNP_ITERATIVE

    try:
        # 최초 자세 추정 - revec(rotation), tvec(translation) 추정
        ok, rvec, tvec = cv2.solvePnP(
            obj_3d, img_2d, camera_matrix, dist_coeffs, flags=flag
        )
        if not ok: # 자세 추정이 불가능하면 특징점만 리턴
            return False, None, None, valid_idx

        if len(valid_idx) >= 5: # 특징점 개수가 5개 이상인 경우 정확성 높은 rvec, tvec로 덮어씌우기
            ok2, rvec, tvec = cv2.solvePnP(
                obj_3d, img_2d, camera_matrix, dist_coeffs,
                rvec=rvec, tvec=tvec, useExtrinsicGuess=True, # 최초 추정값을 초기값으로 주어 정답의 근접성을 높이고 정확성 향상
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        # Reprojection error 검증 - 구한 자세를 재투영하여 검증
        proj, _ = cv2.projectPoints(obj_3d, rvec, tvec, camera_matrix, dist_coeffs)
        err = np.linalg.norm(proj.reshape(-1, 2) - img_2d, axis=1).mean() # 재투영하여 실제 YOLO 픽셀값과의 거리 차이 계산
        
        # 차량은 스케일이 크므로 임계값 완화
        err_threshold = 50 if cls_name == 'car' else 30
        if err > err_threshold: # 오차가 임계값을 넘으면 기각하여 특징점만 반환
            return False, None, None, valid_idx

        return True, rvec, tvec, valid_idx
    except cv2.error:
        return False, None, None, valid_idx

def rvec_to_euler_deg(rvec):
    # 회전 벡터를 오일러 각도(Roll, Pitch, Yaw)로 변환
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll  = np.degrees(np.arctan2( R[2,1], R[2,2]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        yaw   = np.degrees(np.arctan2( R[1,0], R[0,0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1,2], R[1,1]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        yaw   = 0.0
    return roll, pitch, yaw

# ============================================================
# 2. 3D 투영 및 시각화 (opencv - projectPoints 함수)
# 구한 3D 자세를 바탕으로 2D 화면 위에 직육면체 렌더링
# ============================================================
def draw_axes(img, rvec, tvec, K, dist, length=0.3):
    # 각 객체의 중심에서 나가는 3축 그리기 (X: red, Y: green, Z: blue)
    axes_3d = np.array([
        [0, 0, 0], # 원점
        [length, 0, 0], # X
        [0, length, 0], # Y
        [0, 0, length], # Z
    ], dtype=np.float32)
    proj, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2).astype(int)
    o, x, y, z = proj

    # 픽셀 좌표가 이미지 밖으로 나가는 것을 방지
    h, w = img.shape[:2]
    def in_bounds(p):
        return -w < p[0] < 2*w and -h < p[1] < 2*h
    if all(in_bounds(p) for p in [o, x]):
        cv2.line(img, tuple(o), tuple(x), (0, 0, 255),   3, cv2.LINE_AA)
    if all(in_bounds(p) for p in [o, y]):
        cv2.line(img, tuple(o), tuple(y), (0, 255, 0),   3, cv2.LINE_AA)
    if all(in_bounds(p) for p in [o, z]):
        cv2.line(img, tuple(o), tuple(z), (255, 100, 0), 3, cv2.LINE_AA)
    cv2.circle(img, tuple(o), 4, (255, 255, 255), -1)

# ============================================================
# 3. 3D 시각화
# ============================================================
def draw_3d_cylinder_like(img, cls_name, rvec, tvec, K, dist): # [기둥, 볼라드] - 실린더 모형 그리기
    # 볼라드, 기둥은 원통형으로 시각화 하기 위해 분리
    if cls_name not in ['bollard', 'pole']:
        return
    model = CLASS_3D_MODEL[cls_name]
    H = model[:, 1].max()
    W = model[:, 0].max() - model[:, 0].min()
    R = W / 2

    N = 12 # 원통을 표현하기 위한 다각형(12점) 투영
    top_ring = []
    bot_ring = []
    for k in range(N):
        theta = 2 * np.pi * k / N
        x = R * np.sin(theta)
        z = R * np.cos(theta)
        top_ring.append([x, H, z])
        bot_ring.append([x, 0, z])
    all_pts = np.array(top_ring + bot_ring, dtype=np.float32)
    proj, _ = cv2.projectPoints(all_pts, rvec, tvec, K, dist) # 3D -> 2D로 투영
    pts = proj.reshape(-1, 2).astype(int)
    top_pts = pts[:N]
    bot_pts = pts[N:]

    line_color = (200, 255, 200) if cls_name == 'bollard' else (255, 200, 255)
    cv2.polylines(img, [top_pts], True, line_color, 1, cv2.LINE_AA) # 상단 원
    cv2.polylines(img, [bot_pts], True, line_color, 1, cv2.LINE_AA) # 하단 원
    for k in [0, N//4, N//2, 3*N//4]:
        cv2.line(img, tuple(top_pts[k]), tuple(bot_pts[k]),
                 line_color, 1, cv2.LINE_AA)

def draw_3d_scooter(img, rvec, tvec, K, dist): # [스쿠터] - 뼈대 그리기
    pts3d = np.array([
        [-0.275, 1.05,  0.00],
        [ 0.275, 1.05,  0.00],
        [ 0.000, 0.18,  0.55],
        [ 0.000, 0.18, -0.60],
        [ 0.000, 1.05,  0.00],
    ], dtype=np.float32)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    pts = proj.reshape(-1, 2).astype(int)
    color = (255, 220, 100)
    cv2.line(img, tuple(pts[0]), tuple(pts[1]), color, 2, cv2.LINE_AA)
    cv2.line(img, tuple(pts[2]), tuple(pts[3]), color, 2, cv2.LINE_AA)
    cv2.line(img, tuple(pts[4]), tuple(pts[2]), color, 2, cv2.LINE_AA)

def draw_3d_car_box(img, rvec, tvec, K, dist): # [자동차] - 바닥면 직사각형 + 상단 윈드실드 사각형
    # 바퀴 접지점 4개를 베이스로 한 직육면체
    base = CAR_3D[:4].copy()  # FL, FR, RR, RL (Y=0)
    top  = base.copy()
    top[:, 1] = CAR_BOX_HEIGHT  # Y를 1.45m로

    # 윈드실드 상단 2점 (캐빈 시각화)
    ws = CAR_3D[4:6].copy()

    all_pts = np.vstack([base, top, ws]).astype(np.float32)
    proj, _ = cv2.projectPoints(all_pts, rvec, tvec, K, dist)
    pts = proj.reshape(-1, 2).astype(int)

    base_pts = pts[0:4]   # 0-3 바닥면 4점 필셀 좌표
    top_pts  = pts[4:8]   # 4-7 천장면 4점 픽셀 좌표
    ws_pts   = pts[8:10]  # 8-9 윈드실드 2점 픽셀 좌표

    color_body  = (100, 255, 100)   # 차체
    color_front = (0, 255, 255)     # 전방 강조 (앞축)
    color_ws    = (255, 200, 0)     # 윈드실드

    # 바닥면
    cv2.polylines(img, [base_pts], True, color_body, 2, cv2.LINE_AA)
    # 천장면
    cv2.polylines(img, [top_pts],  True, color_body, 2, cv2.LINE_AA)
    # 수직 기둥 4개
    for k in range(4):
        cv2.line(img, tuple(base_pts[k]), tuple(top_pts[k]),
                 color_body, 2, cv2.LINE_AA)

    # ✅ 전방(앞축) 강조: FL-FR 라인
    cv2.line(img, tuple(base_pts[0]), tuple(base_pts[1]),
             color_front, 3, cv2.LINE_AA)
    cv2.line(img, tuple(top_pts[0]),  tuple(top_pts[1]),
             color_front, 3, cv2.LINE_AA)

    # ✅ 윈드실드 라인 (A필러 상단 연결)
    cv2.line(img, tuple(ws_pts[0]), tuple(ws_pts[1]),
             color_ws, 2, cv2.LINE_AA)
    # A필러 (윈드실드 → 앞축 상단 모서리)
    cv2.line(img, tuple(ws_pts[0]), tuple(top_pts[0]),
             color_ws, 1, cv2.LINE_AA)
    cv2.line(img, tuple(ws_pts[1]), tuple(top_pts[1]),
             color_ws, 1, cv2.LINE_AA)

def draw_3d_bbox(img, cls_name, rvec, tvec, K, dist): # 클래스명에 따라 알맞는 Draw 함수 호출
    if cls_name in ['bollard', 'pole']:
        draw_3d_cylinder_like(img, cls_name, rvec, tvec, K, dist)
    elif cls_name == 'scooter':
        draw_3d_scooter(img, rvec, tvec, K, dist)
    elif cls_name == 'car':
        draw_3d_car_box(img, rvec, tvec, K, dist)

# ============================================================
# 2D 키포인트 - YOLO Pose의 특징점 찍기
# ============================================================
def draw_keypoints_minimal(img, kp_xy, cls_name):
    cfg = CLASS_CONFIG.get(cls_name)
    if cfg is None or cfg['n_kpts'] == 0:
        return
    n = cfg['n_kpts']
    for idx in range(min(n, len(kp_xy))):
        x, y = kp_xy[idx]
        if x > 0 and y > 0:
            color = KPT_COLORS[idx % len(KPT_COLORS)]
            radius = 4 if idx >= 4 else 3
            cv2.circle(img, (int(x), int(y)), radius, color, -1)

# ============================================================
# 4. 메인 그리기 (추론 및 영상 합성)
# ============================================================
def draw_predictions(img, result, names, allowed, K, dist): # YOLO 프레임 단위 추론 결과를 순화하며 그리는 함수
    if len(result.boxes) == 0:
        return img, 0, 0

    # 텐서를 CPU 메모리로 옮기고 Numpy 배열로 변환
    boxes = result.boxes.xyxy.cpu().numpy()
    clss  = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    has_kpts = result.keypoints is not None and len(result.keypoints) > 0
    kpts_xy = result.keypoints.xy.cpu().numpy() if has_kpts else None
    kpts_conf = None
    if has_kpts and hasattr(result.keypoints, 'conf') and result.keypoints.conf is not None:
        kpts_conf = result.keypoints.conf.cpu().numpy()

    drawn = 0
    pnp_count = 0

    # 화면에 탐지된 모든 객체 순회
    for i, (box, cls, conf) in enumerate(zip(boxes, clss, confs)):
        cls_name = names[cls]
        if cls_name not in allowed:
            continue
        drawn += 1
        x1, y1, x2, y2 = box.astype(int)

        if cls_name in NO_KPT_CLASSES or not has_kpts: # 허용된 클래스가 아니면 스킵
            continue

        if SHOW_KEYPOINTS:
            draw_keypoints_minimal(img, kpts_xy[i], cls_name)

        # 3D 포즈 추정 대상이면 solvePnP 연산 진입
        if cls_name in PNP_CLASSES:
            kp_conf_i = kpts_conf[i] if kpts_conf is not None else None
            ok, rvec, tvec, used = solve_pnp_for_object(
                cls_name, kpts_xy[i], kp_conf_i, K, dist
            )
            if ok:
                pnp_count += 1
                draw_3d_bbox(img, cls_name, rvec, tvec, K, dist) # 3D 박스 그리기

                # 차량은 축 길이를 크게 - 볼라드같은 작은 객체보다 축 길이를 늘리기
                axis_len = 1.0 if cls_name == 'car' else 0.3
                draw_axes(img, rvec, tvec, K, dist, length=axis_len) # X,Y,Z 축 그리기

                # Depth 라벨
                dist_m = float(np.linalg.norm(tvec))

                # 텍스트 라벨을 띄우기 위한 앵커 포인트 설정
                if cls_name == 'car':
                    label_anchor_3d = np.array(
                        [[0, CAR_BOX_HEIGHT + 0.2, CAR_WS_Z]],
                        dtype=np.float32
                    )
                else:
                    label_anchor_3d = np.array(
                        [[0, CLASS_3D_MODEL[cls_name][:,1].max(), 0]],
                        dtype=np.float32
                    )

                # 3D 허공의 텍스트 앵커를 2D 좌표계로 투영
                origin_proj, _ = cv2.projectPoints(
                    label_anchor_3d, rvec, tvec, K, dist
                )
                # 텍스트가 화면 밖으로 잘리지 않도록 좌표 제한(clip)
                ox, oy = origin_proj.reshape(-1).astype(int)
                ox = int(np.clip(ox, 10, img.shape[1]-150))
                oy = int(np.clip(oy - 10, 20, img.shape[0]-10))

                # 레이블 내용 조합 후 화면에 그리기
                if cls_name == 'car':
                    _, _, yaw = rvec_to_euler_deg(rvec)
                    label = f"{cls_name} D={dist_m:.2f}m Y={yaw:+.0f}" # 차량은 YAW 정보까지 출력
                else:
                    label = f"{cls_name} D={dist_m:.2f}m"

                cv2.putText(img, label, (ox, oy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(img, label, (ox, oy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)

    return img, drawn, pnp_count

# ============================================================
# 5. UI 오버레이 - HUD & Legend
# ============================================================
def draw_legend(img): # 우측 상단 범례 설정
    items = [
        ("Bollard/Pole 6pt + PnP",  (0,200,255)),
        ("Scooter 4pt + PnP",       (255,200,0)),
        ("Car 6pt + PnP (4wh+2ws)", (100,255,100)),
        ("Axes: X=R, Y=G, Z=B",     (255,255,255)),
    ]
    x0, y0 = 10, 10
    bw, bh = 290, 105
    overlay = img.copy()
    cv2.rectangle(overlay, (x0,y0), (x0+bw, y0+bh), (40,40,40), -1)
    img[:] = cv2.addWeighted(overlay, 0.7, img, 0.3, 0) # 반투명 효과
    cv2.rectangle(img, (x0,y0), (x0+bw, y0+bh), (255,255,255), 1)
    for i,(t,c) in enumerate(items):
        cy = y0 + 20 + i*20
        cv2.line(img, (x0+8,cy), (x0+30,cy), c, 3)
        cv2.putText(img, t, (x0+38, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return img

def draw_hud(img, frame_idx, total, fps_now, n_det, n_pnp, conf_th): # 좌측 상단 프레임률 등의 HUD UI
    h, w = img.shape[:2]
    x0 = w - 270; y0 = 10
    bw, bh = 260, 100
    overlay = img.copy()
    cv2.rectangle(overlay, (x0,y0), (x0+bw, y0+bh), (40,40,40), -1)
    img[:] = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    cv2.rectangle(img, (x0,y0), (x0+bw, y0+bh), (255,255,255), 1)
    lines = [
        f"Frame: {frame_idx}/{total}",
        f"FPS: {fps_now:.1f}  CONF>={conf_th:.2f}",
        f"Detected: {n_det}",
        f"6DoF (PnP): {n_pnp}",
    ]
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x0+10, y0+22 + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return img

# ============================================================
# 실행
# ============================================================
def main():
    global CAMERA_MATRIX

    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(WEIGHT_PATH) # YOLO 모델 로드
    names = model.names

    print(f"\nModel classes: {list(names.values())}")
    print(f"CONF threshold: {CONF}")
    print(f"PnP classes: {PNP_CLASSES}")
    print(f"Car: 6-point (4 wheels + 2 windshield A-pillars)")
    print(f"  Tread={CAR_TREAD}m, Wheelbase={CAR_WHEELBASE}m, "
          f"WS_H={CAR_WS_HEIGHT}m\n")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {VIDEO_PATH}")

    # 원본 영상 스펙 추출
    src_fps  = cap.get(cv2.CAP_PROP_FPS)
    src_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps  = src_fps / FRAME_SKIP

    # 캘리브레이션 행렬 초기화
    CAMERA_MATRIX = init_camera_matrix(src_w, src_h)
    print(f"Camera Matrix (default approx):")
    print(CAMERA_MATRIX)
    print(f"⚠️  For accurate PnP, replace fx, fy with calibrated values.\n")

    print(f"Video: {src_w}x{src_h} @ {src_fps:.1f}FPS, {total_fr} frames")
    print(f"Output: {out_fps:.1f}FPS (skip={FRAME_SKIP})\n")

    # 비디오 저장 설정
    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(SAVE_PATH, fourcc, out_fps, (src_w, src_h))

    stats = {cls: 0 for cls in ALL_CLASSES}
    frame_idx = 0
    processed = 0
    t_start = time.time(); t_prev = t_start; fps_now = 0.0

    print("Processing... (press 'q' or ESC to stop)\n")

    while True: # 순회 루프
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if (frame_idx-1) % FRAME_SKIP != 0:
            continue

        result = model.predict(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]
        img_out, n_drawn, n_pnp = draw_predictions(
            frame.copy(), result, names, ALL_CLASSES,
            CAMERA_MATRIX, DIST_COEFFS
        )

        if len(result.boxes) > 0:
            for cls_idx in result.boxes.cls.cpu().numpy().astype(int):
                cn = names[cls_idx]
                if cn in stats: stats[cn] += 1

        t_now = time.time()
        inst = 1.0 / max(t_now - t_prev, 1e-6)
        fps_now = inst if fps_now == 0 else (0.9*fps_now + 0.1*inst)
        t_prev = t_now

        img_out = draw_legend(img_out)
        img_out = draw_hud(img_out, frame_idx, total_fr, fps_now,
                           n_drawn, n_pnp, CONF)

        if writer is not None:
            writer.write(img_out)

        if SHOW_WINDOW:
            preview = img_out
            if RESIZE_PREVIEW != 1.0:
                preview = cv2.resize(img_out, None, fx=RESIZE_PREVIEW, fy=RESIZE_PREVIEW,
                                     interpolation=cv2.INTER_AREA)
            cv2.imshow("YOLO Pose + 6DoF", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[Interrupted by user]")
                break

        processed += 1
        if processed % 100 == 0:
            elapsed = t_now - t_start
            pct = frame_idx/total_fr*100 if total_fr > 0 else 0
            print(f"  [{pct:5.1f}%] frame {frame_idx}/{total_fr} | "
                  f"processed {processed} | avg {processed/elapsed:.1f} FPS")

    # 자원 해제
    cap.release()
    if writer is not None: writer.release()
    if SHOW_WINDOW: cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n✅ Done in {elapsed:.1f}s | {processed} frames @ avg {processed/elapsed:.1f} FPS")
    if SAVE_VIDEO:
        print(f"   Saved → {SAVE_PATH}")
    print(f"\nDetection stats (CONF>={CONF}):")
    for cls, cnt in stats.items():
        print(f"  {cls:12s}: {cnt}")


if __name__ == '__main__':
    main()
