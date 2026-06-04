import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 기본 환경 설정
WEIGHT_PATH = str(Path(__file__).parent / 'best_seg.pt')
VIDEO_PATH  = 'C:/python_project/yolo_seg/media/road1.mp4'
SAVE_PATH   = 'C:/python_project/yolo_seg/media/output.mp4'

IMG_SIZE = 640 # 모델 입력 해상도 (640 x 640)
CONF     = 0.7 # 객체 탐지 신뢰도 임계값 (이 값 미만은 탐지하지 않음)
IOU      = 0.5 # NMS 시 박스 중복 제거 기준 IoU

SHOW_WINDOW    = True   # 결과를 윈도우 창으로 띄울지 여부
SAVE_VIDEO     = False  # 결과 영상을 파일로 저장할지 여부
FRAME_SKIP     = 1      # 프레임 스킵 수 (값을 높이면 배속 효과)
RESIZE_PREVIEW = 1.0    # 미리보기 창 배율 (1.0이면 원본 해상도)

SHOW_BOX   = True  # 바운딩박스 표시 여부
SHOW_MASK  = True  # 세그멘테이션 마스크 표시 여부
SHOW_LABEL = True  # 클래스 라벨 텍스트 표시 여부

MASK_ALPHA = 0.45  # 마스크 합성 시 투명도 (0이면 안보임, 1이면 불투명)
UI_SCALE   = 0.7   # 전역 UI 배율 (작은 화면일수록 작게)

# 클래스 목록 정의
ALL_CLASSES = ['person', 'tree']  # 모델이 학습한 전체 클래스

# 클래스별 색상 설정
CLASS_COLORS = {
    'person': (0, 165, 255), # 주황색
    'tree': (100, 255, 100) # 연두색
}
DEFAULT_COLOR = (200, 200, 200) # 목록에 없는 클래스의 기본 색상

# 1. 마스크 합성 함수
def overlay_mask(img, mask, color, alpha): # 마스크 영역의 픽셀만 골라 alpha 블렌딩
    region = mask > 0
    if not np.any(region):
        return img
    roi = img[region].astype(np.float32) # 마스크 안쪽 픽셀만 추출 (N, 3) 형태
    color_arr = np.array(color, dtype=np.float32)
    img[region] = (roi * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)
    return img

# 2. 마스크 외곽선 그리기 함수
def draw_mask_contour(img, mask, color, ui): # 마스크 경계를 또렷하게 표현하기 위해 윤곽선을 따로 그린다
    contours, _ = cv2.findContours( # 마스크에서 외곽 윤곽선 좌표들을 찾는다
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    thickness = max(1, int(2 * ui)) # 선 두께를 화면 배율에 비례
    cv2.drawContours(img, contours, -1, color, thickness, cv2.LINE_AA)
    return img

# 3. 2D 바운딩박스 및 라벨 그리기 함수
def draw_box(img, box, cls_name, conf, color, ui): # 작은 화면을 고려하여 코너 강조 스타일로 박스를 그린다
    x1, y1, x2, y2 = box.astype(int) # 박스 좌표를 정수형으로 변환한다
    thickness = max(1, int(2 * ui))  # 선 두께를 화면 배율에 비례
    font_scale = 0.45 * ui           # 글자 크기를 화면 배율에 비례

    # 박스 네 모서리에 짧은 코너 선을 그리기
    corner = max(12, int((x2 - x1) * 0.18))
    points = [
        ((x1, y1), (x1 + corner, y1), (x1, y1 + corner)),  # 좌상단
        ((x2, y1), (x2 - corner, y1), (x2, y1 + corner)),  # 우상단
        ((x1, y2), (x1 + corner, y2), (x1, y2 - corner)),  # 좌하단
        ((x2, y2), (x2 - corner, y2), (x2, y2 - corner)),  # 우하단
    ]
    for center, a, b in points:
        cv2.line(img, center, a, color, thickness, cv2.LINE_AA)
        cv2.line(img, center, b, color, thickness, cv2.LINE_AA)

    if SHOW_LABEL: # 라벨 표시 허용 시
        label = cls_name + ' ' + str(int(conf * 100)) + '%'
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        pad = int(4 * ui)

        # 라벨 칩의 위쪽 좌표를 정하고 화면 밖이면 박스 안쪽으로 내린다
        top = y1 - (th + pad * 2)
        bottom = y1
        if top < 0:
            top = y1
            bottom = y1 + (th + pad * 2)

        # 칩 배경을 채우고 그 위에 검은색 글자를 올린다
        cv2.rectangle(img, (x1, top), (x1 + tw + pad * 2, bottom), color, -1)
        cv2.putText(
            img, label, (x1 + pad, bottom - pad),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA
        )
    return img

# 4. 메인 그리기 함수
def draw_predictions(img, result, names, ui):
    # 탐지된 객체가 없으면 원본을 그대로 반환한다
    if result.boxes is None or len(result.boxes) == 0:
        return img, 0

    # 박스 좌표와 클래스 번호, 신뢰도를 넘파이 배열로 변환한다
    boxes = result.boxes.xyxy.cpu().numpy()
    clss  = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    # 세그멘테이션 마스크가 존재하는지 확인한다
    has_mask = result.masks is not None
    # 마스크가 있으면 각 객체별 마스크 데이터를 가져온다
    masks = result.masks.data.cpu().numpy() if has_mask else None

    drawn = 0  # 실제로 그린 객체 수를 센다

    # 탐지된 모든 객체를 하나씩 순회한다
    for i in range(len(boxes)):
        cls_name = names[clss[i]]                          # 클래스 이름
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)  # 클래스 색상
        drawn += 1

        # 마스크 표시가 켜져 있고 마스크가 존재하면 마스크를 그린다
        if SHOW_MASK and has_mask and i < len(masks):
            # 모델 출력 마스크 크기를 원본 영상 크기에 맞게 리사이즈한다
            mask = cv2.resize(
                masks[i], (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            # 0.5를 기준으로 이진화하여 객체 영역만 1로 만든다
            mask = (mask > 0.5).astype(np.uint8)

            # 마스크 영역을 반투명 색으로 덧칠한다
            img = overlay_mask(img, mask, color, MASK_ALPHA)
            # 마스크 경계를 윤곽선으로 또렷하게 표시한다
            img = draw_mask_contour(img, mask, color, ui)

        # 박스 표시가 켜져 있으면 바운딩박스와 라벨을 그린다
        if SHOW_BOX:
            img = draw_box(img, boxes[i], cls_name, confs[i], color, ui)

    return img, drawn

# 5. 실행 메인
def main():
    # YOLO 세그멘테이션 모델을 로드한다
    model = YOLO(WEIGHT_PATH)
    names = model.names

    # 입력 영상을 연다
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError('Cannot open video: ' + VIDEO_PATH)

    # 원본 영상의 해상도와 프레임 정보를 읽는다
    src_fps  = cap.get(cv2.CAP_PROP_FPS)
    src_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps  = src_fps / FRAME_SKIP

    # 해상도에 비례하여 UI 배율을 자동으로 정한다 (0.5에서 1.0 사이)
    ui_scale = float(np.clip(src_w / 1280.0, 0.5, 1.0))

    # 영상 저장이 켜져 있으면 비디오 writer를 준비한다
    writer = None
    if SAVE_VIDEO:
        Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(SAVE_PATH, fourcc, out_fps, (src_w, src_h))

    frame_idx = 0   # 현재까지 읽은 프레임 번호

    # 영상이 끝날 때까지 프레임을 반복 처리한다
    while True:
        ret, frame = cap.read()
        # 더 이상 읽을 프레임이 없으면 반복을 종료한다
        if not ret:
            break

        frame_idx += 1
        # 프레임 스킵 설정에 따라 일부 프레임은 건너뛴다
        if (frame_idx - 1) % FRAME_SKIP != 0:
            continue

        # 현재 프레임에 대해 세그멘테이션 추론을 수행한다
        result = model.predict(
            frame, imgsz=IMG_SIZE, conf=CONF, iou=IOU, verbose=False
        )[0]

        # 추론 결과를 원본 복사본 위에 시각화한다
        img_out, n_drawn = draw_predictions(frame.copy(), result, names, ui_scale)

        # 저장이 켜져 있으면 결과 프레임을 파일에 기록한다
        if writer is not None:
            writer.write(img_out)

        # 윈도우 표시가 켜져 있으면 결과를 창으로 띄운다
        if SHOW_WINDOW:
            preview = img_out
            # 미리보기 배율이 1이 아니면 창 크기를 조정한다
            if RESIZE_PREVIEW != 1.0:
                preview = cv2.resize(
                    img_out, None, fx=RESIZE_PREVIEW, fy=RESIZE_PREVIEW,
                    interpolation=cv2.INTER_AREA
                )
            cv2.imshow('YOLO Segmentation', preview)

            # q 또는 ESC 키를 누르면 반복을 종료한다
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    # 사용한 자원을 모두 해제한다
    cap.release()
    if writer is not None:
        writer.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()