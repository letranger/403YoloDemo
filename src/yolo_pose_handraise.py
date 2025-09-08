import argparse
import time
from collections import deque
from typing import Deque, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except Exception:  # Jetson 可能需 apt 裝 opencv
    cv2 = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

from ultralytics import YOLO


COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

COCO_SKELETON_PAIRS = [
    # 軀幹與手臂
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    # 腿部
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    # 頭部到肩
    (0, 5), (0, 6),
    # 眼耳（可選）
    (1, 3), (2, 4), (1, 0), (2, 0),
]


def init_tts_engine() -> Optional[object]:
    # 優先使用 pyttsx3，若失敗回退到 espeak-ng（以 subprocess 方式）
    if pyttsx3 is not None:
        try:
            engine = pyttsx3.init()
            return engine
        except Exception:
            pass
    return None


def speak_text(text: str, engine: Optional[object]) -> None:
    if engine is not None:
        try:
            engine.say(text)
            engine.runAndWait()
            return
        except Exception:
            pass
    # 回退：使用 espeak-ng（若可用）
    try:
        import subprocess

        subprocess.run(["espeak-ng", text], check=False)
    except Exception:
        pass


def wrist_higher_than_shoulder(
    keypoints: np.ndarray,
    wrist_index: int,
    shoulder_index: int,
    img_h: int,
    margin_ratio: float = 0.03,
) -> bool:
    # keypoints: (17, 3) -> x, y, conf
    wrist = keypoints[wrist_index]
    shoulder = keypoints[shoulder_index]
    # 若任一點置信度低，直接不判定
    if wrist[2] < 0.2 or shoulder[2] < 0.2:
        return False
    margin = img_h * margin_ratio
    return wrist[1] < shoulder[1] - margin


def _valid_kp(pt: np.ndarray, img_w: int, img_h: int, conf_thr: float = 0.2, edge_margin: int = 2) -> bool:
    x, y, c = pt
    if c < conf_thr:
        return False
    # 過濾位在畫面外或過於靠邊（常見 0,0 雜訊）
    if not (edge_margin <= x < img_w - edge_margin):
        return False
    if not (edge_margin <= y < img_h - edge_margin):
        return False
    return True


def draw_pose_and_state(
    frame: np.ndarray,
    keypoints: np.ndarray,
    left_wave: bool,
    right_wave: bool,
    img_w: int,
    img_h: int,
) -> None:
    # 畫骨架連線（只連結有效關節）
    for a, b in COCO_SKELETON_PAIRS:
        pa = keypoints[a]
        pb = keypoints[b]
        if _valid_kp(pa, img_w, img_h) and _valid_kp(pb, img_w, img_h):
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 128, 255), 2)
    # 畫關節點
    for idx, (x, y, c) in enumerate(keypoints):
        if _valid_kp(np.array([x, y, c]), img_w, img_h):
            color = (0, 255, 0)
            if idx in (COCO_KEYPOINTS["left_wrist"], COCO_KEYPOINTS["right_wrist"]):
                color = (0, 255, 255)
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    text = f"WaveL: {left_wave} WaveR: {right_wave}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def is_waving(x_positions: Deque[Optional[float]], min_changes: int, min_amplitude: float) -> bool:
    # x_positions 為 0~1 的正規化手腕 X 座標序列（包含 None）
    xs = [x for x in x_positions if x is not None]
    if len(xs) < max(6, min_changes * 2 + 1):
        return False
    diffs = np.diff(xs)
    # 忽略極小移動，避免雜訊
    epsilon = 0.0025
    signs = []
    for d in diffs:
        if abs(d) <= epsilon:
            continue
        signs.append(1 if d > 0 else -1)
    if len(signs) < 2:
        return False
    changes = 0
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            changes += 1
    amplitude = float(max(xs) - min(xs))
    return amplitude >= min_amplitude and changes >= min_changes


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Pose Wave Detector with TTS")
    parser.add_argument("--source", type=str, default="0", help="攝影機索引或影片路徑")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="YOLOv8 Pose 權重")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度閾值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IOU 閾值")
    parser.add_argument("--device", type=str, default="cpu", help="裝置：cpu/-1 或 GPU 索引 0/1...")
    parser.add_argument("--wave-history", type=int, default=14, help="揮手判斷採樣幀數")
    parser.add_argument("--wave-min-changes", type=int, default=2, help="方向反轉次數門檻")
    parser.add_argument("--wave-min-amplitude", type=float, default=0.08, help="X 方向最小振幅（0~1）")
    parser.add_argument("--require-hand-up", dest="require_hand_up", action="store_true")
    parser.add_argument("--no-require-hand-up", dest="require_hand_up", action="store_false")
    parser.set_defaults(require_hand_up=True)
    parser.add_argument("--cooldown", type=float, default=5.0, help="觸發後冷卻秒數")
    parser.add_argument("--speak-text", type=str, default="你好", help="偵測到時要說的話")
    parser.add_argument("--show", action="store_true", help="顯示偵測視窗")
    args = parser.parse_args()

    if cv2 is None:
        raise RuntimeError("OpenCV 未安裝。Jetson 請以 apt 安裝 python3-opencv；macOS 請用 pip 安裝 opencv-python。")

    # 解析 source
    source: Union[str, int]
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    # 載入模型
    model = YOLO(args.model)

    # 初始化 TTS
    tts_engine = init_tts_engine()
    last_trigger_time: float = 0.0

    # 視訊來源
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟視訊來源：{args.source}")

    # 手腕水平位置歷史（0~1），用於揮手偵測
    left_wrist_x: Deque[Optional[float]] = deque(maxlen=args.wave_history)
    right_wrist_x: Deque[Optional[float]] = deque(maxlen=args.wave_history)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            img_h, img_w = frame.shape[:2]

            # 使用 YOLOv8 Pose 推論
            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
                imgsz=max(320, (img_w // 32) * 32),
                half=True if args.device != "cpu" and args.device != "-1" else False,
            )

            left_up = False
            right_up = False
            left_wave = False
            right_wave = False

            if len(results) > 0 and results[0].keypoints is not None:
                # 取第一個人的關節（可擴充為多人體）
                kps = results[0].keypoints
                # kps.xy[0] -> (17, 2); kps.conf[0] -> (17, 1)
                try:
                    kpts_xy = kps.xy[0].cpu().numpy()
                    kpts_conf = kps.conf[0].cpu().numpy().reshape(-1)
                    keypoints = np.concatenate([kpts_xy, kpts_conf[:, None]], axis=1)  # (17,3)

                    left_up = wrist_higher_than_shoulder(
                        keypoints,
                        COCO_KEYPOINTS["left_wrist"],
                        COCO_KEYPOINTS["left_shoulder"],
                        img_h,
                    )
                    right_up = wrist_higher_than_shoulder(
                        keypoints,
                        COCO_KEYPOINTS["right_wrist"],
                        COCO_KEYPOINTS["right_shoulder"],
                        img_h,
                    )
                    # 收集手腕 x（正規化）
                    lw = keypoints[COCO_KEYPOINTS["left_wrist"]]
                    rw = keypoints[COCO_KEYPOINTS["right_wrist"]]
                    left_wrist_x.append(
                        float(lw[0]) / float(img_w) if _valid_kp(lw, img_w, img_h) else None
                    )
                    right_wrist_x.append(
                        float(rw[0]) / float(img_w) if _valid_kp(rw, img_w, img_h) else None
                    )

                    # 判斷是否揮手（需達到方向變化與振幅門檻）
                    left_wave = is_waving(left_wrist_x, args.wave_min_changes, args.wave_min_amplitude)
                    right_wave = is_waving(right_wrist_x, args.wave_min_changes, args.wave_min_amplitude)

                    if args.require_hand_up:
                        left_wave = left_wave and left_up
                        right_wave = right_wave and right_up

                    if args.show:
                        draw_pose_and_state(frame, keypoints, left_wave, right_wave, img_w, img_h)
                except Exception:
                    pass

            triggered = left_wave or right_wave
            now = time.time()
            if triggered and (now - last_trigger_time >= args.cooldown):
                speak_text(args.speak_text, tts_engine)
                last_trigger_time = now

            if args.show:
                cv2.imshow("YOLOv8 Pose - Wave", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


