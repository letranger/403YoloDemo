import argparse
import os
import json
from pathlib import Path
import time
from collections import deque
from typing import Deque, Optional, Tuple, Union, List

import numpy as np
try:
    import torch
except Exception:
    torch = None  # type: ignore

try:
    import cv2
except Exception:  # Jetson 可能需 apt 裝 opencv
    cv2 = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

from ultralytics import YOLO

# 可選：手部關鍵點（僅在有安裝 mediapipe 時可用）
try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None  # type: ignore

# 可選：trt_pose（Jetson 套件）
try:
    from trt_pose.models import resnet18_baseline_att_224x224_A  # type: ignore
    from trt_pose.coco import CocoTopology  # type: ignore
    from trt_pose.parse_objects import ParseObjects  # type: ignore
except Exception:
    resnet18_baseline_att_224x224_A = None  # type: ignore
    CocoTopology = None  # type: ignore
    ParseObjects = None  # type: ignore


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


def _maybe_set_pyttsx3_voice(engine: object, voice_pref: str) -> Optional[str]:
    # voice_pref 可為：'auto_female_zh'、'auto_male_zh'、或任何子字串匹配 voice.id/name
    try:
        voices = engine.getProperty("voices")
    except Exception:
        return None

    def is_zh(v: object) -> bool:
        try:
            langs = getattr(v, "languages", [])
            for l in langs:
                if isinstance(l, bytes) and b"zh" in l.lower():
                    return True
                if isinstance(l, str) and "zh" in l.lower():
                    return True
        except Exception:
            pass
        vid = str(getattr(v, "id", "")).lower()
        vname = str(getattr(v, "name", "")).lower()
        return ("zh" in vid) or ("chinese" in vname) or ("zh" in vname)

    def pick_by_names(candidates: List[str]) -> Optional[object]:
        cl = [c.lower() for c in candidates]
        for v in voices:
            name = str(getattr(v, "name", "")).lower()
            vid = str(getattr(v, "id", "")).lower()
            if any(k in name or k in vid for k in cl):
                return v
        return None

    target_voice = None
    vp = (voice_pref or "").lower()
    if vp == "auto_female_zh":
        # 常見中文女聲名稱：macOS Ting-Ting/Mei-Jia、Windows Huihui/Yaoyao
        target_voice = pick_by_names(["ting", "mei", "hui", "yao", "mei-jia", "ting-ting"]) or None
        if target_voice is None:
            # 退而求其次：第一個中文語音
            for v in voices:
                if is_zh(v):
                    target_voice = v
                    break
    elif vp == "auto_male_zh":
        # 常見中文男聲：Windows Kangkang 等
        target_voice = pick_by_names(["kang", "liang", "han"]) or None
        if target_voice is None:
            for v in voices:
                if is_zh(v):
                    target_voice = v
                    break
    elif vp:
        # 子字串匹配 voice id/name
        target_voice = pick_by_names([vp])

    try:
        if target_voice is not None:
            engine.setProperty("voice", getattr(target_voice, "id", None))
            return str(getattr(target_voice, "id", ""))
    except Exception:
        return None
    return None


def speak_text(text: str, engine: Optional[object], espeak_voice: str = "zh+f3") -> None:
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

        subprocess.run(["espeak-ng", "-v", espeak_voice, text], check=False)
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


class KeypointEMASmoother:
    """對 17 個關節座標做指數移動平均平滑，僅對高於門檻的點更新。"""
    def __init__(self, alpha: float = 0.4, num_points: int = 17):
        self.alpha = float(alpha)
        self.num_points = int(num_points)
        self.xy: Optional[np.ndarray] = None  # (17,2)

    def update(self, xy: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        if self.xy is None or self.xy.shape != xy.shape:
            self.xy = xy.copy()
            return self.xy
        a = self.alpha
        # 只更新有效點，無效點沿用上一幀，避免 0,0 影響
        for i in range(self.num_points):
            if valid_mask[i]:
                self.xy[i] = a * xy[i] + (1.0 - a) * self.xy[i]
        return self.xy.copy()


def detect_heart_pose(
    keypoints: np.ndarray,
    img_w: int,
    img_h: int,
    require_above_head: bool = True,
    proximity_ratio: float = 0.65,
    margin_ratio: float = 0.01,
) -> bool:
    # 以 wrist 與眼睛/鼻子高度判斷在頭上方，且左右手腕彼此接近
    lw = keypoints[COCO_KEYPOINTS["left_wrist"]]
    rw = keypoints[COCO_KEYPOINTS["right_wrist"]]
    nose = keypoints[COCO_KEYPOINTS["nose"]]
    leye = keypoints[COCO_KEYPOINTS["left_eye"]]
    reye = keypoints[COCO_KEYPOINTS["right_eye"]]

    # 置信濾除與邊界過濾
    def ok(p):
        return _valid_kp(p, img_w, img_h)

    if not (ok(lw) and ok(rw)):
        return False

    head_y = None
    for p in (leye, reye, nose):
        if ok(p):
            head_y = p[1] if head_y is None else min(head_y, p[1])
    if head_y is None:
        # 若頭部點不可用，放寬使用肩高度
        ls = keypoints[COCO_KEYPOINTS["left_shoulder"]]
        rs = keypoints[COCO_KEYPOINTS["right_shoulder"]]
        if ok(ls) and ok(rs):
            head_y = min(ls[1], rs[1]) - img_h * 0.08
        else:
            return False

    if require_above_head:
        margin = img_h * margin_ratio
        if not (lw[1] < head_y - margin and rw[1] < head_y - margin):
            return False

    # 接近度：左右手腕距離需相對肩寬足夠小
    ls = keypoints[COCO_KEYPOINTS["left_shoulder"]]
    rs = keypoints[COCO_KEYPOINTS["right_shoulder"]]
    if not (ok(ls) and ok(rs)):
        return False
    shoulder_width = float(abs(rs[0] - ls[0]))
    if shoulder_width <= 1.0:
        return False
    wrists_dist = float(np.linalg.norm(lw[:2] - rw[:2]))
    return wrists_dist <= shoulder_width * proximity_ratio


def detect_middle_finger_mediapipe(hand_landmarks, img_w: int, img_h: int) -> bool:
    # MediaPipe Hands 21 點索引：
    # 指尖: index=8, middle=12, ring=16, little=20；PIP: 6,10,14,18；拇指-tip=4, IP=3
    try:
        lm = hand_landmarks.landmark
        def y(i: int) -> float:
            return float(lm[i].y) * img_h
        margin = img_h * 0.02
        middle_up = y(12) < y(10) - margin
        index_down = y(8) >= y(6) - margin
        ring_down = y(16) >= y(14) - margin
        little_down = y(20) >= y(18) - margin
        thumb_down = y(4) >= y(3) - margin  # 粗略判斷
        return middle_up and index_down and ring_down and little_down and thumb_down
    except Exception:
        return False


def detect_heart_with_hands(
    hands_result,
    img_w: int,
    img_h: int,
    top_band_pct: float = 0.6,
    dist_pct: float = 0.35,
) -> bool:
    # 輔助條件：兩隻手都存在，且兩手食指或拇指靠近頭頂中央
    try:
        if not hands_result or not hands_result.multi_hand_landmarks:
            return False
        lms = hands_result.multi_hand_landmarks
        if len(lms) < 2:
            return False
        pts = []
        for hlm in lms[:2]:
            lm = hlm.landmark
            # 取食指指尖與拇指指尖
            ix = float(lm[8].x) * img_w
            iy = float(lm[8].y) * img_h
            tx = float(lm[4].x) * img_w
            ty = float(lm[4].y) * img_h
            # 取兩者較靠近中心者
            cx = (ix + tx) / 2.0
            cy = (iy + ty) / 2.0
            pts.append((cx, cy))
        if len(pts) < 2:
            return False
        # 在頭上方帶（可調百分比）
        top_band = img_h * top_band_pct
        if not (pts[0][1] < top_band and pts[1][1] < top_band):
            return False
        # 彼此距離接近（以影像寬度比例）
        dist = float(np.linalg.norm(np.array(pts[0]) - np.array(pts[1])))
        return dist < img_w * dist_pct
    except Exception:
        return False


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


def ensure_trt_assets(weight_path: Optional[str], topo_path: Optional[str]) -> Tuple[str, str]:
    # 預設放在 ~/.cache/trt_pose
    default_dir = os.path.join(str(Path.home()), ".cache", "trt_pose")
    os.makedirs(default_dir, exist_ok=True)
    default_weight = os.path.join(default_dir, "resnet18_baseline_att_224x224_A_epoch_249.pth")
    default_topo = os.path.join(default_dir, "human_pose.json")

    weight = os.path.expanduser(weight_path) if weight_path else default_weight
    topo = os.path.expanduser(topo_path) if topo_path else default_topo

    def _download(url: str, dst: str) -> None:
        try:
            import urllib.request

            urllib.request.urlretrieve(url, dst)
        except Exception:
            pass

    if not os.path.exists(weight):
        _download(
            "https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/resnet18_baseline_att_224x224_A_epoch_249.pth",
            weight,
        )
    if not os.path.exists(topo):
        _download(
            "https://raw.githubusercontent.com/NVIDIA-AI-IOT/trt_pose/master/tasks/human_pose/human_pose.json",
            topo,
        )
    return weight, topo


class TrtPoseBackend:
    def __init__(self, weight_path: str, topo_path: str, input_size: int = 224, device: str = "cuda") -> None:
        if resnet18_baseline_att_224x224_A is None or CocoTopology is None or ParseObjects is None:
            raise RuntimeError("trt_pose 未安裝，請先安裝 NVIDIA-AI-IOT/trt_pose")
        with open(topo_path, "r") as f:
            topo_json = json.load(f)
        self.topology = CocoTopology(topo_json)
        self.input_size = int(input_size)
        num_parts = self.topology.num_parts
        num_links = self.topology.num_links
        self.model = resnet18_baseline_att_224x224_A(num_parts, num_links)
        if torch is None:
            raise RuntimeError("需要 torch 以載入 trt_pose 權重，請在 Jetson 安裝對應版本的 torch")
        self.model.load_state_dict(torch.load(weight_path))
        self.model = self.model.eval()
        if device != "cpu":
            self.model = self.model.cuda()
        self.device = device
        self.parse = ParseObjects(self.topology)
        # 正規化參數
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        if device != "cpu":
            mean = mean.cuda()
            std = std.cuda()
        self.mean = mean
        self.std = std
        # 名稱對應
        names = topo_json.get("keypoints", [])
        self.name_to_idx = {str(n): i for i, n in enumerate(names)}

    def infer_keypoints(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        img_h, img_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size))
        t = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t = (t - self.mean) / self.std
        if self.device != "cpu":
            t = t.cuda()
        with torch.no_grad():
            cmap, paf = self.model(t)
        counts, objects, peaks = self.parse(cmap, paf)
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # 預設 conf=0
        if counts[0] < 1:
            return keypoints, cmap
        # 只取第一個人
        i = 0
        # 取輸出特徵圖尺寸用來縮放
        cmap_h = int(cmap.shape[2])
        cmap_w = int(cmap.shape[3])
        # 需要的名稱
        wanted = [
            "nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle",
        ]
        for out_idx, name in enumerate(wanted):
            part_idx = self.name_to_idx.get(name, None)
            if part_idx is None:
                continue
            k = int(objects[0][i][part_idx])
            if k < 0:
                continue
            peak = peaks[0][part_idx][k]
            # peaks: (y, x) 位置，縮放回原圖
            py = float(peak[0]) * img_h / float(cmap_h)
            px = float(peak[1]) * img_w / float(cmap_w)
            keypoints[out_idx, 0] = px
            keypoints[out_idx, 1] = py
            keypoints[out_idx, 2] = 0.9
        return keypoints, cmap


def draw_pose_and_state(
    frame: np.ndarray,
    keypoints: np.ndarray,
    left_wave: bool,
    right_wave: bool,
    img_w: int,
    img_h: int,
    kp_conf_thr: float,
) -> None:
    # 畫骨架連線（只連結有效關節）
    diag = float(np.hypot(img_w, img_h))
    for a, b in COCO_SKELETON_PAIRS:
        pa = keypoints[a]
        pb = keypoints[b]
        if _valid_kp(pa, img_w, img_h, conf_thr=kp_conf_thr) and _valid_kp(pb, img_w, img_h, conf_thr=kp_conf_thr):
            # 忽略極長不合理的線段（常見 0,0 雜訊造成）
            seg_len = float(np.linalg.norm(pa[:2] - pb[:2]))
            if seg_len > diag * 1.1:
                continue
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 128, 255), 2)
    # 畫關節點
    for idx, (x, y, c) in enumerate(keypoints):
        if _valid_kp(np.array([x, y, c]), img_w, img_h, conf_thr=kp_conf_thr):
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
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="YOLOv8/11 Pose 權重")
    parser.add_argument("--conf", type=float, default=0.35, help="置信度閾值（提高可過濾抖動）")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IOU 閾值")
    parser.add_argument("--device", type=str, default="cpu", help="裝置：cpu/-1 或 GPU 索引 0/1...")
    parser.add_argument("--wave-history", type=int, default=14, help="揮手判斷採樣幀數")
    parser.add_argument("--wave-min-changes", type=int, default=2, help="方向反轉次數門檻")
    parser.add_argument("--wave-min-amplitude", type=float, default=0.08, help="X 方向最小振幅（0~1）")
    parser.add_argument("--require-hand-up", dest="require_hand_up", action="store_true")
    parser.add_argument("--no-require-hand-up", dest="require_hand_up", action="store_false")
    parser.add_argument("--handup-margin", type=float, default=0.03, help="手腕高於肩的 y 邊際比例（相對影像高）")
    parser.set_defaults(require_hand_up=True)
    parser.add_argument("--cooldown", type=float, default=5.0, help="觸發後冷卻秒數")
    parser.add_argument("--speak-text", type=str, default="你好", help="偵測到時要說的話")
    parser.add_argument("--heart-text", type=str, default="我也愛你", help="偵測到愛心手勢時要說的話")
    parser.add_argument("--mf-text", type=str, default="請不要駡人", help="偵測到比中指時要說的話")
    # 語音參數
    parser.add_argument("--voice", type=str, default="auto_female_zh", help="pyttsx3 語音：auto_female_zh/auto_male_zh 或輸入部分名稱/id")
    parser.add_argument("--tts-rate", type=int, default=180, help="語速（約 100~250）")
    parser.add_argument("--tts-volume", type=float, default=1.0, help="音量 0.0~1.0")
    parser.add_argument("--espeak-voice", type=str, default="zh+f3", help="espeak-ng 語音，女聲例：zh+f3")
    # 手部偵測（需要 mediapipe；若未安裝則自動忽略）
    parser.add_argument("--use-hands", dest="use_hands", action="store_true")
    parser.add_argument("--no-use-hands", dest="use_hands", action="store_false")
    parser.set_defaults(use_hands=True)
    # 後端選擇：yolo（預設）或 trt（Jetson）
    parser.add_argument("--backend", type=str, default="yolo", choices=["yolo","trt"], help="姿勢後端：yolo 或 trt")
    parser.add_argument("--trt-weight", type=str, default="", help="trt_pose 權重路徑；留空自動下載")
    parser.add_argument("--trt-topo", type=str, default="", help="trt_pose topology 路徑；留空自動下載")
    parser.add_argument("--show", action="store_true", help="顯示偵測視窗")
    # 平滑與抖動濾波
    parser.add_argument("--ema-alpha", type=float, default=0.4, help="關節座標 EMA 平滑係數 0~1（大=跟隨快）")
    parser.add_argument("--min-kp-conf", type=float, default=0.25, help="關節點最小置信度，用於有效點判定")
    args = parser.parse_args()

    if cv2 is None:
        raise RuntimeError("OpenCV 未安裝。Jetson 請以 apt 安裝 python3-opencv；macOS 請用 pip 安裝 opencv-python。")

    # 解析 source
    source: Union[str, int]
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    # 載入模型或後端
    model = None
    trt_backend = None
    if args.backend == "yolo":
        model = YOLO(args.model)
    else:
        w, t = ensure_trt_assets(args.trt_weight or None, args.trt_topo or None)
        trt_backend = TrtPoseBackend(w, t, input_size=224, device=("cpu" if args.device in ["cpu","-1"] else "cuda"))

    # 初始化 TTS
    tts_engine = init_tts_engine()
    if tts_engine is not None:
        try:
            _maybe_set_pyttsx3_voice(tts_engine, args.voice)
            if args.tts_rate:
                tts_engine.setProperty("rate", int(args.tts_rate))
            if args.tts_volume is not None:
                v = float(args.tts_volume)
                v = 1.0 if v > 1.0 else (0.0 if v < 0.0 else v)
                tts_engine.setProperty("volume", v)
        except Exception:
            pass
    last_trigger_time: float = 0.0

    # 視訊來源
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟視訊來源：{args.source}")

    # 手腕水平位置歷史（0~1），用於揮手偵測
    left_wrist_x: Deque[Optional[float]] = deque(maxlen=args.wave_history)
    right_wrist_x: Deque[Optional[float]] = deque(maxlen=args.wave_history)

    # 若可用，初始化 MediaPipe Hands
    hands = None
    if args.use_hands and mp is not None:
        try:
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            hands = None

    smoother = KeypointEMASmoother(alpha=args.ema_alpha)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            img_h, img_w = frame.shape[:2]

            results = None
            keypoints = None
            if args.backend == "yolo":
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    verbose=False,
                    imgsz=max(320, (img_w // 32) * 32),
                    half=True if args.device != "cpu" and args.device != "-1" else False,
                )
            else:
                trt_kpts, _ = trt_backend.infer_keypoints(frame)
                keypoints = trt_kpts

            left_up = False
            right_up = False
            heart_pose = False
            left_wave = False
            right_wave = False

            if (results is not None and len(results) > 0 and results[0].keypoints is not None) or (keypoints is not None):
                # 取第一個人的關節（可擴充為多人體）
                try:
                    if keypoints is None:
                        kps = results[0].keypoints
                        kpts_xy = kps.xy[0].cpu().numpy()
                        kpts_conf = kps.conf[0].cpu().numpy().reshape(-1)
                        valid_mask = (kpts_conf >= float(args.min_kp_conf))
                        smoothed_xy = smoother.update(kpts_xy, valid_mask)
                        keypoints = np.concatenate([smoothed_xy, kpts_conf[:, None]], axis=1)  # (17,3)
                    else:
                        kpts_xy = keypoints[:, :2]
                        kpts_conf = keypoints[:, 2]
                        valid_mask = (kpts_conf >= float(args.min_kp_conf))
                        smoothed_xy = smoother.update(kpts_xy, valid_mask)
                        keypoints = np.concatenate([smoothed_xy, kpts_conf[:, None]], axis=1)

                    left_up = wrist_higher_than_shoulder(
                        keypoints,
                        COCO_KEYPOINTS["left_wrist"],
                        COCO_KEYPOINTS["left_shoulder"],
                        img_h,
                        args.handup_margin,
                    )
                    right_up = wrist_higher_than_shoulder(
                        keypoints,
                        COCO_KEYPOINTS["right_wrist"],
                        COCO_KEYPOINTS["right_shoulder"],
                        img_h,
                        args.handup_margin,
                    )
                    # 收集手腕 X 正規化
                    lw = keypoints[COCO_KEYPOINTS["left_wrist"]]
                    rw = keypoints[COCO_KEYPOINTS["right_wrist"]]
                    left_wrist_x.append(float(lw[0]) / float(img_w) if _valid_kp(lw, img_w, img_h) else None)
                    right_wrist_x.append(float(rw[0]) / float(img_w) if _valid_kp(rw, img_w, img_h) else None)

                    # 判斷是否揮手
                    left_wave = is_waving(left_wrist_x, args.wave_min_changes, args.wave_min_amplitude)
                    right_wave = is_waving(right_wrist_x, args.wave_min_changes, args.wave_min_amplitude)
                    if args.require_hand_up:
                        left_wave = left_wave and left_up
                        right_wave = right_wave and right_up

                    # 愛心手勢
                    heart_pose = detect_heart_pose(keypoints, img_w, img_h)

                    if args.show:
                        draw_pose_and_state(frame, keypoints, left_wave, right_wave, img_w, img_h, args.min_kp_conf)
                except Exception:
                    pass

            triggered = left_wave or right_wave
            now = time.time()

            # 可選：用 MediaPipe Hands 偵測比中指與輔助愛心
            middle_finger = False
            heart_hands = False
            if hands is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hands.process(rgb)
                    if res and res.multi_hand_landmarks:
                        for hlm in res.multi_hand_landmarks:
                            if detect_middle_finger_mediapipe(hlm, img_w, img_h):
                                middle_finger = True
                                break
                        # 若尚未判定愛心，嘗試用手部點輔助
                        if not heart_pose:
                            heart_hands = detect_heart_with_hands(res, img_w, img_h)
                except Exception:
                    pass

            if (now - last_trigger_time) >= args.cooldown:
                if middle_finger:
                    speak_text(args.mf_text, tts_engine, args.espeak_voice)
                    last_trigger_time = now
                elif heart_pose or heart_hands:
                    speak_text(args.heart_text, tts_engine, args.espeak_voice)
                    last_trigger_time = now
                elif triggered:
                    speak_text(args.speak_text, tts_engine, args.espeak_voice)
                    last_trigger_time = now

            if args.show:
                cv2.imshow("YOLOv8 Pose - 請揮手", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        try:
            if hands is not None:
                hands.close()
        except Exception:
            pass
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


