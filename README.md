## 403YoloDemo — YOLOv8 Pose 揮手偵測 + 語音播報（Jetson Nano）

以 YOLOv8 Pose 偵測人體關節，畫出骨架連線，偵測「揮手」後以 TTS 播報（預設：你好）。

### 功能
- **姿態偵測**：Ultralytics YOLOv8 Pose（預設 `yolov8n-pose.pt`）
- **骨架繪製**：COCO 17 點骨架連線，過濾邊界雜訊
- **揮手偵測**：利用手腕 X 方向的振幅與方向反轉次數
- **語音播報**：`pyttsx3`，失敗回退 `espeak-ng`
- **冷卻機制**：觸發後 N 秒內不重複播報

### 安裝（macOS 本機）
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 執行
```bash
python3 src/yolo_pose_handraise.py --source 0 --device cpu --speak-text "你好" --show
```

### Jetson Nano（重點）
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv espeak-ng
python3 -m venv .venv && . .venv/bin/activate
python3 -m pip install --upgrade pip wheel setuptools
# 依 JetPack 版本安裝對應的 PyTorch/torchvision wheel（見 NVIDIA 官方指引）
pip install -r requirements.txt
python3 src/yolo_pose_handraise.py --source 0 --device 0 --speak-text "你好" --show
```

### 參數
- `--wave-history`：揮手判斷採樣幀數（預設 14）
- `--wave-min-changes`：方向反轉次數門檻（預設 2）
- `--wave-min-amplitude`：X 方向最小振幅（0~1，預設 0.08）
- `--require-hand-up/--no-require-hand-up`：是否限定手腕須高於肩（預設啟用）

### 結構
```
403YoloDemo/
  ├─ src/
  │   └─ yolo_pose_handraise.py
  ├─ scripts/
  │   └─ run_yolo_pose.sh
  ├─ requirements.txt
  └─ README.md
```


