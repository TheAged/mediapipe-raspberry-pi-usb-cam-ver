import cv2
import mediapipe as mp
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# --- 設定 ---
MODEL_PATH = 'efficientdet.tflite' 
CAMERA_INDEX = 0 # 確定可以正常顯示畫面的攝影機索引
FRAME_WIDTH = 640 # 設定確定可行的寬度
FRAME_HEIGHT = 480 # 設定確定可行的高度
SCORE_THRESHOLD = 0.5 # 只顯示信心分數大於此閾值的結果
# ---------------

# 初始化 MediaPipe 物件偵測器相關元件
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 設定偵測器選項 (使用 VIDEO 模式進行同步處理)
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO, # 使用 VIDEO 模式
    max_results=5,
    score_threshold=SCORE_THRESHOLD)

# 創建偵測器實例
try:
    detector = ObjectDetector.create_from_options(options)
    print("MediaPipe ObjectDetector 創建成功。")
except Exception as e:
    print(f"創建 ObjectDetector 失敗: {e}")
    exit()

# 初始化 OpenCV 攝影機
cap = cv2.VideoCapture(CAMERA_INDEX)
# 強制設定之前確認可行的解析度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
time.sleep(1) # 等待相機啟動

if not cap.isOpened():
    print(f"錯誤：無法開啟攝影機索引 {CAMERA_INDEX}")
    exit()

print(f"成功開啟攝影機 {CAMERA_INDEX} ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})。按 'q' 鍵結束。")

# 用於計算 FPS
fps_start_time = time.time()
fps_frame_count = 0

# --- 主迴圈 ---
while True:
    # 讀取攝影機畫面
    ret, frame = cap.read()
    if not ret or frame is None:
        print("錯誤：無法讀取攝影機畫面。")
        break

    fps_frame_count += 1

    # 1. 將 OpenCV 的 BGR 影像轉換為 MediaPipe 需要的 RGB 格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. 建立 MediaPipe 的 Image 物件
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 3. 取得目前時間戳記 (毫秒) - VIDEO 模式需要
    timestamp_ms = int(time.time() * 1000)

    # 4. 執行同步偵測 (detect_for_video)
    try:
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    except Exception as e:
        print(f"偵測時發生錯誤: {e}")
        continue # 跳過這一幀

    # 5. 在原始 frame (BGR) 上繪製偵測結果
    image_copy = np.copy(frame) # 複製一份來繪製，避免修改原圖
    if detection_result and detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            # 畫框 (綠色)
            cv2.rectangle(image_copy, start_point, end_point, (0, 255, 0), 2)

            # 準備文字 (類別 + 分數)
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = f"{category_name} ({probability})"

            # 計算文字位置
            text_x = start_point[0]
            text_y = start_point[1] - 10 if start_point[1] > 20 else start_point[1] + 20 # 避免文字超出頂部

            # 畫文字
            cv2.putText(image_copy, result_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 255), 1, cv2.LINE_AA) # 紅色文字

    # 計算並顯示 FPS
    if fps_frame_count % 10 == 0:
        fps_end_time = time.time()
        fps = fps_frame_count / (fps_end_time - fps_start_time)
        print(f"FPS: {fps:.1f}") # 印在終端機
        # 重置以計算下一個區間的 FPS
        # fps_frame_count = 0
        # fps_start_time = time.time()

    # 顯示標註後的畫面
    cv2.imshow('MediaPipe Object Detection', image_copy)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 清理資源 ---
detector.close() # 關閉偵測器
cap.release()
cv2.destroyAllWindows()
print("程式結束。")
