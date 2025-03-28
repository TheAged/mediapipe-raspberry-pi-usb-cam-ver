# -*- coding: utf-8 -*-
"""
使用 MediaPipe Pose 進行跌倒偵測的範例程式碼 (已修正 TypeError 和 AttributeError)

此程式碼透過分析身體關節點的垂直速度、姿態角度、相對高度及狀態持續時間
來嘗試偵測跌倒事件。

依賴庫: opencv-python, mediapipe, numpy

按 'q' 退出, 按 'r' 重置跌倒狀態。

版本日期: 2025-03-28
"""

import cv2
import mediapipe as mp
import time
import math
import numpy as np

# --- MediaPipe 初始化 ---
mp_pose = mp.solutions.pose
# 增加模型複雜度可能有助於提高精度，但會降低速度 (0: lite, 1: full, 2: heavy)
# 在 Raspberry Pi 上建議使用 0 或 1
pose = mp_pose.Pose(
    model_complexity=1,  # 可以嘗試 0 或 1
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 閾值設定 (!!! 非常重要，需要根據實際情況調整 !!!) ---
Y_VELOCITY_THRESHOLD = 0.6 # 垂直速度閾值 (歸一化座標單位/秒)
ANGLE_THRESHOLD = 70 # 身體角度閾值 (度)
HEIGHT_THRESHOLD_FACTOR = 0.8 # 相對高度閾值 (基於畫面高度的比例)
FALL_CONFIRM_DURATION = 1.2 # 狀態維持時間閾值 (秒)

# --- 狀態變數 ---
prev_landmarks = None
prev_time = time.time()
fall_detected = False
fall_timer_start = None # 計時器，記錄進入疑似跌倒狀態的起始時間

# --- 影片來源 ---
# 使用 webcam (通常是 0 或 1)
# 嘗試使用 V4L2 API (在某些 Linux 系統上可能更穩定)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("無法使用 V4L2 開啟 webcam, 嘗試預設 API...")
    cap = cv2.VideoCapture(0)
# 或者使用影片檔案
# cap = cv2.VideoCapture("your_video.mp4")

if not cap.isOpened():
    print("錯誤：無法開啟影片來源")
    exit()

# 嘗試設定較低的解析度以提高幀率 (可選)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("按 'q' 退出, 按 'r' 重置跌倒狀態")
print("--- 開始偵測 ---")

# 用於判斷是否為檔案的標誌 (只需要判斷一次)
is_file_source = False
try:
    # 嘗試獲取總幀數
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # 如果 frame_count 大於 1 (給點餘裕)，則認為是檔案
    if frame_count > 1:
        is_file_source = True
        print(f"偵測到影片檔案來源 (總幀數: {frame_count})")
    else:
        print("偵測到 Webcam 來源 (或無法獲取總幀數)")
except Exception as e:
    print(f"獲取幀數時出錯 ({e}), 假設為 Webcam 來源")
    is_file_source = False


while True:
    success, image = cap.read()
    if not success:
        # 根據是否為檔案來源決定是結束還是繼續
        if is_file_source:
            print("影片檔案讀取完畢或發生錯誤。")
            break # 如果是檔案，結束迴圈
        else:
            print("無法讀取 Webcam 畫面，繼續嘗試...")
            time.sleep(0.5) # Webcam 暫停一下再試
            continue

    # --- 畫面鏡像翻轉 (僅針對 Webcam) ---
    if not is_file_source: # 如果確定不是檔案 (即判斷為 webcam)
        image = cv2.flip(image, 1) # 進行水平翻轉 (鏡像)

    image_height, image_width, _ = image.shape

    # 將 BGR 圖片轉換為 RGB (MediaPipe 需要 RGB)
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 進行姿態偵測
    results = pose.process(image_rgb)

    # 將圖片標記回可寫入，並轉回 BGR 以便使用 OpenCV 繪圖
    image.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    current_time = time.time()
    delta_t = current_time - prev_time
    if delta_t <= 0:
        delta_t = 0.01 # 避免除以零

    # 初始化本幀的計算變數和跌倒指標狀態
    current_hip_y = None
    vertical_velocity = None
    angle_deg = None
    potential_fall_indicators = {'high_vy': False, 'is_horizontal': False, 'low_height': False}

    # 如果偵測到姿態關節點
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # --- 1. 計算臀部中心點 Y 座標 和 垂直速度 ---
        try:
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
                current_hip_y = (left_hip.y + right_hip.y) / 2
                if prev_landmarks and delta_t > 0:
                    prev_left_hip = prev_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    prev_right_hip = prev_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    if prev_left_hip.visibility > 0.5 and prev_right_hip.visibility > 0.5:
                        prev_hip_y = (prev_left_hip.y + prev_right_hip.y) / 2
                        vertical_velocity = (current_hip_y - prev_hip_y) / delta_t
                if vertical_velocity is not None and vertical_velocity > Y_VELOCITY_THRESHOLD:
                    potential_fall_indicators['high_vy'] = True
                if current_hip_y is not None and current_hip_y > HEIGHT_THRESHOLD_FACTOR:
                    potential_fall_indicators['low_height'] = True
        except IndexError: pass
        except Exception as e: print(f"計算臀部 Y 座標或速度時出錯: {e}")

        # --- 2. 計算身體姿態角度 ---
        try:
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            if (left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and
                left_hip.visibility > 0.5 and right_hip.visibility > 0.5):
                shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
                shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
                hip_center_x = (left_hip.x + right_hip.x) / 2
                hip_center_y = (left_hip.y + right_hip.y) / 2
                delta_x = shoulder_center_x - hip_center_x
                delta_y = shoulder_center_y - hip_center_y
                angle_rad = math.atan2(delta_x, delta_y)
                angle_deg = abs(math.degrees(angle_rad))
                if angle_deg is not None and angle_deg > ANGLE_THRESHOLD:
                    potential_fall_indicators['is_horizontal'] = True
        except IndexError: pass
        except Exception as e: print(f"計算身體角度時出錯: {e}")

        # --- 跌倒判斷邏輯 ---
        is_potentially_falling = potential_fall_indicators['high_vy'] or \
                                 (potential_fall_indicators['is_horizontal'] and potential_fall_indicators['low_height'])
        if is_potentially_falling and not fall_detected:
            if fall_timer_start is None:
                fall_timer_start = current_time
                print(f"[{time.strftime('%H:%M:%S')}] 偵測到疑似跌倒，開始計時...")
            elif (current_time - fall_timer_start) >= FALL_CONFIRM_DURATION:
                fall_detected = True
                print(f"[{time.strftime('%H:%M:%S')}] !!! 確認跌倒 !!!")
        elif not is_potentially_falling and not fall_detected:
            if fall_timer_start is not None:
                print(f"[{time.strftime('%H:%M:%S')}] 恢復正常姿態，重置計時器。")
                fall_timer_start = None

        # --- 繪製關節點和連線 ---
        mp_drawing.draw_landmarks(
            image_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        prev_landmarks = results.pose_landmarks.landmark
        prev_time = current_time
    else: # 沒有偵測到人
        if fall_timer_start is not None:
            print(f"[{time.strftime('%H:%M:%S')}] 目標消失，重置計時器。")
            fall_timer_start = None
        prev_landmarks = None
        prev_time = current_time
        cv2.putText(image_bgr, "No Person Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- 在畫面上顯示狀態文字 ---
    status_text = "Status: Normal"
    color = (0, 255, 0) # Green
    if fall_detected:
        status_text = "Status: FALL DETECTED!"
        color = (0, 0, 255) # Red
    elif fall_timer_start is not None:
        elapsed_time = current_time - fall_timer_start
        status_text = f"Status: Potential Fall... ({elapsed_time:.1f}s)"
        color = (0, 165, 255) # Orange
    cv2.putText(image_bgr, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # --- (可選) 顯示除錯資訊 ---
    debug_text_y = 80
    vv_str = f"{vertical_velocity:.2f}" if vertical_velocity is not None else "N/A"
    cv2.putText(image_bgr, f"Vert Vel: {vv_str}", (10, debug_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    angle_str = f"{angle_deg:.1f}" if angle_deg is not None else "N/A"
    cv2.putText(image_bgr, f"Angle: {angle_str}", (10, debug_text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    hip_y_str = f"{current_hip_y:.2f}" if current_hip_y is not None else "N/A"
    cv2.putText(image_bgr, f"Low H: {potential_fall_indicators['low_height']} (Y:{hip_y_str})", (10, debug_text_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(image_bgr, f"Flags: V={potential_fall_indicators['high_vy']}, H={potential_fall_indicators['is_horizontal']}, L={potential_fall_indicators['low_height']}", (10, debug_text_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    fps = 1.0 / delta_t if delta_t > 0 else 0
    cv2.putText(image_bgr, f"FPS: {fps:.1f}", (image_width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # --- 顯示結果畫面 ---
    cv2.imshow('MediaPipe Pose Fall Detection', image_bgr)

    # --- 按鍵控制 ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("收到 'q'，退出程式。")
        break
    elif key == ord('r'):
        fall_detected = False
        fall_timer_start = None
        print("跌倒狀態已被手動重置。")

# --- 釋放資源 ---
print("--- 結束偵測，釋放資源 ---")
cap.release()
cv2.destroyAllWindows()
pose.close()
print("資源已釋放。")
