import sys

print(f"Python 可執行檔路徑: {sys.executable}") 

try:
    import mediapipe as mp
    print(f"成功匯入 MediaPipe 版本: {mp.__version__}")
except ImportError:
    print("錯誤：無法匯入 MediaPipe。")
    print("請確認您已在啟用虛擬環境 (venv) 後執行過 'pip install mediapipe'")
    mp = None
except Exception as e:
    print(f"匯入 MediaPipe 時發生其他錯誤: {e}")
    mp = None

try:
    import cv2
    # 檢查是否能讀取版本號 (確認是有效的 OpenCV 物件)
    cv_version = getattr(cv2, '__version__', '無法讀取版本') 
    print(f"成功匯入 OpenCV 版本: {cv_version}") 
except ImportError:
    print("錯誤：無法匯入 OpenCV (cv2)。")
    print("請確認 python3-opencv 套件已安裝 (sudo apt install python3-opencv)")
    cv2 = None
except Exception as e:
    print(f"匯入 OpenCV 時發生其他錯誤: {e}")
    cv2 = None

print("-" * 30)
if mp and cv2:
    print("結論：基本的 MediaPipe 和 OpenCV 函式庫看起來已安裝且可匯入。")
    print("下一步是需要成功下載有效的 .tflite 模型檔案才能執行物件偵測。")
else:
    print("結論：請先解決上述匯入錯誤。")
