# model.py
import numpy as np
import cv2
from ultralytics import YOLO

# YOLOv11モデルのロード
model = YOLO(r"C:\Users\reine\OneDrive\ドキュメント\gorogoro_flask_4\src\gorogoro_yolov11_best.pt")

def remove_background(image_path):
    """
    指定された画像の背景をYOLOv11のセグメンテーション機能を使用して除去します。

    :param image_path: 画像のパス
    :return: 背景除去された画像
    """
    # 画像の読み込み
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCVのBGRをRGBに変換

    # 推論を行う
    results = model.predict(source=img_rgb, conf=0.20)  # 信頼度を0.20に変更

    # マスクの取得
    if results and results[0].masks is not None:
        masks = results[0].masks.xy  # セグメンテーションマスクを取得
        mask_combined = np.zeros(img.shape[:2], dtype=np.uint8)  # マスクの初期化

        # 各マスクを結合
        for mask in masks:
            if len(mask) > 0:
                cv2.fillPoly(mask_combined, [np.array(mask, dtype=np.int32)], color=1)

        # 背景を除去するために元の画像とマスクを掛け合わせる
        img_background_removed = img_rgb * mask_combined[:, :, np.newaxis]

        return img_background_removed  # 背景除去された画像を返す
    else:
        return None  # マスクが存在しない場合
