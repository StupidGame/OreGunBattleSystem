import glob
import os
import cv2
import numpy as np

def main():
 files = glob.glob("./*.jpg")
 for file in files:
    # 画像を読み込む。
  img = cv2.imread("sample.png")

  # グレースケールに変換する。
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 2値化する。
  thresh, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

  # 輪郭を抽出する。
  contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # マスクを作成する。
  mask = np.zeros_like(binary)

  # 輪郭内部 (透明化しない画素) を255で塗りつぶす。
  cv2.drawContours(mask, contours, -1, color=255, thickness=-1)

  # RGBA に変換する。
  rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

  # マスクをアルファチャンネルに設定する。
  rgba[..., 3] = mask

  # 保存する。
  cv2.imwrite(file, rgba)


if __name__ == "__main__":
    main()
