from email.mime import base
import os, glob
from PIL import Image
import numpy as np
import uuid
import sys
import cv2
import matplotlib.pyplot as plt

def remove_bg(
    path,
    BLUR = 21,
    CANNY_THRESH_1 = 10,
    CANNY_THRESH_2 = 200,
    MASK_DILATE_ITER = 10,
    MASK_ERODE_ITER = 10,
    MASK_COLOR = (0.0,0.0,1.0),
):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

    mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
    img         = img.astype('float32') / 255.0                 #  for easy blending

    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
    masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

    c_blue, c_green, c_red = cv2.split(img)

    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))

    return img_a

def main():
    filepath_list = glob.glob(input_path + '/*.jpg') # .pngファイルをリストで取得する
    for filepath in filepath_list:
        img_fin = remove_bg(    
            path = filepath,
            BLUR = 21,
            CANNY_THRESH_1 = 5,
            CANNY_THRESH_2 = 70,
            MASK_DILATE_ITER = 10,
            MASK_ERODE_ITER = 6,
        )
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img_fin)
        fig.savefig(filepath)
        for i in range(0, 36):
            uuidgen = str(uuid.uuid4())
            basename  = os.path.basename(filepath) # ファイルパスからファイル名を取得
            save_filepath = out_path + '/' + basename [:-4] + '-' + uuidgen + '.png' # 保存ファイルパスを作成
            img = Image.open(filepath)
            img = img.convert('RGBA') # RGBA(png)→RGB(jpg)へ変換
            img = img.rotate(i*10)
            img.save(save_filepath, "PNG", quality=95)
            print(filepath, '->', save_filepath)
        if flag_delete_original_files:
            os.remove(filepath)
            print('delete', filepath)

if __name__ == '__main__':
    input_path = str(sys.argv[1]) # オリジナルpngファイルがあるフォルダを指定
    out_path = input_path # 変換先のフォルダを指定
    flag_delete_original_files = False # 元ファイルを削除する場合は、True指定
    main()