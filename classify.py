#classify.py
import sys
from PIL import Image
#pip install は pillow
from keras.models import load_model
import numpy as np
import kirinuki

def main():
    name = sys.argv[1]
    #print(name)
    image = Image.open(name).convert('RGB')
    image = image.resize((64,64))
    model = load_model("data/model/model.h5")
    np_image = np.array(image)
    np_image = np_image / 255
    np_image = np_image[np.newaxis, :, :, :]
    result = model.predict(np_image)
    #print(result)
    if result[0][0] > result[0][1]:
        print("gunpla")
    else:
        print("not gunpla")

if __name__ == "__main__":
    input_path = str(sys.argv[1]) # オリジナルpngファイルがあるフォルダを指定
    out_path = input_path # 変換先のフォルダを指定
    flag_delete_original_files = False # 元ファイルを削除する場合は、True指定
    kirinuki.main(input_path)
    main()
