
from keras.models import load_model
import numpy as np
import cv2
import pyocr
from PIL import Image

sec = {}

tick = 1.0/30.0

engines = pyocr.get_available_tools()
engine = engines[0]

# ストリーム処理にして，録画しながら計測したい．
cap = cv2.VideoCapture('data/yamada2.mov')

# 無限ループ
while(True):

    ret, frame = cap.read()    

    if not ret: # 終了処理
        break

    # 画像のサイズを取得,表示。グレースケールの場合,shape[:2]
    h, w, _ = frame.shape[:3]

    # ページ番号をフォーカス 
    # 要：自動化
    w_center = (w//100)*93
    h_center = (h//1000)*999

    # # カメラ画像の整形
    im = frame[h_center:h_center+60, w_center-30:w_center+30] # トリミング

    # 判定結果を上位3番目まで表示させる 

    cv2.imwrite('checkpt.png', im)
    
    num = engine.image_to_string(Image.open('checkpt.png'), lang='eng', builder=pyocr.builders.DigitBuilder(tesseract_layout=6))
    
    print(num) # 「Test Message」が出力される

    if num.isdecimal():
        p_num = int(num)
    else:
        p_num = -1

    if p_num == -1:
        continue
    elif p_num in sec:
        sec[p_num] += tick
    else:
        sec[p_num] = tick

sec = sorted(sec.items())
sec = dict((x, y) for x, y in sec)

print(sec)

sum = 0.0
for i in sec.values():
    sum += i

print(sum)

cap.release() # カメラを解放
cv2.destroyAllWindows() # ウィンドウを消す