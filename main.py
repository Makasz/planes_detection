import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import random
DIR = "C:/Users/Makasz/PycharmProjects/Test01/raw"
moz = []
for filename in os.listdir(DIR):
    print("Processing file " + filename)
    img_color = cv2.imread(DIR +"/" + filename)
    img_raw = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img_raw, 9, 30, 20)
    kernel = np.ones((5, 5), np.uint8)
    v = np.median(img)
    sigma = 0.8
    lower = int(max(30, (1.0 - sigma) * v))
    upper = int(min(200, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper, L2gradient=True)
    img_dil, img_erd = [], edges

    for i in range(15):
        img_dil = cv2.dilate(img_erd, kernel, iterations=1)
        img_erd = cv2.erode(img_dil, kernel, iterations=1)
    img_cnt, cnts, hierarchy = cv2.findContours(img_erd.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_color2 = cv2.cvtColor(img_erd, cv2.COLOR_GRAY2BGR)
    for i in range(len(cnts)):
        moments = cv2.moments(cnts[i])
        #print(moments['mu02'])
        if moments['mu02'] < 50000.0:
            print("Text detected!")
            continue
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255));
        color2 = (255-color[0],255-color[1],255-color[2]);
        cv2.drawContours(img_color, cnts, i, color, cv2.FILLED)
        cv2.circle(img_color, (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])), 5, color2, -1)
    img_color = cv2.bilateralFilter(img_color, 5, 30, 20)
    img_final = img_color

    cv2.imwrite("final\\" + filename, img_final)
    moz.append(img_final)
#plt.imshow(moz, cmap='color')
# f, axarr = plt.subplots(9, 2)
# for x in range(18):
#     print(moz[x]);
#     axarr[x % 9, x % 2].plot(moz[x])
#     axarr[x % 9, x % 2].set_title("")
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
