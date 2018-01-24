import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import random

DIR = "C:/Users/Makasz/PycharmProjects/Test01/raw"
moz = []

for filename in os.listdir(DIR):
    print("Processing file " + filename)
    img = cv2.imread(DIR + "/" + filename)
    v = np.median(img)
    height = len(img)
    width = len(img[0])
    kernel = np.ones((5, 5), np.uint8)
    bitmap = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bitmap = cv2.adaptiveThreshold(bitmap, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 151, 20)  # Mozna użyc adaptive/otsu threshold

    img_dil, img_erd = [], bitmap

    for i in range(15):
        img_dil = cv2.dilate(img_erd, kernel, iterations=1)
        img_erd = cv2.erode(img_dil, kernel, iterations=1)

    bitmap = img_erd

    img_cnt, cnts, hierarchy = cv2.findContours(bitmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #bitmap = cv2.cvtColor(bitmap, cv2.COLOR_GRAY2BGR)
    MIN_LETTER_SIZE = height * width / 2000
    print(len(cnts))
    for i in range(len(cnts)):
        moments = cv2.moments(cnts[i])
        # if moments['mu02'] < 50000.0:
        #     continue
        if cv2.contourArea(cnts[i]) < MIN_LETTER_SIZE or moments['mu02'] < 50000.0:
            cv2.drawContours(bitmap, cnts, i, (255, 255, 255), cv2.FILLED)
        else:
            x, y, w, h = cv2.boundingRect(cnts[i])
            #cv2.rectangle(bitmap, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite("final\\" + str(i) + "_" + filename , bitmap[y: y + h, x: x + w])
    img_final = bitmap
    cv2.imwrite("final\\" + filename, img_final)
    moz.append(img_final)
