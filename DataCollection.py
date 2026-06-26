import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Data/W"
count = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        ratio = h / w
        if ratio > 1:
            k = imgSize / h
            widnew = math.ceil(w * k)
            imgRsz = cv2.resize(imgCrop, (widnew, imgSize))
            wGap = math.ceil((imgSize - widnew) / 2)
            imgWhite[:, wGap:widnew + wGap] = imgRsz

        else:
            k = imgSize / w
            hgtnew = math.ceil(h * k)
            imgRsz = cv2.resize(imgCrop, (imgSize, hgtnew))
            hGap = math.ceil((imgSize - hgtnew) / 2)
            imgWhite[hGap:hgtnew + hGap, :] = imgRsz

        cv2.imshow("White Image", imgWhite)

    cv2.imshow("Image", img)


    key = cv2.waitKey(1)
    if key == ord('s'):
        count += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Image Saved: {count}")
