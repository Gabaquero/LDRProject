import cv2
import numpy as np
import utlis


def getLaneCurve(img):
    imgCopy = img.copy()
    ##1
    imgThres = utlis.thresholding(img)
    ##2
    h, w, c = img.shape
    points = utlis.valTrackbars()
    imgWarp = utlis.WarpImg(imgThres, points, w, h)
    imgWarpPoints = utlis.drawPoints(imgCopy, points)
    ##3
    basePoint, imgHist = utlis.getHistogram(imgWarp, display=True)

    cv2.imshow('Thres', imgThres)
    cv2.imshow('Warp', imgWarp)
    cv2.imshow('Warp Points', imgWarpPoints)
    cv2.imshow('Histogram', imgHist)
    return None


if __name__ == '__main__':
    cap = cv2.VideoCapture('Images/vid1.mp4')
    initialTrackBarVals = [105, 80, 20, 214]
    utlis.initializeTrackbars(initialTrackBarVals)
    frameCounter = 0
    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        success, img = cap.read()
        img = cv2.resize(img, (480, 240))
        getLaneCurve(img)

        cv2.imshow('Vid', img)
        cv2.waitKey(1)
