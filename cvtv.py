#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import cv2 as cv


def augment_detect_circle_channel(image, image_gray, color):
    BLUR = 5
    image_blur = cv.medianBlur(image_gray, BLUR)

    DOWNSCALE = 1
    MIN_DIST = 20
    HIGH_THRESHOLD = 100
    ACCUMULATOR_THRESHOLD = 200
    circles = cv.HoughCircles(image_blur, cv.HOUGH_GRADIENT, DOWNSCALE, MIN_DIST, param1=HIGH_THRESHOLD, param2=ACCUMULATOR_THRESHOLD, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))

    COLOR_BORDER = color
    COLOR_CENTER = color
    for i in circles[0,:]:
        # border
        cv.circle(image, (i[0], i[1]), i[2], COLOR_BORDER, 2)
        # center
        cv.circle(image, (i[0], i[1]), 2, COLOR_CENTER, 3)


def augment_detect_circle(image):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    BLUR = 5
    image_blur = cv.medianBlur(image_gray, BLUR)

    DOWNSCALE = 1
    MIN_DIST = 20
    HIGH_THRESHOLD = 100
    ACCUMULATOR_THRESHOLD = 200
    circles = cv.HoughCircles(image_blur, cv.HOUGH_GRADIENT, DOWNSCALE, MIN_DIST, param1=HIGH_THRESHOLD, param2=ACCUMULATOR_THRESHOLD, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))

    COLOR_BORDER = (0, 255, 0)
    COLOR_CENTER = (0, 0, 255)
    for i in circles[0,:]:
        # border
        cv.circle(image, (i[0], i[1]), i[2], COLOR_BORDER, 2)
        # center
        cv.circle(image, (i[0], i[1]), 2, COLOR_CENTER, 3)


def augment_detect_corner(image):
    # convert to format for cornerHarris
    image_data_gray = np.float32(cv.cvtColor(image, cv.COLOR_BGR2GRAY))

    # mark corners
    BLOCKSIZE = 2
    KSIZE = 3
    K = 0.04
    detected = cv.cornerHarris(image_data_gray, BLOCKSIZE, KSIZE, K)

    # increase size of markers
    detected = cv.dilate(detected, None)

    # overlay detected onto image
    THRESHOLD = 0.01
    COLOR = [0, 0, 255]
    image[detected > THRESHOLD * detected.max()] = COLOR


def main():
    filename = "/home/keith/tmp/cvtv.png"
    image = cv.imread(filename)

    if False:
        augment_detect_corner(image)

    if False:
        augment_detect_circle(image)

    if True:
        image_blue = image.copy()[:,:,0]
        image_green = image.copy()[:,:,1]
        image_red = image.copy()[:,:,2]
        augment_detect_circle_channel(image, image_blue, [100, 100, 255]) # blue
        augment_detect_circle_channel(image, image_green, [100, 255, 100]) # green
        augment_detect_circle_channel(image, image_red, [255, 100, 100]) # red

    cv.imshow("output", image)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
