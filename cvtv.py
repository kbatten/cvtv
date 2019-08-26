#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import cv2 as cv


def augment_detect_corner(image):
    image_gray = np.float32(cv.cvtColor(image, cv.COLOR_BGR2GRAY))

    BLOCKSIZE = 2
    KSIZE = 3
    K = 0.04
    detected = cv.cornerHarris(image_gray, BLOCKSIZE, KSIZE, K)

    THRESHOLD = 0.01
    COLOR = [0, 0, 255]
    image[detected > THRESHOLD * detected.max()] = COLOR


def main():
    filename = "/home/keith/tmp/cvtv.png"
    image = cv.imread(filename)

    augment_detect_corner(image)

    cv.imshow("output", image)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
