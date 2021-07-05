#!/usr/bin/python3

import cv2 as cv
import numpy as np
import sys

from src.extract_cards import extract_cards
from src.extract_symbols import extract_symbols
from src.utils import show_image, show_bbox, distance, display_image

"""
Automatic Dobble player

Sources

Display contours :
https://stackoverflow.com/questions/28677544/how-do-i-display-the-contours-of-an-image-using-opencv-python

FloodFill :
https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

Object extraction :
https://towardsdatascience.com/extracting-circles-and-long-edges-from-images-using-opencv-and-python-236218f0fee4
"""

DEBUG = False


def main():
    """
    Extract both cards
    Extract symbols of both cards
    Find best match and display
    """
    while True:
        # Load image
        image = cv.imread(sys.argv[1])

        # Extract cards from image
        card1, card2 = extract_cards(image, DEBUG)

        # Extract symbols from each card
        symbols1 = extract_symbols(card1, debug=DEBUG)
        symbols2 = extract_symbols(card2, debug=DEBUG)

        # Distances matrix : m[i][j] = distance(symbols1[i], symbols2[j])
        distances = np.array(
            [
                np.array([distance(s1, s2) for s2 in symbols2])
                for s1 in symbols1
            ]
        )

        # Find best match
        min_distance = np.where(distances == np.amin(distances))
        min_pos = (min_distance[0][0], min_distance[1][0])
        s1 = symbols1[min_pos[0]]
        s2 = symbols2[min_pos[1]]

        # Display
        result_image = display_image(s1, s2, image)
        cv.imshow("Display", result_image)
        if cv.waitKey(1) == 27:
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
