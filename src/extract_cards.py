import cv2 as cv
import numpy as np

from src.mask import cards_mask
from src.utils import show_contours, show_image


def extract_cards(image, debug=False):
    """
    Detect 2 cards in @image and return 2 images, each one containing one card
    """
    # Compute mask containing cards
    mask = cards_mask(image, debug=debug)
    if debug:
        show_image("Mask", mask)

    # Find contours of both cards
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    circle1 = contours[0].reshape(-1, 2)
    circle2 = contours[1].reshape(-1, 2)
    if debug:
        show_contours([circle1, circle2], image)

    # Erase anything which isn't a card
    image[mask == 0] = np.array([0, 0, 0])

    # Extract both cards
    b1 = cv.boundingRect(circle1)
    b2 = cv.boundingRect(circle2)
    card1 = image[b1[1] : b1[1] + b1[3], b1[0] : b1[0] + b1[2]]
    card2 = image[b2[1] : b2[1] + b2[3], b2[0] : b2[0] + b2[2]]
    if debug:
        show_image("Card 1", card1)
        show_image("Card 2", card2)

    return card1, card2
