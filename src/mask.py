import cv2 as cv

from src.utils import show_image


def cards_mask(image, debug=False):
    """
    Generate 2D mask for card presence on @image
    """
    # Threshold
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    threshold = cv.inRange(gray, 170, 255)
    if debug:
        show_image("Threshold", threshold)

    # Floodfill
    tmp = threshold.copy()
    points = [
        (0, 0),  # Bottom left
        (tmp.shape[1] // 2, 0),  # Bottom middle
        (tmp.shape[1] - 1, tmp.shape[0] - 1),  # Top right
    ]
    for p in points:
        cv.floodFill(tmp, None, p, 255)

    if debug:
        show_image("Flood fill", tmp)

    # Invert floodfilled image
    inverted = cv.bitwise_not(tmp)
    if debug:
        show_image("Inverted", inverted)

    # Combine the two images to get the foreground.
    return threshold | inverted
