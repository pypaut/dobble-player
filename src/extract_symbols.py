import cv2 as cv

from src.utils import show_image, show_bbox


def extract_symbols(card, debug=False):
    """
    Extract a list of symbols from @card
    """
    # Threshold
    threshold = cv.inRange(card, (1, 1, 1), (220, 220, 220))
    if debug:
        show_image("Threshold card", threshold)

    # Morphology
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    threshold = cv.erode(threshold, kernel, iterations=1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    threshold = cv.dilate(threshold, kernel, iterations=1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    for _ in range(5):
        threshold = cv.morphologyEx(threshold, cv.MORPH_CLOSE, kernel)
    if debug:
        show_image("Morphology", threshold)

    # Find contours of symbols
    contours, hierarchy = cv.findContours(
        threshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
    )

    # Remove border contours
    boxes = [
        cv.boundingRect(c)
        for c in contours
        if cv.boundingRect(c)[0] > 0 and cv.boundingRect(c)[1] > 0
    ]
    if debug:
        show_bbox(boxes, card)

    # Remove boxes inside other boxes
    boxes = [boxes[i] for i in range(len(boxes)) if hierarchy[0][i][3] == -1]

    # Remove if more than 8 symbols
    while len(boxes) > 8:
        boxes_areas = [b[2] * b[3] for b in boxes]
        min_area = min(boxes_areas)
        min_index = boxes_areas.index(min_area)
        boxes.remove(boxes[min_index])

    if debug:
        show_bbox(boxes, card)

    # Symbols
    symbols = [card[b[1] : b[1] + b[3], b[0] : b[0] + b[2]] for b in boxes]

    # Resize symbols
    symbols = [cv.resize(s, (32, 32)) for s in symbols]

    return symbols
