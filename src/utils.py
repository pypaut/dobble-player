import cv2 as cv
import numpy as np


def show_image(title, image):
    """
    Wrapper around cv2.imshow(), for convenience
    """
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyWindow(title)


def show_contours(contours, image):
    """
    Display @image with highlighted @contours
    """
    display_image = image.copy()
    for c in contours:
        for (x, y) in c:
            cv.circle(display_image, (x, y), 1, (255, 0, 0), 3)

    show_image("Contours", display_image)


def show_bbox(boxes, image):
    """
    Display @image with highlighted bounding boxes for each element of
    @contours
    """
    display_image = image.copy()
    for b in boxes:
        cv.rectangle(display_image, b, (255, 0, 0))

    show_image("Bounding boxes", display_image)


def distance(s1, s2):
    """
    Custom distance between two symbols
    """
    rotations = [np.rot90(s2, k=i) for i in range(4)]
    distances = [np.linalg.norm(s1 - r) for r in rotations]
    return min(distances)


def display_image(s1, s2, image):
    """
    Compute final display image
    """
    display_image = np.concatenate((s1, s2), axis=1)
    display_image = cv.resize(display_image, (image.shape[1], image.shape[0]))
    display_image = np.concatenate((image, display_image), axis=0)
    return display_image
