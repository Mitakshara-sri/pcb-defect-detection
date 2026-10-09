import cv2
import numpy as np


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    corrected = cv2.divide(gray, background, scale=128)
    return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
