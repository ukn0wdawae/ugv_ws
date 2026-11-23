"""Coqui detection in a camera image (colour blob: lime-green body + yellow vocal sac)."""
import cv2
import numpy as np


def detect_coqui(bgr):
    """Return (found, bbox (x, y, w, h), centre (u, v), area_px)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([36, 120, 100]), np.array([80, 255, 255]))
    mask_sac = cv2.inRange(hsv, np.array([16, 150, 150]), np.array([36, 255, 255]))
    sac_dilated = cv2.dilate(mask_sac, np.ones((7, 7), np.uint8), iterations=2)
    combined = cv2.bitwise_or(mask_green, mask_sac)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8 or area > 40000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        sac = int(np.count_nonzero(sac_dilated[y:y + h, x:x + w]))
        grn = int(np.count_nonzero(mask_green[y:y + h, x:x + w]))
        if grn >= 5 and sac >= 3:
            score = float(area) + 3.0 * sac
            if score > best_score:
                best_score, best = score, (x, y, w, h, int(area))
    if best is None:
        return False, None, None, 0
    x, y, w, h, area = best
    return True, (x, y, w, h), (x + w // 2, y + h // 2), area


def annotate(bgr, found, bbox, lines):
    img = bgr.copy()
    h, w = img.shape[:2]
    cv2.line(img, (w // 2 - 15, h // 2), (w // 2 + 15, h // 2), (200, 200, 200), 1)
    cv2.line(img, (w // 2, h // 2 - 15), (w // 2, h // 2 + 15), (200, 200, 200), 1)
    if found and bbox is not None:
        x, y, bw, bh = bbox
        cv2.rectangle(img, (max(0, x - 6), max(0, y - 6)), (min(w - 1, x + bw + 6), min(h - 1, y + bh + 6)),
                      (0, 255, 0), 2)
    band = 22 * len(lines) + 8
    cv2.rectangle(img, (0, 0), (w, band), (20, 20, 20), -1)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(img, text, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return img
