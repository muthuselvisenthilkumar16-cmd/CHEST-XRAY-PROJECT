
import cv2
import numpy as np


def assess_image_quality(gray, measurements):
    """
    Preliminary image-quality screening.
    Thresholds are configurable examples, not clinical standards.
    """
    height, width = gray.shape
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    reasons = []

    # Example criteria to be tested and calibrated on suitable images.
    if width < 128 or height < 128:
        reasons.append("Image resolution is too low.")

    if brightness < 25:
        reasons.append("Image may be too dark.")
    elif brightness > 230:
        reasons.append("Image may be too bright.")

    if contrast < 10:
        reasons.append("Image contrast is very low.")

    if blur_score < 20:
        reasons.append("Image may be blurry.")

    status = "Unfit" if reasons else "Fit"

    quality_metrics = {
        "Image width (pixels)": width,
        "Image height (pixels)": height,
        "Mean brightness": round(brightness, 2),
        "Contrast (standard deviation)": round(contrast, 2),
        "Blur score (Laplacian variance)": round(blur_score, 2),
    }

    if not reasons:
        reasons.append(
            "No configured quality checks were triggered. "
            "This does not establish medical suitability."
        )

    return status, reasons, quality_metrics
