
import cv2
import numpy as np


def load_image(uploaded_file):
    """Read an uploaded image and return RGB and grayscale versions."""
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to read this image file.")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return rgb, gray


def process_image(gray):
    """Run the project's image-processing techniques."""
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    enhanced = cv2.equalizeHist(denoised)

    threshold_value, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Basic intensity-based segmentation demonstration.
    # This is not guaranteed to isolate the lungs anatomically.
    segmented = cv2.bitwise_and(enhanced, enhanced, mask=otsu)

    sobel_x = cv2.Sobel(
        enhanced, cv2.CV_64F, 1, 0, ksize=3
    )
    sobel_y = cv2.Sobel(
        enhanced, cv2.CV_64F, 0, 1, ksize=3
    )
    sobel = cv2.convertScaleAbs(
        cv2.magnitude(sobel_x, sobel_y)
    )

    scharr_x = cv2.Scharr(
        enhanced, cv2.CV_64F, 1, 0
    )
    scharr_y = cv2.Scharr(
        enhanced, cv2.CV_64F, 0, 1
    )
    scharr = cv2.convertScaleAbs(
        cv2.magnitude(scharr_x, scharr_y)
    )

    edge_threshold = 50
    sobel_edge_percent = float(
        np.mean(sobel > edge_threshold) * 100
    )
    scharr_edge_percent = float(
        np.mean(scharr > edge_threshold) * 100
    )

    measurements = {
        "Mean brightness": float(np.mean(gray)),
        "Contrast (standard deviation)": float(np.std(gray)),
        "Otsu threshold": float(threshold_value),
        "Segmentation foreground (%)": float(
            np.mean(otsu > 0) * 100
        ),
        "Sobel edge pixels (%)": sobel_edge_percent,
        "Scharr edge pixels (%)": scharr_edge_percent,
    }

    images = {
        "Original grayscale": gray,
        "Noise reduction": denoised,
        "Contrast enhancement": enhanced,
        "Otsu thresholding": otsu,
        "Segmentation demonstration": segmented,
        "Sobel edge detection": sobel,
        "Scharr edge detection": scharr,
    }

    return images, measurements