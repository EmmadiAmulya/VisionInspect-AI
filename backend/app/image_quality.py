import cv2


def analyze_image_quality(image_path: str, blur_threshold: float = 20.0):
    """
    Analyze the uploaded image before AI inspection.
    Returns basic quality information.
    """

    image = cv2.imread(image_path)

    # 1. Check whether image can be opened
    if image is None:
        return {
            "valid": False,
            "quality_ok": False,
            "quality": "Poor",
            "message": "Image could not be read"
        }

    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]

    # 2. Check minimum resolution
    min_width = 100
    min_height = 100

    resolution_ok = width >= min_width and height >= min_height

    # 3. Calculate brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    # 4. Calculate blur using Laplacian variance
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 5. Basic brightness check
    brightness_ok = 30 <= brightness <= 230

    # 6. Basic blur check
    blur_ok = blur_score >= blur_threshold

    # Final quality decision
    quality_ok = resolution_ok and brightness_ok and blur_ok

    if quality_ok:
        quality = "Good"
    else:
        quality = "Poor"

    return {
        "valid": True,
        "width": width,
        "height": height,
        "channels": channels,
        "brightness": round(brightness, 2),
        "blur_score": round(blur_score, 2),
        "resolution_ok": resolution_ok,
        "brightness_ok": brightness_ok,
        "blur_ok": blur_ok,
        "quality": quality,
        "quality_ok": quality_ok
    }