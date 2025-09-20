import os
import numpy as np
import cv2

from src.config import INPUT_PATH, PREPROCESSED_PATH, TARGET_SIZE

# def resize_dir(dirname:str, crop=False) -> None:

#     """
#     resize image to desired size
#     - take into account possible orientation of raw images
#         - resize to width of 512 and heigh 256
    
#     crop: if image contains other thigs than just cultivation tray, it will be cropped automatically
#     - not recommended, does not work perfectly and for some trays it fails spectacularly
#     """

#     files = []

#     if not os.path.exists(PREPROCESSED_PATH):
#         os.makedirs(PREPROCESSED_PATH)
#     input_dir = INPUT_PATH / dirname
#     for file in os.listdir(input_dir):
#         print(file)
#         files.append(file)
#         full_path = os.path.join(input_dir, file)
#         img = cv2.imread(full_path)
#         if img is None:
#             continue  # Skip if image loading fails
#         file = file.split('/')[-1][:-4]
        
#         # Check dimensions and rotate if height > width
#         height, width = img.shape[:2]
#         if height > width:
#             img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
#         if crop:
#             img = _crop_single_tray_(img, show=False)
        
#         resized = cv2.resize(img, TARGET_SIZE)
#         cv2.imwrite(f'{PREPROCESSED_PATH}/{file}_resized.jpg', resized)

def resize_single(filename:str, crop=False) -> None:

    img = cv2.imread(INPUT_PATH / filename)

    file = filename.split('.')[0] # get filename without extension  

    height, width = img.shape[:2]
    if height > width:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if crop:
        img = _crop_single_tray_(img, show=False)
    
    resized = cv2.resize(img, TARGET_SIZE)
    cv2.imwrite(f'{PREPROCESSED_PATH}/{file}_resized.jpg', resized)

def _crop_single_tray_(img):
    
    """
    Crop image to the bounding box of the single tray.
    Input: BGR image (numpy array).
    Output: Cropped BGR image containing the tray.
    """
    # Preprocess image
    thresh = _preprocess_image_(img)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No tray found!")
        return img  # Return original image if no tray detected

    # Assume largest contour is the tray
    tray_contour = max(contours, key=cv2.contourArea)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(tray_contour)

    # Add padding to avoid cutting edges (e.g., 5 pixels)
    padding = 1
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)

    # Crop image
    cropped_img = img[y:y+h, x:x+w]

    return cropped_img

def _preprocess_image_(img, blur_ksize=5, thresh_block_size=11):
    """
    Preprocess image for tray detection: convert to grayscale, blur, and threshold.
    Returns binary image for contour detection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    # Adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresh_block_size, 2
    )
    # Morphological operations to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh