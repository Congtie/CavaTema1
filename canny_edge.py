import cv2 as cv
import numpy as np
import os
import sys

def show_image(title, image):
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    # Resize window to a reasonable size, e.g., 800x600 or scaled down
    h, w = image.shape[:2]
    scale = 800 / max(h, w)
    cv.resizeWindow(title, int(w * scale), int(h * scale))
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def process_image(image_path):
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Convert to grayscale (needed for final output)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Convert to HSV color space directly from original image
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    def nothing(x):
        pass

    cv.namedWindow('HSV Tuning', cv.WINDOW_NORMAL)
    cv.createTrackbar('H Min', 'HSV Tuning', 0, 179, nothing)
    cv.createTrackbar('S Min', 'HSV Tuning', 0, 255, nothing)
    cv.createTrackbar('V Min', 'HSV Tuning', 0, 255, nothing)
    cv.createTrackbar('H Max', 'HSV Tuning', 179, 179, nothing)
    cv.createTrackbar('S Max', 'HSV Tuning', 255, 255, nothing)
    cv.createTrackbar('V Max', 'HSV Tuning', 255, 255, nothing)

    # --- CONFIGURATION ---
    INVERT_MASK = True # Set to True if you are selecting the BACKGROUND (Table/Wall)
    # ---------------------

    # Define HSV range based on user feedback (Brown Background)
    lower = np.array([0, 50, 20])
    upper = np.array([30, 255, 200])

    mask = cv.inRange(hsv, lower, upper)
    
    # Show the mask
    show_image("HSV Mask", mask)
    
    if INVERT_MASK:
        mask = cv.bitwise_not(mask)
        show_image("Inverted Mask (Board)", mask)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    # Open to remove noise, Close to fill holes
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    
    show_image("Final Mask", mask)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area, largest first
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    
    board_contour = None
    approx = None
    
    image_area = img.shape[0] * img.shape[1]
    
    # Debug visualization
    debug_img = img.copy()

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 0.05 * image_area:
            continue
            
        # Geometric filters
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        # Board should be roughly square (allow some perspective distortion)
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue
            
        # Solidity check (Board should be solid convex shape)
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        if solidity < 0.7: # Allow some irregularities but reject weird concave shapes
            continue

        # Use the Convex Hull for approximation!
        # This fixes the issue where pieces on the edge break the contour
        perimeter = cv.arcLength(hull, True)
        curr_approx = cv.approxPolyDP(hull, 0.02 * perimeter, True)
        
        # Draw all candidates in BLUE
        if len(curr_approx) == 4:
            cv.drawContours(debug_img, [curr_approx], -1, (255, 0, 0), 2)
            
            # Pick the first one (largest) as the board
            if board_contour is None:
                board_contour = hull # Use the hull as the board contour
                approx = curr_approx

    if board_contour is None:
        print(f"No valid board contour (4 corners) found for {image_path}")
        show_image("Debug Candidates", debug_img)
        return

    # Reshape approx to (4, 2)
    pts = approx.reshape(4, 2)

    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left
    
    (top_left, top_right, bottom_right, bottom_left) = rect

    # Visualization: Draw selected in GREEN
    cv.drawContours(debug_img, [approx], -1, (0, 255, 0), 3)
    for point in rect:
        cv.circle(debug_img, tuple(point.astype(int)), 20, (0, 0, 255), -1)
    
    show_image("Detected Corners", debug_img)

    width = 1600
    height = 1600
    
    destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype="float32")

    M = cv.getPerspectiveTransform(rect, destination_of_puzzle)

    result = cv.warpPerspective(gray, M, (width, height))
    
    show_image("Result", result)

if __name__ == "__main__":
    # Default to 1_01.jpg if no argument provided
    input_image = "antrenare/5_20.jpg"
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
    
    process_image(input_image)
