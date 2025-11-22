import cv2 as cv
import numpy as np
import os

def show_image(title, image):
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    h, w = image.shape[:2]
    scale = 800 / max(h, w)
    cv.resizeWindow(title, int(w * scale), int(h * scale))
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def extrage_careu(img):
    # Convert to HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Define HSV range based on user feedback (Brown Background)
    lower = np.array([0, 50, 20])
    upper = np.array([30, 255, 200])

    mask = cv.inRange(hsv, lower, upper)
    
    # Invert mask (selecting background)
    mask = cv.bitwise_not(mask)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    
    board_contour = None
    approx = None
    image_area = img.shape[0] * img.shape[1]
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 0.05 * image_area:
            continue
            
        hull = cv.convexHull(cnt)
        perimeter = cv.arcLength(hull, True)
        curr_approx = cv.approxPolyDP(hull, 0.02 * perimeter, True)
        
        if len(curr_approx) == 4:
            board_contour = hull
            approx = curr_approx
            break

    if board_contour is None:
        # Fallback: return resized image if detection fails
        return cv.resize(img, (1600, 1600))

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
    
    width = 1600
    height = 1600
    
    destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype="float32")

    M = cv.getPerspectiveTransform(rect, destination_of_puzzle)
    
    # Warp the original color image
    result = cv.warpPerspective(img, M, (width, height))
    
    return result

lines_horizontal=[]
# Folosim linspace pentru a împărți 1600 în 16 secțiuni egale (17 linii)
for i in np.linspace(0, 1600, 17):
    y = int(i)
    # Corecție pentru ultima linie să fie în interiorul imaginii
    if y == 1600: y = 1599
    
    l=[]
    l.append((0,y))
    l.append((1599,y))
    lines_horizontal.append(l)

lines_vertical=[]
for i in np.linspace(0, 1600, 17):
    x = int(i)
    if x == 1600: x = 1599
    
    l=[]
    l.append((x,0))
    l.append((x,1599))
    lines_vertical.append(l)

# Directorul cu imagini
input_folder = 'antrenare'

if os.path.exists(input_folder):
    files = os.listdir(input_folder)
    for file in files:
        if file.endswith('.jpg'):
            img_path = os.path.join(input_folder, file)
            img = cv.imread(img_path)
            
            if img is None:
                continue
                
            result = extrage_careu(img)
            
            for line in lines_vertical: 
                cv.line(result, line[0], line[1], (0, 255, 0), 5)
            for line in lines_horizontal: 
                cv.line(result, line[0], line[1], (0, 0, 255), 5)
            
            show_image('img', result)
else:
    print(f"Folderul {input_folder} nu există.")
