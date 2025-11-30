import cv2 as cv
import numpy as np
import os

# Importăm funcția extrage_careu din codul principal
def extrage_careu(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 50, 150)
    
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    board_contour = None
    approx = None
    image_area = img.shape[0] * img.shape[1]
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        
        if area < 0.05 * image_area:
            continue
            
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue
            
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        if hull_area == 0: 
            continue
        solidity = float(area) / hull_area
        if solidity < 0.7:
            continue

        perimeter = cv.arcLength(hull, True)
        curr_approx = cv.approxPolyDP(hull, 0.02 * perimeter, True)
        
        if len(curr_approx) == 4:
            if board_contour is None:
                board_contour = hull
                approx = curr_approx
                break

    if board_contour is None:
        print("Nu am gasit conturul tablei!")
        return img

    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    padding = 50
    width = 1600 + 2 * padding
    height = 1600 + 2 * padding
    
    destination_of_puzzle = np.array([
        [padding, padding],
        [width - padding, padding],
        [width - padding, height - padding],
        [padding, height - padding]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(rect, destination_of_puzzle)
    warped = cv.warpPerspective(img, M, (width, height))
    
    return warped

# Încarcă imaginea 1_00 originală
img_path = 'antrenare/1_00.jpg'
img = cv.imread(img_path)
if img is None:
    print("Nu pot încărca imaginea!")
    exit()

# Extragem careul
board = extrage_careu(img)

# Dacă extragerea a eșuat, folosim imaginea originală
if board.shape == img.shape:
    print("Extragerea careului a eșuat, folosesc imaginea originală redimensionată")
    # Resize la dimensiunea standard
    board = cv.resize(img, (1700, 1700))

cv.imwrite('board_extracted.jpg', board)
print(f"Salvat: board_extracted.jpg - Dimensiuni: {board.shape}")

img = board.copy()

# Încarcă template-urile pentru bonus
template_1 = cv.imread('templates/1/01.jpg', cv.IMREAD_GRAYSCALE)
template_2 = cv.imread('templates/2/02.jpg', cv.IMREAD_GRAYSCALE)

if template_1 is None:
    print("Nu găsesc template-ul templates/1/01.jpg!")
    exit()
if template_2 is None:
    print("Nu găsesc template-ul templates/2/02.jpg!")
    exit()

print(f"Template bonus_1: {template_1.shape}")
print(f"Template bonus_2: {template_2.shape}")

# Convertim imaginea la grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detectăm pătratele bonus +1
res1 = cv.matchTemplate(gray, template_1, cv.TM_CCOEFF_NORMED)
max_val_1 = np.max(res1)
print(f"Valoare maximă pentru +1: {max_val_1:.3f}")

threshold = 0.7
loc1 = np.where(res1 >= threshold)

print(f"\nPătrate bonus +1 detectate: {len(loc1[0])}")
for pt in zip(*loc1[::-1]):
    cv.rectangle(img, pt, (pt[0] + template_1.shape[1], pt[1] + template_1.shape[0]), (0, 255, 0), 3)
    cv.putText(img, '+1', (pt[0] + 10, pt[1] + 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Detectăm pătratele bonus +2
res2 = cv.matchTemplate(gray, template_2, cv.TM_CCOEFF_NORMED)
max_val_2 = np.max(res2)
print(f"Valoare maximă pentru +2: {max_val_2:.3f}")

loc2 = np.where(res2 >= threshold)

print(f"Pătrate bonus +2 detectate: {len(loc2[0])}")
for pt in zip(*loc2[::-1]):
    cv.rectangle(img, pt, (pt[0] + template_2.shape[1], pt[1] + template_2.shape[0]), (255, 0, 0), 3)
    cv.putText(img, '+2', (pt[0] + 10, pt[1] + 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Salvăm rezultatul
cv.imwrite('test_bonus_detection.jpg', img)
print("\nSalvat: test_bonus_detection.jpg")
