import cv2 as cv
import numpy as np
import os

def pos_to_coord(row, col):
    """Converteste pozitia (row, col) in coordonata (ex: 1, 1 -> '1A')"""
    return f"{row}{chr(64 + col)}"

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

def load_bonus_templates():
    templates = []
    for bonus_type in ['1', '2']:
        template_dir = f'templates/{bonus_type}'
        if os.path.exists(template_dir):
            for file in os.listdir(template_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(template_dir, file)
                    img = cv.imread(path)
                    if img is not None:
                        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                        templates.append((bonus_type, img_hsv))
                        print(f"Loaded template: {bonus_type} - {file} - size: {img.shape}")
    return templates

def detecteaza_bonus_template(board_hsv, bonus_templates, threshold=0.6):
    bonus_1_coords = []
    bonus_2_coords = []
    
    if not bonus_templates:
        print("Niciun template bonus găsit!")
        return set(), set()
    
    padding = 50
    cell_size = 100
    
    print(f"\nDetectare bonusuri cu threshold={threshold}")
    print(f"Parcurg grid 16x16...")
    
    for row in range(1, 17):
        for col in range(1, 17):
            y = padding + (row - 1) * cell_size
            x = padding + (col - 1) * cell_size
            
            cell_hsv = board_hsv[y:y+cell_size, x:x+cell_size]
            
            if cell_hsv.shape[0] != cell_size or cell_hsv.shape[1] != cell_size:
                continue
            
            best_score = 0
            best_label = None
            
            for label, template_hsv in bonus_templates:
                # Template matching pe fiecare canal HSV
                res_h = cv.matchTemplate(cell_hsv[:,:,0], template_hsv[:,:,0], cv.TM_CCOEFF_NORMED)
                res_s = cv.matchTemplate(cell_hsv[:,:,1], template_hsv[:,:,1], cv.TM_CCOEFF_NORMED)
                res_v = cv.matchTemplate(cell_hsv[:,:,2], template_hsv[:,:,2], cv.TM_CCOEFF_NORMED)
                
                score = (res_h[0,0] + res_s[0,0] + res_v[0,0]) / 3.0
                
                if score > best_score:
                    best_score = score
                    best_label = label
            
            if best_score >= threshold:
                coord = pos_to_coord(row, col)
                if best_label == '1':
                    bonus_1_coords.append(coord)
                    print(f"  Bonus +1 detectat la {coord} (score: {best_score:.3f})")
                elif best_label == '2':
                    bonus_2_coords.append(coord)
                    print(f"  Bonus +2 detectat la {coord} (score: {best_score:.3f})")
    
    return set(bonus_1_coords), set(bonus_2_coords)

# Main
print("Procesare joc 1 (1_00.jpg)...")

# Încarcă imaginea
img = cv.imread('antrenare/1_00.jpg')
if img is None:
    print("Nu pot încărca imaginea antrenare/1_00.jpg!")
    exit()

print(f"Imagine încărcată: {img.shape}")

# Extrage careul
board = extrage_careu(img)
print(f"Careu extras: {board.shape}")

# Salvează careul
cv.imwrite('board_1_00_extracted.jpg', board)
print("Salvat: board_1_00_extracted.jpg")

# Convertim la HSV
board_hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)

# Încarcă template-urile bonus
bonus_templates = load_bonus_templates()
print(f"\nTemplate-uri bonus încărcate: {len(bonus_templates)}")

if len(bonus_templates) == 0:
    print("\nNICIUN TEMPLATE BONUS GĂSIT!")
    print("Asigură-te că există:")
    print("  - templates/1/01.jpg (pentru bonus +1)")
    print("  - templates/2/02.jpg (pentru bonus +2)")
    exit()

# Detectează bonusurile
bonus_1_set, bonus_2_set = detecteaza_bonus_template(board_hsv, bonus_templates, threshold=0.6)

print(f"\n=== REZULTATE ===")
print(f"Bonus +1: {len(bonus_1_set)} pătrate")
print(f"Bonus +2: {len(bonus_2_set)} pătrate")

if len(bonus_1_set) > 0:
    print(f"\nPozițiile bonus +1: {sorted(bonus_1_set)}")
if len(bonus_2_set) > 0:
    print(f"\nPozițiile bonus +2: {sorted(bonus_2_set)}")
