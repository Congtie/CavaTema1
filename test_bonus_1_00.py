import cv2 as cv
import numpy as np
import os
import sys

# Adăugăm funcțiile necesare din detectie_piese_qwirkle.py
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

def load_templates(template_folder):
    templates = []
    if not os.path.exists(template_folder):
        print(f"Folderul {template_folder} nu exista!")
        return templates
    
    for root, dirs, files in os.walk(template_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                img = cv.imread(path)
                if img is not None:
                    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                    
                    rel_path = os.path.relpath(root, template_folder)
                    if rel_path == '.':
                        label = os.path.splitext(file)[0]
                    else:
                        label = os.path.basename(root)
                        
                    templates.append((label, img_hsv))
                    print(f"  Loaded: {label} - {file} ({img.shape[0]}x{img.shape[1]})")
    return templates

def detecteaza_bonus_template(board_hsv, board_bgr, bonus_templates, threshold=0.5):
    bonus_1_coords = []
    bonus_2_coords = []
    
    bonus_temps = [(label, temp) for label, temp in bonus_templates if label in ['1', '2']]
    
    if not bonus_temps:
        print("Niciun template bonus găsit!")
        return set(), set(), board_bgr
    
    output_img = board_bgr.copy()
    
    padding = 50
    cell_size = 100
    extract_size = 150
    
    print(f"\nDetectare bonusuri (threshold={threshold}):")
    detected_count = 0
    
    for row in range(1, 17):
        for col in range(1, 17):
            y_center = padding + (row - 1) * cell_size + cell_size // 2
            x_center = padding + (col - 1) * cell_size + cell_size // 2
            
            half_extract = extract_size // 2
            y1 = y_center - half_extract
            y2 = y_center + half_extract
            x1 = x_center - half_extract
            x2 = x_center + half_extract
            
            if y1 < 0 or y2 > board_hsv.shape[0] or x1 < 0 or x2 > board_hsv.shape[1]:
                continue
            
            cell_hsv = board_hsv[y1:y2, x1:x2]
            
            if cell_hsv.shape[0] != extract_size or cell_hsv.shape[1] != extract_size:
                continue
            
            best_score = 0
            best_label = None
            
            for label, template_hsv in bonus_temps:
                if template_hsv.shape[0] != extract_size or template_hsv.shape[1] != extract_size:
                    template_hsv_resized = cv.resize(template_hsv, (extract_size, extract_size))
                else:
                    template_hsv_resized = template_hsv
                
                res_h = cv.matchTemplate(cell_hsv[:,:,0], template_hsv_resized[:,:,0], cv.TM_CCOEFF_NORMED)
                res_s = cv.matchTemplate(cell_hsv[:,:,1], template_hsv_resized[:,:,1], cv.TM_CCOEFF_NORMED)
                res_v = cv.matchTemplate(cell_hsv[:,:,2], template_hsv_resized[:,:,2], cv.TM_CCOEFF_NORMED)
                
                score = (res_h[0,0] + res_s[0,0] + res_v[0,0]) / 3.0
                
                if score > best_score:
                    best_score = score
                    best_label = label
            
            if best_score >= threshold:
                coord = pos_to_coord(row, col)
                y = padding + (row - 1) * cell_size
                x = padding + (col - 1) * cell_size
                
                if best_label == '1':
                    bonus_1_coords.append(coord)
                    cv.rectangle(output_img, (x, y), (x + cell_size, y + cell_size), (0, 255, 0), 3)
                    cv.putText(output_img, '+1', (x + 10, y + 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    print(f"  +1 la {coord} (score: {best_score:.3f})")
                    detected_count += 1
                elif best_label == '2':
                    bonus_2_coords.append(coord)
                    cv.rectangle(output_img, (x, y), (x + cell_size, y + cell_size), (255, 0, 0), 3)
                    cv.putText(output_img, '+2', (x + 10, y + 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    print(f"  +2 la {coord} (score: {best_score:.3f})")
                    detected_count += 1
    
    print(f"Total detectate: {detected_count}")
    return set(bonus_1_coords), set(bonus_2_coords), output_img

# MAIN
print("="*60)
print("Test detecție bonusuri pe 1_00.jpg")
print("="*60)

# Încarcă imaginea deja procesată (careul extras)
img = cv.imread('detectate/1_00_detected.jpg')
if img is None:
    print("Nu pot încărca detectate/1_00_detected.jpg!")
    print("Rulează mai întâi codul principal pentru a genera imaginea.")
    sys.exit(1)

print(f"\n1. Imagine procesată încărcată: {img.shape}")

# Nu mai e nevoie de extragere careu, imaginea e deja careul extras
board = img

# Convertim la HSV
board_hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)

# Încarcă template-urile
print("\n3. Încărcare template-uri:")
templates = load_templates('templates')
print(f"   Total: {len(templates)} template-uri")

# Filtrăm doar bonusurile
bonus_temps = [(label, temp) for label, temp in templates if label in ['1', '2']]
print(f"   Bonusuri: {len(bonus_temps)} template-uri")

if len(bonus_temps) == 0:
    print("\nERROR: Niciun template bonus găsit!")
    print("Verifică că există:")
    print("  - templates/1/01.jpg")
    print("  - templates/2/02.jpg")
    sys.exit(1)

# Detectează bonusurile
print("\n4. Detecție bonusuri:")
bonus_1_set, bonus_2_set, bonus_img = detecteaza_bonus_template(board_hsv, board, templates, threshold=0.5)

print(f"\n{'='*60}")
print(f"REZULTATE:")
print(f"{'='*60}")
print(f"Bonus +1: {len(bonus_1_set)} pătrate")
print(f"Bonus +2: {len(bonus_2_set)} pătrate")

# Salvăm imaginea
cv.imwrite('detectate/1_00_bonus.jpg', bonus_img)
print(f"\nSalvat: detectate/1_00_bonus.jpg")
print("\nDeschide imaginea pentru a vedea pătratele bonus marcate!")
