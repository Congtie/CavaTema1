import cv2 as cv
import numpy as np
import os

# Functia pentru extragere careu (copied from detectie_piese_qwirkle.py)
def extrage_careu(img):
    """Extrage tabla de joc folosind HSV masking si transformare perspectiva"""
    
    # Convert to HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Define HSV range for background (Brown/Table)
    lower = np.array([0, 50, 20])
    upper = np.array([30, 255, 200])

    mask = cv.inRange(hsv, lower, upper)
    
    # Invert mask to get the board
    mask = cv.bitwise_not(mask)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area, largest first
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    
    board_contour = None
    approx = None
    
    image_area = img.shape[0] * img.shape[1]

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 0.05 * image_area:
            continue
            
        # Geometric filters
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        # Board should be roughly square
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue
            
        # Solidity check
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        if hull_area == 0: 
            continue
        solidity = float(area) / hull_area
        if solidity < 0.7:
            continue

        # Use the Convex Hull for approximation
        perimeter = cv.arcLength(hull, True)
        curr_approx = cv.approxPolyDP(hull, 0.02 * perimeter, True)
        
        if len(curr_approx) == 4:
            if board_contour is None:
                board_contour = hull
                approx = curr_approx
                break

    if board_contour is None:
        print("Nu am gasit conturul tablei!")
        return None

    # Reshape approx to (4, 2)
    pts = approx.reshape(4, 2)

    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    width = 1600
    height = 1600
    
    destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype="float32")

    M = cv.getPerspectiveTransform(rect, destination_of_puzzle)

    result = cv.warpPerspective(img, M, (width, height))
    
    return result

# Functia pentru extragere celula
def extrage_celula(board, row, col, margin=0):
    GRID_SIZE = 16
    IMG_SIZE = 1600
    CELL_SIZE = IMG_SIZE // GRID_SIZE
    
    x1 = col * CELL_SIZE + margin
    y1 = row * CELL_SIZE + margin
    x2 = (col + 1) * CELL_SIZE - margin
    y2 = (row + 1) * CELL_SIZE - margin
    
    return board[y1:y2, x1:x2]

# Mapare pozitie la (row, col)
def pozitie_la_coord(pozitie):
    col_letter = pozitie[-1]
    row_number = int(pozitie[:-1])
    col = ord(col_letter) - ord('A')
    row = row_number - 1
    return row, col

# --- CONFIGURARE ---

# Lista de extractii: (Fisier, Pozitie, Tip/Folder)
extractii = [
    # CERCURI
    ('antrenare/1_20.jpg', '6B', 'cerc'),
    ('antrenare/2_20.jpg', '13P', 'cerc'),
    ('antrenare/3_20.jpg', '9G', 'cerc'),
    ('antrenare/4_20.jpg', '16O', 'cerc'),
    ('antrenare/5_20.jpg', '7P', 'cerc'),
    ('antrenare/5_20.jpg', '11L', 'cerc'),
    ('antrenare/4_19.jpg', '4D', 'cerc'),
    ('antrenare/4_19.jpg', '4E', 'cerc'), # Adaugat pentru a corecta confuzia cu patrat
    ('antrenare/4_19.jpg', '15O', 'cerc'),
    ('antrenare/4_19.jpg', '13O', 'cerc'),
    ('antrenare/4_19.jpg', '4G', 'cerc'),
    ('antrenare/5_20.jpg', '4F', 'cerc'),
    ('antrenare/5_20.jpg', '7O', 'cerc'),
    ('antrenare/5_20.jpg', '14L', 'cerc'),
    
    # PATRATE
    ('antrenare/5_20.jpg', '14C', 'patrat'),
    ('antrenare/5_20.jpg', '15A', 'patrat'),
    ('antrenare/5_20.jpg', '15B', 'patrat'),
    ('antrenare/5_20.jpg', '15C', 'patrat'),
    ('antrenare/5_20.jpg', '16C', 'patrat'),
    ('antrenare/5_20.jpg', '1K', 'patrat'),
    ('antrenare/5_20.jpg', '4O', 'patrat'),
    ('antrenare/5_20.jpg', '10M', 'patrat'),
    ('antrenare/5_20.jpg', '14K', 'patrat'),
    ('antrenare/5_20.jpg', '3F', 'patrat'),
    ('antrenare/4_19.jpg', '3G', 'patrat'),
    ('antrenare/4_19.jpg', '7J', 'patrat'),
    
    # TRIFOI
    ('antrenare/5_20.jpg', '11G', 'trifoi'),
    ('antrenare/5_20.jpg', '12G', 'trifoi'),
    ('antrenare/5_20.jpg', '13G', 'trifoi'),
    ('antrenare/5_20.jpg', '14G', 'trifoi'),
    ('antrenare/5_20.jpg', '15G', 'trifoi'),
    ('antrenare/5_20.jpg', '16G', 'trifoi'),
]

base_output_folder = 'templates'

# --- EXECUTIE ---

# Contoare pentru fiecare tip pentru a numi fisierele unic
counters = {}

# Mai intai stergem fisierele generate anterior de acest script (optional, dar bun pentru curatenie)
# Vom sterge doar fisierele care incep cu "manual_" pentru a nu sterge template-urile originale daca exista
# Sau putem suprascrie. Hai sa folosim un prefix "extra_" pentru aceste template-uri noi.

for filename, pos, label in extractii:
    if not os.path.exists(filename):
        print(f"Fisierul {filename} nu exista!")
        continue
        
    # Initializam contorul pentru acest label
    if label not in counters:
        counters[label] = 1
    
    img = cv.imread(filename)
    if img is None:
        print(f"Nu pot citi {filename}")
        continue
        
    print(f"Procesez {filename} [{pos}] -> {label}...")
    board = extrage_careu(img)
    
    if board is None:
        print(f"Nu am putut extrage tabla din {filename}")
        continue
    
    # NU MAI APLICAM MASCA - salvam cu fundalul original (maro) pentru compatibilitate
    # hsv_board = cv.cvtColor(board, cv.COLOR_BGR2HSV)
    # lower = np.array([0, 50, 20])
    # upper = np.array([30, 255, 200])
    # mask_board = cv.inRange(hsv_board, lower, upper)
    # mask_pieces = cv.bitwise_not(mask_board)
    
    # Curatare masca
    # kernel = np.ones((3,3), np.uint8)
    # mask_pieces = cv.morphologyEx(mask_pieces, cv.MORPH_OPEN, kernel)
    
    # masked_board = cv.bitwise_and(board, board, mask=mask_pieces)
    
    row, col = pozitie_la_coord(pos)
    # Folosim board-ul original (nemascat)
    cell = extrage_celula(board, row, col)
    
    # Cream folderul daca nu exista
    target_folder = os.path.join(base_output_folder, label)
    os.makedirs(target_folder, exist_ok=True)
    
    # Nume fisier: extra_{label}_{counter}.jpg
    # Verificam sa nu suprascriem fisiere existente care nu sunt ale noastre, sau folosim un nume unic
    output_filename = os.path.join(target_folder, f"extra_{label}_{counters[label]}.jpg")
    
    cv.imwrite(output_filename, cell)
    print(f"  Salvat: {output_filename}")
    
    counters[label] += 1

print("Gata!")
