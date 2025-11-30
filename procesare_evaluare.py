import cv2 as cv
import numpy as np
import os
import sys

# Import functii din detectie_piese_qwirkle.py
sys.path.insert(0, '.')
from detectie_piese_qwirkle import extrage_careu, detect_color, match_cell, load_templates, SHAPE_TO_CODE

# Constants
GRID_SIZE = 16
IMG_SIZE = 1700  # cu padding
CELL_SIZE = 100
CELL_OFFSET = 50  # padding
PATCH_SIZE = 138
MATCH_THRESHOLD = 0.40

# Cream folder pentru output evaluare
output_folder = 'evaluare_output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Incarcam template-urile
templates = load_templates('templates')
print(f"Am incarcat {len(templates)} template-uri.\n")

input_folder = 'evaluare/fake_test'

if os.path.exists(input_folder):
    files = [f for f in os.listdir(input_folder) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Procesez {len(files)} imagini din evaluare...\n")
    
    for file in files:
        path = os.path.join(input_folder, file)
        img = cv.imread(path)
        if img is None: 
            print(f"Nu am putut citi imaginea: {path}")
            continue
        
        print(f"Procesare: {file}")
        board = extrage_careu(img)
        
        gray_board = cv.cvtColor(board, cv.COLOR_BGR2GRAY)
        hsv_board = cv.cvtColor(board, cv.COLOR_BGR2HSV)
        
        detected_count = 0
        detected_pieces = []
        
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x1 = CELL_OFFSET + col * CELL_SIZE
                y1 = CELL_OFFSET + row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                
                MARGIN = (PATCH_SIZE - CELL_SIZE) // 2
                patch_x1 = max(0, x1 - MARGIN)
                patch_y1 = max(0, y1 - MARGIN)
                patch_x2 = min(IMG_SIZE, x2 + MARGIN)
                patch_y2 = min(IMG_SIZE, y2 + MARGIN)
                
                cell_gray = gray_board[y1:y2, x1:x2]
                cell_hsv = hsv_board[patch_y1:patch_y2, patch_x1:patch_x2]
                cell_hsv = cv.resize(cell_hsv, (PATCH_SIZE, PATCH_SIZE))
                
                dark_pixels = np.sum(cell_gray < 100)
                dark_ratio = dark_pixels / cell_gray.size
                has_valid_dark_content = dark_ratio > 0.40
                
                is_match, score, name = match_cell(cell_hsv, templates, 0.55)
                color = detect_color(cell_hsv) if is_match else None
                
                if name and ('sus' in name.lower() or 'jos' in name.lower()):
                    is_match = False
                
                is_piece = is_match and has_valid_dark_content
                
                if is_piece:
                    detected_count += 1
                    col_letter = chr(65 + col)
                    row_number = row + 1
                    cell_coord = f"{row_number}{col_letter}"
                    shape_code = SHAPE_TO_CODE.get(name, 0)
                    detected_pieces.append((cell_coord, shape_code, color))
        
        print(f"  Piese detectate: {detected_count}")
        
        # Salvam fisierul txt
        txt_path = os.path.join(output_folder, file.replace('.jpg', '.txt'))
        with open(txt_path, 'w') as f:
            for coord, shape_code, color in detected_pieces:
                f.write(f"{coord} {shape_code}{color}\n")
            f.write(f"{detected_count}\n")
        print(f"  Salvat: {txt_path}\n")

else:
    print(f"Folderul {input_folder} nu exista!")
