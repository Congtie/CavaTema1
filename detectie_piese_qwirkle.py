import cv2 as cv
import numpy as np
import os

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
        return img

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

def load_templates(template_folder):
    templates = []
    if not os.path.exists(template_folder):
        print(f"Folderul {template_folder} nu exista!")
        return templates
    
    for file in os.listdir(template_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # LE INCLUDEM PE TOATE, inclusiv 'sus', pentru a le putea filtra negativ mai tarziu
            path = os.path.join(template_folder, file)
            img = cv.imread(path)
            if img is not None:
                # Convertim la HSV in loc de grayscale
                img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                templates.append((file, img_hsv))
    return templates

def match_cell(cell_hsv, templates, threshold=0.6):
    best_score = -1
    best_template_name = None
    
    h_cell, w_cell, _ = cell_hsv.shape
    
    for name, tmpl in templates:
        # Redimensionam template-ul la dimensiunea celulei
        tmpl_resized = cv.resize(tmpl, (w_cell, h_cell))
        
        # Rotim template-ul 0, 90, 180, 270 grade
        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated_tmpl = tmpl_resized
            elif angle == 90:
                rotated_tmpl = cv.rotate(tmpl_resized, cv.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_tmpl = cv.rotate(tmpl_resized, cv.ROTATE_180)
            elif angle == 270:
                rotated_tmpl = cv.rotate(tmpl_resized, cv.ROTATE_90_COUNTERCLOCKWISE)
            
            # Match pe toate canalele HSV (sau doar pe V si S)
            # Optiune 1: Match pe canalul V (Value - luminozitate)
            res_v = cv.matchTemplate(cell_hsv[:,:,2], rotated_tmpl[:,:,2], cv.TM_CCOEFF_NORMED)
            score_v = np.max(res_v)
            
            # Optiune 2: Match pe canalul S (Saturation - saturatie)
            res_s = cv.matchTemplate(cell_hsv[:,:,1], rotated_tmpl[:,:,1], cv.TM_CCOEFF_NORMED)
            score_s = np.max(res_s)
            
            # Combinam scorurile (media sau max)
            score = (score_v + score_s) / 2.0
            
            if score > best_score:
                best_score = score
                best_template_name = name
                
    return best_score > threshold, best_score, best_template_name

# --- MAIN ---

GRID_SIZE = 16
IMG_SIZE = 1600
CELL_SIZE = IMG_SIZE // GRID_SIZE
MATCH_THRESHOLD = 0.40

input_folder = 'antrenare'
test_file = '1_13.jpg'

if os.path.exists(input_folder):
    files = [test_file]
    
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(input_folder, file)
            img = cv.imread(path)
            if img is None: 
                print(f"Nu am putut citi imaginea: {path}")
                continue
            
            print(f"Procesare: {file}")
            board = extrage_careu(img)
            
            # Board deja este 1600x1600 din extrage_careu
            gray_board = cv.cvtColor(board, cv.COLOR_BGR2GRAY)
            hsv_board = cv.cvtColor(board, cv.COLOR_BGR2HSV)
            
            output = board.copy()
            detected_count = 0
            
            # Incarcam template-urile
            templates = load_templates('templates')
            print(f"  Am incarcat {len(templates)} template-uri.")
            
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    x1 = col * CELL_SIZE
                    y1 = row * CELL_SIZE
                    x2 = x1 + CELL_SIZE
                    y2 = y1 + CELL_SIZE
                    
                    # Extract celula
                    cell_hsv = hsv_board[y1:y2, x1:x2]
                    cell_gray = gray_board[y1:y2, x1:x2]
                    
                    # Verificare intensitate - o piesa ar trebui sa aiba zone intunecate
                    cell_median = np.median(cell_gray)
                    dark_pixels = np.sum(cell_gray < 100)
                    dark_ratio = dark_pixels / cell_gray.size
                    
                    # O piesa Qwirkle are parte neagra (forma) si parte colorata
                    # Dark ratio ar trebui sa fie peste 35% (pentru a elimina linii de grid si umbre)
                    has_valid_dark_content = dark_ratio > 0.35
                    
                    # Template Matching pe HSV
                    is_match, score, name = match_cell(cell_hsv, templates, 0.60)  # Scadem la 0.60 pentru piese mai dificile
                    
                    # FILTRARE NEGATIVA: ignoram 1sus, 2sus, 1jos, 2jos
                    if name and ('sus' in name.lower() or 'jos' in name.lower()):
                        is_match = False
                    
                    # Combinam: Match SI are continut intunecat valid
                    is_piece = is_match and has_valid_dark_content
                    
                    center_x = x1 + CELL_SIZE // 2
                    center_y = y1 + CELL_SIZE // 2
                    
                    if is_piece:
                        # PIESA DETECTATA
                        detected_count += 1
                        
                        # Calculam coordonatele pentru debug (A-P, 1-16)
                        col_letter = chr(65 + col)
                        row_number = row + 1
                        cell_coord = f"{row_number}{col_letter}"
                        
                        cv.circle(output, (center_x, center_y), 20, (0, 255, 0), -1)
                        
                        # Afisam coordonata, numele template-ului si scorul
                        debug_text = f"{cell_coord} {name.split('.')[0][:4]}"
                        cv.putText(output, debug_text, (center_x-30, center_y-25), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                        debug_text2 = f"S:{score:.2f} D:{dark_ratio:.2f}"
                        cv.putText(output, debug_text2, (center_x-30, center_y+35), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    else:
                        # GOL - afisam scorurile pentru debugging
                        cv.circle(output, (center_x, center_y), 3, (0, 0, 255), -1)
                        
                        # Pentru 15C, 15D debugging
                        col_letter = chr(65 + col)
                        row_number = row + 1
                        if row_number == 15 and col_letter in ['C', 'D']:
                            debug_text = f"{row_number}{col_letter}: S:{score:.2f} D:{dark_ratio:.2f}"
                            cv.putText(output, debug_text, (center_x-30, center_y-10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            
            print(f"  Piese detectate: {detected_count}")
            
            # Desenam grila
            lines = np.linspace(0, IMG_SIZE, GRID_SIZE + 1)
            for val in lines:
                p = int(val)
                cv.line(output, (p, 0), (p, IMG_SIZE), (0, 255, 0), 2)
                cv.line(output, (0, p), (IMG_SIZE, p), (0, 255, 0), 2)
            
            # Adaugam etichetele pentru coloane (A-P) si randuri (1-16)
            font = cv.FONT_HERSHEY_SIMPLEX
            for i in range(GRID_SIZE):
                # Coloane (A-P)
                col_label = chr(65 + i)  # A=65 in ASCII
                x_pos = int((i + 0.5) * CELL_SIZE)
                cv.putText(output, col_label, (x_pos - 10, 30), font, 0.8, (255, 255, 0), 2)
                
                # Randuri (1-16)
                row_label = str(i + 1)
                y_pos = int((i + 0.5) * CELL_SIZE)
                cv.putText(output, row_label, (10, y_pos + 10), font, 0.8, (255, 255, 0), 2)
            
            output_path = path.replace('.jpg', '_detected.jpg')
            cv.imwrite(output_path, output)
            print(f"  Imaginea cu piese detectate a fost salvata: {output_path}")
else:
    print(f"Folderul {input_folder} nu exista!")
