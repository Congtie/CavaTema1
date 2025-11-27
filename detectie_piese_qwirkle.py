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

    # Adaugam padding pentru a capta si marginile
    padding = 50  # pixeli extra pe fiecare parte
    width = 1600 + 2 * padding
    height = 1600 + 2 * padding
    
    destination_of_puzzle = np.array([
        [padding, padding],
        [width - padding, padding],
        [width - padding, height - padding],
        [padding, height - padding]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(rect, destination_of_puzzle)

    result = cv.warpPerspective(img, M, (width, height))
    
    return result

def load_templates(template_folder, target_size=100):
    templates = []
    if not os.path.exists(template_folder):
        print(f"Folderul {template_folder} nu exista!")
        return templates
    
    # Citim din toate subfolderele si din root
    for root, dirs, files in os.walk(template_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                img = cv.imread(path)
                if img is not None:
                    # Nu redimensionam, pastram dimensiunea originala
                    # Convertim la HSV
                    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                    
                    # Determinam eticheta (label) pe baza folderului
                    rel_path = os.path.relpath(root, template_folder)
                    if rel_path == '.':
                        # Fisier in root (ex: 1sus.jpg -> label=1sus)
                        label = os.path.splitext(file)[0]
                    else:
                        # Fisier in subfolder (ex: templates/cerc/x.jpg -> label=cerc)
                        label = os.path.basename(root)
                        
                    templates.append((label, img_hsv))
    return templates

def match_cell(cell_hsv, templates, threshold=0.6, top_k=5, debug_coord=None):
    # Lista pentru a stoca toate potrivirile: (scor, label)
    all_matches = []
    
    for label, tmpl in templates:
        # Nu redimensionam - matchTemplate functioneaza cu dimensiuni diferite
        
        best_tmpl_score = -1
        
        # Rotim template-ul 0, 90, 180, 270 grade
        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated_tmpl = tmpl
            elif angle == 90:
                rotated_tmpl = cv.rotate(tmpl, cv.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_tmpl = cv.rotate(tmpl, cv.ROTATE_180)
            elif angle == 270:
                rotated_tmpl = cv.rotate(tmpl, cv.ROTATE_90_COUNTERCLOCKWISE)
            
            # Match pe canalul V (Grayscale)
            res_v = cv.matchTemplate(cell_hsv[:,:,2], rotated_tmpl[:,:,2], cv.TM_CCOEFF_NORMED)
            score = np.max(res_v)
            
            if score > best_tmpl_score:
                best_tmpl_score = score
        
        all_matches.append((best_tmpl_score, label))
    
    # Sortam descrescator dupa scor
    all_matches.sort(key=lambda x: x[0], reverse=True)
    
    # DEBUG: Afisam toate scorurile pentru debug_coord
    if debug_coord:
        print(f"\n  === Scoruri pentru {debug_coord} ===")
        for score, label in all_matches:
            print(f"    {label}: {score:.3f}")
    
    # Filtram cele sub prag
    valid_matches = [m for m in all_matches if m[0] > threshold]
    
    if not valid_matches:
        return False, 0.0, None
        
    # REGULA DE AUR (Exact Match): Daca avem o potrivire aproape perfecta, o luam pe aia
    if valid_matches[0][0] > 0.95:
        return True, valid_matches[0][0], valid_matches[0][1]
        
    # Luam top K
    top_matches = valid_matches[:top_k]
    
    # Votare (suma scorurilor pentru fiecare label)
    votes = {}
    for score, label in top_matches:
        if label not in votes:
            votes[label] = 0
        votes[label] += score
        
    # Gasim castigatorul
    best_label = max(votes, key=votes.get)
    
    # Returnam cel mai mare scor individual pentru referinta
    best_score = top_matches[0][0]
                
    return True, best_score, best_label

# --- MAIN ---

GRID_SIZE = 16
IMG_SIZE = 1700  # 1600 + 2*50 padding
CELL_SIZE = 1600 // GRID_SIZE  # Celula ramane 100x100
CELL_OFFSET = 50  # Offset pentru prima celula (padding)
PATCH_SIZE = 138  # Dimensiunea patch-ului extras din imagine
TEMPLATE_SIZE = 100  # Dimensiunea la care redimensionam template-urile
MARGIN = (PATCH_SIZE - CELL_SIZE) // 2  # Margin pe fiecare parte
MATCH_THRESHOLD = 0.40

# Cream folderul pentru imagini detectate
output_folder = 'detectate'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Incarcam template-urile o singura data (la dimensiunea lor originala)
templates = load_templates('templates')
print(f"Am incarcat {len(templates)} template-uri.\n")

input_folder = 'antrenare'

if os.path.exists(input_folder):
    # Excludem fisierele generate anterior
    files = [f for f in os.listdir(input_folder) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
             and not ('_detected' in f or '_board' in f)]
    
    print(f"Procesez {len(files)} imagini...\n")
    
    # Test doar pe 4_19.jpg
    files = [f for f in files if f == '4_19.jpg' or f == '5_20.jpg']
    
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(input_folder, file)
            img = cv.imread(path)
            if img is None: 
                print(f"Nu am putut citi imaginea: {path}")
                continue
            
            print(f"\nProcesare: {file}")
            board = extrage_careu(img)
            
            # Nume fisier fara extensie pentru folder debug
            base_name = os.path.splitext(file)[0]
            
            # Board deja este 1600x1600 din extrage_careu
            gray_board = cv.cvtColor(board, cv.COLOR_BGR2GRAY)
            hsv_board = cv.cvtColor(board, cv.COLOR_BGR2HSV)
            
            output = board.copy()
            detected_count = 0
            
            # Template-urile sunt deja incarcate global
            
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    # Adaugam offset-ul pentru padding
                    x1 = CELL_OFFSET + col * CELL_SIZE
                    y1 = CELL_OFFSET + row * CELL_SIZE
                    x2 = x1 + CELL_SIZE
                    y2 = y1 + CELL_SIZE
                    
                    # Extract patch mai mare pentru matching mai bun
                    patch_x1 = max(0, x1 - MARGIN)
                    patch_y1 = max(0, y1 - MARGIN)
                    patch_x2 = min(IMG_SIZE, x2 + MARGIN)
                    patch_y2 = min(IMG_SIZE, y2 + MARGIN)
                    
                    # Extract celula pentru verificare continut
                    cell_gray = gray_board[y1:y2, x1:x2]
                    
                    # Extract patch pentru matching
                    cell_hsv = hsv_board[patch_y1:patch_y2, patch_x1:patch_x2]
                    # Resize doar pentru a normaliza dimensiunea in caz de margini
                    cell_hsv = cv.resize(cell_hsv, (PATCH_SIZE, PATCH_SIZE))
                    
                    # Verificare intensitate - o piesa ar trebui sa aiba zone intunecate
                    cell_median = np.median(cell_gray)
                    dark_pixels = np.sum(cell_gray < 100)
                    dark_ratio = dark_pixels / cell_gray.size
                    
                    # Calculam coordonatele pentru debug (A-P, 1-16)
                    col_letter = chr(65 + col)
                    row_number = row + 1
                    cell_coord = f"{row_number}{col_letter}"
                    
                    # O piesa Qwirkle are parte neagra (forma) si parte colorata
                    # Dark ratio ar trebui sa fie peste 40% (pentru a elimina linii de grid si umbre)
                    has_valid_dark_content = dark_ratio > 0.40
                    
                    # Template Matching pe HSV (V + S)
                    debug_param = cell_coord if cell_coord in ["16O", "4G"] else None
                    is_match, score, name = match_cell(cell_hsv, templates, 0.55, debug_coord=debug_param)
                    
                    # DEBUG pentru 16O si 4G
                    if cell_coord in ["16O", "4G"]:
                        print(f"\n=== DEBUG {cell_coord} ===")
                        print(f"  Score: {score:.3f}")
                        print(f"  Is_match: {is_match}")
                        print(f"  Template name: {name}")
                        print(f"  Dark ratio: {dark_ratio:.3f}")
                        print(f"  Has valid dark content: {has_valid_dark_content}")
                        print(f"  Final is_piece: {is_match and has_valid_dark_content}")
                    
                    # FILTRARE NEGATIVA: ignoram 1sus, 2sus, 1jos, 2jos
                    if name and ('sus' in name.lower() or 'jos' in name.lower()):
                        is_match = False
                    
                    # Combinam: Match SI are continut intunecat valid
                    is_piece = is_match and has_valid_dark_content
                    
                    # DEBUG: Salvam patch-urile pentru 15O si 16O
                    if cell_coord in ["15O", "16O", "15L", "4D", "4G"]:
                        debug_folder = os.path.join(output_folder, 'debug_patches', base_name)
                        os.makedirs(debug_folder, exist_ok=True)
                        
                        # Salvam patch-ul HSV convertit la BGR
                        patch_bgr = cv.cvtColor(cell_hsv, cv.COLOR_HSV2BGR)
                        patch_path = os.path.join(debug_folder, f"{cell_coord}_patch_138x138.jpg")
                        cv.imwrite(patch_path, patch_bgr)
                        print(f"  Salvat patch: {cell_coord}_patch_138x138.jpg")
                    
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
                        
                        # Afisam coordonata si numele template-ului
                        template_name = name
                        debug_text = f"{cell_coord} {template_name}"
                        cv.putText(output, debug_text, (center_x-30, center_y-25), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                        debug_text2 = f"S:{score:.2f} D:{dark_ratio:.2f}"
                        cv.putText(output, debug_text2, (center_x-30, center_y+35), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    else:
                        # GOL - afisam scorurile pentru debugging
                        cv.circle(output, (center_x, center_y), 3, (0, 0, 255), -1)
            
            print(f"  Piese detectate: {detected_count}")
            
            # Desenam grila doar pe zona board-ului (ignoram padding-ul)
            for i in range(GRID_SIZE + 1):
                p = CELL_OFFSET + i * CELL_SIZE
                # Linii verticale
                cv.line(output, (p, CELL_OFFSET), (p, CELL_OFFSET + GRID_SIZE * CELL_SIZE), (0, 255, 0), 2)
                # Linii orizontale
                cv.line(output, (CELL_OFFSET, p), (CELL_OFFSET + GRID_SIZE * CELL_SIZE, p), (0, 255, 0), 2)
            
            # Adaugam etichetele pentru coloane (A-P) si randuri (1-16)
            font = cv.FONT_HERSHEY_SIMPLEX
            for i in range(GRID_SIZE):
                # Coloane (A-P)
                col_label = chr(65 + i)  # A=65 in ASCII
                x_pos = CELL_OFFSET + int((i + 0.5) * CELL_SIZE)
                cv.putText(output, col_label, (x_pos - 10, CELL_OFFSET - 10), font, 0.8, (255, 255, 0), 2)
                
                # Randuri (1-16)
                row_label = str(i + 1)
                y_pos = CELL_OFFSET + int((i + 0.5) * CELL_SIZE)
                cv.putText(output, row_label, (CELL_OFFSET - 30, y_pos + 10), font, 0.8, (255, 255, 0), 2)
            
            # Salvam in folderul separat
            output_path = os.path.join(output_folder, file.replace('.jpg', '_detected.jpg'))
            cv.imwrite(output_path, output)
            print(f"  Salvat: {output_path}")
else:
    print(f"Folderul {input_folder} nu exista!")
