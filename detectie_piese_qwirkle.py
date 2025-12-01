import cv2 as cv
import numpy as np
import os

# Debug mode - False pentru procesare rapida fara imagini
DEBUG_MODE = True

# ===== FUNCTII PENTRU BONUS SI SCOR =====

def coord_to_pos(coord):
    """Convertește coordonată (ex: '2G') în (row, col) numeric (1-16, 1-16)"""
    row = int(coord[:-1])
    col_letter = coord[-1]
    col = ord(col_letter) - ord('A') + 1
    return row, col

def pos_to_coord(row, col):
    """Convertește (row, col) numeric în coordonată (ex: '2G')"""
    col_letter = chr(ord('A') + col - 1)
    return f"{row}{col_letter}"

def detecteaza_bonus_pattern(piese_coords):
    """
    Detectează pătratele bonus (+1 și +2) pe baza pieselor inițiale.
    
    Tabla este împărțită în 4 cadrane, fiecare cu propriile reguli:
    - Cadran 1 (1-8, A-H): verifică 2B
    - Cadran 2 (1-8, I-P): verifică 2J
    - Cadran 3 (9-16, A-H): verifică 10B
    - Cadran 4 (9-16, I-P): verifică 10J
    """
    piese_set = set(piese_coords)
    
    bonus_1_set = set()
    bonus_2_set = set()
    
    # CADRAN 1 (1-8, A-H) - verifică 2B
    if '2B' not in piese_set:
        # 2B NU este ocupat
        bonus_2_set.update(['2B', '7G'])
        bonus_1_set.update(['6B', '5C', '4D', '3E', '2F', '7C', '6D', '5E', '4F', '3G'])
    else:
        # 2B ESTE ocupat
        bonus_2_set.update(['7B', '2G'])
        bonus_1_set.update(['2C', '3D', '4E', '5F', '6G', '3B', '4C', '5D', '6E', '7F'])
    
    # CADRAN 2 (1-8, I-P) - verifică 2J
    if '2J' not in piese_set:
        # 2J NU este ocupat
        bonus_2_set.update(['2J', '7O'])
        bonus_1_set.update(['6J', '5K', '4L', '3M', '2N', '7K', '6L', '5M', '4N', '3O'])
    else:
        # 2J ESTE ocupat
        bonus_2_set.update(['7J', '2O'])
        bonus_1_set.update(['2K', '3L', '4M', '5N', '6O', '3J', '4K', '5L', '6M', '7N'])
    
    # CADRAN 3 (9-16, A-H) - verifică 10B
    if '10B' not in piese_set:
        # 10B NU este ocupat
        bonus_2_set.update(['10B', '15G'])
        bonus_1_set.update(['14B', '13C', '12D', '11E', '10F', '15C', '14D', '13E', '12F', '11G'])
    else:
        # 10B ESTE ocupat
        bonus_2_set.update(['10G', '15B'])
        bonus_1_set.update(['10C', '11D', '12E', '13F', '14G', '11B', '12C', '13D', '14E', '15F'])
    
    # CADRAN 4 (9-16, I-P) - verifică 10J
    if '10J' not in piese_set:
        # 10J NU este ocupat
        bonus_2_set.update(['10J', '15O'])
        bonus_1_set.update(['14J', '13K', '12L', '11M', '10N', '15K', '14L', '13M', '12N', '11O'])
    else:
        # 10J ESTE ocupat
        bonus_2_set.update(['10O', '15J'])
        bonus_1_set.update(['10K', '11L', '12M', '13N', '14O', '11J', '12K', '13L', '14M', '15N'])
    
    return bonus_1_set, bonus_2_set

def calculeaza_scor_mutare(piese_noi, toate_piesele, bonus_1_set, bonus_2_set):
    """
    Calculează scorul pentru piesele plasate într-o mutare.
    
    Reguli:
    - Fiecare linie formată/extinsă = număr piese din linie (inclusiv cele noi)
    - Bonus Qwirkle: linie de 6 piese = +6 puncte bonus
    - Pătrate bonus: +1 sau +2 puncte dacă piesa nouă e pe pătrat bonus
    """
    if not piese_noi:
        return 0
    
    scor_total = 0
    
    # Construim dictionar cu pozitiile tuturor pieselor
    board = {}
    for coord, shape, color in toate_piesele:
        row, col = coord_to_pos(coord)
        board[(row, col)] = (coord, shape, color)
    
    # Set pentru a evita să contăm aceeași linie de mai multe ori
    linii_calculate = set()
    
    # Pentru fiecare piesă nouă
    piese_noi_coords = {coord_to_pos(coord) for coord, _, _ in piese_noi}
    
    for coord, shape, color in piese_noi:
        row, col = coord_to_pos(coord)
        
        # Verificăm liniile orizontale și verticale
        for direction in ['H', 'V']:  # Horizontal, Vertical
            if direction == 'H':
                # Linie orizontală (aceeași row, diferite cols)
                linie = [(row, c) for c in range(1, 17) if (row, c) in board]
                linie.sort(key=lambda x: x[1])
            else:
                # Linie verticală (aceeași col, diferite rows)
                linie = [(r, col) for r in range(1, 17) if (r, col) in board]
                linie.sort(key=lambda x: x[0])
            
            # Găsim secvența continuă care conține piesa curentă
            secventa = []
            for pos in linie:
                if not secventa or (direction == 'H' and pos[1] == secventa[-1][1] + 1) or \
                   (direction == 'V' and pos[0] == secventa[-1][0] + 1):
                    secventa.append(pos)
                elif (row, col) in secventa:
                    break
                else:
                    secventa = [pos]
            
            # Scor doar dacă linia are mai mult de 1 piesă și nu a fost deja calculată
            if len(secventa) > 1 and (row, col) in secventa:
                # Identificator unic pentru linie
                linie_id = (direction, tuple(secventa))
                
                # Debug
                if coord in ['3L', '4L', '6L']:
                    deja_calculata = linie_id in linii_calculate
                    print(f"  [CALC DEBUG] {coord}: linie {direction} cu {len(secventa)} piese, deja calculata? {deja_calculata}")
                
                if linie_id not in linii_calculate:
                    linii_calculate.add(linie_id)
                    
                    puncte_linie = len(secventa)
                    
                    # Bonus Qwirkle (6 piese)
                    if len(secventa) == 6:
                        puncte_linie += 6
                    
                    scor_total += puncte_linie
        
        # Bonus pentru pătrat special (doar pentru piese noi)
        if coord in bonus_1_set:
            scor_total += 1
        elif coord in bonus_2_set:
            scor_total += 2
    
    # Dacă nicio linie nu a fost formată (piese singure), scorul = număr de piese
    if scor_total == 0:
        scor_total = len(piese_noi)
    
    return scor_total

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

def detect_color(cell_hsv, debug_coord=None):
    """Detecteaza culoarea piesei din centrul celulei (fără margin)"""
    h, w, _ = cell_hsv.shape
    center_y, center_x = h // 2, w // 2
    
    # Strategie îmbunătățită: în loc să luăm doar centrul (care poate fi umbră),
    # luăm mostre din 4 puncte în jurul centrului și alegem pixelii cei mai luminoși
    sample_points = [
        (center_y - 15, center_x),      # sus
        (center_y + 15, center_x),      # jos
        (center_y, center_x - 15),      # stânga
        (center_y, center_x + 15),      # dreapta
    ]
    
    # Colectăm valorile HSV de la fiecare punct (regiune 5x5)
    all_pixels = []
    for py, px in sample_points:
        if 0 <= py < h and 0 <= px < w:
            region = cell_hsv[max(0, py-2):min(h, py+3), max(0, px-2):min(w, px+3)]
            all_pixels.extend(region.reshape(-1, 3))
    
    # Convertim la numpy array
    all_pixels = np.array(all_pixels)
    
    # Filtram doar pixelii cu V (brightness) peste un prag - aceștia reprezintă culoarea, nu umbra
    bright_pixels = all_pixels[all_pixels[:, 2] > 60]  # V > 60
    
    if len(bright_pixels) == 0:
        # Fallback: folosim toți pixelii dacă nu găsim nimic luminos
        bright_pixels = all_pixels
    
    # Calculam mediana valorilor HSV din pixelii luminoși
    h_median = np.median(bright_pixels[:, 0])
    s_median = np.median(bright_pixels[:, 1])
    v_median = np.median(bright_pixels[:, 2])
    
    # Debug pentru coordonate specifice
    if debug_coord and debug_coord in ['15B', '13B', '14B']:
        print(f"  [COLOR DEBUG {debug_coord}] H={h_median:.1f}, S={s_median:.1f}, V={v_median:.1f} (bright pixels: {len(bright_pixels)})")
    
    # Detectare White - saturatie foarte scazuta
    if s_median <= 50 and v_median >= 60:
        if debug_coord and debug_coord in ['15B', '13B', '14B']:
            print(f"  [COLOR DEBUG {debug_coord}] -> Detected: W (White)")
        return "W"
    
    # Detectie pe baza Hue cu range-uri ajustate
    # Red (rosu pur) - prioritate maxima pentru a evita confuzia cu orange
    if h_median >= 165:  # 165-180
        if debug_coord and debug_coord in ['15B', '13B', '14B']:
            print(f"  [COLOR DEBUG {debug_coord}] -> Detected: R (Red)")
        return "R"  # Red
    
    # Orange (portocaliu)
    if h_median <= 18:  # 0-18
        if debug_coord and debug_coord in ['15B', '13B', '14B']:
            print(f"  [COLOR DEBUG {debug_coord}] -> Detected: O (Orange)")
        return "O"  # Orange
    
    # Yellow (galben)
    if 18 < h_median <= 35:  # 19-35
        if debug_coord and debug_coord in ['15B', '13B', '14B']:
            print(f"  [COLOR DEBUG {debug_coord}] -> Detected: Y (Yellow)")
        return "Y"  # Yellow
    
    # Green (verde)
    if 40 <= h_median <= 80:  # 40-80
        if debug_coord and debug_coord in ['15B', '13B', '14B']:
            print(f"  [COLOR DEBUG {debug_coord}] -> Detected: G (Green)")
        return "G"  # Green
    
    # Blue (albastru)
    if 90 < h_median <= 130:  # 91-130
        if debug_coord and debug_coord in ['15B', '13B', '14B']:
            print(f"  [COLOR DEBUG {debug_coord}] -> Detected: B (Blue)")
        return "B"  # Blue
    
    # Red in middle range (pentru cazuri ambigue)
    if 155 <= h_median < 165:  # 155-164
        if debug_coord and debug_coord in ['15B', '13B', '14B']:
            print(f"  [COLOR DEBUG {debug_coord}] -> Detected: R (Red)")
        return "R"  # Red
    
    # Zona ambigua intre yellow-green si blue-purple
    if 35 < h_median < 40:
        result = "Y" if h_median < 38 else "G"
        if debug_coord and debug_coord in ['15B', '13B', '14B']:
            print(f"  [COLOR DEBUG {debug_coord}] -> Detected: {result} (ambiguous yellow-green)")
        return result
    if 80 < h_median <= 90:
        result = "G" if h_median < 85 else "B"
        if debug_coord and debug_coord in ['15B', '13B', '14B']:
            print(f"  [COLOR DEBUG {debug_coord}] -> Detected: {result} (ambiguous green-blue)")
        return result
    
    if debug_coord and debug_coord in ['15B', '13B', '14B']:
        print(f"  [COLOR DEBUG {debug_coord}] -> Detected: ? (Unknown)")
    return "?"  # Unknown (probabil Purple 130-160)


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
PATCH_SIZE = 150  # Dimensiunea patch-ului extras din imagine (marit pentru a prinde mai mult)
TEMPLATE_SIZE = 100  # Dimensiunea la care redimensionam template-urile
MARGIN = (PATCH_SIZE - CELL_SIZE) // 2  # Margin pe fiecare parte (25px pe fiecare parte)
MATCH_THRESHOLD = 0.40

# Mapare nume template -> cod numeric
SHAPE_TO_CODE = {
    'cerc': 1,
    'patrat': 4,
    'romb': 3,
    'trifoi': 2,
    'stea': 6,
    'shuri': 5
}

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
    
    # Sortam fisierele pentru a procesa in ordine (1_00, 1_01, etc.)
    files = sorted(files)
    
    print(f"Procesez {len(files)} imagini...\n")
    
    # Dictionar pentru a pastra piesele din imaginea anterioara + bonus squares per joc
    previous_pieces = set()
    bonus_1_set = set()
    bonus_2_set = set()
    current_game = None
    
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Detectăm numărul jocului (ex: 1_00.jpg -> joc 1)
            game_num = int(file.split('_')[0])
                
            path = os.path.join(input_folder, file)
            img = cv.imread(path)
            if img is None: 
                print(f"Nu am putut citi imaginea: {path}")
                continue
            
            print(f"\nProcesare: {file}")
            
            # Detectăm numărul jocului (ex: 1_00.jpg -> joc 1)
            
            # La fiecare joc nou, resetăm și detectăm bonus-urile din x_00
            if game_num != current_game:
                current_game = game_num
                previous_pieces = set()
                
                # Dacă e primul fișier (_00), detectăm bonus-urile
                if '_00' in file:
                    print(f"  [INFO] Nou joc detectat - calculez patrate bonus...")
            
            board = extrage_careu(img)
            
            # Nume fisier fara extensie pentru folder debug
            base_name = os.path.splitext(file)[0]
            
            # Board deja este 1600x1600 din extrage_careu
            gray_board = cv.cvtColor(board, cv.COLOR_BGR2GRAY)
            hsv_board = cv.cvtColor(board, cv.COLOR_BGR2HSV)
            
            if DEBUG_MODE:
                output = board.copy()
            detected_count = 0
            detected_pieces = []  # Lista de piese pentru fisierul txt
            
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
                    cell_hsv_no_margin = hsv_board[y1:y2, x1:x2]  # Celula fără margin pentru detectare culoare
                    
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
                    is_match, score, name = match_cell(cell_hsv, templates, 0.55)
                    
                    # Detectam culoarea piesei din celula fara margin (evitam liniile verzi)
                    color = detect_color(cell_hsv_no_margin, debug_coord=cell_coord) if is_match else None
                    
                    # FIX: Rombul alb se confunda cu cercul - verificam explicit cercul daca score-ul e apropiat
                    if is_match and name == 'romb' and color == 'W' and score < 0.88:
                        # Verificam explicit template-ul de cerc
                        cerc_templates = [(label, img) for label, img in templates if 'cerc' in label]
                        if cerc_templates:
                            is_cerc, score_cerc, name_cerc = match_cell(cell_hsv, cerc_templates, 0.55)
                            if is_cerc and score_cerc > score - 0.05:  # Daca cercul e aproape cat rombul
                                is_match, score, name = is_cerc, score_cerc, name_cerc
                    
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
                        
                        # Adaugam in lista pentru fisierul txt
                        shape_code = SHAPE_TO_CODE.get(name, 0)
                        detected_pieces.append((cell_coord, shape_code, color))
                        
                        if DEBUG_MODE:
                            cv.circle(output, (center_x, center_y), 20, (0, 255, 0), -1)
                            
                            # Afisam coordonata, culoarea si numele template-ului
                            template_name = name
                            debug_text = f"{cell_coord} {color}{template_name}"
                            cv.putText(output, debug_text, (center_x-30, center_y-25), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            debug_text2 = f"S:{score:.2f} D:{dark_ratio:.2f}"
                            cv.putText(output, debug_text2, (center_x-30, center_y+35), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    else:
                        if DEBUG_MODE:
                            # GOL - afisam scorurile pentru debugging
                            cv.circle(output, (center_x, center_y), 3, (0, 0, 255), -1)
            
            print(f"  Piese detectate: {detected_count}")
            
            if DEBUG_MODE:
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
            
            # Detectam piesele noi (diferenta fata de imaginea anterioara)
            current_pieces = set((coord, shape_code, color) for coord, shape_code, color in detected_pieces)
            
            # La x_00, detectăm bonus-urile dar nu salvăm fișierul
            if '_00' in file:
                # Detectăm bonusurile pe baza pattern-ului de piese
                coords_only = [coord for coord, _, _ in current_pieces]
                bonus_1_set, bonus_2_set = detecteaza_bonus_pattern(coords_only)
                print(f"  [INFO] Bonus +1: {len(bonus_1_set)} patrate, Bonus +2: {len(bonus_2_set)} patrate")
                
                # Nu salvăm fișier txt pentru x_00
                previous_pieces = current_pieces
                continue
            
            # Filtram piesele care au doar forma schimbata (coord+color ramane)
            # Pentru a evita false positives când forma e detectată inconsistent
            previous_coords_colors = {(coord, color) for coord, _, color in previous_pieces}
            new_pieces = set()
            
            for coord, shape_code, color in current_pieces:
                # Verificam daca coordonata+culoarea exista in imaginea anterioara
                if (coord, color) not in previous_coords_colors:
                    # Piesa cu coord+color nou -> e cu adevarat noua
                    new_pieces.add((coord, shape_code, color))
                # Altfel, e aceeasi piesa cu forma detectata diferit -> ignore
            
            # Calculăm scorul pentru această mutare
            scor = calculeaza_scor_mutare(new_pieces, current_pieces, bonus_1_set, bonus_2_set)
            
            # Debug specific pentru 5_01
            if '5_01' in file:
                print(f"\n  ===== DEBUG DETALIAT PENTRU 5_01 =====")
                print(f"  Piese noi: {new_pieces}")
                print(f"  Bonus +1 set: {bonus_1_set}")
                print(f"  Bonus +2 set: {bonus_2_set}")
                for coord, shape, color in new_pieces:
                    is_bonus_1 = coord in bonus_1_set
                    is_bonus_2 = coord in bonus_2_set
                    print(f"    {coord} ({shape}{color}): Bonus +1? {is_bonus_1}, Bonus +2? {is_bonus_2}")
                print(f"  Scor calculat: {scor}")
                print(f"  Scor asteptat (GT): 9")
                print(f"  ==========================================\n")

            # Salvam fisierul txt doar cu piesele NOI
            txt_path = os.path.join(output_folder, file.replace('.jpg', '.txt'))
            with open(txt_path, 'w') as f:
                for coord, shape_code, color in sorted(new_pieces, key=lambda x: (int(x[0][:-1]), x[0][-1])):
                    f.write(f"{coord} {shape_code}{color}\n")
                # Scriem scorul calculat (fără newline la final)
                f.write(f"{scor}")
            print(f"  Salvat: {txt_path} ({len(new_pieces)} piese noi, scor: {scor})")
            
            # Actualizam pentru urmatoarea iteratie
            previous_pieces = current_pieces
            
else:
    print(f"Folderul {input_folder} nu exista!")
