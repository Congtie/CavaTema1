import cv2 as cv
import numpy as np
import os

DEBUG_MODE = True

def coord_to_pos(coord):
    row = int(coord[:-1])
    col_letter = coord[-1]
    col = ord(col_letter) - ord('A') + 1
    return row, col

def pos_to_coord(row, col):
    col_letter = chr(ord('A') + col - 1)
    return f"{row}{col_letter}"

def detecteaza_bonus_pattern(piese_coords):
    piese_set = set(piese_coords)
    
    bonus_1_set = set()
    bonus_2_set = set()
    
    if '2B' not in piese_set:
        bonus_2_set.update(['2B', '7G'])
        bonus_1_set.update(['6B', '5C', '4D', '3E', '2F', '7C', '6D', '5E', '4F', '3G'])
    else:
        bonus_2_set.update(['7B', '2G'])
        bonus_1_set.update(['2C', '3D', '4E', '5F', '6G', '3B', '4C', '5D', '6E', '7F'])
    
    if '2J' not in piese_set:
        bonus_2_set.update(['2J', '7O'])
        bonus_1_set.update(['6J', '5K', '4L', '3M', '2N', '7K', '6L', '5M', '4N', '3O'])
    else:
        bonus_2_set.update(['7J', '2O'])
        bonus_1_set.update(['2K', '3L', '4M', '5N', '6O', '3J', '4K', '5L', '6M', '7N'])
    
    if '10B' not in piese_set:
        bonus_2_set.update(['10B', '15G'])
        bonus_1_set.update(['14B', '13C', '12D', '11E', '10F', '15C', '14D', '13E', '12F', '11G'])
    else:
        bonus_2_set.update(['10G', '15B'])
        bonus_1_set.update(['10C', '11D', '12E', '13F', '14G', '11B', '12C', '13D', '14E', '15F'])
    
    if '10J' not in piese_set:
        bonus_2_set.update(['10J', '15O'])
        bonus_1_set.update(['14J', '13K', '12L', '11M', '10N', '15K', '14L', '13M', '12N', '11O'])
    else:
        bonus_2_set.update(['10O', '15J'])
        bonus_1_set.update(['10K', '11L', '12M', '13N', '14O', '11J', '12K', '13L', '14M', '15N'])
    
    return bonus_1_set, bonus_2_set

def calculeaza_scor_mutare(piese_noi, toate_piesele, bonus_1_set, bonus_2_set):
    if not piese_noi:
        return 0
    
    scor_total = 0
    
    board = {}
    for coord, shape, color in toate_piesele:
        row, col = coord_to_pos(coord)
        board[(row, col)] = (coord, shape, color)
    
    linii_calculate = set()
    
    piese_noi_coords = {coord_to_pos(coord) for coord, _, _ in piese_noi}
    
    for coord, shape, color in piese_noi:
        row, col = coord_to_pos(coord)
        
        for direction in ['H', 'V']:
            if direction == 'H':
                linie = [(row, c) for c in range(1, 17) if (row, c) in board]
                linie.sort(key=lambda x: x[1])
            else:
                linie = [(r, col) for r in range(1, 17) if (r, col) in board]
                linie.sort(key=lambda x: x[0])
            
            secventa = []
            for pos in linie:
                if not secventa or (direction == 'H' and pos[1] == secventa[-1][1] + 1) or \
                   (direction == 'V' and pos[0] == secventa[-1][0] + 1):
                    secventa.append(pos)
                elif (row, col) in secventa:
                    break
                else:
                    secventa = [pos]
            
            if len(secventa) > 1 and (row, col) in secventa:
                linie_id = (direction, tuple(secventa))
                
                if linie_id not in linii_calculate:
                    linii_calculate.add(linie_id)
                    
                    puncte_linie = len(secventa)
                    
                    if len(secventa) == 6:
                        puncte_linie += 6
                    
                    scor_total += puncte_linie
        
        if coord in bonus_1_set:
            scor_total += 1
        elif coord in bonus_2_set:
            scor_total += 2
    
    if scor_total == 0:
        scor_total = len(piese_noi)
    
    return scor_total

def extrage_careu(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower = np.array([0, 50, 20])
    upper = np.array([30, 255, 200])

    mask = cv.inRange(hsv, lower, upper)
    
    mask = cv.bitwise_not(mask)
    
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

    result = cv.warpPerspective(img, M, (width, height))
    
    return result

def detect_color(patch_hsv):
    h, w, _ = patch_hsv.shape
    center_y, center_x = h // 2, w // 2
    
    radius = 5
    region = patch_hsv[max(0, center_y-radius):min(h, center_y+radius), 
                       max(0, center_x-radius):min(w, center_x+radius)]
    
    h_median = np.median(region[:,:,0])
    s_median = np.median(region[:,:,1])
    v_median = np.median(region[:,:,2])
    
    if s_median <= 50 and v_median >= 60:
        return "W"
    
    if h_median <= 22:
        return "O"
    if 22 < h_median <= 38:
        return "Y"
    if 45 <= h_median <= 85:
        return "G"
    if 85 < h_median <= 130:
        return "B"
    if h_median >= 160:
        return "R"
    
    return "?"


def load_templates(template_folder, target_size=100):
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
    return templates

def match_cell(cell_hsv, templates, threshold=0.6, top_k=5, debug_coord=None):
    all_matches = []
    
    for label, tmpl in templates:
        best_tmpl_score = -1
        
        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated_tmpl = tmpl
            elif angle == 90:
                rotated_tmpl = cv.rotate(tmpl, cv.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_tmpl = cv.rotate(tmpl, cv.ROTATE_180)
            elif angle == 270:
                rotated_tmpl = cv.rotate(tmpl, cv.ROTATE_90_COUNTERCLOCKWISE)
            
            res_v = cv.matchTemplate(cell_hsv[:,:,2], rotated_tmpl[:,:,2], cv.TM_CCOEFF_NORMED)
            score = np.max(res_v)
            
            if score > best_tmpl_score:
                best_tmpl_score = score
        
        all_matches.append((best_tmpl_score, label))
    
    all_matches.sort(key=lambda x: x[0], reverse=True)
    
    valid_matches = [m for m in all_matches if m[0] > threshold]
    
    if not valid_matches:
        return False, 0.0, None
        
    if valid_matches[0][0] > 0.95:
        return True, valid_matches[0][0], valid_matches[0][1]
        
    top_matches = valid_matches[:top_k]
    
    votes = {}
    for score, label in top_matches:
        if label not in votes:
            votes[label] = 0
        votes[label] += score
        
    best_label = max(votes, key=votes.get)
    
    best_score = top_matches[0][0]
                
    return True, best_score, best_label

GRID_SIZE = 16
IMG_SIZE = 1700
CELL_SIZE = 1600 // GRID_SIZE
CELL_OFFSET = 50
PATCH_SIZE = 150
TEMPLATE_SIZE = 100
MARGIN = (PATCH_SIZE - CELL_SIZE) // 2
MATCH_THRESHOLD = 0.40

SHAPE_TO_CODE = {
    'cerc': 1,
    'patrat': 4,
    'romb': 3,
    'trifoi': 2,
    'stea': 6,
    'shuri': 5
}

output_folder = 'detectate'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

templates = load_templates('templates')
print(f"Am incarcat {len(templates)} template-uri.\n")

input_folder = 'antrenare'

if os.path.exists(input_folder):
    files = [f for f in os.listdir(input_folder) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
             and not ('_detected' in f or '_board' in f)]
    
    files = sorted(files)
    
    print(f"Procesez {len(files)} imagini...\n")
    
    previous_pieces = set()
    bonus_1_set = set()
    bonus_2_set = set()
    current_game = None
    
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            game_num = int(file.split('_')[0])
                
            path = os.path.join(input_folder, file)
            img = cv.imread(path)
            if img is None: 
                print(f"Nu am putut citi imaginea: {path}")
                continue
            
            print(f"\nProcesare: {file}")
            
            if game_num != current_game:
                current_game = game_num
                previous_pieces = set()
                
                if '_00' in file:
                    print(f"  [INFO] Nou joc detectat - calculez patrate bonus...")
            
            board = extrage_careu(img)
            
            base_name = os.path.splitext(file)[0]
            
            gray_board = cv.cvtColor(board, cv.COLOR_BGR2GRAY)
            hsv_board = cv.cvtColor(board, cv.COLOR_BGR2HSV)
            
            if DEBUG_MODE:
                output = board.copy()
            detected_count = 0
            detected_pieces = []
            
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    x1 = CELL_OFFSET + col * CELL_SIZE
                    y1 = CELL_OFFSET + row * CELL_SIZE
                    x2 = x1 + CELL_SIZE
                    y2 = y1 + CELL_SIZE
                    
                    patch_x1 = max(0, x1 - MARGIN)
                    patch_y1 = max(0, y1 - MARGIN)
                    patch_x2 = min(IMG_SIZE, x2 + MARGIN)
                    patch_y2 = min(IMG_SIZE, y2 + MARGIN)
                    
                    cell_gray = gray_board[y1:y2, x1:x2]
                    
                    cell_hsv = hsv_board[patch_y1:patch_y2, patch_x1:patch_x2]
                    cell_hsv = cv.resize(cell_hsv, (PATCH_SIZE, PATCH_SIZE))
                    
                    cell_median = np.median(cell_gray)
                    dark_pixels = np.sum(cell_gray < 100)
                    dark_ratio = dark_pixels / cell_gray.size
                    
                    col_letter = chr(65 + col)
                    row_number = row + 1
                    cell_coord = f"{row_number}{col_letter}"
                    
                    has_valid_dark_content = dark_ratio > 0.40
                    
                    is_match, score, name = match_cell(cell_hsv, templates, 0.55)
                    
                    color = detect_color(cell_hsv) if is_match else None
                    
                    if is_match and name == 'romb' and color == 'W' and score < 0.88:
                        cerc_templates = [(label, img) for label, img in templates if 'cerc' in label]
                        if cerc_templates:
                            is_cerc, score_cerc, name_cerc = match_cell(cell_hsv, cerc_templates, 0.55)
                            if is_cerc and score_cerc > score - 0.05:
                                is_match, score, name = is_cerc, score_cerc, name_cerc
                    
                    if name and ('sus' in name.lower() or 'jos' in name.lower()):
                        is_match = False
                    
                    is_piece = is_match and has_valid_dark_content
                    
                    center_x = x1 + CELL_SIZE // 2
                    center_y = y1 + CELL_SIZE // 2
                    
                    if is_piece:
                        detected_count += 1
                        
                        col_letter = chr(65 + col)
                        row_number = row + 1
                        cell_coord = f"{row_number}{col_letter}"
                        
                        shape_code = SHAPE_TO_CODE.get(name, 0)
                        detected_pieces.append((cell_coord, shape_code, color))
                        
                        if DEBUG_MODE:
                            cv.circle(output, (center_x, center_y), 20, (0, 255, 0), -1)
                            
                            template_name = name
                            debug_text = f"{cell_coord} {color}{template_name}"
                            cv.putText(output, debug_text, (center_x-30, center_y-25), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            debug_text2 = f"S:{score:.2f} D:{dark_ratio:.2f}"
                            cv.putText(output, debug_text2, (center_x-30, center_y+35), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    else:
                        if DEBUG_MODE:
                            cv.circle(output, (center_x, center_y), 3, (0, 0, 255), -1)
            
            print(f"  Piese detectate: {detected_count}")
            
            if DEBUG_MODE:
                for i in range(GRID_SIZE + 1):
                    p = CELL_OFFSET + i * CELL_SIZE
                    cv.line(output, (p, CELL_OFFSET), (p, CELL_OFFSET + GRID_SIZE * CELL_SIZE), (0, 255, 0), 2)
                    cv.line(output, (CELL_OFFSET, p), (CELL_OFFSET + GRID_SIZE * CELL_SIZE, p), (0, 255, 0), 2)
                
                font = cv.FONT_HERSHEY_SIMPLEX
                for i in range(GRID_SIZE):
                    col_label = chr(65 + i)
                    x_pos = CELL_OFFSET + int((i + 0.5) * CELL_SIZE)
                    cv.putText(output, col_label, (x_pos - 10, CELL_OFFSET - 10), font, 0.8, (255, 255, 0), 2)
                    
                    row_label = str(i + 1)
                    y_pos = CELL_OFFSET + int((i + 0.5) * CELL_SIZE)
                    cv.putText(output, row_label, (CELL_OFFSET - 30, y_pos + 10), font, 0.8, (255, 255, 0), 2)
                
                output_path = os.path.join(output_folder, file.replace('.jpg', '_detected.jpg'))
                cv.imwrite(output_path, output)
                print(f"  Salvat: {output_path}")
            
            current_pieces = set((coord, shape_code, color) for coord, shape_code, color in detected_pieces)
            
            if '_00' in file:
                coords_only = [coord for coord, _, _ in current_pieces]
                bonus_1_set, bonus_2_set = detecteaza_bonus_pattern(coords_only)
                print(f"  [INFO] Bonus +1: {len(bonus_1_set)} patrate, Bonus +2: {len(bonus_2_set)} patrate")
                
                previous_pieces = current_pieces
                continue
            
            previous_coords_colors = {(coord, color) for coord, _, color in previous_pieces}
            new_pieces = set()
            
            for coord, shape_code, color in current_pieces:
                if (coord, color) not in previous_coords_colors:
                    new_pieces.add((coord, shape_code, color))
            
            scor = calculeaza_scor_mutare(new_pieces, current_pieces, bonus_1_set, bonus_2_set)

            txt_path = os.path.join(output_folder, file.replace('.jpg', '.txt'))
            with open(txt_path, 'w') as f:
                for coord, shape_code, color in sorted(new_pieces, key=lambda x: (int(x[0][:-1]), x[0][-1])):
                    f.write(f"{coord} {shape_code}{color}\n")
                f.write(f"{scor}")
            print(f"  Salvat: {txt_path} ({len(new_pieces)} piese noi, scor: {scor})")
            
            previous_pieces = current_pieces
            
else:
    print(f"Folderul {input_folder} nu exista!")
