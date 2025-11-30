import cv2 as cv
import numpy as np

# Încarcă imaginea 1_00 pentru fiecare joc
def detect_bonus_squares(image_path):
    """
    Detectează pătratele bonus (+1 și +2) din imaginea inițială.
    Returnează două seturi: bonus_1 și bonus_2 cu coordonate.
    """
    img = cv.imread(image_path)
    
    # Convertim în HSV pentru detectarea culorilor
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Pătratele +2 sunt roșii
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)
    
    # Pătratele +1 sunt portocalii/galbene
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([30, 255, 255])
    mask_orange = cv.inRange(hsv, lower_orange, upper_orange)
    
    bonus_1 = set()
    bonus_2 = set()
    
    # Analizăm fiecare careuriu pentru a găsi bonusurile
    # TODO: Implementare detectare bonusuri din imagine
    
    return bonus_1, bonus_2


# Algoritmul tău pentru generarea bonusurilor bazat pe simetrie
def generate_bonus_pattern(has_piece_2B, has_piece_2G, has_piece_7B, has_piece_7G):
    """
    Generează pattern-ul de bonusuri bazat pe unde sunt piesele în colțuri.
    
    Regula: Dacă găsim piesa în 2G → bonusul +2 e în 2B (opus diagonal)
           Dacă găsim piesa în 2B → bonusul +2 e în 2G
    """
    bonus_1 = set()
    bonus_2 = set()
    
    # Quadrant 1: 1-8 cu A-H
    if has_piece_2G:
        # +2 în colțul opus: 2B
        bonus_2.add("2B")
        # +1 pe diagonala: 2F, 3E, 4D, 5C, 6B și 7C, 6D, 5E, 4F, 3G, 7G
        for coord in ["2F", "3E", "4D", "5C", "6B", "7C", "6D", "5E", "4F", "3G", "7G"]:
            bonus_1.add(coord)
    
    if has_piece_2B:
        # +2 în colțul opus: 2G
        bonus_2.add("2G")
        # +1 pe diagonala: 3B, 2C, 4C, 3D, 5D, 4E, 5F, 6E, 6G, 7F
        for coord in ["3B", "2C", "4C", "3D", "5D", "4E", "5F", "6E", "6G", "7F"]:
            bonus_1.add(coord)
    
    if has_piece_7B:
        # +2 în colțul opus: 7G
        bonus_2.add("7G")
        # +1 pe diagonală
        for coord in ["7F", "6E", "5D", "4C", "3B", "2C", "3D", "4E", "5F", "6G"]:
            bonus_1.add(coord)
    
    if has_piece_7G:
        # +2 în colțul opus: 7B
        bonus_2.add("7B")
        # +1 pe diagonală
        for coord in ["6B", "7C", "5C", "6D", "4D", "5E", "3E", "4F", "2F", "3G"]:
            bonus_1.add(coord)
    
    # TODO: Extinde pentru celelalte 3 quadrante (I-P cu 1-8, A-H cu 9-16, I-P cu 9-16)
    
    return bonus_1, bonus_2


# Funcție pentru calcularea scorului Qwirkle
def calculate_qwirkle_score(board, new_pieces, bonus_1_coords, bonus_2_coords):
    """
    Calculează scorul conform regulilor Qwirkle.
    
    board: dict {coord: (shape, color)} - toate piesele de pe tablă
    new_pieces: list [(coord, shape, color)] - piesele noi plasate
    bonus_1_coords: set - coordonatele cu bonus +1
    bonus_2_coords: set - coordonatele cu bonus +2
    """
    total_score = 0
    processed_lines = set()
    
    for coord, shape, color in new_pieces:
        row = int(coord[:-1])
        col = coord[-1]
        
        # Verificăm linia orizontală
        horizontal_line = []
        for c in "ABCDEFGHIJKLMNOP":
            check_coord = f"{row}{c}"
            if check_coord in board:
                horizontal_line.append(check_coord)
        
        if len(horizontal_line) > 1 and tuple(sorted(horizontal_line)) not in processed_lines:
            line_score = len(horizontal_line)
            
            # Adaugă bonusuri de pătrat
            for piece_coord in horizontal_line:
                if piece_coord in new_pieces:
                    if piece_coord in bonus_1_coords:
                        line_score += 1
                    elif piece_coord in bonus_2_coords:
                        line_score += 2
            
            # Bonus Qwirkle (6 piese)
            if len(horizontal_line) == 6:
                line_score += 6
            
            total_score += line_score
            processed_lines.add(tuple(sorted(horizontal_line)))
        
        # Verificăm linia verticală
        vertical_line = []
        for r in range(1, 17):
            check_coord = f"{r}{col}"
            if check_coord in board:
                vertical_line.append(check_coord)
        
        if len(vertical_line) > 1 and tuple(sorted(vertical_line)) not in processed_lines:
            line_score = len(vertical_line)
            
            # Adaugă bonusuri de pătrat
            for piece_coord in vertical_line:
                if piece_coord in new_pieces:
                    if piece_coord in bonus_1_coords:
                        line_score += 1
                    elif piece_coord in bonus_2_coords:
                        line_score += 2
            
            # Bonus Qwirkle (6 piese)
            if len(vertical_line) == 6:
                line_score += 6
            
            total_score += line_score
            processed_lines.add(tuple(sorted(vertical_line)))
    
    # Dacă o singură piesă plasată și nu face parte din nicio linie → 1 punct
    if total_score == 0 and len(new_pieces) == 1:
        total_score = 1
        if new_pieces[0][0] in bonus_1_coords:
            total_score += 1
        elif new_pieces[0][0] in bonus_2_coords:
            total_score += 2
    
    return total_score


if __name__ == "__main__":
    # Test
    print("Detectare bonusuri pentru jocurile 1-5...")
    
    for game in range(1, 6):
        print(f"\nJoc {game}:")
        # TODO: Încarcă imaginea x_00.jpg și detectează bonusurile
        # Pentru moment, folosim algoritmul de generare
