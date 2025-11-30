"""
Script pentru detectarea pătratelor bonus (+1 și +2) din imaginile x_00.jpg
Logica: Verificăm unde sunt plasate piesele inițiale pentru a deduce poziția bonusurilor
"""

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
    Detectează pattern-ul de bonus pe baza pieselor plasate.
    
    Formula ta:
    - Dacă piesa e la 2G: +2 la 2B, +1 la: 2F, 3E, 4D, 5C, 6B și 7C, 6D, 5E, 4F, 3G, 7G
    - Dacă piesa e la 2B: +2 la 2G, +1 la: 3B, 2C, 4C, 3D, 5D, 4E, 5F, 6E, 6G, 7F
    
    Generalizăm pentru toate cele 4 cadrane 8x8.
    """
    
    piese_set = set(piese_coords)
    
    bonus_2_coords = []  # Pătrate cu +2
    bonus_1_coords = []  # Pătrate cu +1
    
    # Pentru fiecare cadran 8x8
    for row_base in [1, 9]:  # Cadrane: 1-8 sau 9-16
        for col_base in [1, 9]:  # Cadrane: A-H (1-8) sau I-P (9-16)
            
            # Calculăm pozițiile cheie pentru acest cadran
            pos_2B = pos_to_coord(row_base + 1, col_base + 1)  # 2B în cadran
            pos_2G = pos_to_coord(row_base + 1, col_base + 6)  # 2G în cadran
            
            has_2B = pos_2B in piese_set
            has_2G = pos_2G in piese_set
            
            if has_2G and not has_2B:
                # Pattern A (piesa la 2G): +2 la 2B
                bonus_2_coords.append(pos_2B)
                
                # +1 pe diagonalele din formula ta
                bonus_1_list = [
                    (row_base + 1, col_base + 5),  # 2F
                    (row_base + 2, col_base + 4),  # 3E
                    (row_base + 3, col_base + 3),  # 4D
                    (row_base + 4, col_base + 2),  # 5C
                    (row_base + 5, col_base + 1),  # 6B
                    (row_base + 6, col_base + 2),  # 7C
                    (row_base + 5, col_base + 3),  # 6D
                    (row_base + 4, col_base + 4),  # 5E
                    (row_base + 3, col_base + 5),  # 4F
                    (row_base + 2, col_base + 6),  # 3G
                    (row_base + 6, col_base + 6),  # 7G
                ]
                
                for r, c in bonus_1_list:
                    if 1 <= r <= 16 and 1 <= c <= 16:
                        bonus_1_coords.append(pos_to_coord(r, c))
                
            elif has_2B and not has_2G:
                # Pattern B (piesa la 2B): +2 la 2G
                bonus_2_coords.append(pos_2G)
                
                # +1 pe diagonalele din formula ta
                bonus_1_list = [
                    (row_base + 2, col_base + 1),  # 3B
                    (row_base + 1, col_base + 2),  # 2C
                    (row_base + 3, col_base + 2),  # 4C
                    (row_base + 2, col_base + 3),  # 3D
                    (row_base + 4, col_base + 3),  # 5D
                    (row_base + 3, col_base + 4),  # 4E
                    (row_base + 4, col_base + 5),  # 5F
                    (row_base + 5, col_base + 4),  # 6E
                    (row_base + 5, col_base + 6),  # 6G
                    (row_base + 6, col_base + 5),  # 7F
                ]
                
                for r, c in bonus_1_list:
                    if 1 <= r <= 16 and 1 <= c <= 16:
                        bonus_1_coords.append(pos_to_coord(r, c))
    
    return bonus_1_coords, bonus_2_coords


def main():
    import os
    
    # Citim toate fișierele x_00.txt
    detectate_folder = 'detectate'
    bonus_maps = {}
    
    for game_num in range(1, 6):
        file_path = os.path.join(detectate_folder, f'{game_num}_00.txt')
        
        if not os.path.exists(file_path):
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        piese = []
        for line in lines:
            line = line.strip()
            if not line or line.isdigit():
                continue
            parts = line.split()
            if len(parts) >= 2:
                piese.append(parts[0])
        
        print(f"\nJOC {game_num}:")
        print(f"Piese inițiale: {piese}")
        
        bonus_1, bonus_2 = detecteaza_bonus_pattern(piese)
        
        if bonus_1 or bonus_2:
            bonus_maps[game_num] = {'bonus_1': bonus_1, 'bonus_2': bonus_2}
            print(f"Bonus +1: {bonus_1}")
            print(f"Bonus +2: {bonus_2}")
        else:
            print("Nu s-a putut detecta pattern-ul de bonus")
    
    # Salvăm într-un fișier
    output_file = 'bonus_squares.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for game_num, bonuses in bonus_maps.items():
            f.write(f"Joc {game_num}:\n")
            f.write(f"  +1: {', '.join(bonuses['bonus_1'])}\n")
            f.write(f"  +2: {', '.join(bonuses['bonus_2'])}\n")
            f.write("\n")
    
    print(f"\n✓ Salvat în {output_file}")


if __name__ == '__main__':
    main()
