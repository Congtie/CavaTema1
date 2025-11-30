def compare_annotations(filename_predicted, filename_gt, verbose=0):
    try:
        p = open(filename_predicted, "rt")
        gt = open(filename_gt, "rt")
    except FileNotFoundError:
        return 0, 0, 0, 0
    
    all_lines_p = p.readlines()
    all_lines_gt = gt.readlines()
    p.close()
    gt.close()

    # positions and tiles
    number_lines_p = len(all_lines_p)
    number_lines_gt = len(all_lines_gt)

    match_positions = 1
    match_tiles = 1
    match_score = 1

    for i in range(number_lines_gt - 1):  # Exclude ultima linie (scorul)
        current_pos_gt, current_tile_gt = all_lines_gt[i].split()
        
        if verbose:
            print(f"  GT: {current_pos_gt} {current_tile_gt}")

        try:
            current_pos_p, current_tile_p = all_lines_p[i].split()
            
            if verbose:
                print(f"  Predicted: {current_pos_p} {current_tile_p}")

            if current_pos_p != current_pos_gt:
                match_positions = 0
                if verbose:
                    print(f"  POSITION MISMATCH!")
            if current_tile_p != current_tile_gt:
                match_tiles = 0
                if verbose:
                    print(f"  TILE MISMATCH!")
        except:
            match_positions = 0
            match_tiles = 0
            if verbose:
                print(f"  MISSING LINE!")
    
    try:
        # verify if there are more positions + tiles lines in the prediction file
        current_pos_p, current_tile_p = all_lines_p[number_lines_gt].split()
        match_positions = 0
        match_tiles = 0

        if verbose:
            print(f"  EXTRA LINE: {current_pos_p} {current_tile_p}")
    except:
        pass
    
    # Comparăm scorul (ultima linie)
    try:
        score_p = int(all_lines_p[-1].strip())
        score_gt = int(all_lines_gt[-1].strip())
        
        if score_p != score_gt:
            match_score = 0
            if verbose:
                print(f"  SCORE MISMATCH! Predicted: {score_p}, GT: {score_gt}")
    except:
        match_score = 0
        if verbose:
            print(f"  SCORE ERROR!")

    points_positions = 0.04 * match_positions
    points_tiles = 0.03 * match_tiles
    points_score = 0.01 * match_score

    return points_positions, points_tiles, points_score


# EVALUATION ON TRAINING SET
print("=" * 60)
print("EVALUARE PE SETUL DE ANTRENARE")
print("=" * 60)

# path-uri
predictions_path_root = "detectate/"
gt_path_root = "antrenare/"

# schimba la 1 pentru detalii
verbose = 0
total_points = 0

for game in range(1, 6):  # 1_xx, 2_xx, 3_xx, 4_xx, 5_xx
    print(f"\n--- JOC {game} ---")
    game_points = 0
    
    for move in range(1, 21):  # 01 la 20 (skip _00)
        name_move = str(move)
        if move < 10:
            name_move = '0' + str(move)

        filename_predicted = predictions_path_root + str(game) + '_' + name_move + '.txt'
        filename_gt = gt_path_root + str(game) + '_' + name_move + '.txt'

        game_move = str(game) + '_' + name_move
        points_position = 0
        points_tiles = 0
        points_score = 0

        try:
            points_position, points_tiles, points_score = compare_annotations(filename_predicted, filename_gt, verbose)
        except Exception as e:
            if verbose:
                print(f"Error pentru {game_move}: {e}")

        total_move_points = points_position + points_tiles + points_score
        game_points += total_move_points
        
        # Afișează rezultate detaliate
        status = "OK" if total_move_points == 0.08 else "ERR"
        if verbose or total_move_points < 0.08:
            print(f"{status} {game_move}: Pos={points_position:.2f} Tile={points_tiles:.2f} Score={points_score:.2f} Total={total_move_points:.2f}")

    total_points += game_points
    print(f"  Total joc {game}: {game_points:.2f} / {20 * 0.08:.2f}")

print("\n" + "=" * 60)
print(f"PUNCTAJ TOTAL: {total_points:.2f} / {5 * 20 * 0.08:.2f}")
print(f"Acuratețe: {(total_points / (5 * 20 * 0.08) * 100):.1f}%")
print("=" * 60)
