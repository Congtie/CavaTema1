import cv2 as cv
import numpy as np
import os

def show_image(title, image):
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    h, w = image.shape[:2]
    scale = 800 / max(h, w)
    cv.resizeWindow(title, int(w * scale), int(h * scale))
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def extrage_careu(img):
    # TODO: Implementează funcția de extragere a careului
    # Momentan facem resize la 1600x1600 pentru a simula rezultatul
    return cv.resize(img, (1600, 1600))

lines_horizontal=[]
# Folosim linspace pentru a împărți 1600 în 9 secțiuni egale (10 linii)
for i in np.linspace(0, 1600, 10):
    y = int(i)
    # Corecție pentru ultima linie să fie în interiorul imaginii
    if y == 1600: y = 1599
    
    l=[]
    l.append((0,y))
    l.append((1599,y))
    lines_horizontal.append(l)

lines_vertical=[]
for i in np.linspace(0, 1600, 10):
    x = int(i)
    if x == 1600: x = 1599
    
    l=[]
    l.append((x,0))
    l.append((x,1599))
    lines_vertical.append(l)

# Directorul cu imagini
input_folder = 'antrenare'

if os.path.exists(input_folder):
    files = os.listdir(input_folder)
    for file in files:
        if file.endswith('.jpg'):
            img_path = os.path.join(input_folder, file)
            img = cv.imread(img_path)
            
            if img is None:
                continue
                
            result = extrage_careu(img)
            
            for line in lines_vertical: 
                cv.line(result, line[0], line[1], (0, 255, 0), 5)
            for line in lines_horizontal: 
                cv.line(result, line[0], line[1], (0, 0, 255), 5)
            
            show_image('img', result)
else:
    print(f"Folderul {input_folder} nu există.")
