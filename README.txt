BIBLIOTECI NECESARE:

opencv-python==4.10.0.84
numpy==1.26.4

RULARE:

Detectarea pieselor Qwirkle:
  Script: solutie.py
  Sintaxa: python solutie.py [input_folder] [output_folder]
  
  Exemple:
  - python solutie.py
    (foloseste folder-ul implicit: input=antrenare/, output=detectate/)
  
  - python solutie.py evaluare/fake_test evaluare/fake_test/detectate
    (ruleaza pe fake_test)
  
  Output: fisiere .txt in output_folder cu format:
    <coord> <cod_forma><culoare>
    ...
    <scor>
  
  Unde:
    - coord: ex. 2B, 10J (rand + coloana)
    - cod_forma: 1=cerc, 2=trifoi, 3=romb, 4=patrat, 5=shuri, 6=stea
    - culoare: R=Red, O=Orange, Y=Yellow, G=Green, B=Blue, W=White
    - scor: scorul total al mutarii conform regulilor Qwirkle

Evaluare pe setul de antrenare:
  Script: evalueaza_detectie.py
  Sintaxa: python evalueaza_detectie.py
  Output: afiseaza scorul total si acuratetea (compara detectate/ cu antrenare/)

Evaluare oficiala:
  Script: evaluare/cod_evaluare/evalueaza_solutie.py
  Sintaxa: cd evaluare/cod_evaluare; python evalueaza_solutie.py
  Output: afiseaza punctajul (compara ../../detectate/ cu ../../antrenare/)

STRUCTURA FOLDERELOR:

CavaTema1/
├── solutie.py              (script principal)
├── evalueaza_detectie.py   (evaluare custom)
├── README.txt              (acest fisier)
├── documentatie.tex        (documentatie tehnica LaTeX)
├── antrenare/              (imagini training: 1_00.jpg - 5_20.jpg)
│   └── *.txt               (ground truth)
├── templates/              (template-uri forme Qwirkle)
│   ├── cerc/
│   ├── patrat/
│   ├── romb/
│   ├── trifoi/
│   ├── stea/
│   └── shuri/
├── detectate/              (output-ul scriptului solutie.py)
│   ├── *.txt               (detectii)
│   └── *_detected.jpg      (imagini debug, daca DEBUG_MODE=True)
└── evaluare/
    ├── fake_test/          (imagini test)
    │   ├── *.jpg
    │   └── ground-truth/
    └── cod_evaluare/
        └── evalueaza_solutie.py

REZULTATE:

- Training set (antrenare/): 8.00/8.00 puncte (100%)
- Fake test: 8.00/8.00 puncte (100%)