"""
Configuración centralizada del sistema Niryo Poker Dice Vision.

Este módulo agrupa parámetros de:
    - Conexión y poses del robot Niryo (escaneo, límites del workspace, pick/drop).
    - Visión por computador (dianas, contornos, rangos HSV).
    - Clasificación CNN de caras de dado (ONNX) y disposición de columnas de depósito.

Modificar aquí los valores de calibración tras mediciones en banco; evitar duplicar
constantes en otros archivos.

Secciones:
    ROBOT: IP, pose de escaneo, poses de referencia del workspace, alturas de acercamiento y agarre.
    VISION: Fallback de esquinas en píxeles, umbrales de área de contornos, rangos HSV por color.
    CNN / POKER: Etiquetas de caras, ruta del modelo, tamaño de entrada y umbral de confianza.
    DROP: Posiciones relativas para apilar dados por cara tras el pick-and-place.
"""

import os

# Raíz del proyecto (una carpeta por encima de src/). Las rutas a assets/ se
# resuelven relativas a este directorio, para que el código funcione con
# cualquier CWD — basta con que el layout del repo siga intacto.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSETS_DIR = os.path.join(_PROJECT_ROOT, "assets")

# =============================================================================
# ROBOT — Conexión, workspace y movimientos de agarre
# =============================================================================

# Dirección IP del Niryo One en la red local (ajustar al entorno de laboratorio).
ROBOT_IP = "172.16.126.177"

# Pose de escaneo: la cámara embarcada observa el workspace completo (x, y, z, roll, pitch, yaw).
SCANNING_POSITION = (0.005, -0.13, 0.23, 2.93, 1.5, 1.43)

# Límites cartesianos del workspace medidos con el puntero/teach del robot.
# Orden físico de las esquinas: TL, BL, TR, BR. Se usan en ``relative_to_robot_xy`` para
# mapear coordenadas normalizadas de imagen a (x, y) del robot.
LIMIT_1 = (-0.08, -0.329, 0.129, -0.523, 1.531, -2.056)  # Esquina TL de la imagen
LIMIT_2 = (-0.082, -0.163, 0.129, 3.046, 1.511, 1.506)  # Esquina BL
LIMIT_3 = (0.082, -0.153, 0.129, -2.957, 1.538, 1.745)   # Esquina TR
LIMIT_4 = (0.082, -0.328, 0.129, 2.289, 1.529, 0.68)    # Esquina BR

# Alturas de seguridad y agarre en metros (aprox. 164 mm y 131 mm respecto al plano de trabajo).
APPROACH_Z = 0.164
PICK_Z = 0.131

# =============================================================================
# VISIÓN — Workspace por dianas y segmentación de piezas
# =============================================================================

# Rectángulo en píxeles usado cuando no se detectan las cuatro dianas (orden homografía: TL, TR, BR, BL).
FALLBACK_WORKSPACE_PIXEL_BOUNDS = {
    "x_min": 100,
    "x_max": 540,
    "y_min": 80,
    "y_max": 400,
}

# Filtrado de contornos: área mínima/máxima y epsilon relativo para ``approxPolyDP``.
MIN_AREA = 1000
MAX_AREA = 150000
EPSILON_FACTOR = 0.03

# Rangos HSV para clasificación por color. El rojo usa dos intervalos por el wrap del matiz en OpenCV.
COLOR_RANGES = {
    "RED": [
        {"lower": (0, 70, 50), "upper": (10, 255, 255)},
        {"lower": (160, 70, 50), "upper": (180, 255, 255)},
    ],
    "GREEN": [{"lower": (35, 70, 50), "upper": (85, 255, 255)}],
    "BLUE": [{"lower": (85, 60, 50), "upper": (130, 255, 255)}],
}

# =============================================================================
# CNN / POKER DICE — Clasificación de caras y modelo ONNX
# =============================================================================

# Etiquetas de clase en el mismo orden que la salida softmax del entrenamiento (train.py).
DICE_FACES = ["9", "10", "J", "Q", "K", "A"]

# Ruta al modelo exportado; debe coincidir con la salida de ``train.export_onnx``.
ONNX_MODEL_PATH = os.path.join(_ASSETS_DIR, "models", "dice_classifier.onnx")

# Checkpoint PyTorch del mejor epoch durante el entrenamiento.
CHECKPOINT_PATH = os.path.join(_ASSETS_DIR, "models", "best_model.pth")

# Directorio raíz del dataset de entrenamiento (subcarpetas por cara).
DATASET_DIR = os.path.join(_ASSETS_DIR, "dataset")

# Tamaño espacial de entrada a la red (ancho, alto); debe coincidir con train y classifier.
CNN_INPUT_SIZE = (64, 64)

# Por debajo de este umbral de probabilidad máxima se devuelve UNKNOWN en inferencia.
CNN_CONFIDENCE_THRESHOLD = 0.7

# =============================================================================
# DROP — Columnas y slots para pick-and-place de dados agrupados por cara
# =============================================================================
# Cada cara distinta recibe una columna vertical lógica dentro del workspace. Los dados de la
# misma cara se apilan en distintos slots Y dentro de esa columna. Así, una jugada con varias
# caras (p. ej. TWO_PAIR) usa varias columnas. Tras cada ronda, ``reset`` en la UI libera
# la asignación de columnas.
#
# En poker dice hay como máximo cinco dados; a lo sumo cinco caras distintas → cinco columnas bastan.

# Fracción horizontal [0, 1] del workspace para el eje X de cada columna (de izquierda a derecha en imagen).
DROP_COLUMN_X_REL = [0.10, 0.30, 0.50, 0.70, 0.90]

# Fracción vertical [0, 1] para apilar hasta cinco dados con la misma cara en una columna.
DROP_SLOT_Y_REL = [0.12, 0.30, 0.48, 0.66, 0.84]

# Altura Z del depósito: misma que PICK_Z para que el dado quede apoyado en el plano de trabajo.
DROP_Z = PICK_Z
