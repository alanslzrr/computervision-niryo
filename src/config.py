"""
Configuracion centralizada del sistema Niryo Poker Dice Vision.

Contiene todos los parametros de calibracion del robot, deteccion de vision
y constantes para los nuevos modulos de clasificacion CNN y evaluacion de jugadas.
"""

# ============================================================================
# ROBOT
# ============================================================================

ROBOT_IP = "172.16.126.177"
# Pose de escaneo (camara mirando al workspace)
SCANNING_POSITION = (0.005, -0.13, 0.23, 2.93, 1.5, 1.43)

# Limites cartesianos del workspace (medidos con puntero en el robot).
# Orden fisico: TL, BL, TR, BR. Se usan en relative_to_robot_xy para mapear pixel -> robot.
LIMIT_1 = (-0.08, -0.329, 0.129, -0.523, 1.531, -2.056)  # TL
LIMIT_2 = (-0.082, -0.163, 0.129, 3.046, 1.511, 1.506)  # BL
LIMIT_3 = (0.082, -0.153, 0.129, -2.957, 1.538, 1.745)   # TR
LIMIT_4 = (0.082, -0.328, 0.129, 2.289, 1.529, 0.68)   # BR

# Parametros de pick en metros (originalmente 164 mm y 131 mm)
APPROACH_Z = 0.164
PICK_Z = 0.131

# ============================================================================
# VISION
# ============================================================================

# Bounds en pixeles como fallback cuando no se detectan las 4 dianas
FALLBACK_WORKSPACE_PIXEL_BOUNDS = {
    "x_min": 100,
    "x_max": 540,
    "y_min": 80,
    "y_max": 400,
}

# Deteccion de piezas: area minima/maxima de contornos y factor para approxPolyDP
MIN_AREA = 1000
MAX_AREA = 150000
EPSILON_FACTOR = 0.03

# Rangos HSV para clasificacion de color (rojo usa dos rangos por el wrap de hue)
COLOR_RANGES = {
    "RED": [
        {"lower": (0, 70, 50), "upper": (10, 255, 255)},
        {"lower": (160, 70, 50), "upper": (180, 255, 255)},
    ],
    "GREEN": [{"lower": (35, 70, 50), "upper": (85, 255, 255)}],
    "BLUE": [{"lower": (85, 60, 50), "upper": (130, 255, 255)}],
}

# ============================================================================
# CNN / POKER DICE (nuevos modulos)
# ============================================================================

DICE_FACES = ["9", "10", "J", "Q", "K", "A"]
ONNX_MODEL_PATH = "dice_classifier.onnx"
DATASET_DIR = "dataset/"
CNN_INPUT_SIZE = (64, 64)
CNN_CONFIDENCE_THRESHOLD = 0.7
