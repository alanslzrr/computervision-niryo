# Niryo Poker Dice Vision

Sistema de visión robótica para el robot Niryo One que detecta dados de póker sobre un workspace calibrado, clasifica sus caras con una CNN entrenada desde cero, evalúa la jugada resultante y ejecuta pick-and-place de las piezas detectadas.



---

## Arquitectura

El proyecto está organizado como un conjunto de módulos planos con responsabilidades bien definidas. La frontera entre visión y robot se establece en el espacio normalizado `[0,1]`: todo lo que opera sobre píxeles vive en `vision.py`, todo lo que toca coordenadas físicas vive en `robot.py`.

```
computervision-niryo/
├── config.py         # Constantes: calibración del robot, parámetros de visión, CNN
├── robot.py          # NiryoVisionPicker + mapeo coordenadas normalizadas → robot
├── vision.py         # Detección del workspace (dianas) + pipeline de detección de piezas
├── classifier.py     # Inferencia ONNX para clasificación de caras de dados
├── evaluator.py      # Evaluador de jugadas de póker (Poker, Full House, Escalera, ...)
├── dataset.py        # Utilidades para captura y almacenamiento del dataset
├── ui.py             # Dibujo sobre frames + comandos de terminal + pick-and-place
├── capture.py        # Script standalone para capturar imágenes de entrenamiento
├── train.py          # Entrenamiento de la CNN con PyTorch + export a ONNX
├── poker.py          # Script standalone para identificar jugadas en tiempo real
├── main.py           # Orquestador: robot + visión + comandos + pick-and-place
├── dataset/          # Imágenes de entrenamiento organizadas por clase
│   ├── 9/  10/  J/  Q/  K/  A/
├── best_model.pth    # Checkpoint PyTorch del mejor modelo
├── dice_classifier.onnx  # Modelo exportado para inferencia
└── requirements.txt
```

### Grafo de dependencias

```
config.py           ← sin imports del proyecto
    ↑
robot.py            ← importa de: config
    ↑
vision.py           ← importa de: config
    ↑
classifier.py       ← importa de: config
evaluator.py        ← importa de: config
dataset.py          ← importa de: config
    ↑
ui.py               ← importa de: config, robot, vision
capture.py          ← importa de: config, robot, vision
poker.py            ← importa de: config, robot, vision, capture, classifier, evaluator
    ↑
main.py             ← importa de: config, robot, vision, ui
train.py            ← importa de: config (standalone)
```

`robot.py` y `vision.py` son independientes entre sí. Los scripts standalone (`capture.py`, `poker.py`, `train.py`, `main.py`) son los puntos de entrada del sistema.

### Pipeline de visión

1. **Calibración del workspace** — Se detectan 4 dianas (círculos concéntricos) en las esquinas del tablero mediante `HoughCircles`. Se construye una homografía que mapea píxeles a coordenadas normalizadas `[0,1]×[0,1]`.
2. **Detección de dados** — Dentro del polígono del workspace se aplica Canny + morfología para detectar los contornos de los dados (funciona con dados blancos sobre fondo blanco porque detecta bordes, no brillo). Se excluyen las dianas por proximidad a las esquinas.
3. **Clasificación CNN** — Cada crop se pasa por una CNN de 3 bloques convolucionales (exportada a ONNX) que devuelve una de 6 caras: `9, 10, J, Q, K, A`.
4. **Evaluación de jugada** — La lista de caras detectadas se evalúa contra la jerarquía de póker dice: Póker > Full House > Escalera > Trío > Doble Par > Par > Nada.
5. **Pick-and-place** — Las coordenadas del dado seleccionado se mapean por interpolación bilineal desde `[0,1]` a coordenadas cartesianas del robot usando los 4 `LIMIT_N` calibrados físicamente.

### Arquitectura de la CNN (`DiceCNN`)

| Capa                          | Entrada      | Salida       |
| ----------------------------- | ------------ | ------------ |
| Conv2d + BN + ReLU + MaxPool  | 3 × 64 × 64  | 32 × 32 × 32 |
| Conv2d + BN + ReLU + MaxPool  | 32 × 32 × 32 | 64 × 16 × 16 |
| Conv2d + BN + ReLU + MaxPool  | 64 × 16 × 16 | 128 × 8 × 8  |
| Flatten + Dropout + FC + ReLU | 8.192        | 256          |
| Dropout + FC (logits)         | 256          | 6 clases     |

El modelo se entrena con `CrossEntropyLoss` + Adam, usando data augmentation (rotaciones, flip horizontal, color jitter, affine). La salida son logits sin softmax — `DiceClassifier` aplica el softmax en inferencia.

---

## Instalación

```bash
# Crear venv desde la raíz niryo-lab-vision/
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias
cd computervision-niryo
pip install -r requirements.txt
```

**Dependencias principales:**
- `opencv-python` — procesamiento de imagen
- `numpy` — operaciones numéricas
- `pyniryo` — control del robot Niryo
- `torch` + `torchvision` — entrenamiento de la CNN
- `onnxruntime` — inferencia del modelo exportado
- `scikit-learn` — métricas y train/test split

---

## Comandos

Todos los scripts asumen que estoy dentro de `computervision-niryo/` con el venv activo y el robot conectado en la IP configurada en `config.py` (`ROBOT_IP`).

### 1. Capturar imágenes para el dataset

```bash
# Modo interactivo — detecta dados y pide la etiqueta por cada crop
python capture.py

# Baja la cámara a Z=0.18m para mejor resolución de los crops
python capture.py --low

# Etiqueta fija — todos los crops se guardan como "K"
python capture.py --label K

# Combinado — baja la cámara y etiqueta fija
python capture.py --low --label J

# Modo raw — guarda el frame completo sin etiquetar en dataset/raw/
python capture.py --raw
```

**Controles en la ventana:**
- **ESPACIO** — capturar los crops de todos los dados detectados
- **f** — guardar el frame completo como referencia
- **q** o **ESC** — salir

Los crops se guardan en `dataset/<cara>/<cara>_<timestamp>.png`.

### 2. Entrenar la CNN

```bash
# Entrena 50 epochs, evalúa, imprime confusion matrix y exporta a ONNX
python train.py
```

Al terminar genera:
- `best_model.pth` — checkpoint PyTorch del mejor epoch por val accuracy
- `dice_classifier.onnx` — modelo exportado para inferencia

El script imprime por epoch el loss y accuracy de train/val, y al final una confusion matrix 6×6 con precision/recall/F1 por clase.

### 3. Identificar jugadas en tiempo real

```bash
# Conecta al robot, detecta dados, clasifica caras y evalúa la jugada en vivo
python poker.py
```

En la ventana veo:
- **Bounding box verde** + predicción (`K 85%`) sobre cada dado reconocido
- **Bounding box rojo** + `? XX%` sobre dados por debajo del threshold de confianza
- **HUD superior**: número de dados detectados y caras clasificadas
- **HUD inferior**: la jugada evaluada (Par, Trío, Full House, ...)

Sale con **q** o **ESC**.

### 4. Pick-and-place por comandos (sistema completo)

```bash
# Arranca el sistema de pick-and-place con comandos de terminal
python main.py
```

Comandos disponibles en la terminal:

```
help                      # Lista de comandos
scan                      # Mover a posición de escaneo
home                      # Mover a posición home
open                      # Abrir pinza
clear                     # Resetear estado de colisión
color any|red|green|blue  # Filtrar por color
shape any|circle|square   # Filtrar por forma
select N                  # Seleccionar pieza N de las visibles
pick                      # Ejecutar pick-and-place sobre la seleccionada
status                    # Imprimir estado actual
exit                      # Salir
```

---

## Flujo de trabajo típico

Cuando recibo un nuevo set de dados o quiero re-entrenar el modelo:

```bash
# 1) Capturar imágenes con la cámara baja para mejor resolución
python capture.py --low --label 9     # ~20 imágenes del 9
python capture.py --low --label 10    # ~20 imágenes del 10
python capture.py --low --label J     # ~20 imágenes de la J
python capture.py --low --label Q     # ~20 imágenes de la Q
python capture.py --low --label K     # ~20 imágenes de la K
python capture.py --low --label A     # ~20 imágenes de la A

# 2) Entrenar la CNN
python train.py

# 3) Probar clasificación en vivo
python poker.py
```

Si quiero añadir más imágenes después, simplemente vuelvo a ejecutar `capture.py` y `train.py` — el script carga todas las PNG existentes en `dataset/<cara>/`.

---

## Caso de uso

**Niryo Poker Dice Vision** es un sistema autónomo que simula una mesa de póker con dados especiales. El flujo completo es:

1. **El robot se posiciona** sobre el workspace en la pose de escaneo y captura una imagen desde su cámara integrada.
2. **La visión detecta el workspace** mediante las 4 dianas de las esquinas y construye la homografía para mapear píxeles a coordenadas físicas del robot.
3. **Se localizan los dados** dentro del workspace usando detección de bordes Canny + morfología, filtrando cualquier contorno que coincida con las dianas.
4. **La CNN clasifica cada dado** en una de seis caras (`9, 10, J, Q, K, A`). El modelo corre via ONNX Runtime con una latencia de pocos milisegundos por crop.
5. **El evaluador de jugadas** analiza la lista de caras detectadas y determina la mejor combinación según la jerarquía del póker dice: Póker (5 iguales), Full House (3+2), Escalera (5 consecutivas), Trío, Doble Par, Par o Nada.
6. **El robot ejecuta el pick-and-place** de los dados que componen la jugada: mueve el brazo a la posición calculada, desciende, cierra la pinza, levanta, y deposita cada pieza en la zona correspondiente según la categoría de la jugada detectada.

El sistema integra todo el pipeline visto en el curso: segmentación y detección clásica con OpenCV (Canny, HoughCircles, homografías), clasificación con redes neuronales convolucionales entrenadas desde cero en PyTorch, despliegue eficiente del modelo via ONNX Runtime, y control robótico en tiempo real mediante `pyniryo`. Es un ejemplo completo de cómo combinar visión clásica con deep learning para resolver una tarea de manipulación robótica con semántica compleja.
