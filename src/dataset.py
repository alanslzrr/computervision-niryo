"""
Utilidades para captura y almacenamiento de imagenes de entrenamiento.

Permite capturar crops de dados desde la camara del robot, etiquetarlos
manualmente y guardarlos organizados por clase para entrenar la CNN.

TODO: Integrar como comando 'dataset' en ui.process_command para
      activar el modo de captura interactivo desde la terminal del sistema.
"""

import os
import time
from typing import List

import cv2
import numpy as np

from config import DATASET_DIR, DICE_FACES


def ensure_dataset_dirs(dataset_dir: str = DATASET_DIR) -> None:
    """Crea la estructura de directorios para el dataset: una carpeta por cada cara de dado.

    Estructura resultante:
        dataset/
            9/
            10/
            J/
            Q/
            K/
            A/
    """
    for face in DICE_FACES:
        path = os.path.join(dataset_dir, face)
        os.makedirs(path, exist_ok=True)
    print(f"[DATASET] Directorios creados en {dataset_dir}")


def save_crop(crop: np.ndarray, label: str, dataset_dir: str = DATASET_DIR) -> str:
    """Guarda un crop etiquetado como PNG en dataset_dir/<label>/.

    Args:
        crop: Imagen BGR del dado recortado.
        label: Cara del dado (debe estar en DICE_FACES).
        dataset_dir: Directorio raiz del dataset.

    Returns:
        Ruta completa del archivo guardado.

    Raises:
        ValueError: Si el label no es una cara valida.
    """
    if label not in DICE_FACES:
        raise ValueError(f"Label '{label}' no valido. Opciones: {DICE_FACES}")

    folder = os.path.join(dataset_dir, label)
    os.makedirs(folder, exist_ok=True)

    timestamp = int(time.time() * 1000)
    filename = f"{label}_{timestamp}.png"
    filepath = os.path.join(folder, filename)

    cv2.imwrite(filepath, crop)
    return filepath


def get_dataset_summary(dataset_dir: str = DATASET_DIR) -> dict:
    """Retorna un resumen del numero de imagenes por clase en el dataset.

    TODO: Usar para verificar balance de clases antes de entrenar.
    """
    summary = {}
    for face in DICE_FACES:
        folder = os.path.join(dataset_dir, face)
        if os.path.isdir(folder):
            count = len([f for f in os.listdir(folder) if f.endswith(".png")])
        else:
            count = 0
        summary[face] = count
    return summary


def capture_dataset_mode(picker, detect_fn, workspace_corners: np.ndarray) -> None:
    """Modo interactivo de captura de dataset.

    Captura frames, detecta objetos, muestra cada crop y pide al usuario
    que lo etiquete con la cara correspondiente. Guarda los crops etiquetados.

    Args:
        picker: Instancia de NiryoVisionPicker (para capture_frame).
        detect_fn: Funcion de deteccion (vision.detect_objects).
        workspace_corners: Esquinas del workspace para la deteccion.

    TODO:
        - Implementar el loop interactivo completo
        - Mostrar ventana con crop ampliado para facilitar etiquetado
        - Opcion de descartar crops malos (tecla 's' para skip)
        - Mostrar resumen de imagenes capturadas al salir
    """
    ensure_dataset_dirs()

    print("\n[DATASET] Modo captura activado")
    print(f"  Caras validas: {DICE_FACES}")
    print("  Escribe la cara del dado para etiquetar, 's' para skip, 'q' para salir")

    frame = picker.capture_frame()
    objects = detect_fn(frame, workspace_corners)

    if not objects:
        print("[DATASET] No se detectaron objetos. Asegurate de que hay dados en el workspace.")
        return

    for i, obj in enumerate(objects, start=1):
        x, y, w, h = obj["bounding_rect"]
        crop = frame[y:y+h, x:x+w]

        cv2.imshow("Crop para etiquetar", crop)
        cv2.waitKey(1)

        label = input(f"  [{i}/{len(objects)}] Etiqueta para este dado (cara): ").strip().upper()

        if label == "Q":
            print("[DATASET] Saliendo del modo captura")
            break
        if label == "S":
            print("  Saltado")
            continue
        if label not in DICE_FACES:
            print(f"  Label '{label}' no valido, saltando")
            continue

        filepath = save_crop(crop, label)
        print(f"  Guardado: {filepath}")

    cv2.destroyWindow("Crop para etiquetar")

    summary = get_dataset_summary()
    print("\n[DATASET] Resumen actual:")
    for face, count in summary.items():
        print(f"  {face}: {count} imagenes")
