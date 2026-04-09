"""
Herramientas para crear y revisar el dataset de imágenes de dados (entrenamiento CNN).

Proporciona creación de árbol de carpetas por clase, guardado de recortes PNG y un modo
interactivo opcional para etiquetar detecciones. El flujo principal de captura en laboratorio
está implementado de forma más completa en ``capture.py``; este módulo puede reutilizarse para
scripts auxiliares o integración futura con ``ui.process_command``.
"""

import os
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from config import DATASET_DIR, DICE_FACES

if TYPE_CHECKING:
    from robot import NiryoVisionPicker


def ensure_dataset_dirs(dataset_dir: str = DATASET_DIR) -> None:
    """Crea bajo ``dataset_dir`` una subcarpeta por cada etiqueta en ``DICE_FACES``.

    Args:
        dataset_dir: Directorio raíz del dataset (por defecto ``config.DATASET_DIR``).
    """
    for face in DICE_FACES:
        path = os.path.join(dataset_dir, face)
        os.makedirs(path, exist_ok=True)
    print(f"[DATASET] Directorios creados en {dataset_dir}")


def save_crop(crop: np.ndarray, label: str, dataset_dir: str = DATASET_DIR) -> str:
    """Guarda un recorte etiquetado como PNG en ``dataset_dir/<label>/``.

    Args:
        crop: Imagen BGR del dado.
        label: Una de las caras en ``DICE_FACES``.
        dataset_dir: Raíz del dataset.

    Returns:
        Ruta absoluta o relativa del archivo escrito.

    Raises:
        ValueError: Si ``label`` no es una clase válida.
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
    """Cuenta imágenes ``.png`` por clase para comprobar balance antes de entrenar.

    Returns:
        Diccionario ``cara -> número de archivos``.
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


def capture_dataset_mode(picker: "NiryoVisionPicker", detect_fn, workspace_corners: np.ndarray) -> None:
    """Modo mínimo de captura por consola sobre una sola captura de frame.

    Detecta objetos con ``detect_fn`` (típicamente ``vision.detect_objects``), muestra cada
    recorte y pide la etiqueta por teclado. Para salir se usa la letra ``Q`` mayúscula; tener en
    cuenta la colisión con la cara ``Q`` (reina) — en ese caso conviene usar ``capture.py``.

    Args:
        picker: Instancia conectada con ``capture_frame``.
        detect_fn: Función ``(frame, workspace_corners) -> lista de objetos``.
        workspace_corners: Matriz de esquinas del workspace en píxeles.

    Note:
        Flujo simplificado; para captura continua y teclas en ventana preferir ``capture.run_capture``.
    """
    ensure_dataset_dirs()

    print("\n[DATASET] Modo captura activado")
    print(f"  Caras validas: {DICE_FACES}")
    print("  Etiqueta por dado; 'S' skip; 'Q' salir (mayúscula)")

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
