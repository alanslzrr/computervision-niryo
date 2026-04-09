"""
Captura de datos para entrenar la CNN de caras de dado (cámara del Niryo).

El robot adopta la pose de escaneo o, con ``--low``, una pose más baja para mayor resolución
en los recortes. ``extract_dice_crops`` segmenta dados por bordes (Canny + morfología) dentro
del polígono del workspace, excluyendo zonas cercanas a las dianas.

Modos de línea de comandos::

    python capture.py                  # Etiqueta interactiva por crop (terminal)
    python capture.py --raw            # Frames completos en dataset/raw/
    python capture.py --label J        # Todos los crops con la misma cara
    python capture.py --low            # Z=0.18 m para crops más grandes
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np

from config import DATASET_DIR, DICE_FACES, ROBOT_IP, SCANNING_POSITION
from robot import NiryoVisionPicker
from vision import detect_workspace_from_dianas, fallback_workspace_corners


RAW_DIR = os.path.join(DATASET_DIR, "raw")
CROP_PADDING = 15  # pixeles de margen extra alrededor del dado

# Limites de area para dados (mucho mas pequenos que piezas del lab anterior)
DICE_MIN_AREA = 400
DICE_MAX_AREA = 8000

# Pose de captura: misma XY que scan pero mas baja para mejor resolucion.
# A Z=0.18 el dado ocupa ~80-100px en vez de ~40px.
# Las dianas pueden salir del frame, pero no se necesitan para capturar fotos.
CAPTURE_POSITION = (
    SCANNING_POSITION[0],   # x
    SCANNING_POSITION[1],   # y
    0.18,                   # z — mas bajo que scan (0.23)
    SCANNING_POSITION[3],   # roll
    SCANNING_POSITION[4],   # pitch
    SCANNING_POSITION[5],   # yaw
)


def setup_dirs(label: str = None) -> str:
    """Crea si no existe el directorio de destino (clase o ``raw``) y devuelve su ruta."""
    if label:
        target = os.path.join(DATASET_DIR, label)
    else:
        target = RAW_DIR
    os.makedirs(target, exist_ok=True)
    return target


def generate_filename(prefix: str = "capture") -> str:
    """Devuelve ``{prefix}_{timestamp_ms}.png`` para nombres únicos."""
    ts = int(time.time() * 1000)
    return f"{prefix}_{ts}.png"


def extract_dice_crops(frame: np.ndarray, workspace_corners: np.ndarray) -> List[Tuple[np.ndarray, tuple]]:
    """Segmenta dados por bordes dentro del workspace y devuelve recortes con padding.

    Estrategia: máscara poligonal del workspace (erosionada para ignorar el marco), Canny sobre
    gris suavizado, morfología para cerrar contornos, filtrado por ``DICE_MIN_AREA`` /
    ``DICE_MAX_AREA`` y exclusión de regiones cerca de las esquinas (dianas).

    Returns:
        Lista de ``(crop_bgr, (x, y, w, h))`` con ``w,h`` del rectángulo ampliado por ``CROP_PADDING``.
    """
    h_frame, w_frame = frame.shape[:2]

    # 1) Mascara del interior del workspace (excluye marco oscuro)
    ws_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
    ws_poly = workspace_corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(ws_mask, [ws_poly], 255)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ws_mask = cv2.erode(ws_mask, erode_kernel, iterations=1)

    # 2) Canny sobre gris enmascarado — detecta bordes independientemente del brillo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, 30, 100)

    # Aplicar mascara del workspace
    edges = cv2.bitwise_and(edges, ws_mask)

    # 3) Dilatar + cerrar para unir bordes en contornos solidos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops = []

    # Radio de exclusion alrededor de cada diana (pixeles)
    DIANA_EXCLUSION_RADIUS = 50

    for contour in contours:
        area = cv2.contourArea(contour)
        if not (DICE_MIN_AREA <= area <= DICE_MAX_AREA):
            continue

        # Centroide del contorno
        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        # Descartar si esta cerca de alguna diana
        is_diana = False
        for corner in workspace_corners:
            dx = cx - corner[0]
            dy = cy - corner[1]
            if (dx * dx + dy * dy) ** 0.5 < DIANA_EXCLUSION_RADIUS:
                is_diana = True
                break
        if is_diana:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Crop con padding
        x1 = max(0, x - CROP_PADDING)
        y1 = max(0, y - CROP_PADDING)
        x2 = min(w_frame, x + w + CROP_PADDING)
        y2 = min(h_frame, y + h + CROP_PADDING)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crops.append((crop, (x1, y1, x2 - x1, y2 - y1)))

    return crops


def draw_detections(display: np.ndarray, crops: List[Tuple[np.ndarray, tuple]]) -> None:
    """Superpone rectángulos verdes y numeración 1..N sobre ``display``."""
    for i, (_, (x, y, w, h)) in enumerate(crops, start=1):
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display, str(i), (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def run_capture(raw_mode: bool = False, fixed_label: str = None, low: bool = False) -> None:
    """Bucle principal: vista en vivo, detección de dados y guardado según modo.

    Args:
        raw_mode: Si True, ``ESPACIO`` guarda el frame completo en ``dataset/raw/``.
        fixed_label: Si se indica (p. ej. ``"J"``), todos los crops van a esa carpeta de clase.
        low: Si True, detecta workspace en pose de scan y baja a ``CAPTURE_POSITION`` para capturar.

    Teclas en la ventana ``Captura Niryo``: espacio = guardar según modo; ``f`` = frame de referencia;
    ``q``/ESC = salir. Al terminar se imprime un resumen por carpeta de cara.
    """
    picker = NiryoVisionPicker(ROBOT_IP)
    workspace_corners = None

    try:
        picker.connect()

        if low:
            # Primero ir a scan normal para detectar dianas del workspace
            picker.move_scan()
            frame_for_ws = picker.capture_frame()
            ws_detected = detect_workspace_from_dianas(frame_for_ws)
            if ws_detected is not None:
                workspace_corners = ws_detected
                print("[CAPTURE] Dianas detectadas desde scan, guardando para referencia")
            else:
                workspace_corners = fallback_workspace_corners()
                print("[CAPTURE] Dianas no detectadas, usando fallback")

            # Ahora bajar a la pose de captura (mas cerca, mejor resolucion)
            from pyniryo import PoseObject
            picker.robot.move(PoseObject(*CAPTURE_POSITION))
            print(f"[CAPTURE] Pose baja Z={CAPTURE_POSITION[2]}m (mejor resolucion)")
        else:
            picker.move_scan()

        # Asegurar pinza abierta para no obstruir la vista
        try:
            picker.robot.release_with_tool()
        except Exception:
            pass

        cv2.namedWindow("Captura Niryo", cv2.WINDOW_NORMAL)

        count = 0
        print("\n=== Modo captura ===")
        print("  ESPACIO  -> capturar crops de dados detectados")
        print("  f        -> guardar frame completo (referencia)")
        print("  q / ESC  -> salir")
        if raw_mode:
            setup_dirs()
            print(f"  Modo raw: frames completos en {RAW_DIR}")
        elif fixed_label:
            if fixed_label not in DICE_FACES:
                print(f"  [WARN] Label '{fixed_label}' no esta en DICE_FACES {DICE_FACES}")
            setup_dirs(fixed_label)
            print(f"  Label fijo: todos los crops como '{fixed_label}'")
        else:
            for face in DICE_FACES:
                setup_dirs(face)
            print(f"  Caras validas: {DICE_FACES}")
            print("  Tras capturar se pide etiqueta por terminal para cada crop")
        print()

        while True:
            frame = picker.capture_frame()

            # Detectar workspace
            detected = detect_workspace_from_dianas(frame)
            if detected is not None:
                if workspace_corners is None:
                    workspace_corners = detected
                else:
                    workspace_corners = 0.75 * workspace_corners + 0.25 * detected

            if workspace_corners is None:
                workspace_corners = fallback_workspace_corners()

            ws = workspace_corners.astype(np.float32)

            # Detectar dados
            crops = extract_dice_crops(frame, ws)

            # Display con detecciones
            display = frame.copy()
            draw_detections(display, crops)
            ws_status = "WS: dianas" if detected is not None else "WS: fallback"
            cv2.putText(display, f"Dados: {len(crops)} | Fotos: {count} | {ws_status} | ESPACIO=capturar | q=salir",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.imshow("Captura Niryo", display)
            key = cv2.waitKey(100) & 0xFF

            if key == ord("q") or key == 27:
                break

            # 'f' = guardar frame completo como referencia
            if key == ord("f"):
                ref_dir = setup_dirs("_frames")
                filepath = os.path.join(ref_dir, generate_filename("frame"))
                cv2.imwrite(filepath, frame)
                print(f"  Frame completo guardado: {filepath}")

            # ESPACIO = capturar crops
            if key == ord(" "):
                if not crops:
                    print("  No se detectaron dados en el workspace")
                    continue

                if raw_mode:
                    # Modo raw: guardar frame completo
                    filepath = os.path.join(RAW_DIR, generate_filename())
                    cv2.imwrite(filepath, frame)
                    count += 1
                    print(f"  [{count}] Frame guardado: {filepath}")

                elif fixed_label:
                    # Guardar todos los crops con la misma etiqueta
                    target_dir = setup_dirs(fixed_label)
                    for i, (crop, _) in enumerate(crops, start=1):
                        filepath = os.path.join(target_dir, generate_filename(fixed_label))
                        cv2.imwrite(filepath, crop)
                        count += 1
                        print(f"  [{count}] Crop {i} -> {fixed_label}: {filepath}")

                else:
                    # Modo interactivo: etiquetar cada crop
                    for i, (crop, (x, y, w, h)) in enumerate(crops, start=1):
                        # Mostrar crop ampliado
                        crop_big = cv2.resize(crop, (200, 200), interpolation=cv2.INTER_NEAREST)
                        cv2.imshow("Crop", crop_big)
                        cv2.waitKey(1)

                        label = input(
                            f"  Crop {i}/{len(crops)} ({w}x{h}px) "
                            f"etiqueta ({'/'.join(DICE_FACES)}, s=skip): "
                        ).strip().upper()

                        if label == "S":
                            print("    Saltado")
                            continue
                        if label not in DICE_FACES:
                            print(f"    '{label}' no valida, saltando")
                            continue

                        target_dir = setup_dirs(label)
                        filepath = os.path.join(target_dir, generate_filename(label))
                        cv2.imwrite(filepath, crop)
                        count += 1
                        print(f"    [{count}] Guardado: {filepath}")

                    try:
                        cv2.destroyWindow("Crop")
                    except Exception:
                        pass

        # Resumen final
        print(f"\nTotal crops guardados: {count}")
        print("Resumen del dataset:")
        for face in DICE_FACES:
            folder = os.path.join(DATASET_DIR, face)
            if os.path.isdir(folder):
                n = len([f for f in os.listdir(folder) if f.endswith(".png")])
                if n > 0:
                    print(f"  {face}: {n} imagenes")

    except KeyboardInterrupt:
        print("\nInterrumpido")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        picker.safe_shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Captura de fotos desde el robot Niryo")
    parser.add_argument("--raw", action="store_true",
                        help="Guardar frames completos sin etiquetar en dataset/raw/")
    parser.add_argument("--label", type=str, default=None,
                        help="Etiqueta fija para todos los crops (ej: --label J)")
    parser.add_argument("--low", action="store_true",
                        help="Bajar camara a Z=0.18m para mejor resolucion de crops")
    args = parser.parse_args()

    run_capture(
        raw_mode=args.raw,
        fixed_label=args.label.upper() if args.label else None,
        low=args.low,
    )
