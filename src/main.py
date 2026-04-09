"""
Sistema de vision para robot Niryo: deteccion, seleccion y pick-and-place de piezas.

OBJETIVO:
    Implementar un sistema de vision robotica que permita detectar piezas (circulares
    y cuadradas) por color (rojo, verde, azul) sobre un workspace calibrado, y
    recogerlas de forma automatizada mediante el robot Niryo One.

METODOLOGIA:
    - Calibracion del workspace mediante deteccion de 4 dianas con HoughCircles.
    - Segmentacion por color en HSV y deteccion de contornos para localizar piezas.
    - Clasificacion por forma (circularidad y numero de vertices) y por color (promedio HSV).
    - Transformacion perspectiva pixel -> coordenadas relativas -> coordenadas Cartesianas del robot.
    - Control interactivo por terminal con filtros y seleccion manual de la pieza a recoger.


"""

import queue
import sys
import threading

import cv2
import numpy as np

from config import ROBOT_IP
from robot import NiryoVisionPicker
from ui import (
    apply_filters,
    draw_hud,
    draw_objects,
    draw_workspace_overlay,
    input_worker,
    print_terminal_help,
    process_command,
)
from vision import detect_objects, detect_workspace_from_dianas, fallback_workspace_corners


def run() -> None:
    """Ejecuta el bucle principal del laboratorio de piezas con forma/color.

    Conecta al Niryo, posiciona en escaneo, lanza un hilo de lectura de comandos por terminal y
    muestra ``Niryo Vision Menu`` con overlay del workspace, detecciones y HUD. Los comandos
    delegan en ``ui.process_command`` (pick, filtros, salida, etc.).
    """
    picker = NiryoVisionPicker(ROBOT_IP)
    cmd_queue: "queue.Queue[str]" = queue.Queue()

    input_thread = threading.Thread(target=input_worker, args=(cmd_queue,), daemon=True)

    try:
        picker.connect()
        picker.move_scan()

        try:
            picker.robot.release_with_tool()
        except Exception:
            pass

        print_terminal_help()
        input_thread.start()

        cv2.namedWindow("Niryo Vision Menu", cv2.WINDOW_NORMAL)

        running = True
        while running:
            frame = picker.capture_frame()

            # 1) Detectar workspace por dianas
            detected = detect_workspace_from_dianas(frame)
            detected_live = detected is not None

            if detected is not None:
                if picker.workspace_corners is None:
                    picker.workspace_corners = detected
                else:
                    # Suavizado para evitar jitter de deteccion
                    picker.workspace_corners = 0.75 * picker.workspace_corners + 0.25 * detected

            if picker.workspace_corners is None:
                picker.workspace_corners = fallback_workspace_corners()

            workspace_corners = picker.workspace_corners.astype(np.float32)

            # 2) Detectar piezas dentro del workspace
            all_objects = detect_objects(frame, workspace_corners)
            visible_objects = apply_filters(all_objects, picker.filter_color, picker.filter_shape)

            if picker.selected_index is not None and (
                picker.selected_index < 1 or picker.selected_index > len(visible_objects)
            ):
                picker.selected_index = None

            # 3) Render
            draw_workspace_overlay(frame, workspace_corners, detected_live)
            draw_objects(frame, visible_objects, picker.selected_index)
            draw_hud(frame, picker, visible_objects)

            cv2.imshow("Niryo Vision Menu", frame)
            cv2.waitKey(1)

            # 4) Procesar comandos de terminal
            while not cmd_queue.empty():
                cmd_line = cmd_queue.get_nowait()
                running = process_command(cmd_line, picker, visible_objects, workspace_corners)
                if not running:
                    break

    except KeyboardInterrupt:
        print("Interrumpido por teclado")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        picker.safe_shutdown()


if __name__ == "__main__":
    run()
