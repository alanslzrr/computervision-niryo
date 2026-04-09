"""
Niryo Poker Dice Vision — Identificacion de jugadas en tiempo real.

Conecta al robot en posicion de escaneo, detecta dados en el workspace,
clasifica cada cara con la CNN (ONNX) y evalua la jugada de poker.

Uso:
    python poker.py
"""

import sys

import cv2
import numpy as np

from capture import extract_dice_crops
from classifier import DiceClassifier
from config import ROBOT_IP
from evaluator import evaluate_hand
from robot import NiryoVisionPicker
from vision import detect_workspace_from_dianas, fallback_workspace_corners


# Colores BGR para el overlay
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)


def run() -> None:
    picker = NiryoVisionPicker(ROBOT_IP)
    classifier = DiceClassifier()
    workspace_corners = None

    if classifier.session is None:
        print("[WARN] No se encontro dice_classifier.onnx — ejecuta train.py primero")
        print("       Los dados se detectaran pero no se clasificaran")

    try:
        picker.connect()
        picker.move_scan()

        try:
            picker.robot.release_with_tool()
        except Exception:
            pass

        cv2.namedWindow("Poker Dice Vision", cv2.WINDOW_NORMAL)

        print("\n=== Poker Dice Vision ===")
        print("  q / ESC  -> salir")
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

            # Clasificar cada dado
            results = []
            for crop, bbox in crops:
                face, conf = classifier.classify(crop)
                results.append((face, conf, bbox))

            # Evaluar jugada con las caras detectadas
            detected_faces = [face for face, conf, _ in results if face != "UNKNOWN"]
            hand_name, hand_rank, hand_desc = evaluate_hand(detected_faces)

            # --- Render ---
            display = frame.copy()

            # Dibujar workspace
            ws_i = ws.astype(int)
            cv2.polylines(display, [ws_i.reshape((-1, 1, 2))], True, YELLOW, 1)

            # Dibujar cada dado con su clasificacion
            for i, (face, conf, (x, y, w, h)) in enumerate(results, start=1):
                if face != "UNKNOWN":
                    color = GREEN
                    label = f"{face} {conf:.0%}"
                else:
                    color = RED
                    label = f"? {conf:.0%}"

                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display, label, (x, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # HUD superior
            ws_status = "dianas" if detected is not None else "fallback"
            cv2.putText(display, f"Dados: {len(crops)} | Caras: {len(detected_faces)} | WS: {ws_status}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

            # Jugada (parte inferior)
            if detected_faces:
                hand_color = ORANGE if hand_rank >= 4 else GREEN if hand_rank >= 2 else WHITE
                cv2.putText(display, f"JUGADA: {hand_desc}",
                            (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

            cv2.imshow("Poker Dice Vision", display)
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\nInterrumpido")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        picker.safe_shutdown()


if __name__ == "__main__":
    run()
