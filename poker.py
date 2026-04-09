"""
Niryo Poker Dice Vision — Identificacion de jugadas y pick-and-place de dados.

Conecta al robot en posicion de escaneo, detecta dados en el workspace,
clasifica cada cara con la CNN (ONNX), evalua la jugada de poker y permite
recoger los dados por comandos de terminal (mismo patron que lab1).

Uso:
    python poker.py

Comandos:
    help | scan | home | open | clear | select N | pick | pick all | status | exit
"""

import queue
import sys
import threading
import time
from typing import List, Tuple

import cv2
import numpy as np
from pyniryo import PoseObject

from capture import extract_dice_crops
from classifier import DiceClassifier
from config import APPROACH_Z, PICK_Z, ROBOT_IP
from evaluator import evaluate_hand
from robot import NiryoVisionPicker, relative_to_robot_xy
from vision import detect_workspace_from_dianas, fallback_workspace_corners, pixel_to_relative


# Colores BGR para el overlay
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)


# ============================================================================
# PICK-AND-PLACE DE DADOS
# ============================================================================

def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Devuelve el centro (cx, cy) de un bounding rect (x, y, w, h)."""
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


def pick_dice(
    picker: NiryoVisionPicker,
    classifier: DiceClassifier,
    target_face: str,
    target_cx: int,
    target_cy: int,
    workspace_corners: np.ndarray,
) -> None:
    """Recoge el dado indicado por (target_face, target_cx, target_cy).

    Antes de mover el brazo se vuelve a la pose de scan, se captura un frame
    fresco, se re-detectan las dianas, se re-extraen y re-clasifican los dados,
    y se elige el dado con la cara objetivo mas cercano al centroide original.
    Luego ejecuta approach -> pick -> grasp -> approach -> home -> release -> scan.
    """
    print("[PICK] Volviendo a scan y re-capturando frame...")
    picker.move_scan()
    fresh_frame = picker.capture_frame()

    # Re-detectar workspace
    fresh_ws = detect_workspace_from_dianas(fresh_frame)
    if fresh_ws is not None:
        ws = fresh_ws
        print("[PICK] Workspace re-detectado desde dianas frescas")
    else:
        ws = workspace_corners
        print("[PICK] Dianas no detectadas, usando workspace anterior")

    ws_f = ws.astype(np.float32)

    # Re-detectar y re-clasificar dados
    crops = extract_dice_crops(fresh_frame, ws_f)

    best_match = None
    best_dist = float("inf")
    for crop, bbox in crops:
        face, conf = classifier.classify(crop)
        if face != target_face:
            continue
        cx, cy = bbox_center(bbox)
        dx = cx - target_cx
        dy = cy - target_cy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_match = (face, conf, cx, cy)

    if best_match is None:
        print(f"[PICK] ERROR: No se encontro dado '{target_face}' en re-deteccion. Abortando.")
        return

    _, conf, cx, cy = best_match
    x_rel, y_rel = pixel_to_relative(cx, cy, ws_f)
    x, y = relative_to_robot_xy(x_rel, y_rel)

    approach_pose = PoseObject(x, y, APPROACH_Z, picker.pick_roll, picker.pick_pitch, picker.pick_yaw)
    pick_pose = PoseObject(x, y, PICK_Z, picker.pick_roll, picker.pick_pitch, picker.pick_yaw)

    print(
        f"[PICK] Recogiendo dado '{target_face}' (conf {conf:.0%})\n"
        f"       pixel original=({target_cx},{target_cy}) -> fresco=({cx},{cy}) delta={best_dist:.1f}px\n"
        f"       rel=({x_rel:.3f},{y_rel:.3f}) robot_xy=({x:.4f},{y:.4f},{PICK_Z:.4f})"
    )

    picker.robot.move(approach_pose)
    picker.robot.move(pick_pose)
    picker.robot.grasp_with_tool()
    time.sleep(0.2)
    picker.robot.move(approach_pose)

    picker.move_home()
    picker.robot.release_with_tool()
    picker.move_scan()


# ============================================================================
# TERMINAL COMMANDS
# ============================================================================

def print_help() -> None:
    print("\nComandos disponibles:")
    print("  help")
    print("  scan       - Mover a posicion de escaneo")
    print("  home       - Mover a home oficial")
    print("  open       - Abrir pinza")
    print("  clear      - Limpiar flag de colision (retirar obstaculo antes)")
    print("  select N   - Seleccionar dado N de los visibles")
    print("  pick       - Pick-and-place del dado seleccionado")
    print("  pick all   - Recoger todos los dados detectados (uno a uno)")
    print("  status     - Estado actual")
    print("  exit       - Salir")


def input_worker(cmd_queue: "queue.Queue[str]") -> None:
    """Hilo que lee comandos de la terminal y los pone en la cola para el hilo principal."""
    while True:
        try:
            line = input("cmd> ").strip()
        except EOFError:
            break
        cmd_queue.put(line)


def process_command(
    cmd_line: str,
    picker: NiryoVisionPicker,
    classifier: DiceClassifier,
    current_dice: List[Tuple[str, float, tuple]],
    workspace_corners: np.ndarray,
) -> bool:
    """Procesa un comando de terminal. Retorna False si se debe salir del bucle principal."""
    if not cmd_line:
        return True

    parts = cmd_line.strip().split()
    cmd = parts[0].lower()

    if cmd == "help":
        print_help()
        return True

    if cmd == "exit":
        return False

    if cmd == "scan":
        print("[CMD] Ir a SCANNING_POSITION")
        picker.move_scan()
        return True

    if cmd == "home":
        print("[CMD] Ir a HOME oficial")
        picker.move_home()
        return True

    if cmd == "open":
        print("[CMD] Abrir pinza")
        picker.robot.release_with_tool()
        return True

    if cmd == "clear":
        print("[CMD] Limpiando colision (retira el obstaculo antes)")
        try:
            picker.clear_collision()
        except Exception as e:
            print(f"[ERROR] Fallo al limpiar colision: {e}")
        return True

    if cmd == "status":
        labels = ", ".join(f"{i}:{face}" for i, (face, _, _) in enumerate(current_dice, start=1))
        print(f"Status -> visibles={len(current_dice)} [{labels}], sel={picker.selected_index}")
        return True

    if cmd == "select" and len(parts) == 2:
        try:
            idx = int(parts[1])
        except ValueError:
            print("Uso: select N")
            return True
        if 1 <= idx <= len(current_dice):
            picker.selected_index = idx
            face, conf, _ = current_dice[idx - 1]
            print(f"Seleccionado dado {idx}: {face} ({conf:.0%})")
        else:
            print(f"Indice fuera de rango. Visibles: {len(current_dice)}")
        return True

    if cmd == "pick":
        # pick all
        if len(parts) == 2 and parts[1].lower() == "all":
            valid = [(face, bbox_center(bbox)) for face, conf, bbox in current_dice if face != "UNKNOWN"]
            if not valid:
                print("No hay dados clasificados para recoger")
                return True
            print(f"\n[PICK ALL] Recogiendo {len(valid)} dado(s)...")
            for i, (face, (cx, cy)) in enumerate(valid, start=1):
                print(f"\n[PICK ALL] {i}/{len(valid)}: dado '{face}' en ({cx},{cy})")
                try:
                    pick_dice(picker, classifier, face, cx, cy, workspace_corners)
                except Exception as e:
                    print(f"[ERROR] Fallo en pick {i}: {e}")
                    print("  Si hubo colision: retira el obstaculo y ejecuta 'clear'")
                    try:
                        picker.move_scan()
                    except Exception:
                        pass
                    break
            picker.selected_index = None
            return True

        # pick simple
        if picker.selected_index is None:
            print("Primero usa: select N")
            return True
        if not (1 <= picker.selected_index <= len(current_dice)):
            print("La seleccion ya no es valida. Vuelve a seleccionar.")
            picker.selected_index = None
            return True

        face, conf, bbox = current_dice[picker.selected_index - 1]
        if face == "UNKNOWN":
            print("Dado seleccionado clasificado como UNKNOWN — no se puede recoger")
            return True
        cx, cy = bbox_center(bbox)
        try:
            pick_dice(picker, classifier, face, cx, cy, workspace_corners)
        except Exception as e:
            print(f"[ERROR] Fallo en pick/place: {e}")
            print("  Si hubo colision: retira el obstaculo y ejecuta 'clear'")
            try:
                picker.move_scan()
            except Exception:
                pass
        picker.selected_index = None
        return True

    print("Comando no reconocido. Usa: help")
    return True


# ============================================================================
# MAIN LOOP
# ============================================================================

def run() -> None:
    picker = NiryoVisionPicker(ROBOT_IP)
    classifier = DiceClassifier()
    workspace_corners = None
    cmd_queue: "queue.Queue[str]" = queue.Queue()

    if classifier.session is None:
        print("[WARN] No se encontro dice_classifier.onnx — ejecuta train.py primero")
        print("       Los dados se detectaran pero no se clasificaran ni recogeran")

    input_thread = threading.Thread(target=input_worker, args=(cmd_queue,), daemon=True)

    try:
        picker.connect()
        picker.move_scan()

        try:
            picker.robot.release_with_tool()
        except Exception:
            pass

        cv2.namedWindow("Poker Dice Vision", cv2.WINDOW_NORMAL)

        print("\n=== Poker Dice Vision ===")
        print_help()
        print("\n  Ventana OpenCV: q / ESC -> salir")
        input_thread.start()

        running = True
        while running:
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

            # Detectar y clasificar dados
            crops = extract_dice_crops(frame, ws)
            results: List[Tuple[str, float, tuple]] = []
            for crop, bbox in crops:
                face, conf = classifier.classify(crop)
                results.append((face, conf, bbox))

            # Validar seleccion contra el frame actual
            if picker.selected_index is not None and (
                picker.selected_index < 1 or picker.selected_index > len(results)
            ):
                picker.selected_index = None

            # Evaluar jugada
            detected_faces = [face for face, conf, _ in results if face != "UNKNOWN"]
            hand_name, hand_rank, hand_desc = evaluate_hand(detected_faces)

            # --- Render ---
            display = frame.copy()

            ws_i = ws.astype(int)
            cv2.polylines(display, [ws_i.reshape((-1, 1, 2))], True, YELLOW, 1)

            for i, (face, conf, (x, y, w, h)) in enumerate(results, start=1):
                if picker.selected_index == i:
                    color = ORANGE
                elif face != "UNKNOWN":
                    color = GREEN
                else:
                    color = RED

                if face != "UNKNOWN":
                    label = f"{i}:{face} {conf:.0%}"
                else:
                    label = f"{i}:? {conf:.0%}"

                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display, label, (x, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # HUD superior
            ws_status = "dianas" if detected is not None else "fallback"
            sel_str = str(picker.selected_index) if picker.selected_index else "-"
            cv2.putText(
                display,
                f"Dados: {len(crops)} | Caras: {len(detected_faces)} | WS: {ws_status} | sel: {sel_str}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2,
            )

            # Jugada
            if detected_faces:
                hand_color = ORANGE if hand_rank >= 4 else GREEN if hand_rank >= 2 else WHITE
                cv2.putText(display, f"JUGADA: {hand_desc}",
                            (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

            cv2.imshow("Poker Dice Vision", display)
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q") or key == 27:
                break

            # Procesar comandos pendientes
            while not cmd_queue.empty():
                cmd_line = cmd_queue.get_nowait()
                running = process_command(cmd_line, picker, classifier, results, ws)
                if not running:
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
