"""
Interfaz de usuario del laboratorio de piezas (``main.py``).

Responsabilidades:
    - Filtros por color/forma sobre la lista de objetos detectados.
    - Dibujo del polígono del workspace, piezas, HUD y estado de filtros/selección.
    - Lectura de comandos en hilo aparte y ejecución de pick con re-detección en frame fresco.

No utilizar este módulo para el flujo de poker dice; ese está en ``poker.py``.
"""

import queue
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from pyniryo import PoseObject

from config import APPROACH_Z, PICK_Z
from robot import NiryoVisionPicker, relative_to_robot_xy
from vision import detect_objects, detect_workspace_from_dianas, pixel_to_relative


# ============================================================================
# FILTROS Y DIBUJO
# ============================================================================

def apply_filters(objects: List[Dict], color_filter: str, shape_filter: str) -> List[Dict]:
    """Filtra la lista de objetos por color y/o forma. ANY significa no filtrar por ese criterio."""
    filtered = objects
    if color_filter != "ANY":
        filtered = [o for o in filtered if o["color"] == color_filter]
    if shape_filter != "ANY":
        filtered = [o for o in filtered if o["shape"] == shape_filter]
    return filtered


def draw_workspace_overlay(frame: np.ndarray, corners: np.ndarray, detected_live: bool) -> None:
    """Dibuja el poligono del workspace, las 4 esquinas (TL, TR, BR, BL) y un indicador
    de si el workspace se detecto por dianas o se usa fallback."""
    corners_i = corners.astype(int)

    # Poligono del workspace detectado
    cv2.polylines(frame, [corners_i.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 255), thickness=2)

    # Marcar las 4 dianas detectadas (centros)
    for i, (x, y) in enumerate(corners_i):
        cv2.circle(frame, (x, y), 8, (0, 255, 255), 2)
        label = ["TL", "TR", "BR", "BL"][i]
        cv2.putText(frame, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    status = "WS: dianas detectadas" if detected_live else "WS: usando fallback temporal"
    cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_objects(frame: np.ndarray, objects: List[Dict], selected_index: Optional[int]) -> None:
    """Dibuja rectangulos, centroides y etiquetas (color + forma) para cada pieza detectada.
    La pieza seleccionada se resalta en naranja."""
    for i, obj in enumerate(objects, start=1):
        cx, cy = obj["centroid"]
        x, y, w, h = obj["bounding_rect"]

        color_bgr = (0, 255, 0)
        if selected_index == i:
            color_bgr = (0, 165, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"{i}:{obj['color']} {obj['shape']}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_hud(frame: np.ndarray, picker: NiryoVisionPicker, visible_objects: List[Dict]) -> None:
    """Dibuja el HUD con instrucciones de control, filtros activos y numero de piezas visibles."""
    lines = [
        "Control: help | scan | home | open | clear | color X | shape X | select N | pick | exit",
        f"Filtro color={picker.filter_color} forma={picker.filter_shape} | visibles={len(visible_objects)} | sel={picker.selected_index}",
    ]

    y = 52
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
        y += 24


# ============================================================================
# TERMINAL COMMANDS
# ============================================================================

def print_terminal_help() -> None:
    """Imprime la lista de comandos disponibles en la terminal."""
    print("\nComandos disponibles:")
    print("  help")
    print("  scan")
    print("  home")
    print("  open")
    print("  clear     - Resetea estado de colision (quitar obstaculo antes)")
    print("  color any|red|green|blue")
    print("  shape any|circle|square")
    print("  select N")
    print("  pick")
    print("  status")
    print("  exit")


def input_worker(cmd_queue: "queue.Queue[str]") -> None:
    """Hilo que lee comandos de la terminal y los pone en la cola para el hilo principal."""
    while True:
        try:
            line = input("cmd> ").strip()
        except EOFError:
            break
        cmd_queue.put(line)


def pick_selected_object(picker: NiryoVisionPicker, obj: Dict, workspace_corners: np.ndarray) -> None:
    """Ejecuta pick-and-place de una pieza del laboratorio de formas/colores.

    Re-captura imagen en pose de escaneo, vuelve a estimar workspace y objetos, empareja por
    color+forma y cercanía al centroide previo, convierte a coordenadas robot y realiza
    approach → pick → grasp → home → release → scan.
    """
    target_color = obj["color"]
    target_shape = obj["shape"]
    original_cx, original_cy = obj["centroid"]

    # --- 1) Re-capturar frame fresco desde la posicion de escaneo ---
    print("[PICK] Re-capturando frame para coordenadas frescas...")
    fresh_frame = picker.capture_frame()

    # --- 2) Re-detectar workspace con dianas frescas ---
    fresh_ws = detect_workspace_from_dianas(fresh_frame)
    if fresh_ws is not None:
        ws = fresh_ws
        print("[PICK] Workspace re-detectado desde dianas frescas")
    else:
        ws = workspace_corners
        print("[PICK] Dianas no detectadas, usando workspace anterior")

    # --- 3) Re-detectar objetos con el frame y workspace frescos ---
    fresh_objects = detect_objects(fresh_frame, ws)

    # --- 4) Buscar la pieza por color, forma y proximidad al centroide original ---
    best_match = None
    best_dist = float("inf")
    for o in fresh_objects:
        if o["color"] == target_color and o["shape"] == target_shape:
            dx = o["centroid"][0] - original_cx
            dy = o["centroid"][1] - original_cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_match = o

    if best_match is None:
        print(f"[PICK] ERROR: No se encontro {target_color} {target_shape} en re-deteccion. Abortando.")
        return

    cx, cy = best_match["centroid"]
    x_rel, y_rel = pixel_to_relative(cx, cy, ws)
    x, y = relative_to_robot_xy(x_rel, y_rel)

    approach_pose = PoseObject(x, y, APPROACH_Z, picker.pick_roll, picker.pick_pitch, picker.pick_yaw)
    pick_pose = PoseObject(x, y, PICK_Z, picker.pick_roll, picker.pick_pitch, picker.pick_yaw)

    print(
        f"[PICK] Recogiendo {target_color} {target_shape}\n"
        f"       pixel original=({original_cx},{original_cy}) -> fresco=({cx},{cy}) delta={best_dist:.1f}px\n"
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


def process_command(
    cmd_line: str,
    picker: NiryoVisionPicker,
    visible_objects: List[Dict],
    workspace_corners: np.ndarray,
) -> bool:
    """Ejecuta un comando del menú de piezas (filtros, movimiento, pick, salida).

    Args:
        cmd_line: Entrada del usuario (p. ej. ``color red``, ``pick``).
        picker: Estado compartido con ``main``.
        visible_objects: Lista ya filtrada mostrada en el último frame.
        workspace_corners: Esquinas del workspace para homografía en pick.

    Returns:
        False si se debe terminar el bucle (``exit``); True para continuar.
    """
    if not cmd_line:
        return True

    parts = cmd_line.strip().split()
    cmd = parts[0].lower()

    if cmd == "help":
        print_terminal_help()
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
        print("[CMD] Limpiando colision (asegurate de haber retirado el obstaculo)")
        try:
            picker.clear_collision()
        except Exception as e:
            print(f"[ERROR] Fallo al limpiar colision: {e}")
        return True

    if cmd == "status":
        ws_state = "detectado" if picker.workspace_corners is not None else "fallback"
        print(
            f"Status -> ws={ws_state}, color={picker.filter_color}, "
            f"shape={picker.filter_shape}, visibles={len(visible_objects)}, sel={picker.selected_index}"
        )
        return True

    if cmd == "color" and len(parts) == 2:
        val = parts[1].upper()
        if val in {"ANY", "RED", "GREEN", "BLUE"}:
            picker.filter_color = val
            picker.selected_index = None
            print(f"[CMD] Filtro color -> {val}")
        else:
            print("Uso: color any|red|green|blue")
        return True

    if cmd == "shape" and len(parts) == 2:
        val = parts[1].upper()
        if val in {"ANY", "CIRCLE", "SQUARE"}:
            picker.filter_shape = val
            picker.selected_index = None
            print(f"[CMD] Filtro forma -> {val}")
        else:
            print("Uso: shape any|circle|square")
        return True

    if cmd == "select" and len(parts) == 2:
        try:
            idx = int(parts[1])
        except ValueError:
            print("Uso: select N")
            return True

        if 1 <= idx <= len(visible_objects):
            picker.selected_index = idx
            obj = visible_objects[idx - 1]
            print(f"Seleccionada pieza {idx}: {obj['color']} {obj['shape']}")
        else:
            print(f"Indice fuera de rango. Visibles: {len(visible_objects)}")
        return True

    if cmd == "pick":
        if picker.selected_index is None:
            print("Primero usa: select N")
            return True
        if not (1 <= picker.selected_index <= len(visible_objects)):
            print("La seleccion actual ya no es valida. Vuelve a seleccionar.")
            picker.selected_index = None
            return True

        target = visible_objects[picker.selected_index - 1]
        try:
            pick_selected_object(picker, target, workspace_corners)
        except Exception as e:
            print(f"[ERROR] Fallo en pick/place: {e}")
            print(
                "  Si hubo colision: retira el obstaculo y ejecuta 'clear' para resetear el robot."
            )
            try:
                picker.move_scan()
            except Exception:
                pass
        return True

    print("Comando no reconocido. Usa: help")
    return True
