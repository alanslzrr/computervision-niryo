"""
Control del robot Niryo y mapeo de coordenadas de imagen a espacio cartesiano del robot.

Responsabilidades:
    - Conexión, calibración y captura desde la cámara del brazo.
    - Movimientos de escaneo, home y pick-and-place asociados al laboratorio de visión.
    - Conversión de coordenadas relativas ``[0, 1] × [0, 1]`` (salida de ``vision.pixel_to_relative``)
      a ``(x, y)`` en metros mediante interpolación bilineal sobre las cuatro esquinas calibradas.

Dependencias: ``pyniryo`` para ``NiryoRobot`` y ``PoseObject``.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from pyniryo import NiryoRobot, PoseObject

from config import (
    LIMIT_1,
    LIMIT_2,
    LIMIT_3,
    LIMIT_4,
    SCANNING_POSITION,
)


def pose_from_tuple(p: Tuple[float, float, float, float, float, float]) -> PoseObject:
    """Construye un ``PoseObject`` a partir de una tupla (x, y, z, roll, pitch, yaw).

    Args:
        p: Seis valores en metros y radianes según la convención de pyniryo.

    Returns:
        Instancia lista para pasar a ``NiryoRobot.move``.
    """
    return PoseObject(*p)


class NiryoVisionPicker:
    """Orquestador de visión + robot: filtros de UI, workspace, columnas de depósito y captura.

    Attributes:
        ip: Dirección IP del robot.
        robot: Instancia de ``NiryoRobot`` tras ``connect``, o None si no hay conexión.
        filter_color: Filtro de color para piezas del lab. anterior: ``ANY`` | ``RED`` | ``GREEN`` | ``BLUE``.
        filter_shape: Filtro de forma: ``ANY`` | ``CIRCLE`` | ``SQUARE``.
        selected_index: Índice 1-based de la pieza/dado seleccionado en la lista filtrada, o None.
        pick_roll, pick_pitch, pick_yaw: Orientación de herramienta reutilizada en picks (desde scan).
        workspace_corners: Matriz (4, 2) float32 con esquinas TL, TR, BR, BL en píxeles, o None.
        face_columns: Asignación cara de dado → índice de columna de depósito (módulo poker).
        face_slot_counts: Contador de slots usados por cara para apilar en columna.
    """

    def __init__(self, ip: str) -> None:
        self.ip = ip
        self.robot: Optional[NiryoRobot] = None

        self.filter_color = "ANY"   # ANY | RED | GREEN | BLUE
        self.filter_shape = "ANY"   # ANY | CIRCLE | SQUARE
        self.selected_index: Optional[int] = None

        self.pick_roll = SCANNING_POSITION[3]
        self.pick_pitch = SCANNING_POSITION[4]
        self.pick_yaw = SCANNING_POSITION[5]

        # Workspace detectado desde dianas (TL, TR, BR, BL)
        self.workspace_corners: Optional[np.ndarray] = None

        # Asignación de columnas de drop por cara (poker dice)
        self.face_columns: dict = {}
        self.face_slot_counts: dict = {}

    def reset_drops(self) -> None:
        """Reinicia columnas y contadores de slots entre rondas de poker dice."""
        self.face_columns = {}
        self.face_slot_counts = {}
        print("[DROP] Columnas y slots reseteados")

    def assign_drop_slot(self, face: str) -> Tuple[int, int]:
        """Reserva la siguiente celda (columna, slot) para una cara de dado.

        La primera aparición de una cara obtiene la siguiente columna libre; sucesivos dados de la
        misma cara incrementan el slot vertical dentro de esa columna.

        Args:
            face: Etiqueta de cara (p. ej. ``"J"``).

        Returns:
            ``(col_idx, slot_idx)`` índices sobre ``config.DROP_COLUMN_X_REL`` y ``DROP_SLOT_Y_REL``.
        """
        if face not in self.face_columns:
            self.face_columns[face] = len(self.face_columns)
        col_idx = self.face_columns[face]
        slot_idx = self.face_slot_counts.get(face, 0)
        self.face_slot_counts[face] = slot_idx + 1
        return col_idx, slot_idx

    def connect(self) -> None:
        """Conecta al robot, ejecuta calibración automática e intenta detectar la herramienta."""
        print(f"Conectando al robot: {self.ip}")
        self.robot = NiryoRobot(self.ip)
        try:
            self.robot.clear_collision_detected()
        except Exception:
            pass
        self.robot.calibrate_auto()
        print("Robot calibrado")

        try:
            self.robot.update_tool()
            print("Herramienta detectada")
        except Exception as e:
            print(f"[WARN] No se pudo detectar herramienta automaticamente: {e}")

    def move_scan(self) -> None:
        """Mueve el brazo a la pose de escaneo definida en ``config.SCANNING_POSITION``."""
        self.robot.move(pose_from_tuple(SCANNING_POSITION))

    def move_home(self) -> None:
        """Mueve el robot a la pose home del firmware."""
        self.robot.move_to_home_pose()

    def clear_collision(self) -> None:
        """Limpia el flag de colisión y recalibra. Retirar el obstáculo antes de llamar."""
        if self.robot is None:
            raise RuntimeError("Robot no conectado")
        print("[COLLISION] Limpiando flag de colision y re-calibrando...")
        self.robot.clear_collision_detected()
        self.robot.calibrate_auto()
        print("[COLLISION] Estado de colision restablecido. Robot listo.")

    def safe_shutdown(self) -> None:
        """Intenta abrir la pinza, ir a home y cerrar la conexión de forma ordenada."""
        if self.robot is None:
            return

        print("Cierre seguro...")
        try:
            try:
                self.robot.release_with_tool()
            except Exception:
                pass
            try:
                self.move_home()
            except Exception as e:
                print(f"[WARN] No se pudo mover a home durante cierre: {e}")
        finally:
            self.robot.close_connection()
            self.robot = None
            print("Robot desconectado")

    def capture_frame(self) -> np.ndarray:
        """Obtiene un fotograma BGR desde la cámara comprimida del robot.

        Returns:
            Imagen ``numpy`` en BGR, misma convención que OpenCV.

        Raises:
            RuntimeError: Si la decodificación JPEG falla.
        """
        img_compressed = self.robot.get_img_compressed()
        frame = cv2.imdecode(np.frombuffer(img_compressed, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("No se pudo decodificar la imagen de la camara")
        return frame


def relative_to_robot_xy(x_rel: float, y_rel: float) -> Tuple[float, float]:
    """Mapea coordenadas normalizadas del workspace a ``(x, y)`` del robot en metros.

    Interpolación bilineal entre las proyecciones de las cuatro esquinas (``LIMIT_1`` … ``LIMIT_4``).
    El eje X de la imagen va espejado respecto al sistema del robot (véase comentarios inline).

    Args:
        x_rel: Fracción horizontal en [0, 1] (izquierda → derecha en imagen).
        y_rel: Fracción vertical en [0, 1].

    Returns:
        Par ``(x, y)`` en metros para construir ``PoseObject`` con la Z deseada.
    """
    # Eje X de la cámara espejado respecto al robot: izquierda imagen ↔ derecha robot.
    p_tl = np.array(LIMIT_4[:2], dtype=float)
    p_bl = np.array(LIMIT_3[:2], dtype=float)
    p_tr = np.array(LIMIT_1[:2], dtype=float)
    p_br = np.array(LIMIT_2[:2], dtype=float)

    top = p_tl * (1.0 - x_rel) + p_tr * x_rel
    bottom = p_bl * (1.0 - x_rel) + p_br * x_rel
    xy = top * (1.0 - y_rel) + bottom * y_rel

    return float(xy[0]), float(xy[1])
