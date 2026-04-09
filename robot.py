"""
Control del robot Niryo y mapeo de coordenadas relativas a cartesianas del robot.

Gestiona conexion, calibracion, captura de imagen y movimientos de pick-and-place.
El mapeo relative_to_robot_xy convierte coordenadas normalizadas [0,1] (generadas
por vision.pixel_to_relative) a posiciones cartesianas del robot.
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
    """Convierte una tupla de 6 valores (x,y,z,roll,pitch,yaw) en un PoseObject del Niryo."""
    return PoseObject(*p)


class NiryoVisionPicker:
    """Controlador del robot Niryo con integracion de vision. Gestiona conexion, captura de
    imagen y ejecucion de movimientos de pick-and-place sobre piezas detectadas."""

    def __init__(self, ip: str):
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

    def connect(self) -> None:
        """Conecta al robot, ejecuta calibracion automatica e intenta detectar la herramienta."""
        print(f"Conectando al robot: {self.ip}")
        self.robot = NiryoRobot(self.ip)
        # Limpiar colision pendiente de sesiones anteriores para permitir movimientos
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
        """Mueve el robot a la pose de escaneo donde la camara enfoca el workspace."""
        self.robot.move(pose_from_tuple(SCANNING_POSITION))

    def move_home(self) -> None:
        """Mueve el robot a su posicion home oficial."""
        self.robot.move_to_home_pose()

    def clear_collision(self) -> None:
        """Resetea el estado de error tras una colision detectada.
        IMPORTANTE: Retirar siempre el obstaculo antes de invocar.
        Tras limpiar, se ejecuta calibracion automatica para re-sincronizar."""
        if self.robot is None:
            raise RuntimeError("Robot no conectado")
        print("[COLLISION] Limpiando flag de colision y re-calibrando...")
        self.robot.clear_collision_detected()
        self.robot.calibrate_auto()
        print("[COLLISION] Estado de colision restablecido. Robot listo.")

    def safe_shutdown(self) -> None:
        """Cierre seguro: libera pinza, vuelve a home y desconecta el robot."""
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
        """Captura un frame desde la camara del robot y lo decodifica como imagen BGR."""
        img_compressed = self.robot.get_img_compressed()
        frame = cv2.imdecode(np.frombuffer(img_compressed, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("No se pudo decodificar la imagen de la camara")
        return frame


def relative_to_robot_xy(x_rel: float, y_rel: float) -> Tuple[float, float]:
    """Convierte coordenadas relativas [0..1]x[0..1] a (x, y) Cartesianas del robot mediante
    interpolacion bilineal. Tiene en cuenta el espejado del eje X entre camara y robot.
    Vease los comentarios sobre LIMIT_1..4 para el mapeo fisico de esquinas."""
    # Eje X de la camara esta espejado respecto al robot:
    #   izqda en imagen = dcha del robot, y viceversa.
    p_tl = np.array(LIMIT_4[:2], dtype=float)   # img-TL -> robot dcha-lejos
    p_bl = np.array(LIMIT_3[:2], dtype=float)   # img-BL -> robot dcha-cerca
    p_tr = np.array(LIMIT_1[:2], dtype=float)   # img-TR -> robot izqda-lejos
    p_br = np.array(LIMIT_2[:2], dtype=float)   # img-BR -> robot izqda-cerca

    top = p_tl * (1.0 - x_rel) + p_tr * x_rel
    bottom = p_bl * (1.0 - x_rel) + p_br * x_rel
    xy = top * (1.0 - y_rel) + bottom * y_rel

    return float(xy[0]), float(xy[1])
