"""
Pipeline de visión para el laboratorio de piezas (círculos y cuadrados de color).

Incluye:
    - Detección del cuadrilátero del workspace mediante cuatro dianas circulares (``HoughCircles``).
    - Homografía pixel → coordenadas normalizadas ``[0, 1]²`` para acoplar con ``robot.relative_to_robot_xy``.
    - Segmentación por rango HSV global + morfología, contornos, filtrado por área y clasificación
      por forma (circularidad, polígono aproximado) y color (promedio HSV en máscara).

Constantes de área y color: ``config`` (``MIN_AREA``, ``MAX_AREA``, ``COLOR_RANGES``, etc.).
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    COLOR_RANGES,
    EPSILON_FACTOR,
    FALLBACK_WORKSPACE_PIXEL_BOUNDS,
    MAX_AREA,
    MIN_AREA,
)


# ============================================================================
# WORKSPACE POR DIANAS
# ============================================================================

def fallback_workspace_corners() -> np.ndarray:
    """Esquinas por defecto del workspace en píxeles si falla la detección por dianas.

    Returns:
        Array ``(4, 2)`` float32 en orden TL, TR, BR, BL según ``FALLBACK_WORKSPACE_PIXEL_BOUNDS``.
    """
    x_min = FALLBACK_WORKSPACE_PIXEL_BOUNDS["x_min"]
    x_max = FALLBACK_WORKSPACE_PIXEL_BOUNDS["x_max"]
    y_min = FALLBACK_WORKSPACE_PIXEL_BOUNDS["y_min"]
    y_max = FALLBACK_WORKSPACE_PIXEL_BOUNDS["y_max"]
    # TL, TR, BR, BL
    return np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ], dtype=np.float32)


def detect_workspace_from_dianas(frame: np.ndarray) -> Optional[np.ndarray]:
    """Localiza cuatro círculos de referencia y los ordena como vértices del workspace.

    Returns:
        Matriz ``(4, 2)`` de centros en orden TL, TR, BR, BL, o None si la geometría no es válida.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)

    h, w = gray.shape[:2]
    min_r = int(min(h, w) * 0.02)
    max_r = int(min(h, w) * 0.12)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(h, w) * 0.15),
        param1=120,
        param2=28,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    centers = np.array([[c[0], c[1]] for c in circles], dtype=np.float32)

    # Esquinas de imagen: TL, TR, BR, BL
    img_corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype=np.float32)

    selected = []
    used = set()

    for corner in img_corners:
        dists = np.linalg.norm(centers - corner, axis=1)
        order = np.argsort(dists)

        chosen = None
        for idx in order:
            if int(idx) in used:
                continue
            chosen = int(idx)
            break

        if chosen is None:
            return None

        used.add(chosen)
        selected.append(centers[chosen])

    selected = np.array(selected, dtype=np.float32)

    # Validacion basica: el area del cuadrilatero debe ser razonable
    quad = selected.reshape((-1, 1, 2)).astype(np.int32)
    area = abs(cv2.contourArea(quad))
    if area < 10000:
        return None

    return selected  # TL, TR, BR, BL


def point_inside_workspace(point: Tuple[int, int], corners: np.ndarray) -> bool:
    """Comprueba si un punto de imagen cae dentro del polígono del workspace.

    Args:
        point: Coordenadas ``(px, py)`` en píxeles.
        corners: Cuatro vértices en orden TL, TR, BR, BL (misma convención que homografía).

    Returns:
        True si el punto está en el interior o sobre el borde (test de OpenCV >= 0).
    """
    poly = corners.reshape((-1, 1, 2)).astype(np.float32)
    return cv2.pointPolygonTest(poly, point, False) >= 0


def pixel_to_relative(px: int, py: int, corners: np.ndarray) -> Tuple[float, float]:
    """Aplica homografía inversa pixel → plano normalizado del workspace.

    Returns:
        Par ``(x_rel, y_rel)`` en ``[0, 1]``, recortado para evitar extrapolaciones fuera del cuadrilátero.
    """
    src = corners.astype(np.float32)
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst) # matriz de homografia

    pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
    rel = cv2.perspectiveTransform(pt, H)[0, 0]

    x_rel = max(0.0, min(1.0, float(rel[0])))
    y_rel = max(0.0, min(1.0, float(rel[1])))
    return x_rel, y_rel


# ============================================================================
# DETECCION DE PIEZAS
# ============================================================================

def classify_shape(num_vertices: int, aspect_ratio: float, circularity: float) -> str:
    """Clasifica la forma a partir de la aproximación poligonal y métricas geométricas.

    Args:
        num_vertices: Vértices del polígono simplificado (``approxPolyDP``).
        aspect_ratio: Ancho / alto del rectángulo delimitador.
        circularity: 4*pi*area/perimetro^2 (métrica de circularidad de OpenCV).

    Returns:
        ``"CIRCLE"``, ``"SQUARE"`` u ``"OTHER"``.
    """
    if circularity > 0.85:
        return "CIRCLE"
    if num_vertices == 4 and 0.85 <= aspect_ratio <= 1.15:
        return "SQUARE"
    return "OTHER"


def detect_color(image: np.ndarray, contour: np.ndarray) -> str:
    """Clasifica el color medio HSV del interior del contorno frente a ``COLOR_RANGES``.

    Returns:
        ``"RED"``, ``"GREEN"``, ``"BLUE"`` o ``"UNKNOWN"`` si ningún rango contiene la media.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.mean(hsv, mask=mask)[:3]

    for color_name, ranges in COLOR_RANGES.items():
        for range_dict in ranges:
            lower = range_dict["lower"]
            upper = range_dict["upper"]
            if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                return color_name

    return "UNKNOWN"


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Binariza por saturación/valor en HSV y limpia con morfología.

    Returns:
        Máscara binaria ``uint8`` lista para ``findContours``.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([180, 255, 255]))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    return binary


def detect_objects(frame: np.ndarray, workspace_corners: np.ndarray) -> List[Dict]:
    """Detecta piezas de laboratorio (círculo/cuadrado, RGB) dentro del polígono del workspace.

    Returns:
        Lista de diccionarios con claves ``shape``, ``color``, ``centroid``, ``bounding_rect``, ``area``,
        ordenados por área descendente.
    """
    binary = preprocess(frame)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue

        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        if not point_inside_workspace((cx, cy), workspace_corners):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        approx = cv2.approxPolyDP(contour, EPSILON_FACTOR * perimeter, True)
        shape = classify_shape(len(approx), aspect_ratio, circularity)
        if shape not in {"CIRCLE", "SQUARE"}:
            continue

        color = detect_color(frame, contour)
        if color not in {"RED", "GREEN", "BLUE"}:
            continue

        objects.append(
            {
                "shape": shape,
                "color": color,
                "centroid": (cx, cy),
                "bounding_rect": (x, y, w, h),
                "area": area,
            }
        )

    objects.sort(key=lambda o: o["area"], reverse=True)
    return objects
