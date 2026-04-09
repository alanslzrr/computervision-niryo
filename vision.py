"""
Pipeline de vision: deteccion del workspace por dianas y deteccion de piezas.

Funciones de calibracion del workspace (HoughCircles), transformacion perspectiva
pixel -> coordenadas relativas [0,1], y deteccion/clasificacion de objetos por
forma (circularidad) y color (HSV).
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
    """Devuelve esquinas de workspace en pixeles cuando no se detectan las dianas. Orden TL, TR, BR, BL."""
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
    """Detecta las 4 dianas del workspace con HoughCircles y las asigna a esquinas (TL, TR, BR, BL)
    segun proximidad a las esquinas de la imagen. Devuelve None si no hay deteccion valida."""
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
    """Comprueba si un punto (px, py) esta dentro del poligono definido por corners."""
    poly = corners.reshape((-1, 1, 2)).astype(np.float32)
    return cv2.pointPolygonTest(poly, point, False) >= 0


def pixel_to_relative(px: int, py: int, corners: np.ndarray) -> Tuple[float, float]:
    """Mapea un pixel (px, py) a coordenadas relativas [0..1]x[0..1] mediante homografia
    con las esquinas TL, TR, BR, BL del workspace. Las coordenadas se recortan al rango [0,1]."""
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
    """Clasifica la forma segun numero de vertices (approx), relacion de aspecto y circularidad.
    Retorna CIRCLE, SQUARE o OTHER."""
    if circularity > 0.85:
        return "CIRCLE"
    if num_vertices == 4 and 0.85 <= aspect_ratio <= 1.15:
        return "SQUARE"
    return "OTHER"


def detect_color(image: np.ndarray, contour: np.ndarray) -> str:
    """Obtiene el color predominante en la region del contorno usando el promedio HSV
    enmascarado. Retorna RED, GREEN, BLUE o UNKNOWN segun COLOR_RANGES."""
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
    """Preprocesa el frame: pasa a HSV, umbraliza para obtener regiones de interes y aplica
    morfologia (apertura + cierre) para limpiar ruido y cerrar huecos en los contornos."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([180, 255, 255]))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    return binary


def detect_objects(frame: np.ndarray, workspace_corners: np.ndarray) -> List[Dict]:
    """Detecta piezas circulares o cuadradas (rojas, verdes o azules) dentro del workspace.
    Filtra por area, clasifica forma y color, y retorna lista ordenada por area descendente."""
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
