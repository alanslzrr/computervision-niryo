"""
Evaluador de jugadas de poker dice.

Dada una lista de caras de dados detectadas (9, 10, J, Q, K, A),
determina la mejor jugada segun la jerarquia de poker dice.

TODO: Integrar con el flujo principal para evaluar automaticamente
      tras clasificar todos los dados visibles.
"""

from collections import Counter
from typing import List, Tuple

from config import DICE_FACES

# Jerarquia de jugadas (mayor rango = mejor jugada)
HAND_RANKINGS = {
    "POKER": 7,       # 5 dados con la misma cara
    "FULL_HOUSE": 6,  # 3 iguales + 2 iguales
    "STRAIGHT": 5,    # 5 caras consecutivas
    "TRIO": 4,        # 3 dados con la misma cara
    "TWO_PAIR": 3,    # 2 pares de dados iguales
    "PAIR": 2,        # 2 dados con la misma cara
    "NOTHING": 1,     # Sin combinacion valida
}


def _is_straight(faces: List[str]) -> bool:
    """Comprueba si las caras forman una escalera (5 consecutivas en el orden de DICE_FACES)."""
    if len(faces) != 5:
        return False
    indices = sorted([DICE_FACES.index(f) for f in faces])
    return indices == list(range(indices[0], indices[0] + 5))


def evaluate_hand(faces: List[str]) -> Tuple[str, int, str]:
    """Evalua la jugada a partir de una lista de caras detectadas.

    Args:
        faces: Lista de strings con las caras detectadas (ej: ["9", "9", "J", "Q", "K"])

    Returns:
        Tupla (hand_name, rank, description) donde:
            - hand_name: nombre de la jugada (POKER, FULL_HOUSE, etc.)
            - rank: valor numerico del rango (1-7)
            - description: descripcion legible de la jugada

    TODO: Decidir como manejar menos de 5 dados (deteccion parcial).
    """
    if not faces:
        return ("NOTHING", 1, "Sin dados detectados")

    counts = Counter(faces)
    freq = sorted(counts.values(), reverse=True)

    # Poker: 5 iguales
    if freq[0] >= 5:
        dominant = counts.most_common(1)[0][0]
        return ("POKER", 7, f"Poker de {dominant}")

    # Full House: 3 + 2
    if len(freq) >= 2 and freq[0] == 3 and freq[1] == 2:
        trio_face = [f for f, c in counts.items() if c == 3][0]
        pair_face = [f for f, c in counts.items() if c == 2][0]
        return ("FULL_HOUSE", 6, f"Full House: trio de {trio_face}, par de {pair_face}")

    # Escalera: 5 consecutivas
    if len(faces) == 5 and len(set(faces)) == 5 and _is_straight(faces):
        return ("STRAIGHT", 5, f"Escalera: {' '.join(sorted(faces, key=lambda f: DICE_FACES.index(f)))}")

    # Trio: 3 iguales
    if freq[0] == 3:
        trio_face = counts.most_common(1)[0][0]
        return ("TRIO", 4, f"Trio de {trio_face}")

    # Doble par
    pairs = [f for f, c in counts.items() if c == 2]
    if len(pairs) >= 2:
        return ("TWO_PAIR", 3, f"Doble par: {pairs[0]} y {pairs[1]}")

    # Par
    if len(pairs) == 1:
        return ("PAIR", 2, f"Par de {pairs[0]}")

    return ("NOTHING", 1, "Nada")
