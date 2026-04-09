"""
Evaluador de jugadas de poker dice.

A partir de una lista de caras detectadas en los dados (``9``, ``10``, ``J``, ``Q``, ``K``, ``A``),
determina la mejor jugada según la jerarquía habitual de poker dice: repoker, poker, full house,
escalera, trío, doble par, par o nada.

Notas:
    - La integración con el flujo en tiempo real se realiza desde ``poker.py`` tras la clasificación CNN.
    - Con menos de cinco dados, solo aplican reglas que no exijan exactamente cinco cartas (p. ej. escalera).
"""

from collections import Counter
from typing import List, Tuple

from config import DICE_FACES

# Valor numérico de rango: mayor es mejor jugada. Coherente con ``evaluate_hand`` (1 = peor, 8 = mejor).
HAND_RANKINGS = {
    "REPOKER": 8,     # Cinco dados con la misma cara
    "POKER": 7,       # Cuatro dados con la misma cara
    "FULL_HOUSE": 6,  # Trío + par
    "STRAIGHT": 5,    # Cinco caras consecutivas en el orden de DICE_FACES
    "TRIO": 4,        # Tres dados con la misma cara
    "TWO_PAIR": 3,    # Dos pares distintos
    "PAIR": 2,        # Un par
    "NOTHING": 1,     # Sin combinación reconocible
}


def _is_straight(faces: List[str]) -> bool:
    """Comprueba si cinco caras forman escalera (cinco índices consecutivos en ``DICE_FACES``).

    Args:
        faces: Lista de exactamente cinco etiquetas de cara.

    Returns:
        True si, ordenadas por posición en ``DICE_FACES``, son cinco índices consecutivos.
    """
    if len(faces) != 5:
        return False
    indices = sorted([DICE_FACES.index(f) for f in faces])
    return indices == list(range(indices[0], indices[0] + 5))


def evaluate_hand(faces: List[str]) -> Tuple[str, int, str]:
    """Evalúa la jugada a partir de las caras detectadas.

    Args:
        faces: Caras observadas, p. ej. ``["9", "9", "J", "Q", "K"]``. Puede contener menos de cinco
            elementos si la detección es parcial; en ese caso no se puede clasificar escalera ni repoker
            completo salvo que la lógica futura replique o descarte dados.

    Returns:
        Tupla ``(hand_name, rank, description)``:
            - **hand_name**: Clave de ``HAND_RANKINGS`` (p. ej. ``"FULL_HOUSE"``).
            - **rank**: Entero 1–8, mayor es mejor (véase ``HAND_RANKINGS``).
            - **description**: Texto breve en español para mostrar en HUD o consola.
    """
    if not faces:
        return ("NOTHING", HAND_RANKINGS["NOTHING"], "Sin dados detectados")

    counts = Counter(faces)
    freq = sorted(counts.values(), reverse=True)

    # Repoker: 5 iguales
    if freq[0] >= 5:
        dominant = counts.most_common(1)[0][0]
        return ("REPOKER", HAND_RANKINGS["REPOKER"], f"Repoker de {dominant}")

    # Poker: 4 iguales
    if freq[0] == 4:
        dominant = counts.most_common(1)[0][0]
        return ("POKER", HAND_RANKINGS["POKER"], f"Poker de {dominant}")

    # Full house: 3 + 2
    if len(freq) >= 2 and freq[0] == 3 and freq[1] == 2:
        trio_face = [f for f, c in counts.items() if c == 3][0]
        pair_face = [f for f, c in counts.items() if c == 2][0]
        return (
            "FULL_HOUSE",
            HAND_RANKINGS["FULL_HOUSE"],
            f"Full House: trio de {trio_face}, par de {pair_face}",
        )

    # Escalera: cinco distintas y consecutivas
    if len(faces) == 5 and len(set(faces)) == 5 and _is_straight(faces):
        ordered = " ".join(sorted(faces, key=lambda f: DICE_FACES.index(f)))
        return ("STRAIGHT", HAND_RANKINGS["STRAIGHT"], f"Escalera: {ordered}")

    # Trío
    if freq[0] == 3:
        trio_face = counts.most_common(1)[0][0]
        return ("TRIO", HAND_RANKINGS["TRIO"], f"Trio de {trio_face}")

    # Doble par
    pairs = [f for f, c in counts.items() if c == 2]
    if len(pairs) >= 2:
        return ("TWO_PAIR", HAND_RANKINGS["TWO_PAIR"], f"Doble par: {pairs[0]} y {pairs[1]}")

    # Par
    if len(pairs) == 1:
        return ("PAIR", HAND_RANKINGS["PAIR"], f"Par de {pairs[0]}")

    return ("NOTHING", HAND_RANKINGS["NOTHING"], "Nada")
