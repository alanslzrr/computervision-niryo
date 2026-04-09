"""
Clasificación de caras de dados de poker mediante modelo CNN en ONNX Runtime.

El archivo ``dice_classifier.onnx`` se genera con ``train.py`` (PyTorch → ONNX). Las clases
siguen el orden de ``config.DICE_FACES``. El preprocesado debe ser idéntico al del entrenamiento
(resize BGR, normalización [0, 1], NCHW).
"""

import os
from typing import Tuple

import cv2
import numpy as np

from config import CNN_CONFIDENCE_THRESHOLD, CNN_INPUT_SIZE, DICE_FACES, ONNX_MODEL_PATH


class DiceClassifier:
    """Inferencia ONNX para asignar una cara de dado a un recorte de imagen.

    Si el archivo del modelo no existe, ``classify`` devuelve ``("UNKNOWN", 0.0)`` sin lanzar error.

    Attributes:
        model_path: Ruta al fichero ``.onnx``.
        session: Sesión ``onnxruntime.InferenceSession`` o None si no hay modelo.
        input_name: Nombre del tensor de entrada según el grafo ONNX.
        output_name: Nombre del tensor de salida (logits).
    """

    def __init__(self, model_path: str = ONNX_MODEL_PATH) -> None:
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None

        if os.path.exists(model_path):
            import onnxruntime as ort
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Prepara un recorte BGR para la entrada de la red.

        Replica el pipeline de ``train.DiceDataset`` / entrenamiento: redimensionado a
        ``CNN_INPUT_SIZE``, división por 255, permutación HWC → CHW y dimensión batch.

        Args:
            crop: Imagen BGR ``uint8`` del dado recortado.

        Returns:
            Tensor float32 con forma ``(1, 3, H, W)``.
        """
        resized = cv2.resize(crop, CNN_INPUT_SIZE)
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(chw, axis=0)
        return batched

    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        """Ejecuta softmax sobre los logits y devuelve la clase ganadora si supera el umbral.

        Args:
            crop: Recorte BGR del dado.

        Returns:
            ``(etiqueta, confianza)``. La etiqueta es ``"UNKNOWN"`` si no hay sesión o si
            ``confianza < CNN_CONFIDENCE_THRESHOLD``.
        """
        if self.session is None:
            return ("UNKNOWN", 0.0)

        tensor = self.preprocess_crop(crop)
        logits = self.session.run([self.output_name], {self.input_name: tensor})[0][0]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])

        if confidence < CNN_CONFIDENCE_THRESHOLD:
            return ("UNKNOWN", confidence)

        return (DICE_FACES[max_idx], confidence)
