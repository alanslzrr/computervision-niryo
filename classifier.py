"""
Clasificador de caras de dados de poker mediante CNN exportada a ONNX.

Carga un modelo ONNX entrenado con PyTorch (train.py) y clasifica crops
de dados en una de seis caras: 9, 10, J, Q, K, A.
"""

import os
from typing import Tuple

import cv2
import numpy as np

from config import CNN_CONFIDENCE_THRESHOLD, CNN_INPUT_SIZE, DICE_FACES, ONNX_MODEL_PATH


class DiceClassifier:
    """Clasificador de caras de dados usando inferencia ONNX Runtime."""

    def __init__(self, model_path: str = ONNX_MODEL_PATH):
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
        """Preprocesa un crop BGR para entrada a la CNN.
        Replica exactamente el pipeline de train.py: resize, /255, HWC->CHW."""
        resized = cv2.resize(crop, CNN_INPUT_SIZE)
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        batched = np.expand_dims(chw, axis=0)       # (1, 3, H, W)
        return batched

    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        """Clasifica un crop de dado y retorna (face_label, confidence)."""
        if self.session is None:
            return ("UNKNOWN", 0.0)

        tensor = self.preprocess_crop(crop)
        logits = self.session.run([self.output_name], {self.input_name: tensor})[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])

        if confidence < CNN_CONFIDENCE_THRESHOLD:
            return ("UNKNOWN", confidence)

        return (DICE_FACES[max_idx], confidence)
