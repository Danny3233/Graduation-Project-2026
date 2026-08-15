from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "sign_recognition"
    / "models"
    / "sign_holistic_model.joblib"
)

SEQUENCE_LENGTH = 30
FRAME_VECTOR_SIZE = 163

MIN_CONFIDENCE = 0.65
MIN_PREDICTION_MARGIN = 0.15


@lru_cache(maxsize=1)
def load_sign_model() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình tại: {MODEL_PATH}. "
            "Bạn cần chuyển video sang JSON holistic "
            "và huấn luyện mô hình mới."
        )

    bundle = joblib.load(MODEL_PATH)

    if "model" not in bundle:
        raise ValueError(
            "File mô hình không chứa khóa 'model'."
        )

    model_sequence_length = bundle.get(
        "sequence_length",
        SEQUENCE_LENGTH,
    )

    model_frame_vector_size = bundle.get(
        "frame_vector_size",
        FRAME_VECTOR_SIZE,
    )

    if model_sequence_length != SEQUENCE_LENGTH:
        raise ValueError(
            "Mô hình sử dụng sequence_length="
            f"{model_sequence_length}, nhưng backend yêu cầu "
            f"{SEQUENCE_LENGTH}."
        )

    if model_frame_vector_size != FRAME_VECTOR_SIZE:
        raise ValueError(
            "Mô hình sử dụng frame_vector_size="
            f"{model_frame_vector_size}, nhưng backend yêu cầu "
            f"{FRAME_VECTOR_SIZE}."
        )

    model = bundle["model"]

    expected_feature_count = (
        SEQUENCE_LENGTH * FRAME_VECTOR_SIZE
    )

    model_feature_count = getattr(
        model,
        "n_features_in_",
        expected_feature_count,
    )

    if model_feature_count != expected_feature_count:
        raise ValueError(
            "Mô hình không tương thích. "
            f"Mô hình cần {model_feature_count} đặc trưng, "
            f"nhưng dữ liệu holistic có "
            f"{expected_feature_count} đặc trưng."
        )

    return bundle


def predict_sign_sequence(
    frames: List[List[float]],
) -> Dict[str, Union[str, float]]:
    sequence = np.asarray(
        frames,
        dtype=np.float32,
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        FRAME_VECTOR_SIZE,
    )

    if sequence.shape != expected_shape:
        raise ValueError(
            "Dữ liệu phải có kích thước "
            f"{SEQUENCE_LENGTH} × {FRAME_VECTOR_SIZE}, "
            f"nhưng nhận được {sequence.shape}."
        )

    if not np.isfinite(sequence).all():
        raise ValueError(
            "Dữ liệu chứa NaN hoặc giá trị vô hạn."
        )

    bundle = load_sign_model()

    model = bundle["model"]
    label_text = bundle.get("label_text", {})

    feature_vector = sequence.reshape(1, -1)

    probabilities = model.predict_proba(
        feature_vector,
    )[0]

    sorted_indices = np.argsort(
        probabilities,
    )[::-1]

    best_index = int(sorted_indices[0])
    best_confidence = float(
        probabilities[best_index]
    )

    second_confidence = 0.0

    if len(sorted_indices) > 1:
        second_index = int(sorted_indices[1])
        second_confidence = float(
            probabilities[second_index]
        )

    prediction_margin = (
        best_confidence - second_confidence
    )

    predicted_label = str(
        model.classes_[best_index]
    )

    predicted_text = label_text.get(
        predicted_label,
        predicted_label,
    )

    return {
        "margin": prediction_margin,
        "label": predicted_label,
        "text": predicted_text,
        "confidence": best_confidence,
    }