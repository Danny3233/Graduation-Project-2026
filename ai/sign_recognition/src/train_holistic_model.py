import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from holistic_features import FRAME_VECTOR_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "sign_sequences_v2"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "sign_recognition"
    / "models"
    / "sign_holistic_model.joblib"
)

SEQUENCE_LENGTH = 30
MIN_SAMPLES_PER_LABEL = 5
MIN_DETECTED_HAND_FRAMES = 20

LABEL_TEXT = {
    "xin_chao": "Xin chào",
    "giup_do": "giúp đỡ", 
    "cam_on": "Cảm ơn",
    "co": "Có",
    "khong": "Không",
    "khong_cho": "không cho",
    "can": "cần",
    "muon": "muốn",
    "di": "đi",
    "uong": "uống",
    "an": "ăn",
    "yeu": "yêu",
    "ban": "bạn",
    "toi": "Tôi",
    "hoc": "học",
    "lam": "làm",
    "nha": "nhà",
    "truong": "trường",
    "lotte_mart": "Lotte Mart",
    "sieu_thi": "siêu thị",
    "ve": "về",
    "den": "đến",
    }

SENTENCE_PATTERNS = {
    ("toi", "yeu", "ban"):
        "Tôi yêu bạn.",

    ("toi", "sieu_thi", "lotte_mart", "di"):
        "Tôi đi siêu thị Lotte Mart.",

    ("toi", "truong", "di"):
        "Tôi đi đến trường.",

    ("toi", "nha", "ve"):
        "Tôi về nhà.",

    ("toi", "an", "muon"):
        "Tôi muốn ăn.",

    ("toi", "uong", "muon"):
        "Tôi muốn uống.",
}

def load_samples() -> Tuple[np.ndarray, np.ndarray, Counter]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {DATA_DIR}"
        )

    features: List[np.ndarray] = []
    labels: List[str] = []
    skipped_files: List[str] = []

    json_files = sorted(DATA_DIR.rglob("*.json"))

    if not json_files:
        raise RuntimeError(
            f"Không tìm thấy file JSON trong: {DATA_DIR}"
        )

    for json_path in json_files:
        try:
            data = json.loads(
                json_path.read_text(encoding="utf-8")
            )

            label = str(data.get("label", "")).strip()

            metadata = data.get("metadata", {})

            detected_hand_frames = int(
                metadata.get(
                    "detected_hand_frames",
                    0,
                )
            )

            frames = np.asarray(
                data.get("frames", []),
                dtype=np.float32,
            )

            expected_shape = (
                SEQUENCE_LENGTH,
                FRAME_VECTOR_SIZE,
            )

            if not label:
                raise ValueError("Thiếu nhãn label.")
            
            if (
                label != "no_sign"
                and detected_hand_frames < MIN_DETECTED_HAND_FRAMES
            ):
                raise ValueError(
                    "Chỉ phát hiện bàn tay trong "
                    f"{detected_hand_frames}/"
                    f"{SEQUENCE_LENGTH} frame; "
                    "yêu cầu ít nhất "
                    f"{MIN_DETECTED_HAND_FRAMES}."
                )

            if frames.shape != expected_shape:
                raise ValueError(
                    f"Shape {frames.shape}, "
                    f"yêu cầu {expected_shape}."
                )

            if not np.isfinite(frames).all():
                raise ValueError(
                    "Dữ liệu chứa NaN hoặc giá trị vô hạn."
                )

            features.append(frames.reshape(-1))
            labels.append(label)

        except Exception as error:
            skipped_files.append(
                f"{json_path}: {error}"
            )

    if skipped_files:
        print("\nCác file bị bỏ qua:")

        for skipped_file in skipped_files:
            print(f"- {skipped_file}")

    if not features:
        raise RuntimeError(
            "Không có mẫu JSON hợp lệ để huấn luyện."
        )

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)

    label_counts = Counter(
        str(label) for label in labels
    )

    return x, y, label_counts


def train_model() -> None:
    print(f"Thư mục dữ liệu: {DATA_DIR}")
    print(
        "Định dạng yêu cầu: "
        f"{SEQUENCE_LENGTH} × {FRAME_VECTOR_SIZE}"
    )

    features, labels, label_counts = load_samples()

    missing_label_text = sorted(
        set(label_counts.keys())
        - set(LABEL_TEXT.keys())
    )

    if missing_label_text:
        raise RuntimeError(
            "Các nhãn chưa có trong LABEL_TEXT: "
            f"{missing_label_text}"
        )

    print("\nSố lượng mẫu:")

    for label, count in sorted(
        label_counts.items()
    ):
        print(f"- {label}: {count}")

    if len(label_counts) < 2:
        raise RuntimeError(
            "Cần ít nhất 2 nhãn khác nhau để huấn luyện. "
            f"Hiện chỉ có: {list(label_counts.keys())}"
        )

    insufficient_labels = [
        label
        for label, count in label_counts.items()
        if count < MIN_SAMPLES_PER_LABEL
    ]

    if insufficient_labels:
        raise RuntimeError(
            "Mỗi nhãn cần tối thiểu "
            f"{MIN_SAMPLES_PER_LABEL} mẫu. "
            f"Các nhãn còn thiếu: {insufficient_labels}"
        )

    minimum_label_count = min(
        label_counts.values()
    )

    number_of_classes = len(label_counts)
    number_of_samples = len(features)

    # Số mẫu kiểm thử dự kiến nếu lấy 20%.
    test_sample_count = int(
        np.ceil(number_of_samples * 0.2)
    )

    # Khi stratify:
    # - Mỗi lớp phải có ít nhất 2 mẫu.
    # - Tập test phải chứa ít nhất 1 mẫu cho mỗi lớp.
    # - Tập train cũng phải chứa ít nhất 1 mẫu cho mỗi lớp.
    can_create_test_set = (
        minimum_label_count >= 2
        and test_sample_count >= number_of_classes
        and (
            number_of_samples - test_sample_count
            >= number_of_classes
        )
    )

    if can_create_test_set:
        x_train, x_test, y_train, y_test = (
            train_test_split(
                features,
                labels,
                test_size=test_sample_count,
                random_state=42,
                stratify=labels,
            )
        )

        has_test_set = True
        print(
            "\nĐã chia dữ liệu:"
            f"\n- Train: {len(x_train)} mẫu"
            f"\n- Test: {len(x_test)} mẫu"
        )
    else:
        print(
            "\nCảnh báo: dữ liệu quá ít để chia "
            "tập huấn luyện và kiểm thử."
        )

        print(
            f"- Tổng số mẫu: {number_of_samples}"
            f"\n- Số nhãn: {number_of_classes}"
            f"\n- Số mẫu test dự kiến: "
            f"{test_sample_count}"
            f"\n- Số mẫu ít nhất trong một nhãn: "
            f"{minimum_label_count}"
        )
        
        print(
            "Mô hình sẽ học bằng toàn bộ dữ liệu."
        )

        x_train = features
        y_train = labels
        x_test = None
        y_test = None

        has_test_set = False

    print("\nĐang huấn luyện mô hình...")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
    )

    test_accuracy = None

    if has_test_set:
        predictions = model.predict(x_test)

        test_accuracy = accuracy_score(
            y_test,
            predictions,
        )

        print(
            "\nĐộ chính xác kiểm thử: "
            f"{test_accuracy:.4f}"
        )

        print("\nBáo cáo phân loại:")

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )
    else:
        training_predictions = model.predict(
            x_train
        )

        training_accuracy = accuracy_score(
            y_train,
            training_predictions,
        )

        print(
            "\nKhông có tập kiểm thử độc lập."
        )
        print(
            "Độ chính xác trên dữ liệu huấn luyện: "
            f"{training_accuracy:.4f}"
        )

    # Huấn luyện lại mô hình cuối cùng bằng toàn bộ dữ liệu.
    # Mô hình phía trên chỉ dùng để đánh giá train/test.
    print(
        "\nĐang huấn luyện mô hình cuối cùng "
        "bằng toàn bộ dữ liệu..."
    )

    final_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    final_model.fit(
        features,
        labels,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_bundle = {
        "model": final_model,
        "label_text": LABEL_TEXT,
        "sentence_patterns": [
            {
                "labels": list(pattern),
                "text": sentence,
            }
            for pattern, sentence in SENTENCE_PATTERNS.items()
        ],
        "sequence_length": SEQUENCE_LENGTH,
        "frame_vector_size": FRAME_VECTOR_SIZE,
        "feature_count": (
            SEQUENCE_LENGTH * FRAME_VECTOR_SIZE
        ),
        "training_samples": len(features),
        "labels": sorted(label_counts.keys()),
        "label_counts": dict(label_counts),
        "has_test_set": has_test_set,
        "test_accuracy": test_accuracy,
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    print("\nĐã lưu mô hình tại:")
    print(MODEL_PATH)


if __name__ == "__main__":
    train_model()