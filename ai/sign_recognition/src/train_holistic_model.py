import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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
MIN_DETECTED_HAND_FRAMES = 15

LABEL_TEXT = {
    "xin_chao": "xin chào",
    "giup_do": "giúp đỡ", 
    "cam_on": "cảm ơn",
    "co": "có",
    "cong_nghe": "công nghệ",
    "khong": "không",
    "khong_cho": "không cho",
    "khong_biet": "không biết",
    "can": "cần",
    "cho_1": "chợ",
    "day": "dạy",
    "muon": "muốn",
    "di": "đi",
    "uong": "uống",
    "an": "ăn",
    "yeu": "yêu",
    "ban": "bạn",
    "ban_1": "bận",
    "buon": "buồn",
    "biet": "biết",
    "toi": "tôi",
    "thich": "thích",
    "hoc": "học",
    "lam": "làm",
    "nha": "nhà",
    "truong": "trường",
    "sieu_thi": "siêu thị",
    "ve": "về",
    "vui": "vui",
    "den": "đến",
    "no_sign": "",
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

def load_samples() -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Counter,
]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {DATA_DIR}"
        )

    features: List[np.ndarray] = []
    labels: List[str] = []
    groups: List[str] = []
    skipped_files: List[str] = []

    json_files = sorted(DATA_DIR.rglob("*.json"))

    if not json_files:
        raise RuntimeError(
            f"Không tìm thấy file JSON trong: {DATA_DIR}"
        )

    for index, json_path in enumerate(
        json_files,
        start=1,
    ):
        print(
            f"[{index}/{len(json_files)}] "
            f"Đang xử lý: {json_path.name}",
            flush=True,
        )
        
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

            print(
                f"    ✓ Hợp lệ | "
                f"label={label} | "
                f"shape={frames.shape} | "
                f"hands={detected_hand_frames}/{SEQUENCE_LENGTH}",
                flush=True,
            )

            source_video = str(
                data.get("source_video", "")
            ).strip()

            if not source_video:
                source_video = json_path.name

            features.append(frames.reshape(-1))
            labels.append(label)

            # Các đoạn cắt từ cùng một video nguồn phải ở cùng
            # một phía train hoặc test để tránh rò rỉ dữ liệu.
            groups.append(
                f"{label}:{source_video}"
            )

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
    group_array = np.asarray(groups)

    label_counts = Counter(
        str(label) for label in labels
    )

    return x, y, group_array, label_counts


def create_group_aware_split(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
):
    rng = np.random.default_rng(42)

    train_indices = []
    test_indices = []

    unique_labels = sorted(
        set(labels.tolist())
    )

    for label in unique_labels:
        label_indices = np.where(
            labels == label
        )[0]

        label_groups = sorted(
            set(groups[label_indices].tolist())
        )

        # Cần ít nhất 2 video nguồn cho nhãn này.
        if len(label_groups) < 2:
            return None

        shuffled_groups = list(label_groups)
        rng.shuffle(shuffled_groups)

        # Ít nhất 1 video cho test,
        # nhưng luôn để lại ít nhất 1 video cho train.
        test_group_count = max(
            1,
            round(len(shuffled_groups) * 0.2),
        )

        test_group_count = min(
            test_group_count,
            len(shuffled_groups) - 1,
        )

        test_groups = set(
            shuffled_groups[:test_group_count]
        )

        for index in label_indices:
            if groups[index] in test_groups:
                test_indices.append(index)
            else:
                train_indices.append(index)

    train_indices = np.asarray(
        train_indices,
        dtype=int,
    )

    test_indices = np.asarray(
        test_indices,
        dtype=int,
    )

    if (
        len(train_indices) == 0
        or len(test_indices) == 0
    ):
        return None

    return train_indices, test_indices


def train_model() -> None:
    print(f"Thư mục dữ liệu: {DATA_DIR}")
    print(
        "Định dạng yêu cầu: "
        f"{SEQUENCE_LENGTH} × {FRAME_VECTOR_SIZE}"
    )

    features, labels, groups, label_counts = load_samples()

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

    number_of_classes = len(label_counts)
    number_of_samples = len(features)

    split_indices = create_group_aware_split(
        features,
        labels,
        groups,
    )

    if split_indices is not None:
        train_indices, test_indices = split_indices

        x_train = features[train_indices]
        x_test = features[test_indices]
        y_train = labels[train_indices]
        y_test = labels[test_indices]

        has_test_set = True

        print(
            "\nĐã chia dữ liệu theo video nguồn:"
            f"\n- Train: {len(x_train)} mẫu"
            f"\n- Test: {len(x_test)} mẫu"
            f"\n- Số nhãn: {number_of_classes}"
        )
    else:
        print(
            "\nCảnh báo: chưa thể tạo tập test độc lập "
            "theo video nguồn."
        )
        print(
            "Mỗi nhãn nên có ít nhất 2 video nguồn khác nhau "
            "để tránh đưa các đoạn cắt gần giống nhau vào cả "
            "train và test."
        )
        print(
            f"- Tổng số mẫu: {number_of_samples}"
            f"\n- Số nhãn: {number_of_classes}"
        )
        print(
            "Mô hình sẽ học bằng toàn bộ dữ liệu; "
            "không báo test accuracy độc lập."
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

        # Chi tiết từng mẫu test
        print("\nChi tiết dự đoán test:")

        for true_label, pred_label in zip(
            y_test,
            predictions,
        ):
            print(
                f"TRUE={true_label:12s} "
                f"PRED={pred_label}"
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

        "sequence_length": SEQUENCE_LENGTH,
        "frame_vector_size": FRAME_VECTOR_SIZE,
        "feature_count": SEQUENCE_LENGTH * FRAME_VECTOR_SIZE,

        "preprocessing": {
            "normalization": "shoulder_center",
            "scale": "shoulder_distance",
            "hand_representation": "left_right_preserved",
        },

        "label_text": LABEL_TEXT,
        "sentence_patterns": [
            {
                "labels": list(pattern),
                "text": sentence,
            }
            for pattern, sentence in SENTENCE_PATTERNS.items()
        ],

        "training_samples": len(features),
        "labels": sorted(label_counts.keys()),
        "label_counts": dict(label_counts),

        "has_test_set": has_test_set,
        "test_accuracy": test_accuracy,

        "evaluation_split": (
            "group_by_source_video"
            if has_test_set
            else None
        ),
        
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