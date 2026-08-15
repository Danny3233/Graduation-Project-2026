import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2
import mediapipe as mp
import numpy as np

from holistic_features import (
    FRAME_VECTOR_SIZE,
    build_holistic_frame_vector,
    extract_face_feature_vector,
    extract_upper_pose_vector,
    flatten_hand_landmarks,
)

SEQUENCE_LENGTH = 30

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_HAND_MODEL_PATH = (
    PROJECT_ROOT
    / "frontend"
    / "public"
    / "models"
    / "hand_landmarker.task"
)

DEFAULT_POSE_MODEL_PATH = (
    PROJECT_ROOT
    / "frontend"
    / "public"
    / "models"
    / "pose_landmarker.task"
)

DEFAULT_FACE_MODEL_PATH = (
    PROJECT_ROOT
    / "frontend"
    / "public"
    / "models"
    / "face_landmarker.task"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "sign_sequences_v2"
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chuyển video ký hiệu thành JSON gồm "
            "tay, thân trên và khuôn mặt, kích thước 30 x 163."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Đường dẫn đến video MP4.",
    )

    parser.add_argument(
        "--label",
        required=True,
        help="Nhãn không dấu, ví dụ: xin_chao.",
    )

    parser.add_argument(
        "--hand-model",
        type=Path,
        default=DEFAULT_HAND_MODEL_PATH,
        help="Đường dẫn đến hand_landmarker.task.",
    )

    parser.add_argument(
        "--pose-model",
        type=Path,
        default=DEFAULT_POSE_MODEL_PATH,
        help="Đường dẫn đến pose_landmarker.task.",
    )

    parser.add_argument(
        "--face-model",
        type=Path,
        default=DEFAULT_FACE_MODEL_PATH,
        help="Đường dẫn đến face_landmarker.task.",
    )

    parser.add_argument(
        "--min-detected-frames",
        type=int,
        default=15,
        help=(
            "Số khung hình tối thiểu phải phát hiện được bàn tay. "
            "Mặc định: 15/30."
        ),
    )

    parser.add_argument(
        "--start-seconds",
        type=float,
        default=None,
        help=(
            "Thời điểm bắt đầu của một lần thực hiện ký hiệu. "
            "Ví dụ: 1.3."
        ),
    )

    parser.add_argument(
        "--end-seconds",
        type=float,
        default=None,
        help=(
            "Thời điểm kết thúc của một lần thực hiện ký hiệu. "
            "Ví dụ: 2.8."
        ),
    )

    return parser.parse_args()


def validate_label(label: str) -> str:
    normalized_label = label.strip().lower()

    if not re.fullmatch(r"[a-z0-9_]+", normalized_label):
        raise ValueError(
            "Nhãn chỉ được chứa chữ thường không dấu, "
            "chữ số và dấu gạch dưới."
        )

    return normalized_label


def build_frame_vector(
    hand_result,
    pose_result,
    face_result,
) -> list[float]:
    left_hand = None
    right_hand = None

    for hand_index, hand_landmarks in enumerate(
        hand_result.hand_landmarks
    ):
        hand_vector = flatten_hand_landmarks(
            hand_landmarks
        )

        hand_name = ""

        if hand_index < len(hand_result.handedness):
            categories = hand_result.handedness[hand_index]

            if categories:
                hand_name = (
                    categories[0]
                    .category_name
                    .strip()
                    .lower()
                )

        if hand_name == "left":
            left_hand = hand_vector
        elif hand_name == "right":
            right_hand = hand_vector
        elif left_hand is None:
            left_hand = hand_vector
        else:
            right_hand = hand_vector

    # Một tay luôn được xem là tay phải chính.
    if left_hand is not None and right_hand is None:
        right_hand = left_hand
        left_hand = None

    pose_landmarks = None

    if pose_result.pose_landmarks:
        pose_landmarks = pose_result.pose_landmarks[0]

    upper_pose = extract_upper_pose_vector(
        pose_landmarks
    )

    face_categories = []

    if (
        face_result is not None
        and face_result.face_blendshapes
    ):
        face_categories = (
            face_result.face_blendshapes[0]
        )

    face_features = extract_face_feature_vector(
        face_categories
    )

    return build_holistic_frame_vector(
        left_hand=left_hand,
        right_hand=right_hand,
        upper_pose=upper_pose,
        face_features=face_features,
    )

def get_video_information(
    capture: cv2.VideoCapture,
) -> tuple[int, float]:
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    if fps <= 0:
        fps = 30.0

    return total_frames, fps


def extract_sequence(
    video_path: Path,
    hand_model_path: Path,
    pose_model_path: Path,
    face_model_path: Path,
    start_seconds: Optional[float] = None,
    end_seconds: Optional[float] = None,
) -> tuple[list[list[float]], dict]:
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Không thể mở video: {video_path}"
        )

    total_frames, fps = get_video_information(capture)

    if total_frames < SEQUENCE_LENGTH:
        capture.release()

        raise ValueError(
            f"Video chỉ có {total_frames} khung hình. "
            f"Cần ít nhất {SEQUENCE_LENGTH} khung hình."
        )
    video_duration_seconds = total_frames / fps

    # Quan trọng: một mẫu huấn luyện chỉ nên chứa MỘT lần thực hiện
    # ký hiệu. Nếu video dài có nhiều lần lặp, dùng --start-seconds
    # và --end-seconds để cắt đúng một lần thực hiện.
    if start_seconds is not None or end_seconds is not None:
        segment_start = (
            0.0 if start_seconds is None
            else max(0.0, float(start_seconds))
        )
        segment_end = (
            video_duration_seconds if end_seconds is None
            else min(video_duration_seconds, float(end_seconds))
        )

        if segment_end <= segment_start:
            capture.release()
            raise ValueError(
                "--end-seconds phải lớn hơn --start-seconds."
            )

        start_frame = int(round(segment_start * fps))
        end_frame = int(round(segment_end * fps))
        start_frame = min(max(start_frame, 0), total_frames - 1)
        end_frame = min(max(end_frame, 0), total_frames - 1)
    else:
        # Chế độ tương thích cũ: bỏ 15% đầu/cuối. Chỉ nên dùng
        # khi video thực sự chỉ chứa một lần thực hiện ký hiệu.
        start_frame = int(total_frames * 0.15)
        end_frame = int(total_frames * 0.85)

    if end_frame - start_frame + 1 < SEQUENCE_LENGTH:
        capture.release()
        raise ValueError(
            "Đoạn video được chọn quá ngắn: "
            f"{end_frame - start_frame + 1} frame. "
            f"Cần ít nhất {SEQUENCE_LENGTH} frame."
        )

    # Resample đều đúng 30 frame trong MỘT lần thực hiện.
    selected_indices = np.linspace(
        start_frame,
        end_frame,
        SEQUENCE_LENGTH,
        dtype=int,
    ).tolist()

    BaseOptions = mp.tasks.BaseOptions
    RunningMode = mp.tasks.vision.RunningMode

    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )

    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = (
        mp.tasks.vision.PoseLandmarkerOptions
    )

    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = (
        mp.tasks.vision.FaceLandmarkerOptions
    )

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(hand_model_path)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(pose_model_path)
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(face_model_path)
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    frames: list[list[float]] = []

    detected_frames = 0
    detected_pose_frames = 0
    detected_face_frames = 0
    detected_face_blendshape_frames = 0

    printed_face_names = False

    current_frame_index = 0
    selected_position = 0
    previous_timestamp_ms = -1

    try:
        with (
            HandLandmarker.create_from_options(
                hand_options
            ) as hand_landmarker,
            PoseLandmarker.create_from_options(
                pose_options
            ) as pose_landmarker,
            FaceLandmarker.create_from_options(
                face_options
            ) as face_landmarker,
        ):
            while selected_position < len(selected_indices):
                success, bgr_frame = capture.read()

                if not success:
                    break

                target_index = selected_indices[
                    selected_position
                ]

                if current_frame_index == target_index:
                    rgb_frame = cv2.cvtColor(
                        bgr_frame,
                        cv2.COLOR_BGR2RGB,
                    )

                    rgb_frame = np.ascontiguousarray(
                        rgb_frame
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )

                    timestamp_ms = int(
                        current_frame_index * 1000 / fps
                    )

                    if timestamp_ms <= previous_timestamp_ms:
                        timestamp_ms = (
                            previous_timestamp_ms + 1
                        )

                    hand_result = (
                        hand_landmarker.detect_for_video(
                            mp_image,
                            timestamp_ms,
                        )
                    )

                    pose_result = (
                        pose_landmarker.detect_for_video(
                            mp_image,
                            timestamp_ms,
                        )
                    )

                    face_result = (
                        face_landmarker.detect_for_video(
                            mp_image,
                            timestamp_ms,
                        )
                    )

                    # Đếm frame nhận dạng được thân trên
                    if pose_result.pose_landmarks:
                        detected_pose_frames += 1

                    # Đếm frame nhận dạng được khuôn mặt
                    if face_result.face_landmarks:
                        detected_face_frames += 1

                    # Đếm frame có dữ liệu biểu cảm khuôn mặt
                    if face_result.face_blendshapes:
                        detected_face_blendshape_frames += 1

                        # Chỉ in danh sách blendshape một lần để kiểm tra
                        if not printed_face_names:
                            first_face_categories = (
                                face_result.face_blendshapes[0]
                            )

                            print("\nCác blendshape nhận được:")

                            for category in first_face_categories:
                                print(
                                    f"- {category.category_name}: "
                                    f"{float(category.score):.4f}"
                                )

                            printed_face_names = True

                    frame_vector = build_frame_vector(
                        hand_result,
                        pose_result,
                        face_result,
                    )

                    if len(frame_vector) != FRAME_VECTOR_SIZE:
                        raise ValueError(
                            "Kích thước frame không hợp lệ: "
                            f"{len(frame_vector)}, "
                            f"yêu cầu {FRAME_VECTOR_SIZE}."
                        )

                    frames.append(frame_vector)

                    if hand_result.hand_landmarks:
                        detected_frames += 1

                    previous_timestamp_ms = timestamp_ms
                    selected_position += 1

                current_frame_index += 1

    finally:
        capture.release()

    if len(frames) != SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Chỉ xử lý được {len(frames)}/"
            f"{SEQUENCE_LENGTH} khung hình."
        )

    metadata = {
        "original_frame_count": total_frames,
        "original_fps": fps,
        "selected_frame_count": len(frames),
        "segment_start_seconds": start_frame / fps,
        "segment_end_seconds": end_frame / fps,
        "segment_duration_seconds": (end_frame - start_frame) / fps,

        "detected_hand_frames": detected_frames,
        "detected_pose_frames": detected_pose_frames,
        "detected_face_frames": detected_face_frames,
        "detected_face_blendshape_frames": (
            detected_face_blendshape_frames
        ),

        "frame_vector_size": FRAME_VECTOR_SIZE,

        "feature_sizes": {
            "left_hand": 63,
            "right_hand": 63,
            "upper_pose": 18,
            "face_blendshapes": 17,
            "hand_presence": 2,
        },
    }

    return frames, metadata

def save_sequence_json(
    video_path: Path,
    label: str,
    frames: list[list[float]],
    metadata: dict,
) -> Path:
    label_directory = OUTPUT_ROOT / label

    label_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    safe_video_name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        video_path.stem,
    )

    output_filename = (
        f"{safe_video_name}_"
        f"{timestamp}_"
        f"{uuid4().hex[:8]}.json"
    )

    output_path = (
        label_directory / output_filename
    )

    output_data = {
        "label": label,
        "frames": frames,
        "source_video": video_path.name,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "metadata": metadata,
    }

    output_path.write_text(
        json.dumps(
            output_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    arguments = parse_arguments()

    video_path = (
        arguments.input
        .expanduser()
        .resolve()
    )

    hand_model_path = (
        arguments.hand_model
        .expanduser()
        .resolve()
    )

    pose_model_path = (
        arguments.pose_model
        .expanduser()
        .resolve()
    )

    face_model_path = (
        arguments.face_model
        .expanduser()
        .resolve()
    )

    label = validate_label(arguments.label)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy video: {video_path}"
        )

    if video_path.suffix.lower() not in {
        ".mp4",
        ".mov",
        ".m4v",
    }:
        raise ValueError(
            "Chỉ hỗ trợ video MP4, MOV hoặc M4V."
        )

    model_paths = {
        "Hand Landmarker": hand_model_path,
        "Pose Landmarker": pose_model_path,
        "Face Landmarker": face_model_path,
    }

    for model_name, current_model_path in (
        model_paths.items()
    ):
        if not current_model_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy {model_name}: "
                f"{current_model_path}"
            )

        if current_model_path.stat().st_size == 0:
            raise ValueError(
                f"File {model_name} đang rỗng: "
                f"{current_model_path}"
            )

    print(f"Đang xử lý: {video_path.name}")
    print(f"Nhãn: {label}")
    print(f"- Hand model: {hand_model_path}")
    print(f"- Pose model: {pose_model_path}")
    print(f"- Face model: {face_model_path}")

    frames, metadata = extract_sequence(
        video_path=video_path,
        hand_model_path=hand_model_path,
        pose_model_path=pose_model_path,
        face_model_path=face_model_path,
        start_seconds=arguments.start_seconds,
        end_seconds=arguments.end_seconds,
    )

    detected_frames = metadata[
        "detected_hand_frames"
    ]

    if (
        label != "no_sign"
        and detected_frames < arguments.min_detected_frames
    ):
        raise RuntimeError(
            "MediaPipe chỉ phát hiện bàn tay trong "
            f"{detected_frames}/{SEQUENCE_LENGTH} "
            "khung hình. Hãy quay video rõ hơn "
            "hoặc giảm --min-detected-frames."
        )

    output_path = save_sequence_json(
        video_path=video_path,
        label=label,
        frames=frames,
        metadata=metadata,
    )

    print("\nChuyển đổi thành công")
    print(f"- Số frame: {len(frames)}")
    print(
        "- Đoạn video: "
        f"{metadata['segment_start_seconds']:.2f}s → "
        f"{metadata['segment_end_seconds']:.2f}s "
        f"({metadata['segment_duration_seconds']:.2f}s)"
    )
    print(
        "- Frame có bàn tay: "
        f"{detected_frames}/{SEQUENCE_LENGTH}"
    )

    print(
        "- Frame có thân trên: "
        f"{metadata['detected_pose_frames']}/"
        f"{SEQUENCE_LENGTH}"
    )

    print(
        "- Frame có khuôn mặt: "
        f"{metadata['detected_face_frames']}/"
        f"{SEQUENCE_LENGTH}"
    )

    print(
        "- Frame có biểu cảm khuôn mặt: "
        f"{metadata['detected_face_blendshape_frames']}/"
        f"{SEQUENCE_LENGTH}"
    )
    
    print(
        "- Kích thước mỗi frame: "
        f"{len(frames[0])}"
    )
    print(f"- File JSON: {output_path}")


if __name__ == "__main__":
    main()