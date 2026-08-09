LANDMARKS_PER_HAND = 21
VALUES_PER_LANDMARK = 3

HAND_VECTOR_SIZE = (
    LANDMARKS_PER_HAND * VALUES_PER_LANDMARK
)  # 63

UPPER_POSE_INDICES = [
    11,  # Vai trái
    12,  # Vai phải
    13,  # Khuỷu tay trái
    14,  # Khuỷu tay phải
    15,  # Cổ tay trái
    16,  # Cổ tay phải
]

UPPER_POSE_VECTOR_SIZE = 18

SELECTED_FACE_BLENDSHAPES = [
    "browInnerUp",
    "browDownLeft",
    "browDownRight",
    "browOuterUpLeft",
    "browOuterUpRight",
    "eyeWideLeft",
    "eyeWideRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "jawOpen",
    "mouthPucker",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthPressLeft",
    "mouthPressRight",
]

FACE_FEATURE_SIZE = len(
    SELECTED_FACE_BLENDSHAPES
)  # 17

FRAME_VECTOR_SIZE = (
    HAND_VECTOR_SIZE * 2
    + UPPER_POSE_VECTOR_SIZE
    + FACE_FEATURE_SIZE
    + 2
)  # 163

def empty_vector(size: int) -> list[float]:
    return [0.0] * size

# 1. Ban tay 
def flatten_hand_landmarks(
    hand_landmarks,
) -> list[float]:
    vector: list[float] = []

    for landmark in hand_landmarks:
        vector.extend(
            [
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
            ]
        )

    if len(vector) != HAND_VECTOR_SIZE:
        raise ValueError(
            f"Mỗi bàn tay phải có "
            f"{HAND_VECTOR_SIZE} giá trị, "
            f"nhưng nhận được {len(vector)}."
        )

    return vector

# 2. Thân trên
def extract_upper_pose_vector(
    pose_landmarks,
) -> list[float]:
    if not pose_landmarks:
        return empty_vector(
            UPPER_POSE_VECTOR_SIZE
        )

    vector: list[float] = []

    for index in UPPER_POSE_INDICES:
        if index >= len(pose_landmarks):
            return empty_vector(
                UPPER_POSE_VECTOR_SIZE
            )

        landmark = pose_landmarks[index]

        vector.extend(
            [
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
            ]
        )

    if len(vector) != UPPER_POSE_VECTOR_SIZE:
        return empty_vector(
            UPPER_POSE_VECTOR_SIZE
        )

    return vector

# 3. Khuôn mặt
def extract_face_feature_vector(
    face_categories,
) -> list[float]:
    scores: dict[str, float] = {}

    for category in face_categories or []:
        category_name = (
            getattr(category, "category_name", None)
            or getattr(category, "display_name", None)
            or ""
        )

        normalized_name = (
            str(category_name)
            .strip()
            .lower()
        )

        category_score = getattr(
            category,
            "score",
            0.0,
        )

        scores[normalized_name] = float(
            category_score or 0.0
        )

    return [
        scores.get(name.lower(), 0.0)
        for name in SELECTED_FACE_BLENDSHAPES
    ]

def build_holistic_frame_vector(
    left_hand,
    right_hand,
    upper_pose,
    face_features,
) -> list[float]:
    has_left_hand = (
        left_hand is not None
        and len(left_hand) == HAND_VECTOR_SIZE
    )

    has_right_hand = (
        right_hand is not None
        and len(right_hand) == HAND_VECTOR_SIZE
    )

    left_vector = (
        left_hand
        if has_left_hand
        else empty_vector(HAND_VECTOR_SIZE)
    )

    right_vector = (
        right_hand
        if has_right_hand
        else empty_vector(HAND_VECTOR_SIZE)
    )

    pose_vector = (
        upper_pose
        if upper_pose is not None
        and len(upper_pose)
        == UPPER_POSE_VECTOR_SIZE
        else empty_vector(
            UPPER_POSE_VECTOR_SIZE
        )
    )

    face_vector = (
        face_features
        if face_features is not None
        and len(face_features)
        == FACE_FEATURE_SIZE
        else empty_vector(
            FACE_FEATURE_SIZE
        )
    )

    frame_vector = [
        *left_vector,
        *right_vector,
        *pose_vector,
        *face_vector,
        1.0 if has_left_hand else 0.0,
        1.0 if has_right_hand else 0.0,
    ]

    if len(frame_vector) != FRAME_VECTOR_SIZE:
        raise ValueError(
            f"Frame phải có "
            f"{FRAME_VECTOR_SIZE} giá trị, "
            f"nhưng nhận được "
            f"{len(frame_vector)}."
        )

    return frame_vector