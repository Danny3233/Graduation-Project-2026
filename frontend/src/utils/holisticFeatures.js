export const LANDMARKS_PER_HAND = 21;
export const VALUES_PER_LANDMARK = 3;

export const HAND_VECTOR_SIZE =
  LANDMARKS_PER_HAND * VALUES_PER_LANDMARK; // 63

export const UPPER_POSE_INDICES = [
  11, // Vai trái
  12, // Vai phải
  13, // Khuỷu tay trái
  14, // Khuỷu tay phải
  15, // Cổ tay trái
  16, // Cổ tay phải
];

export const UPPER_POSE_VECTOR_SIZE =
  UPPER_POSE_INDICES.length *
  VALUES_PER_LANDMARK; // 18

export const SELECTED_FACE_BLENDSHAPES = [
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
];

export const FACE_FEATURE_SIZE =
  SELECTED_FACE_BLENDSHAPES.length; // 17

export const HAND_PRESENCE_SIZE = 2;

export const FRAME_VECTOR_SIZE =
  HAND_VECTOR_SIZE * 2 +
  UPPER_POSE_VECTOR_SIZE +
  FACE_FEATURE_SIZE +
  HAND_PRESENCE_SIZE; // 163

export function createEmptyVector(size) {
  return Array(size).fill(0);
}

export function flattenLandmarks(landmarks) {
  if (!Array.isArray(landmarks)) {
    return [];
  }

  return landmarks.flatMap((landmark) => [
    Number(landmark?.x ?? 0),
    Number(landmark?.y ?? 0),
    Number(landmark?.z ?? 0),
  ]);
}

export function extractUpperPoseVector(
  poseLandmarks,
) {
  if (!Array.isArray(poseLandmarks)) {
    return createEmptyVector(
      UPPER_POSE_VECTOR_SIZE,
    );
  }

  const vector = [];

  for (const index of UPPER_POSE_INDICES) {
    const landmark = poseLandmarks[index];

    if (!landmark) {
      return createEmptyVector(
        UPPER_POSE_VECTOR_SIZE,
      );
    }

    vector.push(
      Number(landmark.x ?? 0),
      Number(landmark.y ?? 0),
      Number(landmark.z ?? 0),
    );
  }

  if (
    vector.length !==
    UPPER_POSE_VECTOR_SIZE
  ) {
    return createEmptyVector(
      UPPER_POSE_VECTOR_SIZE,
    );
  }

  return vector;
}

export function extractFaceFeatureVector(
  faceCategories,
) {
  const scores = new Map();

  for (const category of faceCategories ?? []) {
    const name = String(
      category?.categoryName ?? "",
    )
      .trim()
      .toLowerCase();

    const score = Number(
      category?.score ?? 0,
    );

    scores.set(name, score);
  }

  return SELECTED_FACE_BLENDSHAPES.map(
    (blendshapeName) =>
      scores.get(
        blendshapeName.toLowerCase(),
      ) ?? 0,
  );
}

function normalizeVector(
  vector,
  expectedSize,
) {
  if (
    !Array.isArray(vector) ||
    vector.length !== expectedSize
  ) {
    return createEmptyVector(expectedSize);
  }

  return vector.map((value) => {
    const numericValue = Number(value);

    return Number.isFinite(numericValue)
      ? numericValue
      : 0;
  });
}

export function buildHolisticFrameVector({
  leftHand = null,
  rightHand = null,
  upperPose = null,
  faceFeatures = null,
}) {
  const hasLeftHand =
    Array.isArray(leftHand) &&
    leftHand.length === HAND_VECTOR_SIZE;

  const hasRightHand =
    Array.isArray(rightHand) &&
    rightHand.length === HAND_VECTOR_SIZE;

  const leftHandVector = normalizeVector(
    leftHand,
    HAND_VECTOR_SIZE,
  );

  const rightHandVector = normalizeVector(
    rightHand,
    HAND_VECTOR_SIZE,
  );

  const upperPoseVector = normalizeVector(
    upperPose,
    UPPER_POSE_VECTOR_SIZE,
  );

  const faceVector = normalizeVector(
    faceFeatures,
    FACE_FEATURE_SIZE,
  );

  const frameVector = [
    ...leftHandVector,
    ...rightHandVector,
    ...upperPoseVector,
    ...faceVector,
    hasLeftHand ? 1 : 0,
    hasRightHand ? 1 : 0,
  ];

  if (
    frameVector.length !==
    FRAME_VECTOR_SIZE
  ) {
    throw new Error(
      `Frame phải có ${FRAME_VECTOR_SIZE} giá trị, ` +
        `nhưng nhận được ${frameVector.length}.`,
    );
  }

  return frameVector;
}