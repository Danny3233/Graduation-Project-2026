from pydantic import BaseModel, Field, field_validator


SEQUENCE_LENGTH = 30
FRAME_VECTOR_SIZE = 163


class Landmark(BaseModel):
    x: float
    y: float
    z: float


class HandSampleRequest(BaseModel):
    label: str = Field(
        min_length=1,
        max_length=50,
        pattern=r"^[a-z0-9_]+$",
    )

    # Có thể chứa một hoặc hai bàn tay.
    # Mỗi bàn tay phải có đúng 21 điểm.
    landmarks: list[list[Landmark]]


class SaveSampleResponse(BaseModel):
    message: str
    label: str
    filename: str
    hand_count: int


class SignSequenceRequest(BaseModel):
    label: str = Field(
        min_length=1,
        max_length=50,
        pattern=r"^[a-z0-9_]+$",
    )

    frames: list[list[float]]

    @field_validator("frames")
    @classmethod
    def validate_frames(
        cls,
        frames: list[list[float]],
    ) -> list[list[float]]:
        if len(frames) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Mỗi mẫu phải có đúng "
                f"{SEQUENCE_LENGTH} khung hình."
            )

        for frame_index, frame in enumerate(frames):
            if len(frame) != FRAME_VECTOR_SIZE:
                raise ValueError(
                    f"Khung hình {frame_index + 1} phải có đúng "
                    f"{FRAME_VECTOR_SIZE} giá trị."
                )

        return frames


class SaveSequenceResponse(BaseModel):
    message: str
    label: str
    filename: str
    frame_count: int


class SignPredictionRequest(BaseModel):
    frames: list[list[float]]

    @field_validator("frames")
    @classmethod
    def validate_prediction_frames(
        cls,
        frames: list[list[float]],
    ) -> list[list[float]]:
        if len(frames) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Phải gửi đúng "
                f"{SEQUENCE_LENGTH} khung hình."
            )

        for frame_index, frame in enumerate(frames):
            if len(frame) != FRAME_VECTOR_SIZE:
                raise ValueError(
                    f"Khung hình {frame_index + 1} phải có "
                    f"{FRAME_VECTOR_SIZE} giá trị."
                )

        return frames


class SignPredictionResponse(BaseModel):
    label: str
    text: str
    confidence: float