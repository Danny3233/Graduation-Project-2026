import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.schemas.sign_schema import (
    HandSampleRequest,
    SaveSampleResponse,
    SaveSequenceResponse,
    SignPredictionRequest,
    SignPredictionResponse,
    SignSequenceRequest,
)

from app.services.sign_service import (
    predict_sign_sequence,
)

router = APIRouter(
    prefix="/api/sign",
    tags=["Sign Recognition"],
)

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BACKEND_DIR / "data" / "sign_samples"


@router.post(
    "/samples",
    response_model=SaveSampleResponse,
)
def save_sign_sample(
    sample: HandSampleRequest,
) -> SaveSampleResponse:
    if not sample.landmarks:
        raise HTTPException(
            status_code=400,
            detail="Không phát hiện bàn tay.",
        )

    for hand_index, hand in enumerate(sample.landmarks):
        if len(hand) != 21:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Bàn tay {hand_index + 1} phải có đúng "
                    "21 điểm landmarks."
                ),
            )

    label_directory = DATASET_DIR / sample.label
    label_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc,
    ).strftime("%Y%m%dT%H%M%S%fZ")

    filename = (
        f"{timestamp}_{uuid4().hex[:8]}.json"
    )

    output_path = label_directory / filename

    output_data = sample.model_dump()
    output_data["created_at"] = datetime.now(
        timezone.utc,
    ).isoformat()

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            output_data,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    return SaveSampleResponse(
        message="Đã lưu mẫu ký hiệu",
        label=sample.label,
        filename=filename,
        hand_count=len(sample.landmarks),
    )

@router.post(
    "/predict-sequence",
    response_model=SignPredictionResponse,
)
def predict_sequence(
    request: SignPredictionRequest,
) -> SignPredictionResponse:
    try:
        prediction = predict_sign_sequence(
            request.frames,
        )

        return SignPredictionResponse(
            label=str(prediction["label"]),
            text=str(prediction["text"]),
            confidence=float(
                prediction["confidence"],
            ),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Không thể nhận dạng ký hiệu: "
                f"{error}"
            ),
        ) from error