from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import boto3
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/api/feedback",
    tags=["feedback"],
)

TABLE_NAME = "deaf-usability-feedback"
REGION_NAME = "ap-southeast-1"

dynamodb = boto3.resource(
    "dynamodb",
    region_name=REGION_NAME,
)

table = dynamodb.Table(TABLE_NAME)


class FeedbackRequest(BaseModel):
    ratings: list[int] = Field(
        min_length=5,
        max_length=5,
    )


@router.post("")
def create_feedback(payload: FeedbackRequest):
    if any(
        rating < 1 or rating > 5
        for rating in payload.ratings
    ):
        raise HTTPException(
            status_code=422,
            detail="Mỗi đánh giá phải từ 1 đến 5.",
        )

    feedback_id = str(uuid4())

    average = sum(payload.ratings) / len(
        payload.ratings
    )

    item = {
        "id": feedback_id,
        "ratings": payload.ratings,
        "average": Decimal(
            str(round(average, 2))
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    table.put_item(Item=item)

    return {
        "success": True,
        "id": feedback_id,
        "average": round(average, 2),
        "message": "Đã lưu đánh giá.",
    }


@router.get("/summary")
def get_feedback_summary():
    response = table.scan()
    items = response.get("Items", [])

    while "LastEvaluatedKey" in response:
        response = table.scan(
            ExclusiveStartKey=response[
                "LastEvaluatedKey"
            ]
        )

        items.extend(
            response.get("Items", [])
        )

    if not items:
        return {
            "total": 0,
            "average": 0,
            "question_averages": [
                0,
                0,
                0,
                0,
                0,
            ],
        }

    total = len(items)
    question_totals = [0, 0, 0, 0, 0]

    for item in items:
        ratings = item["ratings"]

        for index, rating in enumerate(ratings):
            question_totals[index] += int(rating)

    question_averages = [
        round(score / total, 2)
        for score in question_totals
    ]

    overall_average = round(
        sum(question_averages)
        / len(question_averages),
        2,
    )

    return {
        "total": total,
        "average": overall_average,
        "question_averages": question_averages,
    }