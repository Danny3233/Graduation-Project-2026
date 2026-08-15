import { useState } from "react";

import { submitUsabilityFeedback } from "../services/feedbackApi";

const QUESTIONS = [
  "Giao diện dễ hiểu",
  "Các chức năng dễ sử dụng",
  "Văn bản hiển thị dễ đọc",
  "Tốc độ phản hồi phù hợp",
  "Hệ thống hữu ích khi giao tiếp thực tế",
];

function UsabilityFeedback() {
  const [ratings, setRatings] = useState(
    Array(QUESTIONS.length).fill(0),
  );

  const [submitted, setSubmitted] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");

  const handleRating = (questionIndex, value) => {
    setRatings((previous) => {
      const updated = [...previous];
      updated[questionIndex] = value;
      return updated;
    });

    setSubmitted(false);
    setError("");
  };

  const handleSubmit = async () => {
    const completed = ratings.every(
      (rating) => rating > 0,
    );

    if (!completed) {
      return;
    }

    try {
      setIsSubmitting(true);
      setError("");
      setSubmitted(false);

      await submitUsabilityFeedback(ratings);

      setSubmitted(true);
      setRatings(
        Array(QUESTIONS.length).fill(0),
      );
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const isComplete = ratings.every(
    (rating) => rating > 0,
  );

  return (
    <section className="usability-section">
      <h2>Phản hồi trải nghiệm</h2>

      <p>
        Sau khi sử dụng hệ thống, hãy đánh giá mức độ
        dễ sử dụng từ 1 đến 5.
      </p>

      <div className="usability-scale">
        <span>1 = Rất khó</span>
        <span>5 = Rất dễ / Rất tốt</span>
      </div>

      {QUESTIONS.map((question, index) => (
        <div
          className="usability-question"
          key={question}
        >
          <p>
            <strong>
              {index + 1}. {question}
            </strong>
          </p>

          <div className="rating-buttons">
            {[1, 2, 3, 4, 5].map((value) => (
              <button
                type="button"
                key={value}
                className={
                  ratings[index] === value
                    ? "rating-button selected"
                    : "rating-button"
                }
                onClick={() =>
                  handleRating(index, value)
                }
              >
                {value}
              </button>
            ))}
          </div>
        </div>
      ))}

      <button
        type="button"
        className="primary-button"
        onClick={handleSubmit}
        disabled={!isComplete || isSubmitting}
      >
        {isSubmitting
          ? "Đang gửi..."
          : "Gửi đánh giá"}
      </button>

      {submitted && (
        <p className="feedback-success">
          Cảm ơn bạn đã đánh giá hệ thống.
        </p>
      )}

      {error && (
        <p className="error-message" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

export default UsabilityFeedback;