const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000"
).replace(/\/$/, "");

export async function submitUsabilityFeedback(
  ratings,
) {
  const response = await fetch(
    `${API_BASE_URL}/api/feedback`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ratings,
      }),
    },
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Không thể gửi đánh giá.",
    );
  }

  return data;
}