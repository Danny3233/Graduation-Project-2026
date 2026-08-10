const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000"
).replace(/\/$/, "");

export async function predictSignSequence(frames) {
  const response = await fetch(
    `${API_BASE_URL}/api/sign/predict-sequence`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        frames,
      }),
    },
  );

  const data = await response.json();

  if (!response.ok) {
    const message =
      typeof data.detail === "string"
        ? data.detail
        : "Không thể nhận dạng ký hiệu.";

    throw new Error(message);
  }

  return data;
}