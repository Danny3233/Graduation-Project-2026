import { useState } from "react";

import SpeechToTextPage from "./pages/SpeechToTextPage";
import TextToSpeechPage from "./pages/TextToSpeechPage";
import SignToTextPage from "./pages/SignToTextPage";

function App() {
  const [selectedFeature, setSelectedFeature] =
    useState("sign-to-text");

  return (
    <main>
      <header>
        <h1>Hệ thống hỗ trợ giao tiếp</h1>

        <p>
          Dành cho cộng đồng người Điếc và khiếm thính
        </p>
      </header>

      <section className="feature-selector">
        <h2>Chọn chức năng</h2>

        <div className="feature-buttons">
          <button
            type="button"
            onClick={() =>
              setSelectedFeature("sign-to-text")
            }
          >
            🖐 Ký hiệu → Văn bản
          </button>

          <button
            type="button"
            onClick={() =>
              setSelectedFeature("speech-to-text")
            }
          >
            🎤 Giọng nói → Văn bản
          </button>

          <button
            type="button"
            onClick={() =>
              setSelectedFeature("text-to-speech")
            }
          >
            🔊 Văn bản → Giọng nói
          </button>
        </div>
      </section>

      {selectedFeature === "sign-to-text" && (
        <SignToTextPage />
      )}

      {selectedFeature === "speech-to-text" && (
        <SpeechToTextPage />
      )}

      {selectedFeature === "text-to-speech" && (
        <TextToSpeechPage />
      )}
    </main>
  );
}

export default App;