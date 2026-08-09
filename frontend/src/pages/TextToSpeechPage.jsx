import { useEffect, useMemo, useState } from "react";

function TextToSpeechPage() {
  const [text, setText] = useState("");
  const [voices, setVoices] = useState([]);
  const [selectedVoiceURI, setSelectedVoiceURI] = useState("");
  const [rate, setRate] = useState(1);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError] = useState("");

  const isSupported =
    "speechSynthesis" in window &&
    "SpeechSynthesisUtterance" in window;

  useEffect(() => {
    if (!isSupported) {
      return undefined;
    }

    const synthesis = window.speechSynthesis;

    function loadVoices() {
      const availableVoices = synthesis.getVoices();

      setVoices(availableVoices);

      const vietnameseVoice = availableVoices.find((voice) =>
        voice.lang.toLowerCase().startsWith("vi"),
      );

      setSelectedVoiceURI((currentVoiceURI) => {
        if (currentVoiceURI) {
          return currentVoiceURI;
        }

        return vietnameseVoice?.voiceURI ?? availableVoices[0]?.voiceURI ?? "";
      });
    }

    loadVoices();
    synthesis.addEventListener("voiceschanged", loadVoices);

    return () => {
      synthesis.removeEventListener("voiceschanged", loadVoices);
      synthesis.cancel();
    };
  }, [isSupported]);

  const vietnameseVoices = useMemo(
    () =>
      voices.filter((voice) =>
        voice.lang.toLowerCase().startsWith("vi"),
      ),
    [voices],
  );

  const displayedVoices =
    vietnameseVoices.length > 0 ? vietnameseVoices : voices;

  function speakText() {
    const normalizedText = text.trim();

    if (!normalizedText) {
      setError("Vui lòng nhập nội dung cần phát giọng nói.");
      return;
    }

    setError("");

    const synthesis = window.speechSynthesis;

    // Hủy nội dung đang chờ để không bị phát lặp.
    synthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(normalizedText);
    const selectedVoice = voices.find(
      (voice) => voice.voiceURI === selectedVoiceURI,
    );

    utterance.lang = selectedVoice?.lang ?? "vi-VN";
    utterance.voice = selectedVoice ?? null;
    utterance.rate = Number(rate);
    utterance.pitch = 1;
    utterance.volume = 1;

    utterance.onstart = () => {
      setIsSpeaking(true);
    };

    utterance.onend = () => {
      setIsSpeaking(false);
    };

    utterance.onerror = (event) => {
      setIsSpeaking(false);

      if (event.error !== "canceled") {
        setError(`Không thể phát giọng nói: ${event.error}`);
      }
    };

    synthesis.speak(utterance);
  }

  function stopSpeaking() {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }

  function clearText() {
    window.speechSynthesis.cancel();
    setText("");
    setError("");
    setIsSpeaking(false);
  }

  if (!isSupported) {
    return (
      <section className="feature-section">
        <h2>Văn bản thành giọng nói</h2>

        <p className="error-message" role="alert">
          Trình duyệt hiện tại không hỗ trợ chức năng phát giọng nói.
        </p>
      </section>
    );
  }

  return (
    <section className="feature-section">
      <h2>Văn bản thành giọng nói</h2>

      <p>
        Người Điếc nhập nội dung cần giao tiếp. Hệ thống sẽ phát nội dung
        thành giọng nói để người nghe hiểu.
      </p>

      <label className="form-label" htmlFor="speech-text">
        Nội dung cần phát
      </label>

      <textarea
        id="speech-text"
        className="text-input"
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="Ví dụ: Xin chào, tôi cần được hỗ trợ."
        rows={6}
        maxLength={1000}
      />

      <div className="character-count">
        {text.length}/1000 ký tự
      </div>

      <div className="settings-grid">
        <div>
          <label className="form-label" htmlFor="voice-select">
            Giọng đọc
          </label>

          <select
            id="voice-select"
            className="select-input"
            value={selectedVoiceURI}
            onChange={(event) =>
              setSelectedVoiceURI(event.target.value)
            }
          >
            {displayedVoices.length === 0 && (
              <option value="">Giọng mặc định của hệ thống</option>
            )}

            {displayedVoices.map((voice) => (
              <option key={voice.voiceURI} value={voice.voiceURI}>
                {voice.name} — {voice.lang}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="form-label" htmlFor="speech-rate">
            Tốc độ đọc: {rate}
          </label>

          <input
            id="speech-rate"
            type="range"
            min="0.5"
            max="1.5"
            step="0.1"
            value={rate}
            onChange={(event) => setRate(event.target.value)}
          />
        </div>
      </div>

      <div className="button-group">
        <button
          type="button"
          className="primary-button"
          onClick={speakText}
          disabled={isSpeaking}
        >
          {isSpeaking ? "Đang phát..." : "Phát giọng nói"}
        </button>

        <button
          type="button"
          className="stop-button"
          onClick={stopSpeaking}
          disabled={!isSpeaking}
        >
          Dừng phát
        </button>

        <button
          type="button"
          className="secondary-button"
          onClick={clearText}
        >
          Xóa nội dung
        </button>
      </div>

      <p className="speech-status" aria-live="polite">
        Trạng thái:{" "}
        <strong>{isSpeaking ? "Đang phát giọng nói" : "Sẵn sàng"}</strong>
      </p>

      {error && (
        <p className="error-message" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

export default TextToSpeechPage;