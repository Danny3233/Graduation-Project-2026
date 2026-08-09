import { useEffect, useRef, useState } from "react";

const BrowserSpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition;

function SpeechToTextPage() {
  const recognitionRef = useRef(null);

  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState("");

  const isSupported = Boolean(BrowserSpeechRecognition);

  useEffect(() => {
    if (!isSupported) {
      return undefined;
    }

    const recognition = new BrowserSpeechRecognition();

    recognition.lang = "vi-VN";
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onresult = (event) => {
      let finalText = "";
      let interimText = "";

      for (
        let resultIndex = event.resultIndex;
        resultIndex < event.results.length;
        resultIndex += 1
      ) {
        const text = event.results[resultIndex][0].transcript;

        if (event.results[resultIndex].isFinal) {
          finalText += `${text} `;
        } else {
          interimText += text;
        }
      }

      if (finalText) {
        setTranscript((currentText) =>
          `${currentText} ${finalText}`.trim(),
        );
      }

      setInterimTranscript(interimText);
    };

    recognition.onerror = (event) => {
      setError(`Lỗi nhận dạng giọng nói: ${event.error}`);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
      setInterimTranscript("");
    };

    recognitionRef.current = recognition;

    return () => {
      recognition.stop();
    };
  }, [isSupported]);

  function startListening() {
    if (!recognitionRef.current || isListening) {
      return;
    }

    try {
      setError("");
      recognitionRef.current.start();
      setIsListening(true);
    } catch (startError) {
      setError(`Không thể bắt đầu microphone: ${startError.message}`);
    }
  }

  function stopListening() {
    if (!recognitionRef.current || !isListening) {
      return;
    }

    recognitionRef.current.stop();
    setIsListening(false);
  }

  function clearTranscript() {
    setTranscript("");
    setInterimTranscript("");
    setError("");
  }

  if (!isSupported) {
    return (
      <section className="speech-section">
        <h2>Giọng nói thành văn bản</h2>

        <p role="alert">
          Trình duyệt hiện tại không hỗ trợ nhận dạng giọng nói.
          Hãy mở ứng dụng bằng Google Chrome.
        </p>
      </section>
    );
  }

  return (
    <section className="speech-section">
      <h2>Giọng nói thành văn bản</h2>

      <p>
        Người nghe nhấn nút microphone và nói tiếng Việt.
        Nội dung sẽ được hiển thị thành văn bản.
      </p>

      <div className="button-group">
        {!isListening ? (
          <button
            type="button"
            className="primary-button"
            onClick={startListening}
          >
            Bắt đầu nghe
          </button>
        ) : (
          <button
            type="button"
            className="stop-button"
            onClick={stopListening}
          >
            Dừng nghe
          </button>
        )}

        <button
          type="button"
          className="secondary-button"
          onClick={clearTranscript}
        >
          Xóa văn bản
        </button>
      </div>

      <p className="listening-status">
        Trạng thái:{" "}
        <strong>
          {isListening ? "Đang nghe..." : "Đã dừng"}
        </strong>
      </p>

      <div className="transcript-box" aria-live="polite">
        {transcript || interimTranscript ? (
          <>
            <span>{transcript}</span>

            {interimTranscript && (
              <span className="interim-text">
                {" "}
                {interimTranscript}
              </span>
            )}
          </>
        ) : (
          <span className="placeholder-text">
            Văn bản nhận dạng sẽ xuất hiện tại đây.
          </span>
        )}
      </div>

      {error && (
        <p className="error-message" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

export default SpeechToTextPage;