import { useEffect, useRef, useState } from "react";

function SpeechToTextPage() {
  const recognitionRef = useRef(null);
  const keepListeningRef = useRef(false);
  const speechStartTimeRef = useRef(null);
  const firstResultCapturedRef = useRef(false);

  const [isSupported] = useState(() => {
    if (typeof window === "undefined") {
      return false;
    }

    return Boolean(
      window.SpeechRecognition ||
        window.webkitSpeechRecognition,
    );
  });

  const [isListening, setIsListening] = useState(false);

  const [finalText, setFinalText] = useState("");
  const [interimText, setInterimText] = useState("");

  const [status, setStatus] = useState(() => {
    if (typeof window === "undefined") {
      return "Trình duyệt không hỗ trợ nhận dạng giọng nói";
    }

    const supported = Boolean(
      window.SpeechRecognition ||
        window.webkitSpeechRecognition,
    );

    return supported
      ? "Sẵn sàng"
      : "Trình duyệt không hỗ trợ nhận dạng giọng nói";
  });

  const [responseTime, setResponseTime] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition ||
      window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      return undefined;
    }

    const recognition = new SpeechRecognition();

    recognition.lang = "vi-VN";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsListening(true);
      setStatus("Micro đang hoạt động");
      setError("");
    };

    recognition.onspeechstart = () => {
      setStatus("Đang nghe giọng nói...");

      speechStartTimeRef.current = performance.now();
      firstResultCapturedRef.current = false;
    };

    recognition.onresult = (event) => {
      let newFinalText = "";
      let newInterimText = "";

      for (
        let index = event.resultIndex;
        index < event.results.length;
        index += 1
      ) {
        const result = event.results[index];
        const transcript = result[0].transcript;

        if (result.isFinal) {
          newFinalText += transcript;
        } else {
          newInterimText += transcript;
        }
      }

      if (
        !firstResultCapturedRef.current &&
        speechStartTimeRef.current !== null &&
        (newFinalText || newInterimText)
      ) {
        const elapsed =
          (performance.now() -
            speechStartTimeRef.current) /
          1000;

        setResponseTime(elapsed);
        firstResultCapturedRef.current = true;
      }

      if (newFinalText.trim()) {
        setFinalText((previousText) => {
          const previous = previousText.trim();
          const current = newFinalText.trim();

          return previous
            ? `${previous} ${current}`
            : current;
        });
      }

      setInterimText(newInterimText);
    };

    recognition.onspeechend = () => {
      setStatus("Đang xử lý...");
    };

    recognition.onerror = (event) => {
      if (event.error === "no-speech") {
        setStatus("Chưa phát hiện giọng nói");
        return;
      }

      if (event.error === "not-allowed") {
        setError(
          "Trình duyệt chưa được cấp quyền sử dụng microphone.",
        );
      } else if (event.error === "audio-capture") {
        setError("Không tìm thấy microphone.");
      } else {
        setError(
          `Lỗi nhận dạng giọng nói: ${event.error}`,
        );
      }

      setStatus("Có lỗi xảy ra");
    };

    recognition.onend = () => {
      if (keepListeningRef.current) {
        setTimeout(() => {
          try {
            recognition.start();
          } catch {
            // Recognition có thể đang tự khởi động lại.
          }
        }, 250);
      } else {
        setIsListening(false);
        setStatus("Đã dừng");
      }
    };

    recognitionRef.current = recognition;

    return () => {
      keepListeningRef.current = false;

      try {
        recognition.abort();
      } catch {
        // Không cần xử lý khi recognition đã dừng.
      }
    };
  }, []);

  const startListening = () => {
    if (!recognitionRef.current) {
      return;
    }

    setError("");
    keepListeningRef.current = true;

    try {
      recognitionRef.current.start();
    } catch {
      setStatus("Micro đang hoạt động");
    }
  };

  const stopListening = () => {
    keepListeningRef.current = false;

    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }

    setIsListening(false);
    setStatus("Đã dừng");
  };

  const clearText = () => {
    setFinalText("");
    setInterimText("");
    setResponseTime(null);
    setError("");
  };

  return (
    <section className="feature-section">
      <h2>Giọng nói chuyển thành văn bản</h2>

      <p>
        Hệ thống nhận giọng nói tiếng Việt từ microphone
        và hiển thị nội dung dưới dạng văn bản theo thời
        gian thực.
      </p>

      <div className="speech-status-grid">
        <p>
          Ngôn ngữ: <strong>Tiếng Việt</strong>
        </p>

        <p>
          Trạng thái: <strong>{status}</strong>
        </p>

        <p>
          Thời gian phản hồi:{" "}
          <strong>
            {responseTime !== null
              ? `${responseTime.toFixed(2)} giây`
              : "--"}
          </strong>
        </p>
      </div>

      {!isSupported && (
        <p className="error-message">
          Trình duyệt hiện tại không hỗ trợ chức năng
          nhận dạng giọng nói.
        </p>
      )}

      <div className="button-group">
        {!isListening ? (
          <button
            type="button"
            className="primary-button"
            onClick={startListening}
            disabled={!isSupported}
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
          onClick={clearText}
          disabled={!finalText && !interimText}
        >
          Xóa văn bản
        </button>
      </div>

      <div className="speech-result-panel">
        <h3>Văn bản nhận dạng</h3>

        <div
          className="continuous-transcript"
          aria-live="polite"
        >
          {!finalText && !interimText ? (
            <span className="placeholder-text">
              Nội dung lời nói sẽ xuất hiện tại đây.
            </span>
          ) : (
            <>
              <span>{finalText}</span>

              {interimText && (
                <span className="interim-transcript">
                  {" "}
                  {interimText}
                </span>
              )}
            </>
          )}
        </div>
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