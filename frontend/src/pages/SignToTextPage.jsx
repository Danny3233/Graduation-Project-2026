import { useEffect, useRef, useState } from "react";

import {
  DrawingUtils,
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";

import { predictSignSequence } from "../services/signApi";

import {
  buildHolisticFrameVector,
  extractFaceFeatureVector,
  extractUpperPoseVector,
  flattenLandmarks,
  FRAME_VECTOR_SIZE,
} from "../utils/holisticFeatures";

const MEDIAPIPE_WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";

const HAND_MODEL_PATH = "/models/hand_landmarker.task";
const POSE_MODEL_PATH = "/models/pose_landmarker.task";
const FACE_MODEL_PATH = "/models/face_landmarker.task";

const SEQUENCE_LENGTH = 30;

// Lấy một frame sau mỗi 0,05 giây.
// 30 frame tương đương khoảng 1,5 giây.
const SAMPLE_INTERVAL_SECONDS = 0.04;

// Gửi một lần dự đoán sau khoảng 1,2 giây.
const PREDICTION_INTERVAL_SECONDS = 0.8;

// Ít nhất 20/30 frame phải nhìn thấy một hoặc hai tay.
const MIN_HAND_FRAMES = 20;

// Ngưỡng tin cậy để frontend chấp nhận kết quả.
const MIN_CONFIDENCE = 0.30;

// Cần hai kết quả liên tiếp giống nhau mới ghép vào văn bản.
const REQUIRED_STABLE_PREDICTIONS = 2;

function SignToTextPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const streamRef = useRef(null);

  const handLandmarkerRef = useRef(null);
  const poseLandmarkerRef = useRef(null);
  const faceLandmarkerRef = useRef(null);

  const animationFrameRef = useRef(null);

  const lastVideoTimeRef = useRef(-1);
  const lastSampleTimeRef = useRef(-1);
  const lastPredictionTimeRef = useRef(-1);

  const frameWindowRef = useRef([]);
  const handPresenceWindowRef = useRef([]);
  const noHandSamplesRef = useRef(0);

  const predictionRequestRef = useRef(false);
  const predictionHistoryRef = useRef([]);

  const lastCommittedLabelRef = useRef("");

  const [modelStatus, setModelStatus] = useState(
    "Đang tải mô hình MediaPipe...",
  );

  const [isCameraActive, setIsCameraActive] = useState(false);
  const [detectedHands, setDetectedHands] = useState(0);

  const [recognitionStatus, setRecognitionStatus] = useState(
    "Mở camera để bắt đầu nhận dạng tự động.",
  );

  const [latestPrediction, setLatestPrediction] =
    useState(null);

  const [recognizedText, setRecognizedText] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;

    async function initializeLandmarkers() {
      let handLandmarker = null;
      let poseLandmarker = null;
      let faceLandmarker = null;

      try {
        const vision = await FilesetResolver.forVisionTasks(
          MEDIAPIPE_WASM_URL,
        );

        handLandmarker =
          await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: HAND_MODEL_PATH,
            },
            runningMode: "VIDEO",
            numHands: 2,
            minHandDetectionConfidence: 0.4,
            minHandPresenceConfidence: 0.4,
            minTrackingConfidence: 0.4,
          });

        poseLandmarker =
          await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: POSE_MODEL_PATH,
            },
            runningMode: "VIDEO",
            numPoses: 1,
            minPoseDetectionConfidence: 0.4,
            minPosePresenceConfidence: 0.4,
            minTrackingConfidence: 0.4,
            outputSegmentationMasks: false,
          });

        faceLandmarker =
          await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: FACE_MODEL_PATH,
            },
            runningMode: "VIDEO",
            numFaces: 1,
            outputFaceBlendshapes: true,
            minFaceDetectionConfidence: 0.4,
            minFacePresenceConfidence: 0.4,
            minTrackingConfidence: 0.4,
          });

        if (!isMounted) {
          handLandmarker.close();
          poseLandmarker.close();
          faceLandmarker.close();
          return;
        }

        handLandmarkerRef.current = handLandmarker;
        poseLandmarkerRef.current = poseLandmarker;
        faceLandmarkerRef.current = faceLandmarker;

        setModelStatus(
          "Các mô hình MediaPipe đã sẵn sàng",
        );
      } catch (initializationError) {
        console.error(initializationError);

        handLandmarker?.close();
        poseLandmarker?.close();
        faceLandmarker?.close();

        setModelStatus(
          "Không thể tải đầy đủ mô hình MediaPipe",
        );

        setError(initializationError.message);
      }
    }

    void initializeLandmarkers();

    return () => {
      isMounted = false;

      if (animationFrameRef.current) {
        cancelAnimationFrame(
          animationFrameRef.current,
        );
      }

      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) => track.stop());
      }

      handLandmarkerRef.current?.close();
      poseLandmarkerRef.current?.close();
      faceLandmarkerRef.current?.close();

      handLandmarkerRef.current = null;
      poseLandmarkerRef.current = null;
      faceLandmarkerRef.current = null;
    };
  }, []);

  function resetRecognitionBuffers() {
    frameWindowRef.current = [];
    handPresenceWindowRef.current = [];
    noHandSamplesRef.current = 0;

    predictionHistoryRef.current = [];

    lastVideoTimeRef.current = -1;
    lastSampleTimeRef.current = -1;
    lastPredictionTimeRef.current = -1;

    predictionRequestRef.current = false;
  }

  async function recognizeWindow(frames) {
    try {
      const result = await predictSignSequence(frames);

      setLatestPrediction(result);

      if (result.confidence < MIN_CONFIDENCE) {
        predictionHistoryRef.current = [];

        setRecognitionStatus(
          `AI đoán gần nhất là “${result.text}”, ` +
            `độ tin cậy ${(
              result.confidence * 100
            ).toFixed(1)}%. ` +
            "Hãy thực hiện ký hiệu rõ hơn.",
        );

        return;
      }

      predictionHistoryRef.current = [
        ...predictionHistoryRef.current,
        result.label,
      ].slice(-REQUIRED_STABLE_PREDICTIONS);

      const history = predictionHistoryRef.current;

      const isStable =
        history.length === REQUIRED_STABLE_PREDICTIONS &&
        history.every((label) => label === result.label);

      if (!isStable) {
        setRecognitionStatus(
          `Đang xác nhận ký hiệu “${result.text}”...`,
        );

        return;
      }

      const isDifferentLabel =
        lastCommittedLabelRef.current !== result.label;

      if (isDifferentLabel) {
        setRecognizedText((currentText) => {
          if (!currentText.trim()) {
            return result.text;
          }

          return `${currentText} ${result.text}`;
        });

        lastCommittedLabelRef.current = result.label;

        setRecognitionStatus(
          `Đã nhận dạng và ghép: “${result.text}”.`,
        );
      } else {
        setRecognitionStatus(
          `Đã nhận dạng: “${result.text}”.`,
        );
      }
    } catch (recognitionError) {
      console.error(recognitionError);

      predictionHistoryRef.current = [];
      setRecognitionStatus("Không thể nhận dạng ký hiệu.");
      setError(recognitionError.message);
    } finally {
      predictionRequestRef.current = false;
    }
  }

  function processVideoFrame() {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    const handLandmarker = handLandmarkerRef.current;
    const poseLandmarker = poseLandmarkerRef.current;
    const faceLandmarker = faceLandmarkerRef.current;

    if (
      !video ||
      !canvas ||
      !handLandmarker ||
      !poseLandmarker ||
      !faceLandmarker ||
      !streamRef.current
    ) {
      return;
    }

    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      animationFrameRef.current =
        requestAnimationFrame(processVideoFrame);

      return;
    }

    if (video.currentTime !== lastVideoTimeRef.current) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const timestamp = video.currentTime * 1000;

      // 1. Phát hiện bàn tay
      const results = handLandmarker.detectForVideo(
        video,
        timestamp,
      );

      // Kiểm tra MediaPipe đang nhận tay trái hay tay phải
      console.log(
        results.handedness?.map(
          (item) => item?.[0]?.categoryName,
        ),
      );

      // 2. Phát hiện cơ thể
      const poseResults = poseLandmarker.detectForVideo(
        video,
        timestamp,
      );

      // 3. Phát hiện khuôn mặt và biểu cảm
      const faceResults = faceLandmarker.detectForVideo(
        video,
        timestamp,
      );

      // Lấy dữ liệu để vừa vẽ, vừa tạo vector AI
      const poseLandmarks =
        poseResults?.landmarks?.[0] ?? null;

      const faceLandmarks =
        faceResults?.faceLandmarks?.[0] ?? null;

      const faceCategories =
        faceResults?.faceBlendshapes?.[0]?.categories ?? [];
      
      const context = canvas.getContext("2d");
      const drawingUtils = new DrawingUtils(context);

      context.clearRect(
        0,
        0,
        canvas.width,
        canvas.height,
      );

      // Vẽ bàn tay
      for (const landmarks of results.landmarks) {
        drawingUtils.drawConnectors(
          landmarks,
          HandLandmarker.HAND_CONNECTIONS,
          {
            color: "#22c55e",
            lineWidth: 4,
          },
        );

        drawingUtils.drawLandmarks(landmarks, {
          color: "#ef4444",
          radius: 3,
        });
      }

      // Vẽ cơ thể
      if (poseLandmarks) {
        drawingUtils.drawConnectors(
          poseLandmarks,
          PoseLandmarker.POSE_CONNECTIONS,
          {
            color: "#2563eb",
            lineWidth: 3,
          },
        );

        drawingUtils.drawLandmarks(
          poseLandmarks,
          {
            color: "#f59e0b",
            radius: 2,
          },
        );
      }

      // Vẽ khuôn mặt
      if (faceLandmarks) {
        drawingUtils.drawConnectors(
          faceLandmarks,
          FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
          {
            color: "#a855f7",
            lineWidth: 2,
          },
        );

        drawingUtils.drawConnectors(
          faceLandmarks,
          FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
          {
            color: "#a855f7",
            lineWidth: 2,
          },
        );

        drawingUtils.drawConnectors(
          faceLandmarks,
          FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
          {
            color: "#a855f7",
            lineWidth: 2,
          },
        );

        drawingUtils.drawConnectors(
          faceLandmarks,
          FaceLandmarker.FACE_LANDMARKS_LIPS,
          {
            color: "#ec4899",
            lineWidth: 2,
          },
        );
      }


      const handCount = results.landmarks.length;

      setDetectedHands((currentCount) =>
        currentCount === handCount
          ? currentCount
          : handCount,
      );

      const shouldSample =
        lastSampleTimeRef.current < 0 ||
        video.currentTime - lastSampleTimeRef.current >=
          SAMPLE_INTERVAL_SECONDS;

      if (shouldSample) {
        /*
        * results là kết quả từ HandLandmarker.
        * poseResults là kết quả từ PoseLandmarker.
        * faceResults là kết quả từ FaceLandmarker.
        */

        let leftHandLandmarks = null;
        let rightHandLandmarks = null;

        results.landmarks.forEach((landmarks, handIndex) => {
          const handName =
            results.handedness?.[handIndex]?.[0]?.categoryName
              ?.trim()
              .toLowerCase() ?? "";

          if (handName === "left") {
            leftHandLandmarks = landmarks;
          } else if (handName === "right") {
            rightHandLandmarks = landmarks;
          } else if (!leftHandLandmarks) {
            leftHandLandmarks = landmarks;
          } else {
            rightHandLandmarks = landmarks;
          }
        });

        // Một tay luôn được lưu vào vùng tay phải.
        if (
          leftHandLandmarks &&
          !rightHandLandmarks
        ) {
          rightHandLandmarks = leftHandLandmarks;
          leftHandLandmarks = null;
        }

        const leftHandVector = leftHandLandmarks
          ? flattenLandmarks(leftHandLandmarks)
          : null;

        const rightHandVector = rightHandLandmarks
          ? flattenLandmarks(rightHandLandmarks)
          : null;

        const upperPoseVector =
          extractUpperPoseVector(poseLandmarks);

        const faceFeatureVector =
          extractFaceFeatureVector(faceCategories);

        const frameVector = buildHolisticFrameVector({
          leftHand: leftHandVector,
          rightHand: rightHandVector,
          upperPose: upperPoseVector,
          faceFeatures: faceFeatureVector,
        });

        if (frameVector.length !== FRAME_VECTOR_SIZE) {
          throw new Error(
            `Frame phải có ${FRAME_VECTOR_SIZE} giá trị, ` +
              `nhưng nhận được ${frameVector.length}.`,
          );
        }

        const hasHand =
          leftHandVector !== null ||
          rightHandVector !== null;

        if (hasHand) {
          noHandSamplesRef.current = 0;
        } else {
          noHandSamplesRef.current += 1;
        }

        if (noHandSamplesRef.current >= 5) {
          frameWindowRef.current = [];
          handPresenceWindowRef.current = [];
          predictionHistoryRef.current = [];

          lastCommittedLabelRef.current = "";

          setLatestPrediction(null);
          setRecognitionStatus(
            "Đưa một hoặc hai bàn tay vào khung hình.",
          );
        } else {
          frameWindowRef.current.push(frameVector);
          handPresenceWindowRef.current.push(hasHand);

          if (frameWindowRef.current.length > SEQUENCE_LENGTH) {
            frameWindowRef.current.shift();
            handPresenceWindowRef.current.shift();
          }
        }
        lastSampleTimeRef.current = video.currentTime;
      }

      const windowIsReady =
        frameWindowRef.current.length === SEQUENCE_LENGTH;

      const validHandFrames =
        handPresenceWindowRef.current.filter(Boolean).length;

      const predictionIntervalReached =
        lastPredictionTimeRef.current < 0 ||
        video.currentTime - lastPredictionTimeRef.current >=
          PREDICTION_INTERVAL_SECONDS;

      const canPredict =
        handCount > 0 &&
        windowIsReady &&
        validHandFrames >= MIN_HAND_FRAMES &&
        predictionIntervalReached &&
        !predictionRequestRef.current;

      if (canPredict) {
        predictionRequestRef.current = true;
        lastPredictionTimeRef.current = video.currentTime;

        const framesToPredict = frameWindowRef.current.map(
          (frame) => [...frame],
        );

        void recognizeWindow(framesToPredict);
      }

      if (handCount === 0) {
        setRecognitionStatus(
          "Đưa một hoặc hai bàn tay vào khung hình.",
        );
      } else if (!windowIsReady) {
        setRecognitionStatus(
          `Đang ghi chuyển động: ${
            frameWindowRef.current.length
          }/${SEQUENCE_LENGTH} khung hình.`,
        );
      }

      lastVideoTimeRef.current = video.currentTime;
    }

    animationFrameRef.current =
      requestAnimationFrame(processVideoFrame);
  }

  async function startCamera() {
    if (!handLandmarkerRef.current ||
        !poseLandmarkerRef.current ||
        !faceLandmarkerRef.current
    ) {
      setError("Mô hình tay, thân trên hoặc khuôn mặt chưa sẵn sàng.");
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Trình duyệt không hỗ trợ camera.");
      return;
    }

    try {
      setError("");
      resetRecognitionBuffers();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          width: {
            ideal: 1280,
          },
          height: {
            ideal: 720,
          },
          facingMode: "user",
        },
      });

      streamRef.current = stream;

      const video = videoRef.current;
      video.srcObject = stream;

      await video.play();

      setIsCameraActive(true);
      setRecognitionStatus(
        "Camera đã mở. Hệ thống đang nhận dạng tự động.",
      );

      animationFrameRef.current =
        requestAnimationFrame(processVideoFrame);
    } catch (cameraError) {
      console.error(cameraError);

      if (cameraError.name === "NotAllowedError") {
        setError("Bạn chưa cấp quyền sử dụng camera.");
      } else if (cameraError.name === "NotFoundError") {
        setError("Không tìm thấy camera.");
      } else {
        setError(`Không thể mở camera: ${cameraError.message}`);
      }
    }
  }

  function stopCamera() {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => track.stop());

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    const canvas = canvasRef.current;

    if (canvas) {
      const context = canvas.getContext("2d");
      context.clearRect(0, 0, canvas.width, canvas.height);
    }

    resetRecognitionBuffers();

    setDetectedHands(0);
    setIsCameraActive(false);
    setRecognitionStatus(
      "Camera đã tắt. Mở camera để nhận dạng.",
    );
  }

  function clearRecognizedText() {
    setRecognizedText("");
    setLatestPrediction(null);

    predictionHistoryRef.current = [];
    lastCommittedLabelRef.current = "";
  }

  return (
    <section className="feature-section">
      <h2>Ký hiệu chuyển thành văn bản</h2>

      <p>
        Sau khi mở camera, hệ thống tự động phân tích bàn tay, cơ thể và khuôn
        mặt để nhận dạng ký hiệu.
      </p>

      <div className="camera-status-grid">
        <p>
          Mô hình: <strong>{modelStatus}</strong>
        </p>

        <p>
          Số bàn tay: <strong>{detectedHands}</strong>
        </p>
      </div>

      <div className="sign-recognition-layout">
        {/* BÊN TRÁI: CAMERA */}
        <div className="sign-camera-column">
          <div className="camera-container">
            <video
              ref={videoRef}
              className="camera-video"
              autoPlay
              muted
              playsInline
            />

            <canvas
              ref={canvasRef}
              className="camera-canvas"
              aria-label="Điểm nhận diện ký hiệu"
            />

            {!isCameraActive && (
              <div className="camera-placeholder">Camera đang tắt</div>
            )}
          </div>

          <div className="button-group">
            {!isCameraActive ? (
              <button
                type="button"
                className="primary-button"
                onClick={startCamera}
                disabled={modelStatus !== "Các mô hình MediaPipe đã sẵn sàng"}
              >
                Mở camera
              </button>
            ) : (
              <button
                type="button"
                className="stop-button"
                onClick={stopCamera}
              >
                Tắt camera
              </button>
            )}
          </div>
        </div>

        {/* BÊN PHẢI: KẾT QUẢ */}
        <div className="recognition-panel">
          <h3>Nhận dạng tự động</h3>

          <p aria-live="polite">
            Trạng thái: <strong>{recognitionStatus}</strong>
          </p>

          {latestPrediction && (
            <div className="prediction-result">
              <p>
                AI đoán gần nhất: <strong>{latestPrediction.text}</strong>
              </p>

              <p>
                Nhãn dữ liệu: <strong>{latestPrediction.label}</strong>
              </p>

              <p>
                Độ tin cậy mô hình:{" "}
                <strong>
                  {(latestPrediction.confidence * 100).toFixed(1)}%
                </strong>
              </p>
            </div>
          )}

          <h3>Văn bản</h3>

          <div className="continuous-transcript" aria-live="polite">
            {recognizedText || (
              <span className="placeholder-text">
                Văn bản nhận dạng sẽ xuất hiện tại đây.
              </span>
            )}
          </div>

          <div className="button-group">
            <button
              type="button"
              className="secondary-button"
              onClick={clearRecognizedText}
              disabled={!recognizedText}
            >
              Xóa văn bản
            </button>
          </div>
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

export default SignToTextPage;