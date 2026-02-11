import { useState, useCallback, useEffect } from "react";
import { preprocessImage, waitForOpenCV } from "@/lib/preprocessing";
import { modelManager, MODEL_CONFIGS } from "@/lib/model-manager";
import type { AppState, PredictionResult } from "@/types";
import { SAMPLE_IMAGE_URL } from "@/constants";

export function useClassifier() {
  const [selectedModel, setSelectedModel] = useState("custom-cnn");
  const [appState, setAppState] = useState<AppState>("idle");
  const [loadProgress, setLoadProgress] = useState(0);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [originalCanvas, setOriginalCanvas] =
    useState<HTMLCanvasElement | null>(null);
  const [claheCanvas, setClaheCanvas] = useState<HTMLCanvasElement | null>(
    null,
  );
  const [opencvReady, setOpencvReady] = useState(false);
  const [imageFile, setImageFile] = useState<File | null>(null);

  const currentConfig = MODEL_CONFIGS[selectedModel]!;

  useEffect(() => {
    waitForOpenCV()
      .then(() => setOpencvReady(true))
      .catch(() => setOpencvReady(false));
  }, []);

  const runAnalysis = useCallback(
    async (file: File) => {
      try {
        setErrorMessage("");
        setResult(null);

        // Preprocess first so the image displays even if model loading fails
        setAppState("preprocessing");
        const config = MODEL_CONFIGS[selectedModel]!;
        const preprocessed = await preprocessImage(file, config.colorMode);
        setOriginalCanvas(preprocessed.originalCanvas);
        setClaheCanvas(preprocessed.claheCanvas);

        if (!modelManager.isLoaded(selectedModel)) {
          setAppState("loading-model");
          setLoadProgress(0);
          await modelManager.loadModel(selectedModel, (progress) => {
            setLoadProgress(Math.round(progress * 100));
          });
        }

        setAppState("analyzing");
        const prediction = await modelManager.predict(
          selectedModel,
          preprocessed.tensorData,
          preprocessed.shape,
        );
        setResult(prediction);
        setAppState("result");
      } catch (err) {
        setErrorMessage(
          err instanceof Error ? err.message : "An unexpected error occurred",
        );
        setAppState("error");
      }
    },
    [selectedModel],
  );

  const handleModelChange = useCallback(
    (modelId: string) => {
      setSelectedModel(modelId);
      setResult(null);
      setOriginalCanvas(null);
      setClaheCanvas(null);
      if (
        appState === "result" ||
        appState === "error" ||
        appState === "ready"
      ) {
        setAppState("idle");
      }
    },
    [appState],
  );

  const handleFileSelect = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) {
        setErrorMessage("Please select a valid image file (JPEG, PNG, etc.)");
        setAppState("error");
        return;
      }
      setImageFile(file);
      runAnalysis(file);
    },
    [runAnalysis],
  );

  const handleSampleImage = useCallback(async () => {
    try {
      setAppState("preprocessing");
      const response = await fetch(SAMPLE_IMAGE_URL);
      const blob = await response.blob();
      const file = new File([blob], "sample-xray.jpg", { type: "image/jpeg" });
      setImageFile(file);
      runAnalysis(file);
    } catch {
      setErrorMessage("Failed to fetch sample image. Please upload your own.");
      setAppState("error");
    }
  }, [runAnalysis]);

  const handleReanalyze = useCallback(() => {
    if (imageFile) {
      runAnalysis(imageFile);
    }
  }, [imageFile, runAnalysis]);

  const isBusy =
    appState === "loading-model" ||
    appState === "analyzing" ||
    appState === "preprocessing";

  return {
    selectedModel,
    appState,
    loadProgress,
    result,
    errorMessage,
    originalCanvas,
    claheCanvas,
    opencvReady,
    imageFile,
    currentConfig,
    isBusy,
    handleModelChange,
    handleFileSelect,
    handleSampleImage,
    handleReanalyze,
  };
}
