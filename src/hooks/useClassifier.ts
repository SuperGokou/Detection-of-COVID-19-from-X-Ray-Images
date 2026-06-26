import { useState, useCallback, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { preprocessImage, waitForOpenCV } from "@/lib/preprocessing";
import { modelManager, MODEL_CONFIGS } from "@/lib/model-manager";
import { computeGradCAM, applyJetColormap, blendHeatmap } from "@/lib/gradcam";
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
  const [opencvError, setOpencvError] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);

  // Grad-CAM state
  const [heatmapCanvas, setHeatmapCanvas] = useState<HTMLCanvasElement | null>(
    null,
  );
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapLoading, setHeatmapLoading] = useState(false);

  // Keep the last preprocessed tensor data for Grad-CAM reuse
  const lastTensorRef = useRef<{
    data: Float32Array;
    shape: [number, number, number, number];
  } | null>(null);

  const currentConfig = MODEL_CONFIGS[selectedModel]!;

  useEffect(() => {
    waitForOpenCV()
      .then(() => {
        setOpencvReady(true);
        setOpencvError(null);
      })
      .catch((err) => {
        setOpencvReady(false);
        setOpencvError(
          err instanceof Error ? err.message : "OpenCV.js failed to load",
        );
      });
  }, []);

  const runAnalysis = useCallback(
    async (file: File) => {
      try {
        setErrorMessage("");
        setResult(null);
        setHeatmapCanvas(null);
        setShowHeatmap(false);

        // Preprocess first so the image displays even if model loading fails
        setAppState("preprocessing");
        const config = MODEL_CONFIGS[selectedModel]!;
        const preprocessed = await preprocessImage(file, config.colorMode);
        setOriginalCanvas(preprocessed.originalCanvas);
        setClaheCanvas(preprocessed.claheCanvas);

        // Store tensor data for later Grad-CAM computation
        lastTensorRef.current = {
          data: preprocessed.tensorData,
          shape: preprocessed.shape,
        };

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

  /** Compute the Grad-CAM heatmap (lazy, called on first toggle). */
  const computeHeatmap = useCallback(() => {
    const config = MODEL_CONFIGS[selectedModel];
    const model = modelManager.getModel(selectedModel);
    const tensorInfo = lastTensorRef.current;

    if (!config || !model || !tensorInfo) return;

    setHeatmapLoading(true);

    // Use requestAnimationFrame to let the UI update (show spinner) before
    // blocking the main thread with the gradient computation.
    requestAnimationFrame(() => {
      try {
        const inputTensor = tf.tensor(
          tensorInfo.data,
          tensorInfo.shape,
        ) as tf.Tensor4D;

        const heatmapData = computeGradCAM(
          model,
          inputTensor,
          config.gradCamTargetLayer,
        );

        inputTensor.dispose();

        const coloredHeatmap = applyJetColormap(heatmapData, 224, 224, 0.5);
        const baseCanvas = claheCanvas || originalCanvas;
        if (baseCanvas) {
          setHeatmapCanvas(blendHeatmap(baseCanvas, coloredHeatmap));
        }
      } catch (err) {
        console.error("Grad-CAM computation failed:", err);
      } finally {
        setHeatmapLoading(false);
      }
    });
  }, [selectedModel, claheCanvas, originalCanvas]);

  /** Toggle heatmap visibility; computes on first toggle. */
  const handleToggleHeatmap = useCallback(() => {
    if (!result) return;

    const next = !showHeatmap;
    setShowHeatmap(next);

    if (next && !heatmapCanvas && !heatmapLoading) {
      computeHeatmap();
    }
  }, [showHeatmap, result, heatmapCanvas, heatmapLoading, computeHeatmap]);

  const handleModelChange = useCallback(
    (modelId: string) => {
      setSelectedModel(modelId);
      setResult(null);
      setOriginalCanvas(null);
      setClaheCanvas(null);
      setHeatmapCanvas(null);
      setShowHeatmap(false);
      lastTensorRef.current = null;
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
    opencvError,
    imageFile,
    currentConfig,
    isBusy,
    heatmapCanvas,
    showHeatmap,
    heatmapLoading,
    handleModelChange,
    handleFileSelect,
    handleSampleImage,
    handleReanalyze,
    handleToggleHeatmap,
  };
}
