import { useState, useCallback, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { preprocessImage, waitForOpenCV } from "@/lib/preprocessing";
import { modelManager, MODEL_CONFIGS } from "@/lib/model-manager";
import {
  AUTO_SOURCE_CNN_ID,
  getAutoSourceArchitecture,
  isAutoSourceRouting,
  sourceRouter,
  type SourceRoutingDecision,
} from "@/lib/source-router";
import { computeGradCAM, applyJetColormap, blendHeatmap } from "@/lib/gradcam";
import type { AppState, PredictionResult } from "@/types";
import { SAMPLE_IMAGE_URL } from "@/constants";

export function useClassifier() {
  const [selectedModel, setSelectedModel] = useState(AUTO_SOURCE_CNN_ID);
  const [activeModelId, setActiveModelId] = useState("custom-cnn");
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

  const [heatmapCanvas, setHeatmapCanvas] = useState<HTMLCanvasElement | null>(
    null,
  );
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapLoading, setHeatmapLoading] = useState(false);

  const lastTensorRef = useRef<{
    data: Float32Array;
    shape: [number, number, number, number];
  } | null>(null);

  const currentConfig = MODEL_CONFIGS[activeModelId]!;

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

        const isAutoRouting = isAutoSourceRouting(selectedModel);
        const selectedArchitecture = getAutoSourceArchitecture(selectedModel);

        setAppState("preprocessing");
        const initialColorMode = isAutoRouting
          ? "grayscale"
          : MODEL_CONFIGS[selectedModel]!.colorMode;
        const initialPreprocessed = await preprocessImage(file, initialColorMode);
        let modelId = selectedModel;
        let preprocessed = initialPreprocessed;
        let currentColorMode = initialColorMode;
        let routingDecision: SourceRoutingDecision | undefined;

        if (isAutoRouting) {
          const routing = await sourceRouter.route(
            initialPreprocessed.originalCanvas,
            selectedArchitecture,
          );
          routingDecision = routing;
          modelId = MODEL_CONFIGS[routing.selectedModelId]
            ? routing.selectedModelId
            : "custom-cnn";
          setActiveModelId(modelId);

          const routedConfig = MODEL_CONFIGS[modelId]!;
          if (routedConfig.colorMode !== initialColorMode) {
            preprocessed = await preprocessImage(file, routedConfig.colorMode);
            currentColorMode = routedConfig.colorMode;
          }
        } else {
          setActiveModelId(modelId);
        }

        setOriginalCanvas(preprocessed.originalCanvas);
        setClaheCanvas(preprocessed.claheCanvas);

        lastTensorRef.current = {
          data: preprocessed.tensorData,
          shape: preprocessed.shape,
        };

        if (!modelManager.isLoaded(modelId)) {
          setAppState("loading-model");
          setLoadProgress(0);
          try {
            await modelManager.loadModel(modelId, (progress) => {
              setLoadProgress(Math.round(progress * 100));
            });
          } catch (err) {
            if (isAutoRouting && modelId !== selectedArchitecture) {
              console.warn(
                `Routed model ${modelId} failed to load; falling back to ${selectedArchitecture}.`,
                err,
              );
              modelId = selectedArchitecture;
              setActiveModelId(modelId);
              const fallbackConfig = MODEL_CONFIGS[modelId]!;
              if (fallbackConfig.colorMode !== currentColorMode) {
                preprocessed = await preprocessImage(file, fallbackConfig.colorMode);
                currentColorMode = fallbackConfig.colorMode;
                setOriginalCanvas(preprocessed.originalCanvas);
                setClaheCanvas(preprocessed.claheCanvas);
                lastTensorRef.current = {
                  data: preprocessed.tensorData,
                  shape: preprocessed.shape,
                };
              }
              setLoadProgress(0);
              await modelManager.loadModel(modelId, (progress) => {
                setLoadProgress(Math.round(progress * 100));
              });
            } else {
              throw err;
            }
          }
        }

        setAppState("analyzing");
        const prediction = await modelManager.predict(
          modelId,
          preprocessed.tensorData,
          preprocessed.shape,
        );
        if (routingDecision) {
          prediction.routing = {
            ...routingDecision,
            selectedModelId: modelId,
          };
        }
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

  const computeHeatmap = useCallback(() => {
    const config = MODEL_CONFIGS[activeModelId];
    const model = modelManager.getModel(activeModelId);
    const tensorInfo = lastTensorRef.current;

    if (!config || !model || !tensorInfo) return;

    setHeatmapLoading(true);

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
  }, [activeModelId, claheCanvas, originalCanvas]);

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
      setActiveModelId(
        isAutoSourceRouting(modelId)
          ? getAutoSourceArchitecture(modelId)
          : modelId,
      );
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
