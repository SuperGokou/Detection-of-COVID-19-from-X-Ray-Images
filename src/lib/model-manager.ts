/**
 * TensorFlow.js model loading, caching, and inference manager.
 */
import * as tf from "@tensorflow/tfjs";

export interface ModelConfig {
  id: string;
  displayName: string;
  description: string;
  path: string;
  colorMode: "grayscale" | "rgb";
  inputShape: [number, number, number, number];
  estimatedSizeMB: number;
  /** Layer name to use as Grad-CAM target (last conv layer before GAP) */
  gradCamTargetLayer: string;
}

export const MODEL_CONFIGS: Record<string, ModelConfig> = {
  "custom-cnn": {
    id: "custom-cnn",
    displayName: "Custom CNN",
    description:
      "A lightweight 4-block CNN trained from scratch on grayscale X-rays with L2 regularization. Fast inference, small model size.",
    path: "models/custom-cnn/model.json",
    colorMode: "grayscale",
    inputShape: [1, 224, 224, 1],
    estimatedSizeMB: 2.7,
    gradCamTargetLayer: "activation_3", // Last ReLU before GAP (14x14x256)
  },
  densenet121: {
    id: "densenet121",
    displayName: "DenseNet121",
    description:
      "DenseNet121 pre-trained on ImageNet, fine-tuned on RGB X-rays. Dense connectivity enables feature reuse across layers.",
    path: "models/densenet121/model.json",
    colorMode: "rgb",
    inputShape: [1, 224, 224, 3],
    estimatedSizeMB: 15,
    gradCamTargetLayer: "relu", // Final activation before GAP (7x7x1024)
  },
  resnet50: {
    id: "resnet50",
    displayName: "ResNet50",
    description:
      "ResNet50 pre-trained on ImageNet, fine-tuned on RGB X-rays. Residual connections allow training of very deep networks.",
    path: "models/resnet50/model.json",
    colorMode: "rgb",
    inputShape: [1, 224, 224, 3],
    estimatedSizeMB: 50,
    gradCamTargetLayer: "conv5_block3_out", // Last residual block (7x7x2048)
  },
};

export interface PredictionResult {
  /** Raw sigmoid output probability */
  probability: number;
  /** Binary prediction label */
  prediction: "POSITIVE" | "NEGATIVE";
  /** Confidence as a percentage (0-100) */
  confidence: number;
}

type ProgressCallback = (progress: number) => void;

class ModelManager {
  private loadedModels: Map<string, tf.LayersModel> = new Map();

  /**
   * Check if a model is already loaded and cached.
   */
  isLoaded(modelId: string): boolean {
    return this.loadedModels.has(modelId);
  }

  /**
   * Load a model by ID with progress reporting.
   * Models are cached in memory after first load.
   */
  async loadModel(
    modelId: string,
    onProgress?: ProgressCallback,
  ): Promise<void> {
    if (this.loadedModels.has(modelId)) {
      onProgress?.(1);
      return;
    }

    const config = MODEL_CONFIGS[modelId];
    if (!config) {
      throw new Error(`Unknown model: ${modelId}`);
    }

    // Resolve model path relative to base URL
    const basePath = import.meta.env.BASE_URL;
    const modelUrl = `${basePath}${config.path}`;

    const model = await tf.loadLayersModel(modelUrl, {
      onProgress: (fraction) => {
        onProgress?.(fraction);
      },
    });

    // Warm up the model with a dummy inference to pre-compile WebGL shaders
    const dummyInput = tf.zeros(config.inputShape);
    const warmupResult = model.predict(dummyInput) as tf.Tensor;
    warmupResult.dispose();
    dummyInput.dispose();

    this.loadedModels.set(modelId, model);
  }

  /**
   * Run inference on preprocessed image data.
   *
   * @param modelId - The model to use for prediction
   * @param tensorData - Preprocessed Float32Array from preprocessing.ts
   * @param shape - Tensor shape [1, 224, 224, channels]
   * @returns Prediction result with probability, label, and confidence
   */
  async predict(
    modelId: string,
    tensorData: Float32Array,
    shape: [number, number, number, number],
  ): Promise<PredictionResult> {
    const model = this.loadedModels.get(modelId);
    if (!model) {
      throw new Error(`Model ${modelId} not loaded. Call loadModel() first.`);
    }

    const inputTensor = tf.tensor(tensorData, shape);

    try {
      const output = model.predict(inputTensor) as tf.Tensor;
      const probability = (await output.data())[0]!;
      output.dispose();

      // Sigmoid output: > 0.5 = POSITIVE (COVID), <= 0.5 = NEGATIVE
      const prediction: "POSITIVE" | "NEGATIVE" =
        probability > 0.5 ? "POSITIVE" : "NEGATIVE";
      const confidence =
        prediction === "POSITIVE"
          ? probability * 100
          : (1 - probability) * 100;

      return { probability, prediction, confidence };
    } finally {
      inputTensor.dispose();
    }
  }

  /**
   * Get the underlying LayersModel instance for direct access (e.g. Grad-CAM).
   */
  getModel(modelId: string): tf.LayersModel | undefined {
    return this.loadedModels.get(modelId);
  }

  /**
   * Dispose all loaded models to free memory.
   */
  disposeAll(): void {
    for (const model of this.loadedModels.values()) {
      model.dispose();
    }
    this.loadedModels.clear();
  }
}

// Singleton instance
export const modelManager = new ModelManager();
