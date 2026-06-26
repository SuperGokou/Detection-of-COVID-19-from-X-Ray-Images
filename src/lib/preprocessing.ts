/**
 * CLAHE preprocessing pipeline using OpenCV.js.
 * Replicates the exact Python pipeline from covid19_run.py:349-367.
 *
 * Python reference:
 *   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
 *   image_uint8 = cv2.normalize(image, None, 0, 255, NORM_MINMAX).astype('uint8')
 *   Grayscale: clahe.apply(image_uint8)
 *   RGB: RGB->YUV, CLAHE on Y channel, YUV->RGB
 */

import { loadOpenCV } from "./opencv-loader";

declare const cv: any;

const IMG_SIZE = 224;

export interface PreprocessingResult {
  /** Preprocessed tensor data as Float32Array, normalized to [0, 1] */
  tensorData: Float32Array;
  /** Shape of the tensor: [1, 224, 224, channels] */
  shape: [number, number, number, number];
  /** Canvas element showing original image resized to 224x224 */
  originalCanvas: HTMLCanvasElement;
  /** Canvas element showing image after CLAHE preprocessing */
  claheCanvas: HTMLCanvasElement;
}

/**
 * Wait for OpenCV.js to be ready. Delegates to the self-hosted loader, which
 * injects the script (base-path aware) and resolves once the WASM runtime is
 * initialized.
 */
export function waitForOpenCV(): Promise<void> {
  return loadOpenCV();
}

/**
 * Load an image file into a canvas resized to 224x224.
 */
function loadImageToCanvas(file: File): Promise<HTMLCanvasElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = IMG_SIZE;
      canvas.height = IMG_SIZE;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);
      URL.revokeObjectURL(img.src);
      resolve(canvas);
    };
    img.onerror = () => {
      URL.revokeObjectURL(img.src);
      reject(new Error("Failed to load image"));
    };
    img.src = URL.createObjectURL(file);
  });
}

/**
 * Process an image for the Custom CNN model (grayscale, 1 channel).
 *
 * Pipeline (matching Python):
 *   1. RGBA -> Grayscale
 *   2. cv2.normalize(NORM_MINMAX) to 0-255 uint8
 *   3. CLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply()
 *   4. Divide by 255.0
 *   5. Return as [1, 224, 224, 1] tensor
 */
function processGrayscale(canvas: HTMLCanvasElement): {
  tensorData: Float32Array;
  claheCanvas: HTMLCanvasElement;
} {
  const src = cv.imread(canvas);
  const gray = new cv.Mat();
  const normalized = new cv.Mat();
  const claheResult = new cv.Mat();

  try {
    // RGBA -> Grayscale
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // cv2.normalize(image, None, 0, 255, NORM_MINMAX).astype('uint8')
    cv.normalize(gray, normalized, 0, 255, cv.NORM_MINMAX, cv.CV_8U);

    // CLAHE(clipLimit=2.0, tileGridSize=(8,8))
    const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
    clahe.apply(normalized, claheResult);
    clahe.delete();

    // Render CLAHE result to a canvas for visualization
    const claheCanvas = document.createElement("canvas");
    claheCanvas.width = IMG_SIZE;
    claheCanvas.height = IMG_SIZE;
    cv.imshow(claheCanvas, claheResult);

    // Convert to Float32Array normalized to [0, 1]
    const totalPixels = IMG_SIZE * IMG_SIZE;
    const tensorData = new Float32Array(totalPixels);
    const data = claheResult.data;
    for (let i = 0; i < totalPixels; i++) {
      tensorData[i] = data[i]! / 255.0;
    }

    return { tensorData, claheCanvas };
  } finally {
    src.delete();
    gray.delete();
    normalized.delete();
    claheResult.delete();
  }
}

/**
 * Process an image for DenseNet121/ResNet50 (RGB, 3 channels).
 *
 * Pipeline (matching Python):
 *   1. RGBA -> RGB
 *   2. cv2.normalize(NORM_MINMAX) to 0-255 uint8
 *   3. RGB -> YUV
 *   4. CLAHE on Y (luminance) channel
 *   5. YUV -> RGB
 *   6. Divide by 255.0
 *   7. Return as [1, 224, 224, 3] tensor
 */
function processRGB(canvas: HTMLCanvasElement): {
  tensorData: Float32Array;
  claheCanvas: HTMLCanvasElement;
} {
  const src = cv.imread(canvas);
  const rgb = new cv.Mat();
  const normalized = new cv.Mat();
  const yuv = new cv.Mat();
  const result = new cv.Mat();

  try {
    // RGBA -> RGB
    cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);

    // cv2.normalize(image, None, 0, 255, NORM_MINMAX).astype('uint8')
    cv.normalize(rgb, normalized, 0, 255, cv.NORM_MINMAX, cv.CV_8U);

    // RGB -> YUV
    cv.cvtColor(normalized, yuv, cv.COLOR_RGB2YUV);

    // Split channels, apply CLAHE to Y channel
    const channels = new cv.MatVector();
    cv.split(yuv, channels);

    const yChannel = channels.get(0);
    const claheY = new cv.Mat();
    const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
    clahe.apply(yChannel, claheY);
    clahe.delete();

    // Replace Y channel with CLAHE result
    channels.set(0, claheY);
    const mergedYUV = new cv.Mat();
    cv.merge(channels, mergedYUV);

    // YUV -> RGB
    cv.cvtColor(mergedYUV, result, cv.COLOR_YUV2RGB);

    // Render CLAHE result to a canvas for visualization
    const claheCanvas = document.createElement("canvas");
    claheCanvas.width = IMG_SIZE;
    claheCanvas.height = IMG_SIZE;
    // Convert RGB to RGBA for canvas display
    const rgba = new cv.Mat();
    cv.cvtColor(result, rgba, cv.COLOR_RGB2RGBA);
    cv.imshow(claheCanvas, rgba);
    rgba.delete();

    // Convert to Float32Array normalized to [0, 1], in RGB order
    const totalPixels = IMG_SIZE * IMG_SIZE;
    const tensorData = new Float32Array(totalPixels * 3);
    const data = result.data;
    for (let i = 0; i < totalPixels; i++) {
      tensorData[i * 3] = data[i * 3]! / 255.0;
      tensorData[i * 3 + 1] = data[i * 3 + 1]! / 255.0;
      tensorData[i * 3 + 2] = data[i * 3 + 2]! / 255.0;
    }

    // Cleanup
    yChannel.delete();
    claheY.delete();
    mergedYUV.delete();
    for (let i = 0; i < channels.size(); i++) {
      channels.get(i).delete();
    }
    channels.delete();

    return { tensorData, claheCanvas };
  } finally {
    src.delete();
    rgb.delete();
    normalized.delete();
    yuv.delete();
    result.delete();
  }
}

/**
 * Preprocess an image file for model inference.
 *
 * @param file - The image file to process
 * @param colorMode - "grayscale" for Custom CNN (1 channel), "rgb" for DenseNet/ResNet (3 channels)
 * @returns Preprocessed tensor data and visualization canvases
 */
export async function preprocessImage(
  file: File,
  colorMode: "grayscale" | "rgb",
): Promise<PreprocessingResult> {
  await waitForOpenCV();

  const originalCanvas = await loadImageToCanvas(file);

  const { tensorData, claheCanvas } =
    colorMode === "grayscale"
      ? processGrayscale(originalCanvas)
      : processRGB(originalCanvas);

  const channels = colorMode === "grayscale" ? 1 : 3;

  return {
    tensorData,
    shape: [1, IMG_SIZE, IMG_SIZE, channels],
    originalCanvas,
    claheCanvas,
  };
}
