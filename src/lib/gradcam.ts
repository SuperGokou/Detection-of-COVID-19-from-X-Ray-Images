/**
 * Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
 *
 * Computes heatmaps highlighting which regions of an input image
 * most influenced the model's prediction. Requires tf.LayersModel
 * for gradient and layer access.
 *
 * Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from
 * Deep Networks via Gradient-based Localization", ICCV 2017.
 */
import * as tf from "@tensorflow/tfjs";

const IMG_SIZE = 224;

/**
 * Find the target convolutional layer in the model.
 * Falls back to the last 4D-output layer if the named layer is not found.
 */
function findTargetLayer(
  model: tf.LayersModel,
  targetName: string,
): tf.layers.Layer {
  try {
    return model.getLayer(targetName);
  } catch {
    // Fallback: find the last layer whose output is 4D (batch, h, w, channels)
    for (let i = model.layers.length - 1; i >= 0; i--) {
      const outputShape = model.layers[i]!.outputShape;
      const shape = Array.isArray(outputShape[0])
        ? (outputShape[0] as number[])
        : (outputShape as number[]);
      if (shape.length === 4) {
        console.warn(
          `Grad-CAM: layer "${targetName}" not found, falling back to "${model.layers[i]!.name}"`,
        );
        return model.layers[i]!;
      }
    }
    throw new Error("Grad-CAM: no suitable convolutional layer found");
  }
}

/**
 * Compute a Grad-CAM heatmap for a given input tensor.
 *
 * Algorithm:
 *   1. Build a feature-extractor sub-model (input -> target conv layer output)
 *   2. Forward pass to get conv feature maps
 *   3. Identify classifier head layers (everything after the target layer)
 *   4. Use tf.grad to compute d(prediction)/d(convFeatures) through the head
 *   5. Global-average-pool the gradients for channel importance weights
 *   6. Weighted sum of feature maps + ReLU + normalize to [0,1]
 *   7. Resize to 224x224
 *
 * @returns Float32Array of length 224*224 with values in [0, 1]
 */
export function computeGradCAM(
  model: tf.LayersModel,
  inputTensor: tf.Tensor4D,
  targetLayerName: string,
): Float32Array {
  return tf.tidy(() => {
    const targetLayer = findTargetLayer(model, targetLayerName);

    // Feature extractor: input -> target conv layer output
    const convModel = tf.model({
      inputs: model.inputs,
      outputs: targetLayer.output as tf.SymbolicTensor,
    });

    // Forward pass to get conv feature maps [1, H, W, C]
    const convOutput = convModel.predict(inputTensor) as tf.Tensor4D;

    // Identify classifier head layers (everything after target in topo order).
    // For all 3 models the head is a simple sequential chain:
    //   [MaxPool+Dropout for CNN] -> GAP -> Dense -> BN -> [Activation] -> Dropout -> Dense(sigmoid)
    const targetIdx = model.layers.indexOf(targetLayer);
    const headLayers = model.layers.slice(targetIdx + 1);

    // Compute gradient of the scalar prediction w.r.t. conv features.
    // layer.apply() on real tensors uses tracked TF.js ops, so tf.grad
    // can back-propagate through them.
    const gradFn = tf.grad((features: tf.Tensor) => {
      let x = features;
      for (const layer of headLayers) {
        x = layer.apply(x) as tf.Tensor;
      }
      // Squeeze to scalar for tf.grad
      return x.reshape([]);
    });

    const grads = gradFn(convOutput); // same shape as convOutput [1, H, W, C]

    // Channel importance weights = global average of gradients over spatial dims
    const weights = grads.mean([1, 2], true); // [1, 1, 1, C]

    // Weighted combination of feature maps + ReLU
    const cam = convOutput.mul(weights).sum(-1).relu(); // [1, H, W]

    // Normalize to [0, 1]
    const max = cam.max();
    const min = cam.min();
    const range = max.sub(min);
    const normalized = tf.where(
      range.greater(tf.scalar(0)),
      cam.sub(min).div(range.add(tf.scalar(1e-8))),
      tf.zerosLike(cam),
    );

    // Resize to IMG_SIZE x IMG_SIZE
    const squeezed = normalized.squeeze([0]) as tf.Tensor2D; // [H, W]
    const expanded = squeezed.expandDims(-1) as tf.Tensor3D; // [H, W, 1]
    const resized = tf.image
      .resizeBilinear(expanded, [IMG_SIZE, IMG_SIZE])
      .squeeze([-1]); // [224, 224]

    return resized.dataSync() as Float32Array;
  });
}

/**
 * Apply a jet colormap to grayscale heatmap data and return an RGBA canvas.
 * Maps 0 (cold/blue) -> 0.5 (green) -> 1 (hot/red).
 * Alpha scales with intensity so cold regions are more transparent.
 */
export function applyJetColormap(
  data: Float32Array,
  width: number,
  height: number,
  alpha: number = 0.5,
): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const imageData = ctx.createImageData(width, height);
  const pixels = imageData.data;

  for (let i = 0; i < data.length; i++) {
    const v = data[i]!;
    let r: number, g: number, b: number;

    if (v < 0.125) {
      r = 0;
      g = 0;
      b = 0.5 + v * 4;
    } else if (v < 0.375) {
      r = 0;
      g = (v - 0.125) * 4;
      b = 1;
    } else if (v < 0.625) {
      r = (v - 0.375) * 4;
      g = 1;
      b = 1 - (v - 0.375) * 4;
    } else if (v < 0.875) {
      r = 1;
      g = 1 - (v - 0.625) * 4;
      b = 0;
    } else {
      r = 1 - (v - 0.875) * 4;
      g = 0;
      b = 0;
    }

    const idx = i * 4;
    pixels[idx] = Math.round(r * 255);
    pixels[idx + 1] = Math.round(g * 255);
    pixels[idx + 2] = Math.round(b * 255);
    // Alpha scales with intensity so cold areas are more transparent
    pixels[idx + 3] = Math.round(v * alpha * 255);
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

/**
 * Blend a heatmap canvas on top of a base image canvas.
 * Returns a new canvas with the composited result.
 */
export function blendHeatmap(
  baseCanvas: HTMLCanvasElement,
  heatmapCanvas: HTMLCanvasElement,
): HTMLCanvasElement {
  const result = document.createElement("canvas");
  result.width = baseCanvas.width;
  result.height = baseCanvas.height;
  const ctx = result.getContext("2d")!;

  // Draw base image
  ctx.drawImage(baseCanvas, 0, 0, result.width, result.height);

  // Overlay heatmap with alpha blending
  ctx.drawImage(heatmapCanvas, 0, 0, result.width, result.height);

  return result;
}
