export const AUTO_SOURCE_ROUTING_ID = "auto-source";
export const AUTO_SOURCE_CNN_ID = "auto-source-custom-cnn";
export const AUTO_SOURCE_DENSENET_ID = "auto-source-densenet121";
export const AUTO_SOURCE_RESNET_ID = "auto-source-resnet50";

export type SourceModelArchitecture =
  | "custom-cnn"
  | "densenet121"
  | "resnet50";

export const AUTO_SOURCE_OPTIONS: Array<{
  id: string;
  label: string;
  architecture: SourceModelArchitecture;
}> = [
  {
    id: AUTO_SOURCE_CNN_ID,
    label: "Auto source routing - Custom CNN",
    architecture: "custom-cnn",
  },
  {
    id: AUTO_SOURCE_DENSENET_ID,
    label: "Auto source routing - DenseNet121",
    architecture: "densenet121",
  },
  {
    id: AUTO_SOURCE_RESNET_ID,
    label: "Auto source routing - ResNet50",
    architecture: "resnet50",
  },
];

export function isAutoSourceRouting(modelId: string): boolean {
  return (
    modelId === AUTO_SOURCE_ROUTING_ID ||
    AUTO_SOURCE_OPTIONS.some((option) => option.id === modelId)
  );
}

export function getAutoSourceArchitecture(
  modelId: string,
): SourceModelArchitecture {
  return (
    AUTO_SOURCE_OPTIONS.find((option) => option.id === modelId)?.architecture ??
    "custom-cnn"
  );
}

export interface SourceFeatureVector {
  mean: number;
  std: number;
  entropy: number;
  edgeDensity: number;
  gradientMean: number;
}

export interface SourceProfile {
  id: string;
  displayName: string;
  description: string;
  defaultModelId?: string;
  modelId?: string;
  models?: Partial<Record<SourceModelArchitecture, string>>;
  sampleCount: number;
  featureMean: SourceFeatureVector;
  featureStd: SourceFeatureVector;
}

export interface SourceRoutingManifest {
  version: number;
  generatedAt: string;
  imageSize: number;
  features: Array<keyof SourceFeatureVector>;
  profiles: SourceProfile[];
}

export interface SourceRoutingDecision {
  selectedSourceId: string;
  selectedSourceName: string;
  selectedModelId: string;
  selectedArchitecture: SourceModelArchitecture;
  confidence: number;
  inputFeatures: SourceFeatureVector;
  scores: Array<{
    sourceId: string;
    sourceName: string;
    modelId: string;
    architecture: SourceModelArchitecture;
    distance: number;
    similarity: number;
  }>;
}

const FEATURE_KEYS: Array<keyof SourceFeatureVector> = [
  "mean",
  "std",
  "entropy",
  "edgeDensity",
  "gradientMean",
];

const DEFAULT_MANIFEST: SourceRoutingManifest = {
  version: 1,
  generatedAt: "2026-07-12T00:00:00.000Z",
  imageSize: 224,
  features: FEATURE_KEYS,
  profiles: [
    {
      id: "brixia",
      displayName: "Brixia",
      description: "DICOM COVID-positive training source from the Brixia dataset.",
      defaultModelId: "resnet50",
      models: {
        "custom-cnn": "custom-cnn",
        densenet121: "densenet121",
        resnet50: "resnet50",
      },
      sampleCount: 300,
      featureMean: {
        mean: 0.406801,
        std: 0.159596,
        entropy: 0.804619,
        edgeDensity: 0.008187,
        gradientMean: 0.026469,
      },
      featureStd: {
        mean: 0.149597,
        std: 0.049376,
        entropy: 0.07887,
        edgeDensity: 0.007102,
        gradientMean: 0.008056,
      },
    },
    {
      id: "prunecxr",
      displayName: "PruneCXR / NIH",
      description: "Healthy negative chest X-ray source used for training negatives.",
      defaultModelId: "densenet121",
      models: {
        "custom-cnn": "custom-cnn",
        densenet121: "densenet121",
        resnet50: "resnet50",
      },
      sampleCount: 300,
      featureMean: {
        mean: 0.529031,
        std: 0.236935,
        entropy: 0.911622,
        edgeDensity: 0.009996,
        gradientMean: 0.036974,
      },
      featureStd: {
        mean: 0.094557,
        std: 0.040231,
        entropy: 0.050155,
        edgeDensity: 0.007072,
        gradientMean: 0.008405,
      },
    },
    {
      id: "covidgr",
      displayName: "COVIDGR",
      description: "External Spanish hospital validation source with positive and negative labels.",
      defaultModelId: "source-covidgr-custom-cnn",
      models: {
        "custom-cnn": "source-covidgr-custom-cnn",
        densenet121: "source-covidgr-densenet121",
        resnet50: "source-covidgr-resnet50",
      },
      sampleCount: 300,
      featureMean: {
        mean: 0.527701,
        std: 0.230302,
        entropy: 0.902748,
        edgeDensity: 0.002899,
        gradientMean: 0.035954,
      },
      featureStd: {
        mean: 0.057051,
        std: 0.027829,
        entropy: 0.03836,
        edgeDensity: 0.003869,
        gradientMean: 0.004962,
      },
    },
  ],
};

function vectorToArray(vector: SourceFeatureVector): number[] {
  return FEATURE_KEYS.map((key) => vector[key]);
}

function euclideanDistance(
  input: SourceFeatureVector,
  profile: SourceProfile,
): number {
  const inputValues = vectorToArray(input);
  const meanValues = vectorToArray(profile.featureMean);
  const stdValues = vectorToArray(profile.featureStd);

  const squared = inputValues.reduce((sum, value, index) => {
    const scale = Math.max(stdValues[index] ?? 0, 0.01);
    const z = (value - (meanValues[index] ?? 0)) / scale;
    return sum + z * z;
  }, 0);

  return Math.sqrt(squared / FEATURE_KEYS.length);
}

function softmaxConfidence(distances: number[], selectedIndex: number): number {
  const similarities = distances.map((distance) => Math.exp(-distance));
  const total = similarities.reduce((sum, value) => sum + value, 0);
  if (total <= 0) return 0;
  return (similarities[selectedIndex] ?? 0) / total;
}

function resolveProfileModelId(
  profile: SourceProfile,
  architecture: SourceModelArchitecture,
): string {
  return (
    profile.models?.[architecture] ??
    profile.defaultModelId ??
    profile.modelId ??
    "custom-cnn"
  );
}

function getImageData(canvas: HTMLCanvasElement): ImageData {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Unable to read image canvas for source routing.");
  }
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

function toGrayscale(imageData: ImageData): Uint8ClampedArray {
  const gray = new Uint8ClampedArray(imageData.width * imageData.height);
  const data = imageData.data;

  for (let i = 0, j = 0; i < data.length; i += 4, j += 1) {
    gray[j] = Math.round(
      (data[i] ?? 0) * 0.299 + (data[i + 1] ?? 0) * 0.587 + (data[i + 2] ?? 0) * 0.114,
    );
  }

  return gray;
}

function computeEntropy(gray: Uint8ClampedArray): number {
  const bins = new Array<number>(32).fill(0);
  for (const pixel of gray) {
    const index = Math.min(31, Math.floor(pixel / 8));
    bins[index] = (bins[index] ?? 0) + 1;
  }

  const total = gray.length || 1;
  return (
    -bins.reduce((sum, count) => {
      if (count === 0) return sum;
      const p = count / total;
      return sum + p * Math.log2(p);
    }, 0) / 5
  );
}

function computeEdgeFeatures(
  gray: Uint8ClampedArray,
  width: number,
  height: number,
): Pick<SourceFeatureVector, "edgeDensity" | "gradientMean"> {
  let edgeCount = 0;
  let gradientTotal = 0;
  let compared = 0;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const center = y * width + x;
      const gx =
        -(gray[center - width - 1] ?? 0) -
        2 * (gray[center - 1] ?? 0) -
        (gray[center + width - 1] ?? 0) +
        (gray[center - width + 1] ?? 0) +
        2 * (gray[center + 1] ?? 0) +
        (gray[center + width + 1] ?? 0);
      const gy =
        -(gray[center - width - 1] ?? 0) -
        2 * (gray[center - width] ?? 0) -
        (gray[center - width + 1] ?? 0) +
        (gray[center + width - 1] ?? 0) +
        2 * (gray[center + width] ?? 0) +
        (gray[center + width + 1] ?? 0);
      const gradient = Math.sqrt(gx * gx + gy * gy) / 4;
      gradientTotal += gradient;
      if (gradient > 60) edgeCount += 1;
      compared += 1;
    }
  }

  return {
    edgeDensity: compared > 0 ? edgeCount / compared : 0,
    gradientMean: compared > 0 ? gradientTotal / compared / 255 : 0,
  };
}

export function extractSourceFeatures(
  canvas: HTMLCanvasElement,
): SourceFeatureVector {
  const imageData = getImageData(canvas);
  const gray = toGrayscale(imageData);

  let sum = 0;
  for (const pixel of gray) sum += pixel;
  const meanRaw = sum / (gray.length || 1);

  let variance = 0;
  for (const pixel of gray) {
    const delta = pixel - meanRaw;
    variance += delta * delta;
  }

  const edgeFeatures = computeEdgeFeatures(
    gray,
    imageData.width,
    imageData.height,
  );

  return {
    mean: meanRaw / 255,
    std: Math.sqrt(variance / (gray.length || 1)) / 255,
    entropy: computeEntropy(gray),
    ...edgeFeatures,
  };
}

class SourceRouter {
  private manifestPromise: Promise<SourceRoutingManifest> | null = null;

  private async loadManifest(): Promise<SourceRoutingManifest> {
    if (this.manifestPromise) return this.manifestPromise;

    this.manifestPromise = fetch(`${import.meta.env.BASE_URL}source-routing.json`)
      .then(async (response) => {
        if (!response.ok) return DEFAULT_MANIFEST;
        return (await response.json()) as SourceRoutingManifest;
      })
      .catch(() => DEFAULT_MANIFEST);

    return this.manifestPromise;
  }

  async route(
    canvas: HTMLCanvasElement,
    architecture: SourceModelArchitecture = "custom-cnn",
  ): Promise<SourceRoutingDecision> {
    const manifest = await this.loadManifest();
    const inputFeatures = extractSourceFeatures(canvas);
    const distances = manifest.profiles.map((profile) =>
      euclideanDistance(inputFeatures, profile),
    );
    const selectedIndex = distances.reduce(
      (bestIndex, distance, index) =>
        distance < (distances[bestIndex] ?? Number.POSITIVE_INFINITY)
          ? index
          : bestIndex,
      0,
    );
    const selected = manifest.profiles[selectedIndex] ?? DEFAULT_MANIFEST.profiles[0]!;

    const scores = manifest.profiles
      .map((profile, index) => {
        const distance = distances[index] ?? Number.POSITIVE_INFINITY;
        return {
          sourceId: profile.id,
          sourceName: profile.displayName,
          modelId: resolveProfileModelId(profile, architecture),
          architecture,
          distance,
          similarity: Math.exp(-distance),
        };
      })
      .sort((a, b) => a.distance - b.distance);

    return {
      selectedSourceId: selected.id,
      selectedSourceName: selected.displayName,
      selectedModelId: resolveProfileModelId(selected, architecture),
      selectedArchitecture: architecture,
      confidence: softmaxConfidence(distances, selectedIndex),
      inputFeatures,
      scores,
    };
  }
}

export const sourceRouter = new SourceRouter();
