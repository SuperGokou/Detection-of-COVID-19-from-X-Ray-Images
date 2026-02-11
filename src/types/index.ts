export type AppState =
  | "idle"
  | "loading-model"
  | "ready"
  | "preprocessing"
  | "analyzing"
  | "result"
  | "error";

export type { PredictionResult, ModelConfig } from "@/lib/model-manager";
