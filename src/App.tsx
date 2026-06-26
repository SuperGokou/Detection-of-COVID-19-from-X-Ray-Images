import { useRef, useEffect } from "react";
import {
  AlertTriangle,
  Loader2,
  Info,
  Play,
  FileStack,
  CheckCircle,
  Clock,
  Activity,
} from "lucide-react";

import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { ModelSelector } from "@/components/ModelSelector";
import { ImageUpload } from "@/components/ImageUpload";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useClassifier } from "@/hooks/useClassifier";

function StatCard({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <div className="bg-white border border-[#e5e7eb] rounded-lg shadow-sm p-6 flex gap-4 items-start">
      <div className="size-12 bg-[#eff6ff] rounded-full flex items-center justify-center shrink-0">
        {icon}
      </div>
      <div className="flex flex-col gap-1">
        <p className="text-sm text-[#6a7282] leading-tight">{label}</p>
        <p className="text-2xl font-bold text-[#1e1e1e]">{value}</p>
        <p className="text-xs text-[#99a1af]">{sub}</p>
      </div>
    </div>
  );
}

export default function App() {
  const {
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
    handleToggleHeatmap,
  } = useClassifier();

  const imageDisplayRef = useRef<HTMLDivElement>(null);

  // Show the appropriate canvas: heatmap overlay when active, else CLAHE/original
  useEffect(() => {
    if (imageDisplayRef.current) {
      imageDisplayRef.current.innerHTML = "";
      const canvas =
        showHeatmap && heatmapCanvas
          ? heatmapCanvas
          : claheCanvas || originalCanvas;
      if (canvas) {
        const clone = document.createElement("canvas");
        clone.width = canvas.width;
        clone.height = canvas.height;
        const ctx = clone.getContext("2d");
        ctx?.drawImage(canvas, 0, 0);
        clone.className = "w-full h-full object-contain";
        imageDisplayRef.current.appendChild(clone);
      }
    }
  }, [originalCanvas, claheCanvas, showHeatmap, heatmapCanvas]);

  const hasImage = !!originalCanvas;
  const isAnalyzing =
    appState === "loading-model" ||
    appState === "preprocessing" ||
    appState === "analyzing";

  return (
    <div className="min-h-screen flex flex-col bg-[#f3f3f3]">
      <Header />

      {/* Hero Section */}
      <section className="bg-white border-b border-[#e5e7eb] px-8 pt-10 pb-10">
        <div className="max-w-[1200px] mx-auto flex gap-8 items-start">
          <div className="flex-1 max-w-[560px]">
            <div className="inline-flex items-center gap-2 bg-[#eff6ff] rounded-full px-3 py-1 mb-4">
              <Info className="size-4 text-[#0579b8]" />
              <span className="text-sm text-[#0579b8]">Research Sandbox Environment</span>
            </div>
            <h2 className="text-[30px] font-bold text-[#1e1e1e] leading-tight mb-4">
              Deploy and benchmark deep learning models for rapid COVID-19 detection.
            </h2>
            <p className="text-lg text-[#4a5565] leading-relaxed">
              A collaborative platform designed to standardize the evaluation of AI models
              on chest radiography. Upload anonymized DICOM/JPEG data to benchmark accuracy
              and inference speed.
            </p>
          </div>

          <div className="flex gap-5 shrink-0">
            <StatCard
              icon={<FileStack className="size-6 text-[#0579b8]" />}
              label="Total Images Tested"
              value="14,205"
              sub="+128 today"
            />
            <StatCard
              icon={<CheckCircle className="size-6 text-[#0579b8]" />}
              label="Model Accuracy (mAP)"
              value="94.8%"
              sub="ResNet50 Baseline"
            />
            <StatCard
              icon={<Clock className="size-6 text-[#0579b8]" />}
              label="Avg. Inference Time"
              value="120ms"
              sub="Per single instance"
            />
          </div>
        </div>
      </section>

      {/* Main Content: Two-Column Layout */}
      <main className="flex-1 px-8 py-8">
        <div className="max-w-[1200px] mx-auto flex gap-8 items-start">
          {/* Left Column: Model Configuration */}
          <div className="w-[380px] shrink-0 flex flex-col gap-6">
            {/* Config Card */}
            <div className="bg-white border border-[#e5e7eb] rounded-lg shadow-sm">
              <div className="px-6 py-4 border-b border-[#f3f4f6]">
                <h3 className="text-lg font-bold text-[#1e1e1e]">Model Configuration</h3>
              </div>
              <div className="px-6 py-6 flex flex-col gap-4">
                <ModelSelector
                  selectedModel={selectedModel}
                  onModelChange={handleModelChange}
                />

                <div className="border-t border-[#f3f4f6] pt-4">
                  <ImageUpload onFileSelect={handleFileSelect} />
                </div>

                <button
                  onClick={() => {
                    if (imageFile) {
                      handleFileSelect(imageFile);
                    } else {
                      handleSampleImage();
                    }
                  }}
                  disabled={isBusy || !opencvReady}
                  className="w-full h-10 bg-[#a51c30] text-white rounded-lg shadow-sm flex items-center justify-center gap-2 text-base hover:bg-[#8a1728] transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                >
                  <Play className="size-4" />
                  Run Analysis
                </button>
              </div>
            </div>

            {/* Privacy Notice */}
            <div className="bg-white border border-[#e5e7eb] rounded-lg shadow-sm px-6 py-5">
              <div className="flex gap-3">
                <Info className="size-5 text-[#0579b8] shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-bold text-[#0579b8]">Privacy Notice</p>
                  <p className="text-xs text-[#0579b8]/80 mt-1 leading-relaxed">
                    Uploaded images are processed in a volatile memory sandbox and are not
                    permanently stored on our servers unless explicitly saved to your
                    research cohort.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Analysis & Visualization */}
          <div className="flex-1 bg-white border border-[#e5e7eb] rounded-lg shadow-sm min-h-[617px]">
            <div className="px-6 py-4 border-b border-[#f3f4f6]">
              <h3 className="text-lg font-bold text-[#1e1e1e]">Analysis &amp; Visualization</h3>
            </div>
            <div className="px-6 py-6">
              <div className="flex gap-6 h-[500px]">
                {/* Left: Image Display */}
                <div className="flex-1 flex flex-col">
                  <div className="flex-1 bg-[#101828] rounded-lg overflow-hidden flex items-center justify-center">
                    {hasImage ? (
                      <div ref={imageDisplayRef} className="w-full h-full flex items-center justify-center" />
                    ) : (
                      <div className="text-[#6a7282] text-sm">
                        No image loaded
                      </div>
                    )}
                  </div>
                  <div className="mt-3 flex items-center justify-between gap-3">
                    <span className="text-sm text-[#6a7282] font-mono min-w-0 truncate">
                      {imageFile ? imageFile.name : "No file selected"}
                    </span>
                    <div className="flex items-center gap-2 shrink-0">
                      {heatmapLoading && (
                        <Loader2 className="size-4 text-[#a51c30] animate-spin" />
                      )}
                      <span className="text-sm text-[#364153] whitespace-nowrap">Show Heatmap (Grad-CAM)</span>
                      <button
                        onClick={handleToggleHeatmap}
                        disabled={!result || heatmapLoading}
                        className={`w-11 h-6 rounded-full relative transition-colors ${
                          !result
                            ? "bg-[#d1d5dc] opacity-50 cursor-not-allowed"
                            : showHeatmap
                              ? "bg-[#a51c30] cursor-pointer"
                              : "bg-[#d1d5dc] cursor-pointer hover:bg-[#b0b5be]"
                        }`}
                      >
                        <div
                          className={`absolute top-1 size-4 bg-white rounded-full shadow-sm transition-[left] ${
                            showHeatmap ? "left-6" : "left-1"
                          }`}
                        />
                      </button>
                    </div>
                  </div>
                </div>

                {/* Right: Results Panel */}
                <div className="flex-1 flex flex-col items-center justify-center">
                  {/* Loading State */}
                  {isAnalyzing && (
                    <div className="text-center">
                      <Loader2 className="size-12 mx-auto text-[#a51c30] animate-spin mb-4" />
                      <p className="text-lg text-[#101828]">
                        {appState === "loading-model"
                          ? `Loading ${currentConfig.displayName}...`
                          : appState === "preprocessing"
                            ? "Preprocessing image..."
                            : "Running inference..."}
                      </p>
                      {appState === "loading-model" && (
                        <div className="mt-4 w-48 mx-auto">
                          <Progress value={loadProgress} />
                          <p className="text-xs text-[#6a7282] mt-1 text-center">
                            {loadProgress}% (~{currentConfig.estimatedSizeMB} MB)
                          </p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Result State */}
                  {appState === "result" && result && (
                    <div className="w-full max-w-xs text-center">
                      <Badge
                        variant={result.prediction === "POSITIVE" ? "destructive" : "success"}
                        className="text-lg px-4 py-2 mb-4"
                      >
                        {result.prediction}
                      </Badge>
                      <p className="text-sm text-[#6a7282] mb-6">
                        COVID-19{" "}
                        {result.prediction === "POSITIVE"
                          ? "indicators detected"
                          : "indicators not detected"}
                      </p>

                      <div className="text-left space-y-3">
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="text-[#364153]">Confidence</span>
                            <span className="font-bold text-[#1e1e1e]">
                              {result.confidence.toFixed(1)}%
                            </span>
                          </div>
                          <Progress
                            value={result.confidence}
                            className={
                              result.prediction === "POSITIVE"
                                ? "[&_[data-slot=progress-indicator]]:bg-destructive"
                                : "[&_[data-slot=progress-indicator]]:bg-success"
                            }
                          />
                        </div>
                        <div className="text-xs text-[#99a1af] pt-2 border-t border-[#f3f4f6]">
                          <p>Model: {currentConfig.displayName}</p>
                          <p>Raw sigmoid: {result.probability.toFixed(6)}</p>
                          <p>Threshold: 0.5</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Error State */}
                  {appState === "error" && (
                    <div className="text-center max-w-xs">
                      <AlertTriangle className="size-12 mx-auto text-red-500 mb-4" />
                      <p className="text-lg text-[#101828] mb-2">Analysis Error</p>
                      <p className="text-sm text-[#6a7282]">{errorMessage}</p>
                    </div>
                  )}

                  {/* Idle State */}
                  {(appState === "idle" || appState === "ready") && !result && (
                    <div className="text-center">
                      <Activity className="size-12 mx-auto text-[#d1d5dc] mb-4" />
                      <p className="text-lg text-[#101828]">Ready to Analyze</p>
                      <p className="text-sm text-[#6a7282] mt-2 max-w-[280px]">
                        Select a model and click "Run Analysis" to see prediction probabilities.
                      </p>
                    </div>
                  )}

                  {/* OpenCV load error — shown whenever the load failed and
                      OpenCV is not ready, independent of appState. */}
                  {!opencvReady && opencvError && (
                    <div className="mt-4 text-center max-w-[280px]">
                      <AlertTriangle className="size-5 mx-auto text-red-500 mb-1" />
                      <p className="text-xs text-red-600">{opencvError}</p>
                    </div>
                  )}

                  {/* OpenCV loading spinner — only while idle and before any error. */}
                  {!opencvReady && !opencvError && appState === "idle" && (
                    <div className="mt-4 text-center">
                      <Loader2 className="size-4 mx-auto text-[#6a7282] animate-spin mb-1" />
                      <p className="text-xs text-[#6a7282]">Loading OpenCV.js WASM...</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
