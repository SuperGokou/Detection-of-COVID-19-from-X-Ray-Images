import { ChevronDown } from "lucide-react";

import { MODEL_CONFIGS } from "@/lib/model-manager";

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (modelId: string) => void;
}

export function ModelSelector({
  selectedModel,
  onModelChange,
}: ModelSelectorProps) {
  return (
    <div className="flex flex-col gap-2">
      <label className="text-sm text-[#364153]">Select Architecture</label>
      <div className="relative">
        <select
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
          className="w-full h-10 appearance-none rounded-lg border border-[#d1d5dc] bg-white px-3 pr-10 text-sm text-[#1e1e1e] outline-none focus:border-[#a51c30] focus:ring-1 focus:ring-[#a51c30] cursor-pointer"
        >
          {Object.values(MODEL_CONFIGS).map((config) => (
            <option key={config.id} value={config.id}>
              {config.displayName} (~{config.estimatedSizeMB} MB, {config.colorMode === "grayscale" ? "Grayscale" : "RGB"})
            </option>
          ))}
        </select>
        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 size-4 text-[#6a7282] pointer-events-none" />
      </div>
    </div>
  );
}
