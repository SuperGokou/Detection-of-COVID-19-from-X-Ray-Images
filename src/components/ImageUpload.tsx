import { useRef, useState, useCallback } from "react";
import { CloudUpload } from "lucide-react";

interface ImageUploadProps {
  onFileSelect: (file: File) => void;
}

export function ImageUpload({ onFileSelect }: ImageUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) onFileSelect(file);
    },
    [onFileSelect],
  );

  return (
    <div className="flex flex-col gap-2">
      <label className="text-sm text-[#364153]">Input Source</label>
      <div
        className={`
          border-2 border-dashed rounded-lg py-8 px-4 text-center transition-colors cursor-pointer
          ${dragOver ? "border-[#a51c30] bg-red-50" : "border-[#d1d5dc] hover:border-[#a51c30]/50"}
        `}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <CloudUpload className="size-10 mx-auto text-[#6a7282] mb-3" />
        <p className="text-sm text-[#101828]">
          Click to upload or drag &amp; drop
        </p>
        <p className="text-xs text-[#6a7282] mt-1">
          DICOM, JPEG, PNG (Max 10MB)
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFileSelect(file);
          }}
        />
      </div>
    </div>
  );
}
