import { useRef, useEffect } from "react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface PreprocessingViewProps {
  originalCanvas: HTMLCanvasElement;
  claheCanvas: HTMLCanvasElement | null;
}

export function PreprocessingView({
  originalCanvas,
  claheCanvas,
}: PreprocessingViewProps) {
  const originalRef = useRef<HTMLDivElement>(null);
  const claheRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (originalRef.current) {
      originalRef.current.innerHTML = "";
      originalCanvas.className = "w-full h-auto rounded-md";
      originalRef.current.appendChild(originalCanvas);
    }
  }, [originalCanvas]);

  useEffect(() => {
    if (claheCanvas && claheRef.current) {
      claheRef.current.innerHTML = "";
      claheCanvas.className = "w-full h-auto rounded-md";
      claheRef.current.appendChild(claheCanvas);
    }
  }, [claheCanvas]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Preprocessing</CardTitle>
        <CardDescription>
          CLAHE contrast enhancement (clipLimit=2.0, tileGrid=8x8)
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-2">
              Original (224x224)
            </p>
            <div ref={originalRef} className="bg-muted rounded-md overflow-hidden" />
          </div>
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-2">
              After CLAHE
            </p>
            <div ref={claheRef} className="bg-muted rounded-md overflow-hidden" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
