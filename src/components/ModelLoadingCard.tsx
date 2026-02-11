import { Loader2 } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import type { AppState } from "@/types";

interface ModelLoadingCardProps {
  appState: AppState;
  loadProgress: number;
  modelDisplayName: string;
  estimatedSizeMB: number;
  hasCanvas: boolean;
}

export function ModelLoadingCard({
  appState,
  loadProgress,
  modelDisplayName,
  estimatedSizeMB,
  hasCanvas,
}: ModelLoadingCardProps) {
  if (appState === "loading-model") {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Loader2 className="size-4 animate-spin" />
            Loading {modelDisplayName}
          </CardTitle>
          <CardDescription>
            Downloading model weights (~{estimatedSizeMB} MB)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Progress value={loadProgress} />
            <p className="text-xs text-muted-foreground text-right">
              {loadProgress}%
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (
    (appState === "preprocessing" || appState === "analyzing") &&
    !hasCanvas
  ) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Loader2 className="size-4 animate-spin" />
            {appState === "preprocessing"
              ? "Preprocessing image..."
              : "Running inference..."}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <Skeleton className="aspect-square rounded-md" />
            <Skeleton className="aspect-square rounded-md" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (appState === "analyzing" && hasCanvas) {
    return (
      <Card>
        <CardContent className="py-6">
          <div className="flex items-center justify-center gap-2 text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            <span className="text-sm">
              Running inference with {modelDisplayName}...
            </span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return null;
}
