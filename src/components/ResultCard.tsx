import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { PredictionResult } from "@/types";

interface ResultCardProps {
  result: PredictionResult;
  modelDisplayName: string;
}

export function ResultCard({ result, modelDisplayName }: ResultCardProps) {
  const isPositive = result.prediction === "POSITIVE";

  return (
    <Card>
      <CardHeader>
        <CardTitle>Classification Result</CardTitle>
        <CardDescription>{modelDisplayName} prediction</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <Badge
              variant={isPositive ? "destructive" : "success"}
              className="text-base px-3 py-1"
            >
              {result.prediction}
            </Badge>
            <span className="text-sm text-muted-foreground">
              COVID-19{" "}
              {isPositive
                ? "indicators detected"
                : "indicators not detected"}
            </span>
          </div>

          <div className="space-y-1">
            <div className="flex justify-between text-sm">
              <span>Confidence</span>
              <span className="font-medium">
                {result.confidence.toFixed(1)}%
              </span>
            </div>
            <Progress
              value={result.confidence}
              className={
                isPositive
                  ? "[&_[data-slot=progress-indicator]]:bg-destructive"
                  : "[&_[data-slot=progress-indicator]]:bg-success"
              }
            />
          </div>

          <div className="text-xs text-muted-foreground pt-2 border-t">
            <p>
              Raw sigmoid output: {result.probability.toFixed(6)} | Threshold:
              0.5
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
