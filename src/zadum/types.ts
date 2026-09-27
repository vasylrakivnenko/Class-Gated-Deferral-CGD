export type CascadeTier = 'tier0_local' | 'tier1_direct' | 'tier2_reasoning';

export interface ClassTierAssignment {
  className: string;
  count: number;
  frequency: number; // percentage of dataset (0 - 1)
  tier0Accuracy: number; // e.g. from local Zadum gate
  tier1Accuracy: number; // from Flash Direct LLM
  tier2Accuracy: number; // from Frontier Reasoning LLM
  assignedTier: CascadeTier;
  overrideTier?: CascadeTier;
  confidenceMargin: number;
  reasoningNote: string;
}

export interface ZadumCascadeConfig {
  taskName: string;
  accuracyBar: number; // e.g. 0.90
  tier0Margin: number; // if tier0 acc >= bar - margin, keep tier 0
  tier1Model: string;
  tier2Model: string;
  classes: Record<string, ClassTierAssignment>;
}

export interface ZadumMetrics {
  totalItems: number;
  tier0Ratio: number;
  tier1Ratio: number;
  tier2Ratio: number;
  blendedAccuracy: number;
  blendedCostPer1k: number;
  baselineCostPer1k: number; // e.g. Gemini 3.1 Pro Frontier ($0.35/1k)
  savingsPercent: number;
  estimatedLatencyMs: number;
}

export interface JsonlValidationResult {
  valid: boolean;
  error?: string;
  warning?: string;
  rowCount?: number;
  fileSizeBytes?: number;
  sampleClasses?: string[];
  classCounts?: Record<string, number>;
  sampleRows?: Array<{ text: string; label: string }>;
}
