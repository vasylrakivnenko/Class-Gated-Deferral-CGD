export type ModelFamily = 'llm' | 'classical' | 'encoder';

export interface BenchmarkRow {
  key: string;
  label: string;
  family: ModelFamily;
  cost: number;
  cost_call: number;
  calls: number;
  changed: boolean | null;
  acc: number;
  lo: number;
  hi: number;
  tin: number;
  tout: number;
  tcached: number;
  secs: number;
  n: number;
  basis: string;
  passes_holm: boolean | null;
  perclass?: Record<string, number>;
  nclasses: number;
  linked: { cost: number; acc: number } | null;
  verdict: string;
  passes: boolean | null;
  diff: number | null;
  p: number | null;
  isref: boolean;
  note: string;
}

export interface BenchmarkTask {
  key: string;
  label: string;
  majority: number;
  bar: number;
  ntest: number;
  ref: string;
  nclasses: number;
  resolution: string;
  rows: BenchmarkRow[];
}

export interface BenchmarkDataset {
  tasks: Record<string, BenchmarkTask>;
  default: string;
}

export interface PricingRow {
  model: string;
  params: string | null;
  provider: 'azure' | 'fireworks' | 'google';
  ctx: number;
  cache: boolean;
  in: number;
  out: number;
  note?: string;
  dup?: string;
  inferred?: string;
}

export interface SampleItem {
  id: string;
  text: string;
  label: string;
}
