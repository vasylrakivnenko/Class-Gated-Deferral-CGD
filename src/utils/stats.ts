/**
 * Statistical utilities for Wilson 95% interval, cost calculations, and Pareto frontier.
 */

export function wilsonInterval(k: number, n: number, z = 1.959963984540054): [number, number] {
  if (n <= 0) return [0, 0];
  const p = k / n;
  const denominator = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denominator;
  const halfSpread = (z * Math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)) / denominator;
  return [Math.max(0, center - halfSpread), Math.min(1, center + halfSpread)];
}

export function formatCost(v: number): string {
  if (v <= 1e-9) return '$0.00 (free)';
  if (v < 0.001) return `$${v.toFixed(5)}`;
  if (v < 0.01) return `$${v.toFixed(4)}`;
  if (v < 1) return `$${v.toFixed(3)}`;
  return `$${v.toFixed(2)}`;
}

export function formatPct(v: number, decimals = 1): string {
  return `${(v * 100).toFixed(decimals)}%`;
}

export interface ParetoPoint {
  cost: number;
  acc: number;
  key: string;
  label: string;
  [key: string]: any;
}

/**
 * Computes Pareto-optimal frontier points where lower cost & higher accuracy dominates.
 */
export function computeParetoFrontier<T extends { cost: number; acc: number }>(rows: T[]): T[] {
  // Sort by cost ascending, then by accuracy descending
  const sorted = rows.slice().sort((a, b) => a.cost - b.cost || b.acc - a.acc);
  const frontier: T[] = [];
  let bestAcc = -Infinity;

  for (const row of sorted) {
    if (row.acc > bestAcc) {
      frontier.push(row);
      bestAcc = row.acc;
    }
  }

  return frontier;
}
