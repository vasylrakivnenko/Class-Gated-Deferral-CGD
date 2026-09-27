import React, { useState, useMemo } from 'react';
import { BenchmarkTask, BenchmarkRow, ModelFamily } from '../types/benchmark';
import { formatCost, formatPct } from '../utils/stats';
import { Info, Sparkles, Filter, Sliders, CheckCircle2, XCircle, ArrowUpRight, HelpCircle } from 'lucide-react';

interface ParetoChartProps {
  task: BenchmarkTask;
  tasks: Record<string, BenchmarkTask>;
  currentTaskKey: string;
  onSelectTask: (key: string) => void;
  liveGeminiCandidate?: {
    model: string;
    modelLabel: string;
    acc: number;
    cost: number;
    tin: number;
    tout: number;
    lo?: number;
    hi?: number;
  } | null;
}

const W = 1240;
const H = 660;
const M = { l: 80, r: 130, t: 40, b: 70 };
const PW = W - M.l - M.r;
const PH = H - M.t - M.b;
const YMIN = 0.20;
const YMAX = 1.02;

const FAM_COLOR: Record<ModelFamily, string> = {
  llm: '#3b82f6', // blue
  classical: '#10b981', // emerald
  encoder: '#f59e0b', // amber
};

const STORY_KEYS = new Set(['tfidf-logreg', 'frozen-embed', 'finetuned-encoder', 'modernbert']);

export const ParetoChart: React.FC<ParetoChartProps> = ({
  task,
  tasks,
  currentTaskKey,
  onSelectTask,
  liveGeminiCandidate,
}) => {
  const [accuracyBar, setAccuracyBar] = useState<number>(task.bar || 0.90);
  const [showML, setShowML] = useState<boolean>(true);
  const [showPareto, setShowPareto] = useState<boolean>(true);
  const [labelMode, setLabelMode] = useState<'story' | 'all' | 'none'>('all');
  const [selectedPoint, setSelectedPoint] = useState<BenchmarkRow | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<BenchmarkRow | null>(null);

  // Sync accuracyBar if task changes
  React.useEffect(() => {
    setAccuracyBar(task.bar);
    setSelectedPoint(null);
  }, [task.key, task.bar]);

  // Points excluding the floor baseline row
  const points = useMemo(() => {
    return task.rows.filter((r) => r.key !== 'majority');
  }, [task.rows]);

  // Scaled coordinates
  const { xOf, yOf, pricedTicks, cMin, cMax } = useMemo(() => {
    const priced = points.map((r) => r.cost).filter((c) => c > 1e-9);
    const minVal = priced.length > 0 ? Math.min(...priced) : 0.01;
    const maxVal = priced.length > 0 ? Math.max(...priced) : 3.0;

    const logMin = Math.log10(minVal / 1.7);
    const logMax = Math.log10(maxVal * 1.9);

    const FREE_X = M.l + 30;
    const LOG_X0 = M.l + 90;
    const LOG_W = M.l + PW - LOG_X0;

    const getX = (cost: number) => {
      if (cost <= 1e-9) return FREE_X;
      return LOG_X0 + ((Math.log10(cost) - logMin) / (logMax - logMin)) * LOG_W;
    };

    const getY = (acc: number) => {
      return M.t + ((YMAX - acc) / (YMAX - YMIN)) * PH;
    };

    // Nice log ticks
    const candidateTicks = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0];
    const validTicks = candidateTicks.filter((t) => t >= minVal * 0.7 && t <= maxVal * 1.3);

    return {
      xOf: getX,
      yOf: getY,
      pricedTicks: validTicks,
      cMin: minVal,
      cMax: maxVal,
    };
  }, [points]);

  // Visible rows
  const visibleRows = useMemo(() => {
    return points.filter((r) => {
      const isML = r.family === 'classical' || r.family === 'encoder';
      return showML || !isML;
    });
  }, [points, showML]);

  // Pareto Frontier Calculation
  const paretoFrontier = useMemo(() => {
    if (!showPareto) return [];
    // Include live candidate if present
    const rowsToTest = [...visibleRows];
    if (liveGeminiCandidate) {
      rowsToTest.push({
        key: 'live-gemini-point',
        label: liveGeminiCandidate.modelLabel,
        family: 'llm',
        cost: liveGeminiCandidate.cost,
        cost_call: liveGeminiCandidate.cost,
        calls: 1,
        changed: null,
        acc: liveGeminiCandidate.acc,
        lo: liveGeminiCandidate.lo || liveGeminiCandidate.acc - 0.03,
        hi: liveGeminiCandidate.hi || liveGeminiCandidate.acc + 0.03,
        tin: liveGeminiCandidate.tin,
        tout: liveGeminiCandidate.tout,
        tcached: 0,
        secs: 0,
        n: 250,
        basis: 'live',
        passes_holm: true,
        nclasses: task.nclasses,
        linked: null,
        verdict: 'Measured live via Gemini API',
        passes: liveGeminiCandidate.acc >= accuracyBar,
        diff: null,
        p: null,
        isref: false,
        note: 'Live measured point from playground',
      });
    }

    const sorted = rowsToTest.slice().sort((a, b) => a.cost - b.cost || b.acc - a.acc);
    const front: typeof rowsToTest = [];
    let bestAcc = -Infinity;

    for (const r of sorted) {
      if (r.acc > bestAcc) {
        front.push(r);
        bestAcc = r.acc;
      }
    }
    return front;
  }, [visibleRows, showPareto, liveGeminiCandidate, accuracyBar, task.nclasses]);

  // Active inspected item
  const activeItem = selectedPoint || hoveredPoint;

  return (
    <div className="space-y-6">
      {/* Controls Bar */}
      <div className="bg-neutral-50 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-4 sm:p-5 shadow-xs">
        <div className="flex flex-wrap items-center justify-between gap-4">
          {/* Dataset Switcher */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400">
              Benchmark Dataset:
            </span>
            <div className="inline-flex rounded-lg p-1 bg-neutral-200/70 dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-700">
              {Object.keys(tasks).map((tKey) => {
                const isSelected = tKey === currentTaskKey;
                return (
                  <button
                    key={tKey}
                    onClick={() => onSelectTask(tKey)}
                    className={`px-3 py-1.5 text-xs sm:text-sm font-medium rounded-md transition-all ${
                      isSelected
                        ? 'bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white shadow-xs'
                        : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
                    }`}
                  >
                    {tasks[tKey].label}
                    <span className="ml-1.5 opacity-60 text-xs">({tasks[tKey].nclasses} classes)</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Labels Toggle */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400">
              Labels:
            </span>
            <div className="inline-flex rounded-lg p-0.5 bg-neutral-200/70 dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-700 text-xs">
              {(['story', 'all', 'none'] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setLabelMode(mode)}
                  className={`px-2.5 py-1 rounded capitalize font-medium transition-colors ${
                    labelMode === mode
                      ? 'bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white shadow-xs'
                      : 'text-neutral-600 dark:text-neutral-400'
                  }`}
                >
                  {mode === 'story' ? 'Story Highlights' : mode}
                </button>
              ))}
            </div>
          </div>

          {/* Accuracy Bar Slider */}
          <div className="flex items-center gap-3 bg-white dark:bg-neutral-800/80 px-3.5 py-1.5 rounded-lg border border-neutral-200 dark:border-neutral-700">
            <label htmlFor="acc-slider" className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
              Accuracy Bar:
            </label>
            <input
              id="acc-slider"
              type="range"
              min="0.50"
              max="1.00"
              step="0.005"
              value={accuracyBar}
              onChange={(e) => setAccuracyBar(parseFloat(e.target.value))}
              className="w-28 sm:w-36 accent-emerald-500 cursor-pointer"
            />
            <span className="font-mono font-bold text-xs sm:text-sm text-emerald-600 dark:text-emerald-400 w-12">
              {(accuracyBar * 100).toFixed(1)}%
            </span>
          </div>

          {/* Toggles */}
          <div className="flex items-center gap-4 text-xs font-medium text-neutral-700 dark:text-neutral-300">
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={showML}
                onChange={(e) => setShowML(e.target.checked)}
                className="rounded border-neutral-300 text-emerald-600 focus:ring-emerald-500"
              />
              <span>Show Zadum-Gemini Gates (TF / E)</span>
            </label>

            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={showPareto}
                onChange={(e) => setShowPareto(e.target.checked)}
                className="rounded border-neutral-300 text-emerald-600 focus:ring-emerald-500"
              />
              <span>Pareto Frontier Line</span>
            </label>
          </div>
        </div>
      </div>

      {/* Main Chart Card */}
      <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-4 sm:p-6 shadow-sm relative overflow-hidden">
        {/* Chart Header Info */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-neutral-100 dark:border-neutral-800/80 gap-2 mb-4">
          <div>
            <h2 className="text-base font-semibold text-neutral-900 dark:text-white flex items-center gap-2">
              <span>Pareto Frontier: Measured Cost vs Accuracy</span>
              <span className="text-xs font-normal text-neutral-500">
                (Held-out N={task.ntest} human-annotated test items, no optimizer contamination)
              </span>
            </h2>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
              Reference benchmark standard: <span className="font-semibold text-neutral-700 dark:text-neutral-300">{task.ref}</span>
            </p>
          </div>

          <div className="flex items-center gap-3 text-xs">
            <span className="flex items-center gap-1.5 text-neutral-600 dark:text-neutral-400">
              <span className="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block"></span> Prompted LLM (Gemini)
            </span>
            <span className="flex items-center gap-1.5 text-neutral-600 dark:text-neutral-400">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block"></span> Zadum-Gemini TF Gate ($0 Cost)
            </span>
            <span className="flex items-center gap-1.5 text-neutral-600 dark:text-neutral-400">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block"></span> Zadum-Gemini E Gate ($0 Cost)
            </span>
            <span className="flex items-center gap-1.5 text-neutral-400">
              <span className="w-2.5 h-2.5 rounded-full border border-neutral-400 inline-block"></span> Disqualified (&lt;Bar)
            </span>
          </div>
        </div>

        {/* SVG Canvas Container */}
        <div className="relative w-full overflow-x-auto">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="w-full h-auto select-none font-sans"
            style={{ minWidth: '780px' }}
          >
            <defs>
              {/* Arrow Marker for GEPA improvements */}
              <marker
                id="gepa-arrow"
                viewBox="0 0 10 10"
                refX="6"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 1 L 8 5 L 0 9 z" fill="#6b7280" />
              </marker>
              <marker
                id="gepa-arrow-gold"
                viewBox="0 0 10 10"
                refX="6"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 1 L 8 5 L 0 9 z" fill="#10b981" />
              </marker>
            </defs>

            {/* Grid Lines - Horizontal Accuracy */}
            {[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map((acc) => {
              const y = yOf(acc);
              return (
                <g key={`y-grid-${acc}`}>
                  <line
                    x1={M.l}
                    y1={y}
                    x2={W - M.r}
                    y2={y}
                    stroke="currentColor"
                    className="text-neutral-200 dark:text-neutral-800"
                    strokeWidth="1"
                    strokeDasharray="2,3"
                  />
                  <text
                    x={M.l - 12}
                    y={y + 4}
                    textAnchor="end"
                    className="text-[11px] fill-neutral-500 dark:text-neutral-400 font-mono"
                  >
                    {(acc * 100).toFixed(0)}%
                  </text>
                </g>
              );
            })}

            {/* Log Grid Lines - Vertical Cost */}
            {pricedTicks.map((cost) => {
              const x = xOf(cost);
              return (
                <g key={`x-grid-${cost}`}>
                  <line
                    x1={x}
                    y1={M.t}
                    x2={x}
                    y2={H - M.b}
                    stroke="currentColor"
                    className="text-neutral-200 dark:text-neutral-800"
                    strokeWidth="1"
                    strokeDasharray="2,3"
                  />
                  <text
                    x={x}
                    y={H - M.b + 18}
                    textAnchor="middle"
                    className="text-[10px] fill-neutral-500 dark:text-neutral-400 font-mono"
                  >
                    ${cost >= 1 ? cost.toFixed(1) : cost.toString().replace(/^0/, '')}
                  </text>
                </g>
              );
            })}

            {/* Zero Cost Column Marker ($0 / Local) */}
            <g>
              <line
                x1={M.l + 30}
                y1={M.t}
                x2={M.l + 30}
                y2={H - M.b}
                stroke="#10b981"
                strokeWidth="1"
                strokeDasharray="3,3"
                opacity="0.3"
              />
              <text
                x={M.l + 30}
                y={H - M.b + 18}
                textAnchor="middle"
                className="text-[11px] font-bold fill-emerald-600 dark:fill-emerald-400 font-mono"
              >
                $0 (Local/Free)
              </text>
            </g>

            {/* Axis Break Mark (Between $0 discrete slot and Log-scale priced models) */}
            <g>
              <line
                x1={M.l + 60}
                y1={H - M.b - 8}
                x2={M.l + 65}
                y2={H - M.b + 8}
                stroke="currentColor"
                className="text-neutral-400"
                strokeWidth="1.5"
              />
              <line
                x1={M.l + 65}
                y1={H - M.b - 8}
                x2={M.l + 70}
                y2={H - M.b + 8}
                stroke="currentColor"
                className="text-neutral-400"
                strokeWidth="1.5"
              />
            </g>

            {/* Axes Lines */}
            <line
              x1={M.l}
              y1={H - M.b}
              x2={W - M.r}
              y2={H - M.b}
              stroke="currentColor"
              className="text-neutral-400 dark:text-neutral-600"
              strokeWidth="1.5"
            />
            <line
              x1={M.l}
              y1={M.t}
              x2={M.l}
              y2={H - M.b}
              stroke="currentColor"
              className="text-neutral-400 dark:text-neutral-600"
              strokeWidth="1.5"
            />

            {/* Axis Titles */}
            <text
              x={M.l + PW / 2}
              y={H - M.b + 42}
              textAnchor="middle"
              className="text-xs font-semibold fill-neutral-700 dark:fill-neutral-300 tracking-wide"
            >
              Measured Cost per 1,000 Classifications ($ Log Scale, priced from actual token count) →
            </text>

            <text
              x={-(M.t + PH / 2)}
              y={24}
              transform="rotate(-90)"
              textAnchor="middle"
              className="text-xs font-semibold fill-neutral-700 dark:fill-neutral-300 tracking-wide"
            >
              Accuracy on Held-Out Test Set (Wilson 95% CI) →
            </text>

            {/* Majority Baseline Reference Line */}
            {task.majority > 0 && (
              <g>
                <line
                  x1={M.l}
                  y1={yOf(task.majority)}
                  x2={W - M.r}
                  y2={yOf(task.majority)}
                  stroke="#ef4444"
                  strokeWidth="1.5"
                  strokeDasharray="6,4"
                  opacity="0.8"
                />
                <text
                  x={W - M.r + 8}
                  y={yOf(task.majority) + 4}
                  className="text-[10px] font-semibold fill-rose-600 dark:fill-rose-400 font-mono"
                >
                  Floor: {(task.majority * 100).toFixed(1)}% (Majority Class)
                </text>
              </g>
            )}

            {/* Accuracy Bar Line */}
            <g>
              <line
                x1={M.l}
                y1={yOf(accuracyBar)}
                x2={W - M.r}
                y2={yOf(accuracyBar)}
                stroke="#10b981"
                strokeWidth="2"
                strokeDasharray="5,5"
              />
              <rect
                x={W - M.r + 6}
                y={yOf(accuracyBar) - 10}
                width={110}
                height={20}
                rx={4}
                className="fill-emerald-500/10 stroke-emerald-500"
                strokeWidth="1"
              />
              <text
                x={W - M.r + 12}
                y={yOf(accuracyBar) + 4}
                className="text-[11px] font-bold fill-emerald-600 dark:fill-emerald-400 font-mono"
              >
                Bar: {(accuracyBar * 100).toFixed(1)}%
              </text>
            </g>

            {/* Pareto Frontier Line */}
            {showPareto && paretoFrontier.length > 1 && (
              <path
                d={paretoFrontier
                  .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xOf(p.cost)} ${yOf(p.acc)}`)
                  .join(' ')}
                fill="none"
                stroke="#10b981"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                opacity="0.6"
              />
            )}

            {/* GEPA Optimization Links (Arrows from baseline prompt -> GEPA prompt) */}
            {visibleRows
              .filter((r) => r.linked)
              .map((r) => {
                const x1 = xOf(r.linked!.cost);
                const y1 = yOf(r.linked!.acc);
                const x2 = xOf(r.cost);
                const y2 = yOf(r.acc);
                return (
                  <line
                    key={`gepa-link-${r.key}`}
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="#9ca3af"
                    strokeWidth="1.5"
                    strokeDasharray="2,2"
                    markerEnd="url(#gepa-arrow)"
                    opacity="0.75"
                  />
                );
              })}

            {/* Data Points and Error Bars */}
            {visibleRows.map((r) => {
              const cx = xOf(r.cost);
              const cy = yOf(r.acc);
              const yLo = yOf(r.lo);
              const yHi = yOf(r.hi);
              const passesBar = r.acc >= accuracyBar;
              const isSelected = activeItem?.key === r.key;
              const color = FAM_COLOR[r.family];

              // Check if we show label
              const isStory = STORY_KEYS.has(r.key) || r.isref || !!r.linked;
              const showText =
                labelMode === 'all' ||
                (labelMode === 'story' && isStory) ||
                isSelected;

              return (
                <g
                  key={r.key}
                  className="cursor-pointer transition-transform"
                  onClick={() => setSelectedPoint(isSelected ? null : r)}
                  onMouseEnter={() => setHoveredPoint(r)}
                  onMouseLeave={() => setHoveredPoint(null)}
                >
                  {/* Wilson 95% Confidence Interval Line */}
                  <line
                    x1={cx}
                    y1={yHi}
                    x2={cx}
                    y2={yLo}
                    stroke={color}
                    strokeWidth="1.5"
                    opacity={isSelected ? 0.9 : 0.4}
                  />
                  {/* Cap top & bottom */}
                  <line
                    x1={cx - 3.5}
                    y1={yHi}
                    x2={cx + 3.5}
                    y2={yHi}
                    stroke={color}
                    strokeWidth="1.5"
                    opacity={isSelected ? 0.9 : 0.4}
                  />
                  <line
                    x1={cx - 3.5}
                    y1={yLo}
                    x2={cx + 3.5}
                    y2={yLo}
                    stroke={color}
                    strokeWidth="1.5"
                    opacity={isSelected ? 0.9 : 0.4}
                  />

                  {/* Halo when hovered/selected */}
                  {isSelected && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={14}
                      fill={color}
                      opacity="0.2"
                      className="animate-pulse"
                    />
                  )}

                  {/* Unchanged GEPA Ring: GEPA ran but kept prompt */}
                  {r.changed === false && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={9}
                      fill="none"
                      stroke={color}
                      strokeWidth="1.5"
                      strokeDasharray="2,2"
                      opacity="0.8"
                    />
                  )}

                  {/* Main Point Dot */}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isSelected ? 7 : 5.5}
                    fill={passesBar ? color : 'white'}
                    stroke={color}
                    strokeWidth={passesBar ? (r.isref ? 2.5 : 1.5) : 2.5}
                    className="transition-all duration-150"
                  />

                  {/* Reference Model Crown Indicator */}
                  {r.isref && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={9}
                      fill="none"
                      stroke="#8b5cf6"
                      strokeWidth="1.5"
                    />
                  )}

                  {/* Label Text */}
                  {showText && (
                    <g transform={`translate(${cx + 8}, ${cy - 4})`}>
                      <rect
                        x="-2"
                        y="-10"
                        width={Math.min(r.label.length * 6.5 + 8, 140)}
                        height="14"
                        rx="3"
                        className="fill-white/80 dark:fill-neutral-900/80"
                      />
                      <text
                        className="text-[10px] sm:text-[11px] font-medium fill-neutral-800 dark:fill-neutral-200"
                        style={{ pointerEvents: 'none' }}
                      >
                        {r.label.length > 22 ? r.label.slice(0, 20) + '…' : r.label}
                      </text>
                    </g>
                  )}
                </g>
              );
            })}

            {/* Live Gemini Candidate Marker if present */}
            {liveGeminiCandidate && (
              <g
                className="cursor-pointer animate-bounce"
                transform={`translate(0, 0)`}
              >
                <circle
                  cx={xOf(liveGeminiCandidate.cost)}
                  cy={yOf(liveGeminiCandidate.acc)}
                  r={18}
                  fill="#10b981"
                  opacity="0.25"
                />
                <circle
                  cx={xOf(liveGeminiCandidate.cost)}
                  cy={yOf(liveGeminiCandidate.acc)}
                  r={7.5}
                  fill="#10b981"
                  stroke="#ffffff"
                  strokeWidth="2.5"
                />
                <g
                  transform={`translate(${xOf(liveGeminiCandidate.cost) + 12}, ${yOf(liveGeminiCandidate.acc) - 10})`}
                >
                  <rect
                    x="0"
                    y="-12"
                    width="145"
                    height="20"
                    rx="4"
                    fill="#10b981"
                  />
                  <text
                    x="6"
                    y="2"
                    className="text-[11px] font-bold fill-white"
                  >
                    ★ {liveGeminiCandidate.modelLabel} (Live)
                  </text>
                </g>
              </g>
            )}
          </svg>
        </div>

        {/* Chart Footer Notes */}
        <div className="mt-4 pt-3 border-t border-neutral-100 dark:border-neutral-800/80 flex flex-wrap items-center justify-between gap-3 text-xs text-neutral-500">
          <div>
            {task.resolution}
          </div>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <span className="font-semibold text-neutral-700 dark:text-neutral-300">Tip:</span>
              Click any point to inspect token usage, Wilson bounds, and McNemar test.
            </span>
          </div>
        </div>
      </div>

      {/* Selected/Hovered Point Inspection Card */}
      {activeItem && (
        <div className="bg-white dark:bg-neutral-900 border-2 border-emerald-500/40 rounded-xl p-5 shadow-md">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-neutral-200 dark:border-neutral-800">
            <div className="flex items-center gap-3">
              <span
                className="w-3.5 h-3.5 rounded-full inline-block"
                style={{ background: FAM_COLOR[activeItem.family] }}
              />
              <h3 className="text-lg font-bold text-neutral-900 dark:text-white">
                {activeItem.label}
              </h3>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-neutral-100 dark:bg-neutral-800 font-mono text-neutral-600 dark:text-neutral-400 capitalize">
                {activeItem.family} {activeItem.isref ? '• Frontier Reference' : ''}
              </span>
            </div>

            <div className="flex items-center gap-3">
              <span
                className={`text-xs px-2.5 py-1 rounded-md font-semibold ${
                  activeItem.acc >= accuracyBar
                    ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300'
                    : 'bg-rose-100 text-rose-800 dark:bg-rose-950/80 dark:text-rose-300'
                }`}
              >
                {activeItem.acc >= accuracyBar ? '✓ Clears Bar' : '✗ Below Bar'}
              </span>
              <button
                onClick={() => setSelectedPoint(null)}
                className="text-xs text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-200"
              >
                ✕ Close
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-4 text-xs sm:text-sm">
            <div className="bg-neutral-50 dark:bg-neutral-800/50 p-3 rounded-lg border border-neutral-200 dark:border-neutral-800">
              <span className="text-neutral-500 text-xs block">Accuracy (Point)</span>
              <span className="text-lg font-bold text-neutral-900 dark:text-white font-mono">
                {formatPct(activeItem.acc)}
              </span>
              <span className="text-[11px] text-neutral-400 block mt-0.5">
                95% Wilson: [{formatPct(activeItem.lo)} — {formatPct(activeItem.hi)}]
              </span>
            </div>

            <div className="bg-neutral-50 dark:bg-neutral-800/50 p-3 rounded-lg border border-neutral-200 dark:border-neutral-800">
              <span className="text-neutral-500 text-xs block">Measured Cost / 1k</span>
              <span className="text-lg font-bold text-emerald-600 dark:text-emerald-400 font-mono">
                {formatCost(activeItem.cost)}
              </span>
              <span className="text-[11px] text-neutral-400 block mt-0.5">
                Basis: {activeItem.basis} (calls: {activeItem.calls})
              </span>
            </div>

            <div className="bg-neutral-50 dark:bg-neutral-800/50 p-3 rounded-lg border border-neutral-200 dark:border-neutral-800">
              <span className="text-neutral-500 text-xs block">Measured Token Profile</span>
              <div className="font-mono text-xs text-neutral-700 dark:text-neutral-300 mt-1 space-y-0.5">
                <div>In: <b>{activeItem.tin.toFixed(1)}</b> tok</div>
                <div>Out: <b>{activeItem.tout.toFixed(1)}</b> tok</div>
              </div>
            </div>

            <div className="bg-neutral-50 dark:bg-neutral-800/50 p-3 rounded-lg border border-neutral-200 dark:border-neutral-800">
              <span className="text-neutral-500 text-xs block">McNemar Mid-P vs Ref</span>
              <div className="text-xs text-neutral-700 dark:text-neutral-300 mt-1">
                {activeItem.isref ? (
                  <span className="text-neutral-400">Baseline Reference</span>
                ) : activeItem.p !== null ? (
                  <div>
                    <div>p = <b>{activeItem.p.toFixed(4)}</b></div>
                    <div className="text-[11px] text-neutral-500 mt-0.5">
                      Diff: {activeItem.diff !== null ? `${(activeItem.diff * 100).toFixed(1)}pp` : '—'}
                    </div>
                  </div>
                ) : (
                  <span className="text-neutral-400">N/A</span>
                )}
              </div>
            </div>
          </div>

          {/* Statistical Verdict & Notes */}
          <div className="mt-4 pt-3 border-t border-neutral-100 dark:border-neutral-800/80 text-xs space-y-2">
            {activeItem.verdict && (
              <div className="flex items-start gap-2">
                <span className="font-semibold text-neutral-600 dark:text-neutral-400 whitespace-nowrap">
                  Statistical Verdict:
                </span>
                <span className="text-neutral-800 dark:text-neutral-200">
                  {activeItem.verdict}
                  {activeItem.passes_holm !== null && (
                    <span className="ml-1 text-[11px] text-neutral-500">
                      (Holm-Bonferroni corrected: {activeItem.passes_holm ? 'Pass' : 'Fail'})
                    </span>
                  )}
                </span>
              </div>
            )}

            {activeItem.note && (
              <div className="flex items-start gap-2">
                <span className="font-semibold text-neutral-600 dark:text-neutral-400 whitespace-nowrap">
                  Note:
                </span>
                <span className="text-neutral-700 dark:text-neutral-300 italic">
                  "{activeItem.note}"
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
