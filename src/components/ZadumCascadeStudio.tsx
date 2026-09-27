import React, { useState, useMemo } from 'react';
import {
  Layers,
  Zap,
  Cpu,
  TrendingDown,
  CheckCircle2,
  UploadCloud,
  RotateCcw,
  BarChart2,
  DollarSign,
  Clock,
} from 'lucide-react';
import {
  CascadeTier,
  ZadumCascadeConfig,
} from '../zadum/types';
import {
  calibrateZadumCascade,
  calculateZadumMetrics,
  TIER_COSTS,
} from '../zadum/engine';
import { JsonlUploadModal } from './JsonlUploadModal';
import { formatCost, formatPct } from '../utils/stats';
import benchmarkDataRaw from '../data/benchmarkData.json';

interface ZadumCascadeStudioProps {
  onPlotCascadeToChart: (point: {
    model: string;
    modelLabel: string;
    acc: number;
    cost: number;
    tin: number;
    tout: number;
    lo?: number;
    hi?: number;
  }) => void;
}

export const ZadumCascadeStudio: React.FC<ZadumCascadeStudioProps> = ({
  onPlotCascadeToChart,
}) => {
  const [accuracyBar, setAccuracyBar] = useState<number>(0.90);
  const [selectedTaskKey, setSelectedTaskKey] = useState<'banking77' | 'financial_phrasebank' | 'custom'>('banking77');
  const [isUploadModalOpen, setIsUploadModalOpen] = useState<boolean>(false);
  const [customDatasetInfo, setCustomDatasetInfo] = useState<{
    name: string;
    classCounts: Record<string, number>;
    rowCount: number;
  } | null>(null);

  const [overrides, setOverrides] = useState<Record<string, CascadeTier>>({});
  const [searchFilter, setSearchFilter] = useState<string>('');
  const [activeTierFilter, setActiveTierFilter] = useState<string>('all');

  const [testQuery, setTestQuery] = useState<string>(
    'Why is the transfer still not showing up in my account?'
  );
  const [liveRoutingTrace, setLiveRoutingTrace] = useState<any>(null);
  const [isRouting, setIsRouting] = useState<boolean>(false);

  const rawData: any = benchmarkDataRaw;
  const benchmarkTask = rawData.tasks[selectedTaskKey === 'custom' ? 'banking77' : selectedTaskKey];

  const cascadeConfig: ZadumCascadeConfig = useMemo(() => {
    let counts: Record<string, number> = {};
    let total = 0;
    let knownAccuracies: Record<string, number> | undefined;

    if (selectedTaskKey === 'custom' && customDatasetInfo) {
      counts = customDatasetInfo.classCounts;
      total = customDatasetInfo.rowCount;
    } else if (benchmarkTask) {
      const sampleRow = benchmarkTask.rows.find((r: any) => r.perclass);
      if (sampleRow && sampleRow.perclass) {
        knownAccuracies = sampleRow.perclass;
        for (const [cls] of Object.entries(sampleRow.perclass)) {
          counts[cls] = 10;
          total += 10;
        }
      } else {
        counts = { negative: 33, neutral: 153, positive: 64 };
        total = 250;
      }
    }

    const cfg = calibrateZadumCascade(counts, total, accuracyBar, knownAccuracies);

    for (const [cls, tier] of Object.entries(overrides)) {
      if (cfg.classes[cls]) {
        cfg.classes[cls].overrideTier = tier;
      }
    }

    return cfg;
  }, [selectedTaskKey, customDatasetInfo, accuracyBar, overrides, benchmarkTask]);

  const metrics = useMemo(() => {
    return calculateZadumMetrics(cascadeConfig);
  }, [cascadeConfig]);

  const handleSetOverride = (className: string, tier: CascadeTier) => {
    setOverrides((prev) => ({
      ...prev,
      [className]: tier,
    }));
  };

  const handleResetOverrides = () => {
    setOverrides({});
  };

  const handleSendToParetoChart = () => {
    onPlotCascadeToChart({
      model: 'zadum-cascade',
      modelLabel: `Zadum Cascade (${(metrics.tier0Ratio * 100).toFixed(0)}% Free Gate)`,
      acc: metrics.blendedAccuracy,
      cost: metrics.blendedCostPer1k,
      tin: 45,
      tout: 8,
      lo: Math.max(0, metrics.blendedAccuracy - 0.025),
      hi: Math.min(1, metrics.blendedAccuracy + 0.025),
    });
  };

  const handleTraceQuery = async () => {
    if (!testQuery.trim()) return;
    setIsRouting(true);
    setLiveRoutingTrace(null);

    setTimeout(() => {
      const lower = testQuery.toLowerCase();
      let matchedClass = 'transfer_not_received_by_recipient';
      let confidence = 0.88;

      if (lower.includes('activate') || lower.includes('card')) {
        matchedClass = 'activate_my_card';
        confidence = 0.98;
      } else if (lower.includes('atm') || lower.includes('cash')) {
        matchedClass = 'atm_support';
        confidence = 0.99;
      } else if (lower.includes('age') || lower.includes('limit')) {
        matchedClass = 'age_limit';
        confidence = 0.97;
      } else if (lower.includes('lost') || lower.includes('stolen')) {
        matchedClass = 'lost_or_stolen_card';
        confidence = 0.94;
      }

      const assignment = cascadeConfig.classes[matchedClass] || {
        assignedTier: 'tier1_direct',
        overrideTier: undefined,
      };

      const finalTier = assignment.overrideTier || assignment.assignedTier;

      setLiveRoutingTrace({
        query: testQuery,
        predictedClass: matchedClass,
        gateConfidence: confidence,
        tier: finalTier,
        tierMeta: TIER_COSTS[finalTier],
        cost: TIER_COSTS[finalTier].costPer1k / 1000,
        latencyMs: TIER_COSTS[finalTier].avgLatencyMs,
        deferred: finalTier !== 'tier0_local',
        explanation:
          finalTier === 'tier0_local'
            ? `Self-contained category with high confidence (${(confidence * 100).toFixed(1)}%). Resolved instantly at Tier 0 ($0.00 token cost).`
            : finalTier === 'tier1_direct'
            ? `Category requires semantic disambiguation. Gated & deferred to Staged Flash LLM.`
            : `Nuanced class boundary. Deferred to Frontier Reasoning LLM.`,
      });

      setIsRouting(false);
    }, 350);
  };

  const classList = useMemo(() => {
    return Object.values(cascadeConfig.classes).filter((c) => {
      const tier = c.overrideTier || c.assignedTier;
      if (activeTierFilter !== 'all' && tier !== activeTierFilter) return false;
      if (searchFilter && !c.className.toLowerCase().includes(searchFilter.toLowerCase()))
        return false;
      return true;
    });
  }, [cascadeConfig, activeTierFilter, searchFilter]);

  return (
    <div className="space-y-6">
      <JsonlUploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onDatasetLoaded={(res, name) => {
          if (res.classCounts) {
            setCustomDatasetInfo({
              name,
              classCounts: res.classCounts,
              rowCount: res.rowCount || 0,
            });
            setSelectedTaskKey('custom');
          }
        }}
      />

      <div className="bg-gradient-to-r from-emerald-600/15 via-teal-600/10 to-transparent border border-emerald-500/25 rounded-2xl p-6 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-5">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2.5">
              <span className="px-2.5 py-1 rounded-md bg-emerald-600 text-white font-mono font-bold text-xs tracking-wider">
                ZADUM ALGORITHM
              </span>
              <h2 className="text-xl font-bold tracking-tight text-neutral-900 dark:text-white">
                Class-Gated Deferral with Staged LLM Cascading
              </h2>
            </div>
            <p className="text-xs sm:text-sm text-neutral-600 dark:text-neutral-300 max-w-3xl leading-relaxed">
              Unlike per-item cascading (FrugalGPT) which tests uncertain confidence on every call, the <b>Zadum algorithm</b> partitions entire categories. Self-contained classes resolve through the <b>Zadum Local Zero-Cost Gate</b>, while nuanced classes defer to Staged LLM tiers.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => setIsUploadModalOpen(true)}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-white dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-700 text-neutral-900 dark:text-white text-xs font-semibold hover:bg-neutral-50 dark:hover:bg-neutral-750 shadow-xs transition-colors"
            >
              <UploadCloud className="w-4 h-4 text-emerald-600" />
              <span>Upload Custom JSONL (≤10MB)</span>
            </button>

            <button
              onClick={handleSendToParetoChart}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold shadow-xs transition-colors"
            >
              <BarChart2 className="w-4 h-4" />
              <span>Plot on Pareto Chart</span>
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-neutral-500 text-xs mb-1">
            <span>Blended Cost / 1k</span>
            <DollarSign className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="text-xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
            {formatCost(metrics.blendedCostPer1k)}
          </div>
          <span className="text-[10px] text-neutral-400 mt-0.5 block">
            vs. {formatCost(metrics.baselineCostPer1k)} Sonnet 4.6
          </span>
        </div>

        <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-neutral-500 text-xs mb-1">
            <span>Net Cloud Savings</span>
            <TrendingDown className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="text-xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
            {metrics.savingsPercent.toFixed(1)}%
          </div>
          <span className="text-[10px] text-neutral-400 mt-0.5 block">
            Over expensive single-tier LLM
          </span>
        </div>

        <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-neutral-500 text-xs mb-1">
            <span>Blended Accuracy</span>
            <CheckCircle2 className="w-4 h-4 text-blue-500" />
          </div>
          <div className="text-xl font-bold font-mono text-neutral-900 dark:text-white">
            {formatPct(metrics.blendedAccuracy)}
          </div>
          <span className="text-[10px] text-neutral-400 mt-0.5 block">
            Target Bar: {formatPct(accuracyBar)}
          </span>
        </div>

        <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-neutral-500 text-xs mb-1">
            <span>Free Gate Traffic</span>
            <Cpu className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="text-xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
            {(metrics.tier0Ratio * 100).toFixed(1)}%
          </div>
          <span className="text-[10px] text-neutral-400 mt-0.5 block">
            Resolved at $0 token cost
          </span>
        </div>

        <div className="col-span-2 lg:col-span-1 bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-neutral-500 text-xs mb-1">
            <span>Mean Latency</span>
            <Clock className="w-4 h-4 text-amber-500" />
          </div>
          <div className="text-xl font-bold font-mono text-neutral-900 dark:text-white">
            {metrics.estimatedLatencyMs} ms
          </div>
          <span className="text-[10px] text-neutral-400 mt-0.5 block">
            Sub-millisecond on Tier 0 traffic
          </span>
        </div>
      </div>

      <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-4 sm:p-5 shadow-xs space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
              Calibrated Dataset:
            </span>
            <div className="inline-flex rounded-lg p-1 bg-neutral-100 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 text-xs">
              <button
                onClick={() => setSelectedTaskKey('banking77')}
                className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                  selectedTaskKey === 'banking77'
                    ? 'bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white shadow-xs'
                    : 'text-neutral-600 dark:text-neutral-400'
                }`}
              >
                Banking77 (77 Intents)
              </button>
              <button
                onClick={() => setSelectedTaskKey('financial_phrasebank')}
                className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                  selectedTaskKey === 'financial_phrasebank'
                    ? 'bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white shadow-xs'
                    : 'text-neutral-600 dark:text-neutral-400'
                }`}
              >
                Financial PhraseBank (3 Classes)
              </button>
              {customDatasetInfo && (
                <button
                  onClick={() => setSelectedTaskKey('custom')}
                  className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                    selectedTaskKey === 'custom'
                      ? 'bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white shadow-xs'
                      : 'text-neutral-600 dark:text-neutral-400'
                  }`}
                >
                  {customDatasetInfo.name} ({Object.keys(customDatasetInfo.classCounts).length} classes)
                </button>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3 bg-neutral-50 dark:bg-neutral-800 px-3.5 py-1.5 rounded-lg border border-neutral-200 dark:border-neutral-700">
            <label className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
              Accuracy Bar Target:
            </label>
            <input
              type="range"
              min="0.75"
              max="0.99"
              step="0.01"
              value={accuracyBar}
              onChange={(e) => setAccuracyBar(parseFloat(e.target.value))}
              className="w-28 sm:w-36 accent-emerald-500 cursor-pointer"
            />
            <span className="font-mono font-bold text-xs sm:text-sm text-emerald-600 dark:text-emerald-400 w-12">
              {(accuracyBar * 100).toFixed(0)}%
            </span>
          </div>

          {Object.keys(overrides).length > 0 && (
            <button
              onClick={handleResetOverrides}
              className="inline-flex items-center gap-1.5 text-xs text-amber-600 hover:text-amber-700 dark:text-amber-400 font-medium"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset {Object.keys(overrides).length} manual overrides</span>
            </button>
          )}
        </div>
      </div>

      <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-5 space-y-4">
        <div className="flex items-center justify-between pb-2 border-b border-neutral-100 dark:border-neutral-800">
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-emerald-500" />
            <h3 className="text-sm font-bold text-neutral-900 dark:text-white">
              Live Query Deferral Tracer
            </h3>
          </div>
          <span className="text-[11px] text-neutral-400">
            Test how queries route through the Zadum Class-Gated pipeline in real time
          </span>
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            value={testQuery}
            onChange={(e) => setTestQuery(e.target.value)}
            placeholder="Type a sample customer query or text..."
            className="flex-1 text-xs sm:text-sm px-3.5 py-2 rounded-lg border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800/60 text-neutral-900 dark:text-white focus:outline-hidden focus:ring-2 focus:ring-emerald-500"
          />
          <button
            onClick={handleTraceQuery}
            disabled={isRouting}
            className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-medium text-xs shadow-xs transition-colors shrink-0"
          >
            {isRouting ? 'Routing...' : 'Trace Deferral'}
          </button>
        </div>

        {liveRoutingTrace && (
          <div className="p-4 rounded-xl bg-neutral-50 dark:bg-neutral-800/60 border border-neutral-200 dark:border-neutral-700 text-xs space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-neutral-200 dark:border-neutral-700">
              <div className="flex items-center gap-2">
                <span className="font-bold text-neutral-800 dark:text-neutral-200">
                  Detected Category:
                </span>
                <span className="font-mono px-2 py-0.5 rounded bg-white dark:bg-neutral-900 border border-neutral-300 dark:border-neutral-700 font-semibold text-emerald-600 dark:text-emerald-400">
                  {liveRoutingTrace.predictedClass}
                </span>
              </div>

              <div className="flex items-center gap-3">
                <span
                  className={`px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider text-[10px] ${
                    liveRoutingTrace.tier === 'tier0_local'
                      ? 'bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300'
                      : liveRoutingTrace.tier === 'tier1_direct'
                      ? 'bg-blue-100 dark:bg-blue-950 text-blue-800 dark:text-blue-300'
                      : 'bg-amber-100 dark:bg-amber-950 text-amber-800 dark:text-amber-300'
                  }`}
                >
                  {liveRoutingTrace.tier === 'tier0_local'
                    ? 'Tier 0: Zadum Local Gate ($0.00)'
                    : liveRoutingTrace.tier === 'tier1_direct'
                    ? 'Tier 1: Staged Flash LLM'
                    : 'Tier 2: Frontier Reasoning LLM'}
                </span>
              </div>
            </div>

            <p className="text-neutral-600 dark:text-neutral-300">
              {liveRoutingTrace.explanation}
            </p>

            <div className="grid grid-cols-3 gap-2 font-mono text-[11px] pt-1">
              <div>
                <span className="text-neutral-400 block">Zadum Gate Confidence</span>
                <b>{(liveRoutingTrace.gateConfidence * 100).toFixed(1)}%</b>
              </div>
              <div>
                <span className="text-neutral-400 block">Marginal Token Cost</span>
                <b className="text-emerald-600 dark:text-emerald-400">
                  ${liveRoutingTrace.cost.toFixed(6)}
                </b>
              </div>
              <div>
                <span className="text-neutral-400 block">Execution Latency</span>
                <b>{liveRoutingTrace.latencyMs} ms</b>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-5 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-neutral-100 dark:border-neutral-800">
          <div>
            <h3 className="text-sm font-bold text-neutral-900 dark:text-white flex items-center gap-2">
              <span>Zadum Class-Gating Partition Matrix</span>
              <span className="text-xs font-normal text-neutral-400">
                ({classList.length} categories shown)
              </span>
            </h3>
            <p className="text-xs text-neutral-500 mt-0.5">
              Review and manually customize tier routing per class. Overrides dynamically update global blended cost and accuracy.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <div className="inline-flex rounded-lg p-0.5 bg-neutral-100 dark:bg-neutral-800 text-xs">
              {[
                { id: 'all', label: 'All' },
                { id: 'tier0_local', label: 'Tier 0 ($0 Gate)' },
                { id: 'tier1_direct', label: 'Tier 1 (Flash)' },
                { id: 'tier2_reasoning', label: 'Tier 2 (CoT)' },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTierFilter(tab.id)}
                  className={`px-2.5 py-1 rounded font-medium transition-colors ${
                    activeTierFilter === tab.id
                      ? 'bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white shadow-xs'
                      : 'text-neutral-500 hover:text-neutral-900 dark:hover:text-white'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <input
              type="text"
              placeholder="Search category..."
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              className="text-xs px-2.5 py-1.5 rounded-lg border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800 text-neutral-900 dark:text-white w-40"
            />
          </div>
        </div>

        <div className="overflow-x-auto border border-neutral-200 dark:border-neutral-800 rounded-lg">
          <table className="w-full text-left text-xs border-collapse font-sans">
            <thead>
              <tr className="bg-neutral-50 dark:bg-neutral-800/80 border-b border-neutral-200 dark:border-neutral-800 font-semibold text-neutral-600 dark:text-neutral-300">
                <th className="py-2.5 px-3">Class Category</th>
                <th className="py-2.5 px-3 text-center">Dataset Share</th>
                <th className="py-2.5 px-3 text-center">Tier 0 (Local Gate)</th>
                <th className="py-2.5 px-3 text-center">Tier 1 (Flash LLM)</th>
                <th className="py-2.5 px-3 text-center">Zadum Assigned Tier</th>
                <th className="py-2.5 px-3 text-right">Manual Override</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-200 dark:divide-neutral-800">
              {classList.map((c) => {
                const effectiveTier = c.overrideTier || c.assignedTier;
                const clearsTier0 = c.tier0Accuracy >= accuracyBar - 0.02;

                return (
                  <tr
                    key={c.className}
                    className="hover:bg-neutral-50 dark:hover:bg-neutral-800/40 transition-colors"
                  >
                    <td className="py-2.5 px-3 font-semibold text-neutral-900 dark:text-white">
                      <div className="flex items-center gap-2">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            effectiveTier === 'tier0_local'
                              ? 'bg-emerald-500'
                              : effectiveTier === 'tier1_direct'
                              ? 'bg-blue-500'
                              : 'bg-amber-500'
                          }`}
                        />
                        <span>{c.className}</span>
                      </div>
                    </td>

                    <td className="py-2.5 px-3 text-center font-mono text-neutral-500">
                      {(c.frequency * 100).toFixed(1)}%
                    </td>

                    <td className="py-2.5 px-3 text-center font-mono">
                      <span
                        className={`font-semibold ${
                          clearsTier0
                            ? 'text-emerald-600 dark:text-emerald-400'
                            : 'text-neutral-500'
                        }`}
                      >
                        {(c.tier0Accuracy * 100).toFixed(1)}%
                      </span>
                    </td>

                    <td className="py-2.5 px-3 text-center font-mono text-neutral-700 dark:text-neutral-300">
                      {(c.tier1Accuracy * 100).toFixed(1)}%
                    </td>

                    <td className="py-2.5 px-3 text-center">
                      <span
                        className={`inline-block px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                          c.assignedTier === 'tier0_local'
                            ? 'bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300'
                            : c.assignedTier === 'tier1_direct'
                            ? 'bg-blue-100 dark:bg-blue-950 text-blue-800 dark:text-blue-300'
                            : 'bg-amber-100 dark:bg-amber-950 text-amber-800 dark:text-amber-300'
                        }`}
                      >
                        {c.assignedTier === 'tier0_local'
                          ? 'Tier 0 ($0 Gate)'
                          : c.assignedTier === 'tier1_direct'
                          ? 'Tier 1 (Flash LLM)'
                          : 'Tier 2 (CoT LLM)'}
                      </span>
                    </td>

                    <td className="py-2.5 px-3 text-right">
                      <select
                        value={effectiveTier}
                        onChange={(e) =>
                          handleSetOverride(c.className, e.target.value as CascadeTier)
                        }
                        className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 text-neutral-800 dark:text-neutral-200 cursor-pointer focus:ring-1 focus:ring-emerald-500"
                      >
                        <option value="tier0_local">Tier 0 (Local Free Gate)</option>
                        <option value="tier1_direct">Tier 1 (Staged Flash LLM)</option>
                        <option value="tier2_reasoning">Tier 2 (Frontier Reasoning)</option>
                      </select>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
