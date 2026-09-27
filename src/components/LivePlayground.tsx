import React, { useState } from 'react';
import { Sparkles, Play, RefreshCw, Layers, Zap, CheckCircle2, XCircle, ArrowRight, TrendingDown, DollarSign } from 'lucide-react';
import samplesData from '../data/samples.json';
import { formatCost, formatPct, wilsonInterval } from '../utils/stats';

interface LivePlaygroundProps {
  onPlotCandidate: (candidate: {
    model: string;
    modelLabel: string;
    acc: number;
    cost: number;
    tin: number;
    tout: number;
    lo?: number;
    hi?: number;
  }) => void;
  currentTaskKey: string;
}

const AVAILABLE_MODELS = [
  {
    id: 'gemini-2.5-flash',
    name: 'Gemini 2.5 Flash',
    tag: 'Lightweight / Fast',
    inPrice: 0.075,
    outPrice: 0.30,
    desc: 'Lightweight high-efficiency workhorse model on GCP.',
  },
  {
    id: 'gemini-3.1-flash-lite',
    name: 'Gemini 3.1 Flash-Lite',
    tag: 'Ultra-budget / Fast',
    inPrice: 0.075,
    outPrice: 0.30,
    desc: 'Lowest cost per token; optimal for high-throughput classification.',
  },
  {
    id: 'gemini-3.8-flash',
    name: 'Gemini 3.8 Flash',
    tag: 'Balanced Frontier',
    inPrice: 0.15,
    outPrice: 0.60,
    desc: 'High precision and fast generation for complex queries.',
  },
  {
    id: 'gemini-3.1-pro-preview',
    name: 'Gemini 3.1 Pro',
    tag: 'Reasoning Frontier',
    inPrice: 1.25,
    outPrice: 5.00,
    desc: 'Frontier reasoning intelligence; higher token rates.',
  },
];

export const LivePlayground: React.FC<LivePlaygroundProps> = ({ onPlotCandidate, currentTaskKey }) => {
  const [selectedModel, setSelectedModel] = useState<string>('gemini-3.1-flash-lite');
  const [task, setTask] = useState<'financial_phrasebank' | 'banking77' | 'cuad_covenant_not_to_sue'>(
    (currentTaskKey === 'cuad_covenant_not_to_sue' || currentTaskKey === 'banking77' || currentTaskKey === 'financial_phrasebank')
      ? currentTaskKey
      : 'cuad_covenant_not_to_sue'
  );
  const [promptType, setPromptType] = useState<'direct' | 'reasoning'>('direct');
  const [inputText, setInputText] = useState<string>('');
  const [expectedLabel, setExpectedLabel] = useState<string>('');
  const [customPrompt, setCustomPrompt] = useState<string>('');
  const [showPromptEditor, setShowPromptEditor] = useState<boolean>(false);

  // Evaluation states
  const [loading, setLoading] = useState<boolean>(false);
  const [batchLoading, setBatchLoading] = useState<boolean>(false);
  const [singleResult, setSingleResult] = useState<any>(null);
  const [batchResult, setBatchResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Load sample on mount or task change
  React.useEffect(() => {
    const list = (samplesData as Record<string, any[]>)[task] || [];
    if (list.length > 0) {
      setInputText(list[0].text);
      setExpectedLabel(list[0].label);
    }
  }, [task]);

  const loadSample = (index: number) => {
    const list = (samplesData as Record<string, any[]>)[task] || [];
    if (list[index]) {
      setInputText(list[index].text);
      setExpectedLabel(list[index].label);
      setSingleResult(null);
      setError(null);
    }
  };

  const handleEvaluateSingle = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/evaluate-live', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          task,
          text: inputText,
          promptType,
          customInstruction: showPromptEditor ? customPrompt : '',
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Server responded with status ${res.status}`);
      }

      const data = await res.json();
      setSingleResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to complete evaluation');
    } finally {
      setLoading(false);
    }
  };

  const handleRunBatch = async () => {
    const list = (samplesData as Record<string, any[]>)[task] || [];
    if (list.length === 0) return;
    setBatchLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/benchmark-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          task,
          samples: list,
          promptType,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Batch benchmark failed`);
      }

      const data = await res.json();
      setBatchResult(data);

      // Compute Wilson interval for batch
      const [lo, hi] = wilsonInterval(data.correctCount, data.sampleCount);

      // Plot candidate onto chart
      onPlotCandidate({
        model: data.model,
        modelLabel: `${data.modelLabel} (${promptType === 'direct' ? 'Direct' : 'CoT'})`,
        acc: data.accuracy,
        cost: data.costPer1k,
        tin: data.avgInTokens,
        tout: data.avgOutTokens,
        lo,
        hi,
      });
    } catch (err: any) {
      setError(err.message || 'Batch evaluation encountered an error');
    } finally {
      setBatchLoading(false);
    }
  };

  const currentModelMeta = AVAILABLE_MODELS.find((m) => m.id === selectedModel) || AVAILABLE_MODELS[0];
  const sampleList = samplesData[task] || [];

  return (
    <div className="space-y-6">
      {/* Intro banner */}
      <div className="bg-gradient-to-r from-emerald-500/10 via-teal-500/5 to-transparent border border-emerald-500/20 rounded-xl p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
              <h2 className="text-base sm:text-lg font-bold text-neutral-900 dark:text-white">
                Live Gemini Downshift Playground
              </h2>
            </div>
            <p className="text-xs sm:text-sm text-neutral-600 dark:text-neutral-400 mt-1 max-w-3xl">
              Execute live classifications via Gemini. Measure real wall-clock latency, input/output token counts, and verify the cost impact of reasoning vs direct prompts.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold px-2.5 py-1 rounded bg-white dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 text-neutral-700 dark:text-neutral-300">
              API Active: @google/genai
            </span>
          </div>
        </div>
      </div>

      {/* Configuration Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Settings */}
        <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-5 space-y-5">
          <h3 className="text-sm font-semibold text-neutral-900 dark:text-white pb-2 border-b border-neutral-100 dark:border-neutral-800">
            1. Select Model & Profile
          </h3>

          {/* Model Selection */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-neutral-600 dark:text-neutral-400 block">
              Gemini Candidate:
            </label>
            <div className="space-y-2">
              {AVAILABLE_MODELS.map((m) => {
                const isSelected = selectedModel === m.id;
                return (
                  <div
                    key={m.id}
                    onClick={() => setSelectedModel(m.id)}
                    className={`p-3 rounded-lg border text-xs cursor-pointer transition-all ${
                      isSelected
                        ? 'border-emerald-500 bg-emerald-50/50 dark:bg-emerald-950/20 text-neutral-900 dark:text-white shadow-xs'
                        : 'border-neutral-200 dark:border-neutral-800 hover:border-neutral-300 dark:hover:border-neutral-700'
                    }`}
                  >
                    <div className="flex items-center justify-between font-semibold">
                      <span>{m.name}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 font-mono">
                        ${m.inPrice}/${m.outPrice}
                      </span>
                    </div>
                    <p className="text-[11px] text-neutral-500 mt-1">{m.desc}</p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Task Selection */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-neutral-600 dark:text-neutral-400 block">
              Benchmark Dataset Task:
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
              <button
                onClick={() => setTask('cuad_covenant_not_to_sue')}
                className={`p-2.5 rounded-lg border text-left font-medium transition-all ${
                  task === 'cuad_covenant_not_to_sue'
                    ? 'border-emerald-500 bg-emerald-50/60 dark:bg-emerald-950/20 text-emerald-900 dark:text-emerald-200'
                    : 'border-neutral-200 dark:border-neutral-800 text-neutral-600 dark:text-neutral-400'
                }`}
              >
                <div className="truncate font-semibold">CUAD Covenant Not To Sue</div>
                <span className="text-[10px] opacity-70">LegalBench • 2 classes (yes/no)</span>
              </button>

              <button
                onClick={() => setTask('financial_phrasebank')}
                className={`p-2.5 rounded-lg border text-left font-medium transition-all ${
                  task === 'financial_phrasebank'
                    ? 'border-emerald-500 bg-emerald-50/60 dark:bg-emerald-950/20 text-emerald-900 dark:text-emerald-200'
                    : 'border-neutral-200 dark:border-neutral-800 text-neutral-600 dark:text-neutral-400'
                }`}
              >
                <div className="font-semibold">Financial PhraseBank</div>
                <span className="text-[10px] opacity-70">3 classes (sentiments)</span>
              </button>

              <button
                onClick={() => setTask('banking77')}
                className={`p-2.5 rounded-lg border text-left font-medium transition-all ${
                  task === 'banking77'
                    ? 'border-emerald-500 bg-emerald-50/60 dark:bg-emerald-950/20 text-emerald-900 dark:text-emerald-200'
                    : 'border-neutral-200 dark:border-neutral-800 text-neutral-600 dark:text-neutral-400'
                }`}
              >
                <div className="font-semibold">Banking77</div>
                <span className="text-[10px] opacity-70">77 customer intents</span>
              </button>
            </div>
          </div>

          {/* Prompt Style: Direct vs Reasoning */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-neutral-600 dark:text-neutral-400">
                Execution Profile (Prompt Strategy):
              </label>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <button
                onClick={() => setPromptType('direct')}
                className={`p-2.5 rounded-lg border text-left font-medium transition-all ${
                  promptType === 'direct'
                    ? 'border-blue-500 bg-blue-50/60 dark:bg-blue-950/20 text-blue-900 dark:text-blue-200'
                    : 'border-neutral-200 dark:border-neutral-800 text-neutral-600 dark:text-neutral-400'
                }`}
              >
                <div className="font-bold flex items-center gap-1">
                  <Zap className="w-3.5 h-3.5 text-blue-500" /> DIRECT
                </div>
                <div className="text-[10px] text-neutral-500 mt-1">
                  dspy.Predict (~5-15 output tokens). Lowest cost per call.
                </div>
              </button>

              <button
                onClick={() => setPromptType('reasoning')}
                className={`p-2.5 rounded-lg border text-left font-medium transition-all ${
                  promptType === 'reasoning'
                    ? 'border-amber-500 bg-amber-50/60 dark:bg-amber-950/20 text-amber-900 dark:text-amber-200'
                    : 'border-neutral-200 dark:border-neutral-800 text-neutral-600 dark:text-neutral-400'
                }`}
              >
                <div className="font-bold flex items-center gap-1">
                  <Layers className="w-3.5 h-3.5 text-amber-500" /> REASONING
                </div>
                <div className="text-[10px] text-neutral-500 mt-1">
                  dspy.ChainOfThought (~150-300 output tokens). Unsolicited thinking tax.
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* Center & Right Column: Input & Live Results */}
        <div className="lg:col-span-2 space-y-6">
          {/* Sample Selector & Input Box */}
          <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-neutral-100 dark:border-neutral-800">
              <h3 className="text-sm font-semibold text-neutral-900 dark:text-white">
                2. Test Input Sentence
              </h3>

              {/* Sample Chips */}
              <div className="flex items-center gap-1.5 overflow-x-auto text-xs py-1">
                <span className="text-[11px] text-neutral-400 mr-1">Load sample:</span>
                {sampleList.slice(0, 5).map((s, i) => (
                  <button
                    key={s.id}
                    onClick={() => loadSample(i)}
                    className="px-2 py-0.5 rounded border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800 hover:bg-neutral-100 dark:hover:bg-neutral-700 font-mono text-[10px]"
                  >
                    #{i + 1}
                  </button>
                ))}
              </div>
            </div>

            <textarea
              rows={3}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Enter text to classify..."
              className="w-full text-xs sm:text-sm p-3 rounded-lg border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800/50 text-neutral-900 dark:text-white focus:ring-2 focus:ring-emerald-500 focus:outline-hidden"
            />

            <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2">
                <span className="text-neutral-500">Expected Ground Truth Label:</span>
                <span className="font-mono font-semibold px-2 py-0.5 rounded bg-neutral-100 dark:bg-neutral-800 text-neutral-800 dark:text-neutral-200">
                  {expectedLabel || '—'}
                </span>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleEvaluateSingle}
                  disabled={loading || batchLoading || !inputText.trim()}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-medium shadow-xs disabled:opacity-50 transition-colors"
                >
                  {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                  <span>Evaluate Single Call</span>
                </button>

                <button
                  onClick={handleRunBatch}
                  disabled={loading || batchLoading}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-neutral-900 hover:bg-neutral-800 text-white dark:bg-white dark:text-neutral-900 dark:hover:bg-neutral-100 font-medium shadow-xs disabled:opacity-50 transition-colors"
                >
                  {batchLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Layers className="w-3.5 h-3.5" />}
                  <span>Batch Test 10 Samples & Plot</span>
                </button>
              </div>
            </div>

            {error && (
              <div className="p-3 rounded-lg bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900 text-rose-800 dark:text-rose-300 text-xs">
                {error}
              </div>
            )}
          </div>

          {/* Single Call Result Display */}
          {singleResult && (
            <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-neutral-100 dark:border-neutral-800">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                  <h4 className="text-sm font-bold text-neutral-900 dark:text-white">
                    Live Call Telemetry & Token Measurements
                  </h4>
                </div>
                <span className="text-xs font-mono text-neutral-500">
                  Latency: {singleResult.latencyMs} ms
                </span>
              </div>

              {/* Prediction vs Ground Truth Match */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 rounded-lg bg-neutral-50 dark:bg-neutral-800/40 border border-neutral-200 dark:border-neutral-700">
                <div className="space-y-1">
                  <span className="text-[11px] uppercase tracking-wider text-neutral-400 block font-semibold">
                    Extracted Prediction
                  </span>
                  <div className="text-base font-bold text-neutral-900 dark:text-white font-mono flex items-center gap-2">
                    <span>{singleResult.predictedLabel}</span>
                    {expectedLabel && (
                      singleResult.predictedLabel.toLowerCase() === expectedLabel.toLowerCase() ? (
                        <span className="inline-flex items-center gap-1 text-xs text-emerald-600 font-medium">
                          <CheckCircle2 className="w-4 h-4" /> Correct Match
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-xs text-rose-500 font-medium">
                          <XCircle className="w-4 h-4" /> Expected: {expectedLabel}
                        </span>
                      )
                    )}
                  </div>
                </div>

                <div className="text-right">
                  <span className="text-[11px] uppercase tracking-wider text-neutral-400 block font-semibold">
                    Measured Cost / 1k Calls
                  </span>
                  <span className="text-base font-bold text-emerald-600 dark:text-emerald-400 font-mono">
                    {formatCost(singleResult.costPer1k)}
                  </span>
                </div>
              </div>

              {/* Token Economics Breakdown */}
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div className="p-3 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-850">
                  <span className="text-neutral-500 block">Prompt (In) Tokens</span>
                  <span className="text-sm font-bold font-mono text-neutral-900 dark:text-white">
                    {singleResult.promptTokens}
                  </span>
                  <span className="text-[10px] text-neutral-400 block mt-0.5">
                    @ ${singleResult.rates.in}/1M
                  </span>
                </div>

                <div className="p-3 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-850">
                  <span className="text-neutral-500 block">Candidates (Out) Tokens</span>
                  <span className="text-sm font-bold font-mono text-neutral-900 dark:text-white">
                    {singleResult.candidatesTokens}
                  </span>
                  <span className="text-[10px] text-neutral-400 block mt-0.5">
                    @ ${singleResult.rates.out}/1M
                  </span>
                </div>

                <div className="p-3 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-850">
                  <span className="text-neutral-500 block">Gemini 3.1 Pro Baseline</span>
                  <span className="text-sm font-bold font-mono text-neutral-900 dark:text-white">
                    $0.35 / 1k
                  </span>
                  <span className="text-[10px] text-emerald-600 font-semibold block mt-0.5">
                    {Math.max(0, (1 - singleResult.costPer1k / 0.35) * 100).toFixed(0)}% cheaper
                  </span>
                </div>
              </div>

              {/* Raw Response Output if reasoning */}
              {promptType === 'reasoning' && (
                <div className="space-y-1 text-xs">
                  <span className="font-semibold text-neutral-500">Chain of Thought Trace:</span>
                  <pre className="p-3 rounded-lg bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 font-mono text-[11px] whitespace-pre-wrap max-h-40 overflow-y-auto">
                    {singleResult.rawOutput}
                  </pre>
                </div>
              )}
            </div>
          )}

          {/* Batch Evaluation Summary */}
          {batchResult && (
            <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-emerald-500/20">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                  <h4 className="text-sm font-bold text-neutral-900 dark:text-white">
                    Batch Empirical Result: {batchResult.modelLabel}
                  </h4>
                </div>
                <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300">
                  {batchResult.sampleCount} Held-out Samples
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800">
                  <span className="text-neutral-500">Sample Accuracy</span>
                  <div className="text-lg font-bold text-neutral-900 dark:text-white font-mono mt-0.5">
                    {formatPct(batchResult.accuracy)}
                  </div>
                  <span className="text-[10px] text-neutral-400">
                    ({batchResult.correctCount}/{batchResult.sampleCount} correct)
                  </span>
                </div>

                <div className="p-3 rounded-lg bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800">
                  <span className="text-neutral-500">Measured Cost / 1k</span>
                  <div className="text-lg font-bold text-emerald-600 dark:text-emerald-400 font-mono mt-0.5">
                    {formatCost(batchResult.costPer1k)}
                  </div>
                  <span className="text-[10px] text-neutral-400">
                    Based on actual token mix
                  </span>
                </div>

                <div className="p-3 rounded-lg bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800">
                  <span className="text-neutral-500">Avg Tokens / Call</span>
                  <div className="text-sm font-bold text-neutral-900 dark:text-white font-mono mt-1">
                    {batchResult.avgInTokens} in / {batchResult.avgOutTokens} out
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800">
                  <span className="text-neutral-500">Average Latency</span>
                  <div className="text-sm font-bold text-neutral-900 dark:text-white font-mono mt-1">
                    {batchResult.avgLatencyMs} ms / call
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between pt-2">
                <span className="text-xs text-neutral-500">
                  ✓ Point automatically rendered on the Pareto chart!
                </span>
                <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                  Check Pareto tab to see frontier positioning →
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
