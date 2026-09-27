import React, { useState } from 'react';
import { AlertTriangle, TrendingDown, Layers, Scale, Sparkles, Cpu, CheckCircle2, ChevronRight, Terminal, Zap } from 'lucide-react';

interface FindingItem {
  id: string;
  badge: string;
  badgeColor: string;
  title: string;
  summary: string;
  table?: {
    headers: string[];
    rows: (string | number)[][];
  };
  details: string[];
  verdict: string;
}

const FINDINGS: FindingItem[] = [
  {
    id: 'cache-threshold',
    badge: 'Pricing Trap #1',
    badgeColor: 'bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300',
    title: 'The Rate Card Said 3x Cheaper. The Measurement Said 1.16x.',
    summary:
      'Claude Haiku 4.5 lists at $1.00 / $5.00 per 1M tokens; Claude Sonnet 4.6 lists at $3.00 / $15.00. The sticker price promises a 3.0x saving. On the real Banking77 task, the measured gap was only 1.16x.',
    table: {
      headers: ['Model', 'Input Tokens', 'Cached Tokens', 'Measured $/1k Calls'],
      rows: [
        ['Claude Sonnet 4.6', '2,641', '1,884 (71%)', '$3.211'],
        ['Claude Haiku 4.5', '2,640', '0 (0%)', '$2.765'],
      ],
    },
    details: [
      'Same prompt, same 77-class classification task, same code path.',
      'Why? Anthropic enforces different minimum cacheable prompt lengths per model tier. Sonnet caches prompts starting at 1,024 tokens; Haiku requires 2,048 tokens and strict prefix alignment.',
      'A workload below a cheaper tier’s caching threshold forfeits discounts, while the pricier model caches the bulk of its prompt, erasing the headline price advantage.',
    ],
    verdict: 'Never pick a model based on advertised sticker rates without measuring prompt cache warmth on your actual prompt length.',
  },
  {
    id: 'reasoning-tax',
    badge: 'Pricing Trap #2',
    badgeColor: 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300',
    title: 'Reasoning Models Bill <think> Blocks As Expensive Output Tokens',
    summary:
      'Nemotron-30B had Fireworks serverless’s lowest advertised sticker price ($0.05/$0.20). But in classification smoke tests, it emitted 1,046 output tokens of unsolicited reasoning for a single one-word answer.',
    table: {
      headers: ['Configuration', 'Prompt Output Tokens', 'Effective Price per 1k'],
      rows: [
        ['Direct Mode (dspy.Predict)', '5 – 35 tokens', '$0.07 / 1k items'],
        ['Reasoning On (CoT / Think)', '260 – 1,046 tokens', '$0.65 – $1.40 / 1k items'],
      ],
    },
    details: [
      'Most provider rate cards charge 3x to 5x more for output tokens than input tokens.',
      'When an LLM generates a lengthy internal thinking trace, you pay output token rates for every internal thinking token.',
      'A cheap reasoning model is frequently 5x to 10x more expensive per item than a frontier model running in Direct mode without unsolicited reasoning.',
    ],
    verdict: 'For production classification, enforce strict DIRECT output signatures (`dspy.Predict`) and disable reasoning blocks unless required.',
  },
  {
    id: 'classical-encoders',
    badge: 'Architecture Win',
    badgeColor: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300',
    title: 'A 68M Parameter Encoder Beats Prompted LLMs at $0 Marginal Cost',
    summary:
      'A compact fine-tuned encoder (ettin-encoder-68m, ~68 million weights) reached 94.0% accuracy on Banking77 and 97.6% on Financial PhraseBank, beating most prompt-only LLMs at zero cloud token cost.',
    table: {
      headers: ['Approach', 'Accuracy (Banking77)', 'Marginal Token Cost', 'Throughput'],
      rows: [
        ['Fine-tuned ettin-68m', '94.0%', '$0.00 / 1k (Local)', '201 items / sec (CPU)'],
        ['GPT-5-nano (Direct)', '86.4%', '$0.20 / 1k (Azure)', '12 items / sec'],
        ['Claude Sonnet 4.6 (Ref)', '82.9%', '$1.14 / 1k (Hosted)', '8 items / sec'],
      ],
    },
    details: [
      'Encoders do not autoregressively generate tokens one-by-one; they compute a single forward pass over the embedding.',
      'Serving ettin-68m on a minimal $5/month CPU instance yields ~50 million classifications per month for pennies.',
      'TF-IDF + Logistic Regression also provides a critical baseline: on Financial PhraseBank it scores 78.4% at microsecond latency.',
    ],
    verdict: 'If you have labeled training data, a fine-tuned encoder is almost always higher accuracy, lower latency, and 100x cheaper than an LLM.',
  },
  {
    id: 'statistical-split',
    badge: 'Evaluation Rigor',
    badgeColor: 'bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300',
    title: 'The Sacred 3-Way Split: Why Test Sets Must Be Contamination-Proof',
    summary:
      'DSPy GEPA and reflection optimizers overfit to whatever validation set they evaluate on. A model score reported on train or val is an illusion.',
    details: [
      'Train Split: Read by GEPA to propose instruction variations and edge-case prompt amendments.',
      'Validation Split: Used by the optimizer to select which candidates survive each iteration.',
      'Test Split (Sacred): Carved out first and NEVER shown to any optimizer or reflection loop.',
      'Financial PhraseBank sentences_allagree: 2,264 sentences where 100% of human annotators agreed. Unanimous human holdouts eliminate noisy label contamination.',
    ],
    verdict: 'Never optimize prompts on the dataset you report accuracy on. The holdout test set must remain untouched.',
  },
  {
    id: 'paired-testing',
    badge: 'Statistics',
    badgeColor: 'bg-purple-100 text-purple-800 dark:bg-purple-950 dark:text-purple-300',
    title: 'McNemar Mid-P & Holm-Bonferroni Correction',
    summary:
      'Comparing models via independent binomial intervals ignores that both models are tested on the exact same sentences. Paired testing is required.',
    details: [
      'With N=250 test items and two models agreeing on 220 items, the effective sample size is 30, not 250.',
      'McNemar mid-p tests whether the discordant errors are symmetric. It avoids both under-rejection of exact tests and normal approximations.',
      'When screening 27 candidate models against the reference standard, testing at alpha=0.05 will produce false winners by random chance. Holm-Bonferroni family-wise error rate control is mandatory.',
    ],
    verdict: 'Statistical rigor protects against declaring a cheap model "equivalent" when sample size was merely insufficient to detect its errors.',
  },
];

export const FindingsViewer: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('cache-threshold');

  const activeFinding = FINDINGS.find((f) => f.id === activeTab) || FINDINGS[0];

  return (
    <div className="space-y-6">
      {/* Intro Header */}
      <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-5 shadow-xs">
        <h2 className="text-lg font-bold text-neutral-900 dark:text-white flex items-center gap-2">
          <span>Empirical Findings & Real-World Traps</span>
        </h2>
        <p className="text-xs sm:text-sm text-neutral-600 dark:text-neutral-400 mt-1 max-w-3xl">
          Non-obvious insights discovered through empirical token measurement and held-out validation that cannot be learned from rate cards, model docs, or standard leaderboards.
        </p>
      </div>

      {/* Main Container */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Navigation Sidebar */}
        <div className="lg:col-span-4 space-y-2">
          {FINDINGS.map((item) => {
            const isSelected = item.id === activeTab;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full text-left p-3.5 rounded-xl border text-xs transition-all ${
                  isSelected
                    ? 'border-emerald-500 bg-emerald-50/50 dark:bg-emerald-950/20 shadow-xs'
                    : 'border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 hover:border-neutral-300 dark:hover:border-neutral-700'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span
                    className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded ${item.badgeColor}`}
                  >
                    {item.badge}
                  </span>
                  {isSelected && <ChevronRight className="w-3.5 h-3.5 text-emerald-600" />}
                </div>
                <h4 className="font-bold text-neutral-900 dark:text-white text-xs sm:text-sm line-clamp-2">
                  {item.title}
                </h4>
              </button>
            );
          })}
        </div>

        {/* Detailed Finding Display */}
        <div className="lg:col-span-8 bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-6 shadow-xs space-y-6">
          <div>
            <span
              className={`text-xs font-bold uppercase tracking-wider px-2.5 py-1 rounded inline-block mb-2 ${activeFinding.badgeColor}`}
            >
              {activeFinding.badge}
            </span>
            <h3 className="text-xl font-bold text-neutral-900 dark:text-white tracking-tight">
              {activeFinding.title}
            </h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-300 mt-2 leading-relaxed">
              {activeFinding.summary}
            </p>
          </div>

          {/* Table if present */}
          {activeFinding.table && (
            <div className="overflow-x-auto border border-neutral-200 dark:border-neutral-800 rounded-lg">
              <table className="w-full text-left text-xs border-collapse font-sans">
                <thead>
                  <tr className="bg-neutral-50 dark:bg-neutral-800 border-b border-neutral-200 dark:border-neutral-800 font-semibold text-neutral-700 dark:text-neutral-300">
                    {activeFinding.table.headers.map((h, i) => (
                      <th key={i} className="py-2.5 px-3">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-200 dark:divide-neutral-800 font-mono text-neutral-900 dark:text-neutral-200">
                  {activeFinding.table.rows.map((row, rIdx) => (
                    <tr key={rIdx} className="hover:bg-neutral-50/50 dark:hover:bg-neutral-800/40">
                      {row.map((cell, cIdx) => (
                        <td key={cIdx} className="py-2.5 px-3">
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Key Insights List */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-neutral-400">
              Detailed Breakdown & Mechanics
            </h4>
            <div className="space-y-2">
              {activeFinding.details.map((detail, idx) => (
                <div key={idx} className="flex items-start gap-2.5 text-xs sm:text-sm text-neutral-700 dark:text-neutral-300">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-2 shrink-0"></span>
                  <span>{detail}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Takeaway Verdict Box */}
          <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-950 dark:text-emerald-200 text-xs sm:text-sm flex items-start gap-3">
            <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <span className="font-bold block mb-0.5">Engineering Recommendation:</span>
              <p>{activeFinding.verdict}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
