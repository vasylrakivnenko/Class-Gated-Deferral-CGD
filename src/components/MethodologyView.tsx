import React from 'react';
import { Scale, CheckCircle, ShieldCheck, Calculator, AlertOctagon, HelpCircle } from 'lucide-react';

export const MethodologyView: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-5 shadow-xs">
        <h2 className="text-lg font-bold text-neutral-900 dark:text-white flex items-center gap-2">
          <Scale className="w-5 h-5 text-emerald-600" />
          <span>Statistical Methodology: What Separates a Benchmark from Marketing</span>
        </h2>
        <p className="text-xs sm:text-sm text-neutral-600 dark:text-neutral-400 mt-1 max-w-3xl">
          Why traditional model comparison benchmarks produce misleading claims, and how Downshift enforces mathematical rigor on cost, evaluation splits, and hypothesis testing.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Card 1: 3-Way Split */}
        <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 font-bold text-sm">
            <ShieldCheck className="w-4 h-4" />
            <span>1. The Sacred 3-Way Split</span>
          </div>
          <p className="text-xs text-neutral-600 dark:text-neutral-300 leading-relaxed">
            GEPA reads the <b>train</b> set to propose prompt instructions, and scores candidate prompts on the <b>val</b> set to select which prompts survive each generation. Both splits are therefore contaminated: a score reported on either is an overfitted training number wearing a disguise.
          </p>
          <div className="p-3 rounded-lg bg-neutral-50 dark:bg-neutral-800/60 font-mono text-[11px] text-neutral-700 dark:text-neutral-300 space-y-1">
            <div>• Train (N=200): Proposes prompt variations</div>
            <div>• Val (N=120): Selection tournament filter</div>
            <div className="text-emerald-600 dark:text-emerald-400 font-bold">
              • Test (N=250): Sacred holdout carved out FIRST; never shown to any optimizer.
            </div>
          </div>
        </div>

        {/* Card 2: Cost is Measured */}
        <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 font-bold text-sm">
            <Calculator className="w-4 h-4" />
            <span>2. Measured Tokens, Never Blended</span>
          </div>
          <p className="text-xs text-neutral-600 dark:text-neutral-300 leading-relaxed">
            The conventional "$/1M tokens, 50/50 blend" figure is meaningless for classification tasks. A sentiment query is typically ~260 input tokens and 5–15 output tokens (an 18:1 ratio). Blends flatter expensive models with cheap input and invert the rankings against the actual cloud invoice.
          </p>
          <div className="p-3 rounded-lg bg-neutral-50 dark:bg-neutral-800/60 font-mono text-[11px] text-neutral-700 dark:text-neutral-300">
            <code>cost_per_1k = 1000 * (mean_tin * price_in + mean_tout * price_out) / 1e6</code>
          </div>
        </div>

        {/* Card 3: Wilson Intervals */}
        <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 font-bold text-sm">
            <Scale className="w-4 h-4" />
            <span>3. Wilson Score Confidence Intervals</span>
          </div>
          <p className="text-xs text-neutral-600 dark:text-neutral-300 leading-relaxed">
            Normal approximation intervals (Wald method) fail badly near 0% and 100% boundaries, frequently producing confidence limits &gt; 100%. Exact Clopper-Pearson intervals are overly conservative and make decisive benchmark differences appear muddy. Wilson score intervals provide asymmetric, boundary-respecting coverage.
          </p>
          <p className="text-[11px] text-neutral-500">
            At N=250 and 95% accuracy, the Wilson 95% interval is approximately [91.5%, 97.2%].
          </p>
        </div>

        {/* Card 4: McNemar Mid-P & Holm-Bonferroni */}
        <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 font-bold text-sm">
            <AlertOctagon className="w-4 h-4" />
            <span>4. Paired McNemar Mid-P & Holm-Bonferroni</span>
          </div>
          <p className="text-xs text-neutral-600 dark:text-neutral-300 leading-relaxed">
            When comparing two models on the same 250 sentences, pairing is critical. If two models agree on 220 items, the test is evaluated on the 30 discordant items. Furthermore, comparing 25+ models against one baseline introduces the multiple testing problem: without Holm-Bonferroni correction, one in three benchmark runs would report a spurious winner by chance.
          </p>
          <div className="text-[11px] text-neutral-500">
            Non-inferiority is formally tested against margin δ = 0.03 (1-sided test against -3pp loss).
          </div>
        </div>
      </div>
    </div>
  );
};
