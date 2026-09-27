import React from 'react';

export const ZadumMethodologyCard: React.FC = () => {
  return (
    <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl p-6 shadow-xs space-y-6">
      <div className="border-b border-neutral-100 dark:border-neutral-800 pb-4">
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-mono text-xs font-bold">
            ARCHITECTURAL ADVANTAGE
          </span>
        </div>
        <h3 className="text-lg font-bold text-neutral-900 dark:text-white mt-1">
          Why Zadum Algorithm (Class-Gating) Outperforms Per-Item Cascading
        </h3>
        <p className="text-xs sm:text-sm text-neutral-600 dark:text-neutral-400 mt-1">
          Comparison against FrugalGPT, GPTCache, and standard confidence thresholding.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="p-4 rounded-xl bg-neutral-50 dark:bg-neutral-800/40 border border-neutral-200 dark:border-neutral-700 space-y-2">
          <span className="text-xs font-bold text-rose-600 dark:text-rose-400 block uppercase tracking-wider">
            Conventional Cascades (Per-Item)
          </span>
          <p className="text-xs text-neutral-600 dark:text-neutral-300 leading-relaxed">
            Per-item systems run a small model on every single query, check the softmax entropy or verbalized confidence score, and defer if uncertain.
          </p>
          <ul className="text-[11px] text-neutral-500 space-y-1 list-disc pl-4">
            <li><b>Double latency on hard items:</b> Runs Model A first, waits, then runs Model B.</li>
            <li><b>Softmax overconfidence:</b> Neural models output uncalibrated high confidence on wrong answers.</li>
            <li><b>Zero token savings on gate:</b> Every call still incurs minimum network and inference overhead.</li>
          </ul>
        </div>

        <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/25 space-y-2">
          <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400 block uppercase tracking-wider">
            Zadum Algorithm (Class-Gated Deferral)
          </span>
          <p className="text-xs text-neutral-600 dark:text-neutral-300 leading-relaxed">
            Separates the category space into self-contained classes vs reasoning classes ahead of time.
          </p>
          <ul className="text-[11px] text-neutral-600 dark:text-neutral-300 space-y-1 list-disc pl-4">
            <li><b>Zero double-hop penalty:</b> Self-contained classes resolve in &lt;5ms on CPU with $0 cloud cost.</li>
            <li><b>Statistical holdout verification:</b> Class boundaries are proven on held-out test splits.</li>
            <li><b>Up to 90% cloud cost reduction:</b> High-frequency routine categories never hit an LLM API.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
