import React from 'react';
import { BarChart3, Layers, DollarSign, Scale, Github, Sparkles } from 'lucide-react';

export type TabType = 'cascade' | 'chart' | 'playground' | 'pricing' | 'methodology';

interface HeaderProps {
  currentTab: TabType;
  onTabChange: (tab: TabType) => void;
  selectedTaskLabel: string;
}

export const Header: React.FC<HeaderProps> = ({ currentTab, onTabChange, selectedTaskLabel }) => {
  return (
    <header className="border-b border-neutral-200 dark:border-neutral-800 bg-white/80 dark:bg-neutral-900/80 backdrop-blur sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="inline-flex items-center justify-center p-1.5 rounded-lg bg-emerald-600/10 text-emerald-600 dark:text-emerald-400 border border-emerald-600/20 font-mono font-bold text-sm tracking-wider">
                ZADUM AI
              </span>
              <h1 className="text-xl font-bold tracking-tight text-neutral-900 dark:text-white">
                Gemini-Optimized Class-Gated Cascade
              </h1>
              <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 border border-neutral-200 dark:border-neutral-700">
                Active Benchmark: {selectedTaskLabel}
              </span>
            </div>
            <p className="text-xs sm:text-sm text-neutral-500 dark:text-neutral-400 mt-1">
              Optimizing high-volume tasks like contract clause review and legal intake: class-gated deferral cuts Gemini costs by up to 90% and delivers sub-5ms latency without sacrificing accuracy.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <a
              href="https://github.com/vasylrakivnenko/zadumai"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-md border border-neutral-300 dark:border-neutral-700 text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
            >
              <Github className="w-3.5 h-3.5" />
              <span>vasylrakivnenko/zadumai</span>
            </a>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex items-center gap-1 sm:gap-2 mt-4 overflow-x-auto pb-1 text-sm border-t border-neutral-100 dark:border-neutral-800/80 pt-3">
          <button
            onClick={() => onTabChange('cascade')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-medium text-xs sm:text-sm whitespace-nowrap transition-all ${
              currentTab === 'cascade'
                ? 'bg-neutral-900 text-white dark:bg-emerald-600 dark:text-white shadow-sm ring-1 ring-neutral-800 dark:ring-emerald-500'
                : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-100 dark:hover:bg-neutral-800/60'
            }`}
          >
            <Layers className={`w-4 h-4 ${currentTab === 'cascade' ? 'text-emerald-400 dark:text-emerald-200' : 'text-emerald-500'}`} />
            <span className="font-semibold">Zadum Cascade Optimizer</span>
            <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-bold tracking-wide ${
              currentTab === 'cascade'
                ? 'bg-emerald-500/30 text-emerald-300 dark:bg-black/30 dark:text-emerald-100 border border-emerald-400/30'
                : 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20'
            }`}>
              90% Cut
            </span>
          </button>

          <button
            onClick={() => onTabChange('chart')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-medium text-xs sm:text-sm whitespace-nowrap transition-all ${
              currentTab === 'chart'
                ? 'bg-neutral-900 text-white dark:bg-white dark:text-neutral-950 shadow-sm'
                : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-100 dark:hover:bg-neutral-800/60'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            <span>Cost vs Accuracy (Pareto)</span>
          </button>

          <button
            onClick={() => onTabChange('playground')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-medium text-xs sm:text-sm whitespace-nowrap transition-all ${
              currentTab === 'playground'
                ? 'bg-emerald-600 text-white shadow-sm'
                : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-100 dark:hover:bg-neutral-800/60'
            }`}
          >
            <Sparkles className="w-4 h-4 text-emerald-300" />
            <span>Live Gemini Downshift</span>
            <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-500/30 text-emerald-100 uppercase tracking-wider font-mono">
              Live API
            </span>
          </button>

          <button
            onClick={() => onTabChange('pricing')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-medium text-xs sm:text-sm whitespace-nowrap transition-all ${
              currentTab === 'pricing'
                ? 'bg-neutral-900 text-white dark:bg-white dark:text-neutral-950 shadow-sm'
                : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-100 dark:hover:bg-neutral-800/60'
            }`}
          >
            <DollarSign className="w-4 h-4" />
            <span>Rate Cards & ROI Simulator</span>
          </button>

          <button
            onClick={() => onTabChange('methodology')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-medium text-xs sm:text-sm whitespace-nowrap transition-all ${
              currentTab === 'methodology'
                ? 'bg-neutral-900 text-white dark:bg-white dark:text-neutral-950 shadow-sm'
                : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-100 dark:hover:bg-neutral-800/60'
            }`}
          >
            <Scale className="w-4 h-4" />
            <span>Statistical Discipline</span>
          </button>
        </nav>
      </div>
    </header>
  );
};
