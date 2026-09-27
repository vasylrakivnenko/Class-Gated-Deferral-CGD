import React from 'react';
import { BarChart3, FlaskConical, DollarSign, BookOpen, Scale, Github, Sparkles } from 'lucide-react';

export type TabType = 'chart' | 'playground' | 'pricing' | 'findings' | 'methodology';

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
                ↓DOWNSHIFT
              </span>
              <h1 className="text-xl font-bold tracking-tight text-neutral-900 dark:text-white">
                Model Pricing & Pareto Frontier
              </h1>
              <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 border border-neutral-200 dark:border-neutral-700">
                Task: {selectedTaskLabel}
              </span>
            </div>
            <p className="text-xs sm:text-sm text-neutral-500 dark:text-neutral-400 mt-1">
              Port classification from expensive frontier models to the cheapest model, prompt, or encoder that clears your accuracy bar.
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
              Live
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
            <span>Rate Cards & Cost Simulator</span>
          </button>

          <button
            onClick={() => onTabChange('findings')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-medium text-xs sm:text-sm whitespace-nowrap transition-all ${
              currentTab === 'findings'
                ? 'bg-neutral-900 text-white dark:bg-white dark:text-neutral-950 shadow-sm'
                : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-100 dark:hover:bg-neutral-800/60'
            }`}
          >
            <BookOpen className="w-4 h-4" />
            <span>Empirical Findings</span>
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
