/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Header, TabType } from './components/Header';
import { ParetoChart } from './components/ParetoChart';
import { LivePlayground } from './components/LivePlayground';
import { PricingTable } from './components/PricingTable';
import { FindingsViewer } from './components/FindingsViewer';
import { MethodologyView } from './components/MethodologyView';
import benchmarkDataRaw from './data/benchmarkData.json';
import { BenchmarkDataset } from './types/benchmark';

const BENCHMARK_DATA: BenchmarkDataset = benchmarkDataRaw as BenchmarkDataset;

export default function App() {
  const [currentTab, setCurrentTab] = useState<TabType>('chart');
  const [currentTaskKey, setCurrentTaskKey] = useState<string>(
    BENCHMARK_DATA.default || Object.keys(BENCHMARK_DATA.tasks)[0]
  );
  const [liveGeminiCandidate, setLiveGeminiCandidate] = useState<any>(null);

  const currentTask = BENCHMARK_DATA.tasks[currentTaskKey] || Object.values(BENCHMARK_DATA.tasks)[0];

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 flex flex-col font-sans selection:bg-emerald-500/20 selection:text-emerald-700 dark:selection:text-emerald-300">
      <Header
        currentTab={currentTab}
        onTabChange={setCurrentTab}
        selectedTaskLabel={currentTask.label}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 sm:py-8">
        {currentTab === 'chart' && (
          <ParetoChart
            task={currentTask}
            tasks={BENCHMARK_DATA.tasks}
            currentTaskKey={currentTaskKey}
            onSelectTask={setCurrentTaskKey}
            liveGeminiCandidate={liveGeminiCandidate}
          />
        )}

        {currentTab === 'playground' && (
          <LivePlayground
            onPlotCandidate={(cand) => {
              setLiveGeminiCandidate(cand);
              setCurrentTab('chart');
            }}
            currentTaskKey={currentTaskKey}
          />
        )}

        {currentTab === 'pricing' && <PricingTable />}

        {currentTab === 'findings' && <FindingsViewer />}

        {currentTab === 'methodology' && <MethodologyView />}
      </main>

      <footer className="border-t border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 py-6 mt-12 text-xs text-neutral-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-bold font-mono text-neutral-700 dark:text-neutral-300">↓ DOWNSHIFT</span>
            <span>—</span>
            <span>Porting tasks down to the cheapest model, prompt, or encoder that clears your bar.</span>
          </div>

          <div className="flex items-center gap-4">
            <a
              href="https://github.com/vasylrakivnenko/zadumai"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-neutral-800 dark:hover:text-neutral-200 underline"
            >
              GitHub: vasylrakivnenko/zadumai
            </a>
            <span>•</span>
            <span>Measured Token Pricing</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
