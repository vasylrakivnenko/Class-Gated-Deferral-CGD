import React, { useState, useMemo } from 'react';
import { PricingRow } from '../types/benchmark';
import { Search, ArrowUpDown, ArrowUp, ArrowDown, Calculator, Sliders, DollarSign, ExternalLink } from 'lucide-react';
import pricingDataRaw from '../data/pricingData.json';

const PRICING_DATA: PricingRow[] = pricingDataRaw as PricingRow[];

export const PricingTable: React.FC = () => {
  const [filterProvider, setFilterProvider] = useState<'all' | 'google'>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [sortKey, setSortKey] = useState<keyof PricingRow | 'blend'>('blend');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');

  // Token Bill Calculator state
  const [inputTokens, setInputTokens] = useState<number>(260);
  const [outputTokens, setOutputTokens] = useState<number>(15);
  const [monthlyVolume, setMonthlyVolume] = useState<number>(1_000_000); // 1 million calls
  const [cacheHitRate, setCacheHitRate] = useState<number>(0.0);

  const filteredAndSorted = useMemo(() => {
    let list = PRICING_DATA.filter((r) => {
      if (filterProvider !== 'all' && r.provider !== filterProvider) return false;
      if (searchTerm && !r.model.toLowerCase().includes(searchTerm.toLowerCase())) return false;
      return true;
    });

    list.sort((a, b) => {
      let valA: any = a[sortKey as keyof PricingRow];
      let valB: any = b[sortKey as keyof PricingRow];

      if (sortKey === 'blend') {
        valA = 0.9 * a.in + 0.1 * a.out;
        valB = 0.9 * b.in + 0.1 * b.out;
      }

      if (valA == null) return sortOrder === 'asc' ? 1 : -1;
      if (valB == null) return sortOrder === 'asc' ? -1 : 1;

      if (typeof valA === 'string') {
        return sortOrder === 'asc'
          ? valA.localeCompare(valB)
          : valB.localeCompare(valA);
      }
      return sortOrder === 'asc' ? valA - valB : valB - valA;
    });

    return list;
  }, [filterProvider, searchTerm, sortKey, sortOrder]);

  const handleSort = (key: keyof PricingRow | 'blend') => {
    if (sortKey === key) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortOrder('asc');
    }
  };

  // Calculator monthly bill formula
  const computeMonthlyCost = (row: PricingRow) => {
    const effectiveInRate = row.cache
      ? row.in * (1 - cacheHitRate * 0.8) // ~80% discount on cached tokens
      : row.in;
    const costPerCall = (inputTokens * effectiveInRate + outputTokens * row.out) / 1_000_000;
    return costPerCall * monthlyVolume;
  };

  // Find cheapest and most expensive for comparison
  const calculatedBills = useMemo(() => {
    return PRICING_DATA.map((row) => ({
      ...row,
      monthlyCost: computeMonthlyCost(row),
    })).sort((a, b) => a.monthlyCost - b.monthlyCost);
  }, [inputTokens, outputTokens, monthlyVolume, cacheHitRate]);

  const cheapest = calculatedBills[0];
  const reference = calculatedBills.find((r) => r.model.includes('Gemini 3.1 Pro')) || calculatedBills[calculatedBills.length - 1];

  return (
    <div className="space-y-6">
      {/* Interactive Bill Simulator Card */}
      <div className="bg-gradient-to-br from-neutral-900 via-neutral-900 to-neutral-950 text-white rounded-xl p-5 sm:p-6 shadow-md border border-neutral-800">
        <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-neutral-800 gap-3">
          <div className="flex items-center gap-2.5">
            <Calculator className="w-5 h-5 text-emerald-400" />
            <div>
              <h3 className="text-base font-bold tracking-tight">
                Interactive Measured Token Bill Simulator
              </h3>
              <p className="text-xs text-neutral-400">
                Classification tasks have a heavy input-to-output skew (e.g. 260 input tokens vs 15 output tokens). Blended rates lie.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right">
              <span className="text-[11px] uppercase tracking-wider text-neutral-400 block">
                Estimated Monthly Savings vs {reference?.model}
              </span>
              <span className="text-lg font-mono font-bold text-emerald-400">
                ${Math.max(0, (reference?.monthlyCost || 0) - (cheapest?.monthlyCost || 0)).toLocaleString(undefined, { maximumFractionDigits: 0 })} / mo
              </span>
            </div>
          </div>
        </div>

        {/* Sliders Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 mt-5 text-xs">
          <div>
            <div className="flex justify-between text-neutral-300 mb-1">
              <span>Input Tokens / Call:</span>
              <span className="font-mono font-bold text-white">{inputTokens} tok</span>
            </div>
            <input
              type="range"
              min="50"
              max="2000"
              step="10"
              value={inputTokens}
              onChange={(e) => setInputTokens(parseInt(e.target.value, 10))}
              className="w-full accent-emerald-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-neutral-500 mt-0.5 font-mono">
              <span>50 (short)</span>
              <span>260 (phrasebank)</span>
              <span>2000 (doc)</span>
            </div>
          </div>

          <div>
            <div className="flex justify-between text-neutral-300 mb-1">
              <span>Output Tokens / Call:</span>
              <span className="font-mono font-bold text-white">{outputTokens} tok</span>
            </div>
            <input
              type="range"
              min="2"
              max="500"
              step="1"
              value={outputTokens}
              onChange={(e) => setOutputTokens(parseInt(e.target.value, 10))}
              className="w-full accent-emerald-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-neutral-500 mt-0.5 font-mono">
              <span>5 (direct)</span>
              <span>15 (label)</span>
              <span>300 (reasoning)</span>
            </div>
          </div>

          <div>
            <div className="flex justify-between text-neutral-300 mb-1">
              <span>Monthly Query Volume:</span>
              <span className="font-mono font-bold text-white">
                {(monthlyVolume / 1_000_000).toFixed(1)}M calls
              </span>
            </div>
            <input
              type="range"
              min="100000"
              max="20000000"
              step="100000"
              value={monthlyVolume}
              onChange={(e) => setMonthlyVolume(parseInt(e.target.value, 10))}
              className="w-full accent-emerald-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-neutral-500 mt-0.5 font-mono">
              <span>100k</span>
              <span>1M</span>
              <span>20M</span>
            </div>
          </div>

          <div>
            <div className="flex justify-between text-neutral-300 mb-1">
              <span>Prompt Cache Warmth:</span>
              <span className="font-mono font-bold text-white">{(cacheHitRate * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="0.9"
              step="0.05"
              value={cacheHitRate}
              onChange={(e) => setCacheHitRate(parseFloat(e.target.value))}
              className="w-full accent-emerald-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-neutral-500 mt-0.5 font-mono">
              <span>0% (cold)</span>
              <span>50%</span>
              <span>90% (warm)</span>
            </div>
          </div>
        </div>

        {/* Top 3 Value Picks */}
        <div className="mt-5 pt-4 border-t border-neutral-800 grid grid-cols-1 sm:grid-cols-3 gap-3">
          {calculatedBills.slice(0, 3).map((item, idx) => (
            <div
              key={item.model}
              className="p-3 rounded-lg bg-neutral-800/60 border border-neutral-700/60 flex items-center justify-between"
            >
              <div>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-400 block">
                  #{idx + 1} Lowest Cost
                </span>
                <span className="text-xs font-bold text-white">{item.model}</span>
                <span className="text-[10px] text-neutral-400 block capitalize">
                  {item.provider}
                </span>
              </div>
              <div className="text-right">
                <span className="text-sm font-bold font-mono text-emerald-400">
                  ${item.monthlyCost.toFixed(2)}
                </span>
                <span className="text-[10px] text-neutral-400 block">/ month</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Pricing Table Section */}
      <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-xl p-5 shadow-xs space-y-4">
        {/* Table Filters & Search */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-1.5 overflow-x-auto text-xs">
            <span className="text-neutral-500 font-semibold mr-1">Platform:</span>
            {[
              { id: 'all', label: `All GCP Models (${PRICING_DATA.length})` },
            ].map((p) => (
              <button
                key={p.id}
                onClick={() => setFilterProvider(p.id as any)}
                className={`px-3 py-1.5 rounded-lg font-medium transition-colors whitespace-nowrap ${
                  filterProvider === p.id
                    ? 'bg-neutral-900 text-white dark:bg-white dark:text-neutral-900 shadow-xs'
                    : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 hover:bg-neutral-200 dark:hover:bg-neutral-700'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>

          <div className="relative w-full sm:w-64">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-neutral-400" />
            <input
              type="text"
              placeholder="Search model name..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full text-xs pl-8 pr-3 py-1.5 rounded-lg border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800 text-neutral-900 dark:text-white focus:outline-hidden focus:ring-1 focus:ring-neutral-400"
            />
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto border border-neutral-200 dark:border-neutral-800 rounded-lg">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-neutral-50 dark:bg-neutral-800/80 border-b border-neutral-200 dark:border-neutral-800 text-neutral-600 dark:text-neutral-300 font-semibold select-none">
                <th
                  onClick={() => handleSort('model')}
                  className="py-2.5 px-3 cursor-pointer hover:bg-neutral-100 dark:hover:bg-neutral-700/50"
                >
                  <div className="flex items-center gap-1">
                    <span>Model</span>
                    {sortKey === 'model' && (sortOrder === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />)}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('params')}
                  className="py-2.5 px-3 cursor-pointer hover:bg-neutral-100 dark:hover:bg-neutral-700/50"
                >
                  <div className="flex items-center gap-1">
                    <span>Params / Architecture</span>
                    {sortKey === 'params' && (sortOrder === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />)}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('provider')}
                  className="py-2.5 px-3 cursor-pointer hover:bg-neutral-100 dark:hover:bg-neutral-700/50"
                >
                  <div className="flex items-center gap-1">
                    <span>Provider</span>
                    {sortKey === 'provider' && (sortOrder === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />)}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('ctx')}
                  className="py-2.5 px-3 text-right cursor-pointer hover:bg-neutral-100 dark:hover:bg-neutral-700/50"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>Context</span>
                    {sortKey === 'ctx' && (sortOrder === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />)}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('cache')}
                  className="py-2.5 px-3 text-center cursor-pointer hover:bg-neutral-100 dark:hover:bg-neutral-700/50"
                >
                  <div className="flex items-center justify-center gap-1">
                    <span>Caching</span>
                    {sortKey === 'cache' && (sortOrder === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />)}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('in')}
                  className="py-2.5 px-3 text-right cursor-pointer hover:bg-neutral-100 dark:hover:bg-neutral-700/50"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>$/1M In</span>
                    {sortKey === 'in' && (sortOrder === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />)}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('out')}
                  className="py-2.5 px-3 text-right cursor-pointer hover:bg-neutral-100 dark:hover:bg-neutral-700/50"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>$/1M Out</span>
                    {sortKey === 'out' && (sortOrder === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />)}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('blend')}
                  className="py-2.5 px-3 text-right cursor-pointer hover:bg-neutral-100 dark:hover:bg-neutral-700/50 bg-neutral-100/50 dark:bg-neutral-800"
                >
                  <div className="flex items-center justify-end gap-1 font-bold text-neutral-900 dark:text-white">
                    <span>$/1M Blend*</span>
                    {sortKey === 'blend' && (sortOrder === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />)}
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-200 dark:divide-neutral-800 font-mono text-neutral-800 dark:text-neutral-200">
              {filteredAndSorted.map((r, i) => {
                const blend = 0.9 * r.in + 0.1 * r.out;
                const isGoogle = r.provider === 'google';
                return (
                  <tr
                    key={`${r.model}-${r.provider}-${i}`}
                    className={`hover:bg-neutral-50 dark:hover:bg-neutral-800/50 transition-colors ${
                      isGoogle ? 'bg-emerald-50/20 dark:bg-emerald-950/10' : ''
                    }`}
                  >
                    <td className="py-2.5 px-3 font-sans font-semibold text-neutral-900 dark:text-white">
                      <div className="flex items-center gap-1.5">
                        {isGoogle && (
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                        )}
                        <span>{r.model}</span>
                      </div>
                      {r.note && (
                        <div className="text-[10px] text-neutral-400 font-normal font-sans line-clamp-1 mt-0.5">
                          {r.note}
                        </div>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-neutral-500 font-sans text-[11px]">
                      {r.params || '—'}
                    </td>
                    <td className="py-2.5 px-3 font-sans capitalize">
                      <span
                        className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold ${
                          r.provider === 'google'
                            ? 'bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300'
                            : r.provider === 'azure'
                            ? 'bg-blue-100 dark:bg-blue-950 text-blue-700 dark:text-blue-300'
                            : 'bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-300'
                        }`}
                      >
                        {r.provider === 'google' ? 'Google' : r.provider === 'azure' ? 'MS Azure' : 'Fireworks'}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      {r.ctx >= 1000000
                        ? `${(r.ctx / 1000000).toFixed(1)}M`
                        : `${(r.ctx / 1000).toFixed(0)}K`}
                    </td>
                    <td className="py-2.5 px-3 text-center font-sans">
                      {r.cache ? (
                        <span className="text-emerald-600 font-bold">Yes</span>
                      ) : (
                        <span className="text-neutral-400">No</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-right">${r.in.toFixed(2)}</td>
                    <td className="py-2.5 px-3 text-right">${r.out.toFixed(2)}</td>
                    <td className="py-2.5 px-3 text-right font-bold text-neutral-900 dark:text-white bg-neutral-50/50 dark:bg-neutral-800/30">
                      ${blend.toFixed(3)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Table Footer Methodology Notes */}
        <div className="text-xs text-neutral-500 space-y-1 pt-2">
          <p>
            *Blend = 0.9 × $/1M in + 0.1 × $/1M out — a realistic classification workload proxy (heavily input-weighted compared to standard 50/50 chat blend).
          </p>
          <p>
            "Caching" indicates the provider bills a discounted rate for repeated input tokens (typically 75–90% off).
          </p>
        </div>
      </div>
    </div>
  );
};
