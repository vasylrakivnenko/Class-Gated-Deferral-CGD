# Zadum-Gemini AI (zadumai)

> **Zero-Cost Local Class Gating & Staged Gemini Deferral**  
> Match or exceed frontier LLM accuracy at a fraction of the cost by partitioning predictable classes to instant, zero-token local gates (`Zadum-Gemini TF`, `Zadum-Gemini E`, `Zadum-Gemini E-Lite`) and deferring complex queries to Google Gemini.

[![Vite](https://img.shields.io/badge/Vite-6.x-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![React](https://img.shields.io/badge/React-19.x-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-4.x-38B2AC?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini_3.1-4285F4?logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)

---

## ⚡ What is Zadum-Gemini AI?

In high-volume classification systems (customer support routing, compliance review, legal contract clauses, sentiment analysis), calling frontier LLMs on every query wastes budget and introduces avoidable latency. Traditional confidence-based routing (like FrugalGPT) requires scoring every individual call with an LLM.

**Zadum AI** introduces **Class-Gated Deferral**:
1. **Zero-Cost Local Gate (`Zadum-Gemini TF` / `Zadum-Gemini E`)**: Resolves high-confidence, self-contained semantic classes locally in under 5ms with **$0 marginal cloud token cost**.
2. **Staged Gemini Deferral**: Only ambiguous or high-liability edge cases are deferred to **Gemini 3.1 Flash-Lite**, **Gemini 2.5 Flash**, or **Gemini 3.1 Pro**.
3. **Provable Statistical Parity**: Decisions are validated using rigorous non-inferiority margins, Wilson 95% confidence intervals, and McNemar mid-p paired hypothesis tests with Holm–Bonferroni correction.

---

## 🚀 Key Features

### 1. Interactive Pareto Frontier Analysis
- Visualizes the true trade-off between **empirical accuracy (%)** and **cost per 1,000 queries ($/1k)**.
- Default **"All"** label mode displaying exact model points, confidence intervals, and Pareto-optimal models.
- Side-by-side comparison of local zero-cost models (`Zadum-Gemini TF`, `Zadum-Gemini E`, `Zadum-Gemini E-Lite`) alongside Gemini 3.1 Pro, Gemini 2.5 Flash, and Gemini 2.5 Flash-Lite.

### 2. Zadum Cascade Studio & Class Gating
- Partition classes into:
  - **Tier 0 (Local Zero-Cost Gate)**: 0 token cost, ~4ms latency.
  - **Tier 1 (Direct Gemini Flash-Lite)**: Ultra-fast lightweight triage.
  - **Tier 2 (Staged Reasoning / Gemini Pro)**: Deep multi-step reasoning for ambiguous edge cases.
- Real-time simulation of aggregate accuracy, weighted cost reduction, latency savings, and monthly run-rate impact.

### 3. Interactive Live Playground
- Test user queries in real-time.
- Visual routing traces showing whether a prompt is handled by the local gate or escalated to Gemini.
- Inspection of predicted probabilities, entropy, and cost breakdown per query.

### 4. Custom Dataset Evaluator (JSONL Upload)
- Upload custom labeled datasets in standard `.jsonl` format.
- Supports files up to **1 MB** (up to **10,000 rows**).
- Includes one-click quick-load sample benchmarks:
  - **Legal Contract Clauses** (CUAD covenant-not-to-sue & audit rights)
  - **Customer Banking Intake** (Banking77 intents)

### 5. Production Pricing & Token Bill Calculator
- Accurate pricing engine calibrated to Google Cloud Platform rate cards.
- Accounts for realistic input-to-output token ratios (e.g., 260 input tokens : 35 output tokens).
- Dynamic monthly savings estimates based on custom query volumes and cache hit rates.

---

## 📊 Benchmark Datasets Included

| Dataset | Domain | Classes | Test Size | Frontier Reference | Zadum-Gemini Gate Result |
|---|---|---|---|---|---|
| **Banking77** | Financial Customer Service | 77 | 3,080 | Gemini 3.1 Pro (93.9%) | **Zadum-Gemini E** matches within non-inferiority margin at **$0 cost** |
| **Financial PhraseBank** | Financial News Sentiment | 3 | 250 | Gemini 3.1 Pro (97.6%) | **Zadum-Gemini TF & E** match human consensus holdouts |
| **CUAD Covenant Not to Sue** | Legal Contract Review | 2 | 308 | Gemini 3.1 Pro (96.1%) | **Zadum-Gemini TF** achieves 96.5% at **$0 marginal cost** |
| **CUAD Audit Rights** | Commercial Contracts | 2 | 308 | Gemini 3.1 Pro (95.8%) | High-throughput local gate with >94% keep rate |

---

## 🛠️ Architecture & Tech Stack

- **Frontend**: React 19, TypeScript, Vite
- **Styling**: Tailwind CSS v4, Lucide Icons
- **Math & Statistics**:
  - Wilson Score Intervals (continuity-corrected)
  - McNemar mid-p paired contingency tests
  - Holm–Bonferroni Family-Wise Error Rate (FWER) control
  - One-sided non-inferiority testing against $-\delta$ margin
- **Algorithms**:
  - `Zadum-Gemini TF`: N-gram TF-IDF linear gate with calibrated softmax
  - `Zadum-Gemini E`: In-domain dense neural representation gate
  - `Zadum-Gemini E-Lite`: Low-footprint embedding representation gate

---

## 💻 Getting Started Locally

### Prerequisites
- Node.js 20+
- npm or pnpm

### Installation

```bash
# Clone the repository
git clone https://github.com/vasylrakivnenko/zadumai.git
cd zadumai

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```

The app will be accessible at `http://localhost:3000`.

### Building for Production

```bash
npm run build
```

---

## 📝 License

Apache-2.0 or MIT. See repository details for specifics.
