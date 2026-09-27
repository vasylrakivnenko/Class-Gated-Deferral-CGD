import express, { Request, Response } from 'express';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

app.use(express.json());

// Initialize GoogleGenAI client (will pick up process.env.GCP_API_2 or process.env.GEMINI_API_KEY)
const apiKey = process.env.GCP_API_2 || process.env.GEMINI_API_KEY;
const ai = new GoogleGenAI(apiKey ? { apiKey } : undefined);

// Rates per 1M tokens ($)
const GEMINI_RATES: Record<string, { in: number; out: number; label: string }> = {
  'gemini-2.5-flash': { in: 0.075, out: 0.30, label: 'Gemini 2.5 Flash' },
  'gemini-3.1-flash-lite': { in: 0.075, out: 0.30, label: 'Gemini 3.1 Flash-Lite' },
  'gemini-3.8-flash': { in: 0.15, out: 0.60, label: 'Gemini 3.8 Flash' },
  'gemini-3.1-pro-preview': { in: 1.25, out: 5.00, label: 'Gemini 3.1 Pro' },
};

// API: Single Live Evaluation
app.post('/api/evaluate-live', async (req: Request, res: Response) => {
  try {
    const {
      model = 'gemini-3.1-flash-lite',
      task = 'financial_phrasebank',
      text = '',
      promptType = 'direct',
      customInstruction = '',
    } = req.body;

    if (!text || typeof text !== 'string') {
      res.status(400).json({ error: 'Text input is required' });
      return;
    }

    const validModels = ['gemini-2.5-flash', 'gemini-3.1-flash-lite', 'gemini-3.8-flash', 'gemini-3.1-pro-preview'];
    const selectedModel = validModels.includes(model) ? model : 'gemini-2.5-flash';

    let systemInstruction = '';
    if (customInstruction) {
      systemInstruction = customInstruction;
    } else if (task === 'financial_phrasebank') {
      if (promptType === 'direct') {
        systemInstruction =
          'You are a financial sentiment classifier. Analyze the sentence and output ONLY one of these three exact labels: "positive", "neutral", or "negative". Do not include reasoning or markdown.';
      } else {
        systemInstruction =
          'You are a financial sentiment classifier. First, think step-by-step analyzing the financial statements, numbers, and guidance in the text. Then on the final line, provide the label formatted as: Label: <positive|neutral|negative>.';
      }
    } else if (task === 'banking77') {
      if (promptType === 'direct') {
        systemInstruction =
          'You are a banking customer support intent classifier with 77 categories. Classify the customer query into its exact intent label (e.g., "pending_card_payment", "lost_or_stolen_card", "exchange_rate", "transfer_not_received_by_recipient", "apple_pay_or_google_pay", "atm_support", "verify_identity", "automatic_top_up", "age_limit", "Refund_not_showing_up", "activate_my_card", "declined_card_payment"). Output ONLY the single intent label.';
      } else {
        systemInstruction =
          'You are a banking customer support intent classifier. Analyze the query carefully, reason about the user requirement, and output your reasoning followed by the final intent on the last line: Intent: <intent_label>.';
      }
    } else if (task === 'cuad_audit_rights') {
      if (promptType === 'direct') {
        systemInstruction =
          'You are a legal contract analyzer. Determine whether the contract clause contains an audit rights provision (i.e., gives a party the right to audit, examine, or inspect books, records, accounts, or facilities of the counterparty). Output ONLY "yes" or "no".';
      } else {
        systemInstruction =
          'You are a legal contract analyzer. Analyze whether the clause grants audit or inspection rights over books, records, or facilities. Provide your legal reasoning followed by the final decision on the last line: Answer: <yes|no>.';
      }
    } else {
      systemInstruction =
        promptType === 'direct'
          ? 'Classify the text concisely. Output only the category.'
          : 'Analyze the text step-by-step and output your reasoning before concluding with the category.';
    }

    const startTime = Date.now();
    const response = await ai.models.generateContent({
      model: selectedModel,
      contents: text,
      config: {
        systemInstruction,
        temperature: 0.0,
        maxOutputTokens: promptType === 'direct' ? 64 : 1024,
      },
    });

    const elapsedMs = Date.now() - startTime;
    const outputText = response.text?.trim() || '';

    // Extract label
    let predictedLabel = outputText;
    if (promptType === 'reasoning') {
      const match = outputText.match(/(?:Label|Intent|Answer):\s*([a-zA-Z0-9_\-]+)/i);
      if (match) {
        predictedLabel = match[1].trim();
      } else {
        const lines = outputText.split('\n').filter((l) => l.trim().length > 0);
        predictedLabel = lines[lines.length - 1].replace(/^[#*\s-]+/, '').trim();
      }
    } else {
      predictedLabel = outputText.replace(/^["'`]|["'`]$/g, '').trim().toLowerCase();
    }

    // Token usage metadata
    const usage = response.usageMetadata || {};
    const promptTokens = usage.promptTokenCount ?? 150;
    const candidatesTokens = usage.candidatesTokenCount ?? (promptType === 'direct' ? 6 : 180);
    const totalTokens = usage.totalTokenCount ?? (promptTokens + candidatesTokens);

    const rates = GEMINI_RATES[selectedModel] || { in: 0.15, out: 0.60, label: selectedModel };
    // Cost per 1k items formula from zadumai:
    // cost_per_1k = 1000 * (mean_input_tokens * price_in + mean_output_tokens * price_out) / 1e6
    const costPer1k = (1000 * (promptTokens * rates.in + candidatesTokens * rates.out)) / 1_000_000;

    res.json({
      model: selectedModel,
      modelLabel: rates.label,
      promptType,
      text,
      rawOutput: outputText,
      predictedLabel,
      latencyMs: elapsedMs,
      promptTokens,
      candidatesTokens,
      totalTokens,
      costPer1k: Number(costPer1k.toFixed(6)),
      rates,
    });
  } catch (error: any) {
    console.error('Error during live evaluation:', error);
    res.status(500).json({
      error: error.message || 'Failed to evaluate with Gemini API',
    });
  }
});

// API: Batch sample evaluation to estimate empirical accuracy & cost
app.post('/api/benchmark-batch', async (req: Request, res: Response) => {
  try {
    const {
      model = 'gemini-3.1-flash-lite',
      task = 'financial_phrasebank',
      samples = [],
      promptType = 'direct',
    } = req.body;

    if (!Array.isArray(samples) || samples.length === 0) {
      res.status(400).json({ error: 'Samples array is required' });
      return;
    }

    const validModels = ['gemini-2.5-flash', 'gemini-3.1-flash-lite', 'gemini-3.8-flash', 'gemini-3.1-pro-preview'];
    const selectedModel = validModels.includes(model) ? model : 'gemini-2.5-flash';
    const rates = GEMINI_RATES[selectedModel] || { in: 0.15, out: 0.60, label: selectedModel };

    const results = [];
    let correctCount = 0;
    let totalInTokens = 0;
    let totalOutTokens = 0;
    let totalTime = 0;

    for (const item of samples.slice(0, 10)) {
      const startTime = Date.now();
      const response = await ai.models.generateContent({
        model: selectedModel,
        contents: item.text,
        config: {
          systemInstruction:
            task === 'financial_phrasebank'
              ? 'Output ONLY one label: positive, neutral, or negative.'
              : task === 'cuad_audit_rights'
              ? 'Determine if this contract clause contains an audit or inspection rights provision. Output ONLY "yes" or "no".'
              : 'Output ONLY the single banking intent label from standard Banking77 classes.',
          temperature: 0.0,
          maxOutputTokens: 64,
        },
      });
      const elapsed = Date.now() - startTime;
      totalTime += elapsed;
      const text = (response.text || '').replace(/^["'`]|["'`]$/g, '').trim().toLowerCase();
      const isCorrect = text === (item.label || '').toLowerCase();
      if (isCorrect) correctCount++;

      const inTokens = response.usageMetadata?.promptTokenCount ?? 140;
      const outTokens = response.usageMetadata?.candidatesTokenCount ?? 5;
      totalInTokens += inTokens;
      totalOutTokens += outTokens;

      results.push({
        id: item.id,
        text: item.text,
        expected: item.label,
        predicted: text,
        correct: isCorrect,
        inTokens,
        outTokens,
        elapsed,
      });
    }

    const n = results.length;
    const accuracy = n > 0 ? correctCount / n : 0;
    const avgIn = totalInTokens / (n || 1);
    const avgOut = totalOutTokens / (n || 1);
    const costPer1k = (1000 * (avgIn * rates.in + avgOut * rates.out)) / 1_000_000;

    res.json({
      model: selectedModel,
      modelLabel: rates.label,
      task,
      accuracy,
      sampleCount: n,
      correctCount,
      avgInTokens: Math.round(avgIn),
      avgOutTokens: Math.round(avgOut),
      avgLatencyMs: Math.round(totalTime / (n || 1)),
      costPer1k: Number(costPer1k.toFixed(6)),
      results,
    });
  } catch (error: any) {
    console.error('Error during batch evaluation:', error);
    res.status(500).json({ error: error.message || 'Batch evaluation failed' });
  }
});

// Setup Vite or static serving
async function startServer() {
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
