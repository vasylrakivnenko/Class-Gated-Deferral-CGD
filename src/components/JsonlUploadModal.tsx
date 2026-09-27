import React, { useState, useRef } from 'react';
import { UploadCloud, AlertCircle, CheckCircle2, FileText, X, ArrowRight } from 'lucide-react';
import { validateJsonlContent, MAX_FILE_SIZE_BYTES } from '../zadum/validator';
import { JsonlValidationResult } from '../zadum/types';

interface JsonlUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onDatasetLoaded: (result: JsonlValidationResult, datasetName: string) => void;
}

export const JsonlUploadModal: React.FC<JsonlUploadModalProps> = ({ isOpen, onClose, onDatasetLoaded }) => {
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [fileName, setFileName] = useState<string>('');
  const [fileSize, setFileSize] = useState<number>(0);
  const [validationResult, setValidationResult] = useState<JsonlValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleProcessFile = (file: File) => {
    setErrorMessage(null);
    setValidationResult(null);
    setFileName(file.name);
    setFileSize(file.size);

    if (file.size > MAX_FILE_SIZE_BYTES) {
      setErrorMessage(
        `File size ${(file.size / (1024 * 1024)).toFixed(2)} MB exceeds the maximum 1MB limit. Please upload a file up to 1MB (up to 10,000 rows).`
      );
      return;
    }

    if (!file.name.endsWith('.jsonl') && !file.name.endsWith('.json')) {
      setErrorMessage(
        `Invalid file extension "${file.name}". The Zadum algorithm requires a line-delimited JSON file (.jsonl) with 1 JSON object per row.`
      );
      return;
    }

    setIsValidating(true);
    const reader = new FileReader();

    reader.onload = (e) => {
      const text = e.target?.result as string;
      const res = validateJsonlContent(text, file.size);
      setIsValidating(false);

      if (!res.valid) {
        setErrorMessage(res.error || 'Validation failed for this JSONL file.');
      } else {
        setValidationResult(res);
      }
    };

    reader.onerror = () => {
      setIsValidating(false);
      setErrorMessage('Failed to read the file from disk.');
    };

    reader.readAsText(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleProcessFile(e.target.files[0]);
    }
  };

  const handleLoadLegalSampleDataset = () => {
    const sampleRows = [
      { text: "Developer agrees that it will not at any time do or cause to be done any act or thing contesting or impairing any part of such right, title and interest in licensor's patents.", label: "covenant_not_to_sue" },
      { text: "Neither party shall disclose, publish, or otherwise disseminate any Confidential Information of the other party to any third party without prior written consent.", label: "confidentiality" },
      { text: "Licensor reserves the right, upon 10 days advance written notice, to audit and inspect licensee's records and premises to verify compliance with royalty obligations.", label: "audit_rights" },
      { text: "Either party may terminate this Agreement immediately upon written notice if the other party breaches any material term and fails to cure within 30 days.", label: "termination" },
      { text: "To the maximum extent permitted by applicable law, in no event shall either party's aggregate liability exceed the total amounts paid under this Agreement in the preceding 12 months.", label: "limitation_of_liability" },
      { text: "Supplier agrees to defend, indemnify and hold harmless Customer against any third-party claims alleging infringement of any copyright, trademark, or patent.", label: "indemnification" },
      { text: "This Agreement shall be governed by, and construed in accordance with, the laws of the State of Delaware, without giving effect to conflict of laws principles.", label: "governing_law" },
      { text: "Neither party may assign or transfer any rights or obligations under this Agreement without the express written consent of the other party.", label: "assignment" },
      { text: "Neither party shall be liable for delays or failures in performance resulting from acts beyond its reasonable control, including natural disasters or civil unrest.", label: "force_majeure" },
      { text: "All notices, requests, demands, and other communications shall be in writing and shall be deemed to have been duly given when delivered personally or by certified mail.", label: "notices" },
    ];

    const lines = sampleRows.map((r) => JSON.stringify(r)).join('\n');
    const res = validateJsonlContent(lines, lines.length);
    setFileName('legal_contract_clauses_sample.jsonl');
    setFileSize(lines.length);
    setValidationResult(res);
    setErrorMessage(null);
  };

  const handleLoadSampleDataset = () => {
    const sampleRows = [
      { text: "How do I activate my new credit card?", label: "activate_my_card" },
      { text: "I lost my wallet with my card in it.", label: "lost_or_stolen_card" },
      { text: "What is the fee for withdrawing money from an ATM abroad?", label: "atm_support" },
      { text: "Why is the refund from Amazon not showing up?", label: "Refund_not_showing_up" },
      { text: "Can I set an automatic recurring top up for my balance?", label: "automatic_top_up" },
      { text: "My card was unexpectedly declined at a grocery checkout.", label: "declined_card_payment" },
      { text: "What is the minimum age limit to open a student account?", label: "age_limit" },
      { text: "How do I verify my passport in the app?", label: "verify_identity" },
      { text: "What is your USD to GBP exchange rate?", label: "exchange_rate" },
      { text: "Can I link my account to Google Pay and Apple Pay?", label: "apple_pay_or_google_pay" },
    ];

    const lines = sampleRows.map((r) => JSON.stringify(r)).join('\n');
    const res = validateJsonlContent(lines, lines.length);
    setFileName('sample_banking_eval.jsonl');
    setFileSize(lines.length);
    setValidationResult(res);
    setErrorMessage(null);
  };

  const handleConfirm = () => {
    if (validationResult && validationResult.valid) {
      onDatasetLoaded(validationResult, fileName.replace(/\.[^/.]+$/, ''));
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs">
      <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl max-w-xl w-full p-6 shadow-2xl space-y-5">
        <div className="flex items-center justify-between pb-3 border-b border-neutral-100 dark:border-neutral-800">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
              <UploadCloud className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-neutral-900 dark:text-white">
                Upload Dataset for Zadum Class-Gating
              </h3>
              <p className="text-xs text-neutral-500 dark:text-neutral-400">
                Calibrate class tiers and test deferral on your JSONL sample (≤1MB, ≤10,000 rows).
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-200"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-3.5 rounded-xl bg-neutral-50 dark:bg-neutral-800/50 border border-neutral-200 dark:border-neutral-700/60 text-xs space-y-1.5">
          <div className="flex items-center gap-1.5 font-semibold text-neutral-800 dark:text-neutral-200">
            <FileText className="w-4 h-4 text-emerald-600" />
            <span>Dataset Requirements (.jsonl format):</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-[11px] text-neutral-600 dark:text-neutral-400 font-mono">
            <div>• Format: Line-delimited JSON (.jsonl)</div>
            <div>• Max File Size: 1 MB</div>
            <div>• Max Records: 10,000 rows</div>
            <div>• Required keys: "text", "label"</div>
          </div>
        </div>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
            dragActive
              ? 'border-emerald-500 bg-emerald-50/40 dark:bg-emerald-950/20'
              : 'border-neutral-300 dark:border-neutral-700 hover:border-neutral-400 dark:hover:border-neutral-600'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".jsonl,.json"
            onChange={handleFileSelect}
            className="hidden"
          />
          <UploadCloud className="w-8 h-8 mx-auto text-neutral-400 mb-2" />
          <p className="text-xs sm:text-sm font-semibold text-neutral-800 dark:text-neutral-200">
            Click to browse or drag and drop your .jsonl file
          </p>
          <p className="text-[11px] text-neutral-400 mt-1">
            Up to 1MB · Max 10,000 rows
          </p>
        </div>

        <div className="space-y-1.5 pt-1 text-xs">
          <div className="text-neutral-500 text-[11px]">Quick load a benchmark sample:</div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={handleLoadLegalSampleDataset}
              className="px-2.5 py-1 rounded-md bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-500/30 text-emerald-700 dark:text-emerald-300 font-medium hover:bg-emerald-100 transition-colors"
            >
              📄 Legal Contract Clauses (.jsonl)
            </button>
            <button
              onClick={handleLoadSampleDataset}
              className="px-2.5 py-1 rounded-md bg-neutral-100 dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-700 text-neutral-700 dark:text-neutral-300 font-medium hover:bg-neutral-200 transition-colors"
            >
              🏦 Customer Banking Intake (.jsonl)
            </button>
          </div>
        </div>

        {errorMessage && (
          <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 text-rose-800 dark:text-rose-300 text-xs flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-rose-600 dark:text-rose-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-bold block">File Format Error</span>
              <p>{errorMessage}</p>
            </div>
          </div>
        )}

        {validationResult && validationResult.valid && (
          <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-900 dark:text-emerald-200 text-xs space-y-2">
            <div className="flex items-center justify-between font-bold">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                <span>Dataset Validated Successfully</span>
              </div>
              <span className="font-mono">{validationResult.rowCount?.toLocaleString()} rows</span>
            </div>
            <div className="flex flex-wrap gap-2 text-[11px] font-mono text-neutral-700 dark:text-neutral-300">
              <span className="px-2 py-0.5 rounded bg-white dark:bg-neutral-800 border border-emerald-500/30">
                Classes: {validationResult.sampleClasses?.length}
              </span>
              <span className="px-2 py-0.5 rounded bg-white dark:bg-neutral-800 border border-emerald-500/30">
                Size: {((fileSize || 0) / 1024).toFixed(1)} KB
              </span>
            </div>
          </div>
        )}

        <div className="flex items-center justify-end gap-3 pt-3 border-t border-neutral-100 dark:border-neutral-800">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-xs font-medium text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-800"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={!validationResult || !validationResult.valid || isValidating}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-medium text-xs shadow-xs disabled:opacity-50 transition-colors"
          >
            <span>Calibrate with Zadum Algorithm</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
