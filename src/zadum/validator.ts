import { JsonlValidationResult } from './types';

export const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB
export const MAX_ROW_COUNT = 10000; // 10,000 rows

/**
 * Validates an uploaded JSONL file string according to Zadum algorithm requirements.
 * - Max size: 10MB
 * - Max rows: 10,000
 * - Must be valid JSON on each line
 * - Each row must contain 'text' (non-empty string) and 'label' (non-empty string)
 */
export function validateJsonlContent(content: string, fileSize: number): JsonlValidationResult {
  if (fileSize > MAX_FILE_SIZE_BYTES) {
    return {
      valid: false,
      error: `File size exceeds the 10MB limit (uploaded: ${(fileSize / (1024 * 1024)).toFixed(2)} MB). Please upload a smaller file.`,
      fileSizeBytes: fileSize,
    };
  }

  const lines = content.split(/\r?\n/).filter((l) => l.trim().length > 0);

  if (lines.length === 0) {
    return {
      valid: false,
      error: 'The uploaded file is empty or contains no valid non-empty lines.',
      rowCount: 0,
      fileSizeBytes: fileSize,
    };
  }

  if (lines.length > MAX_ROW_COUNT) {
    return {
      valid: false,
      error: `File contains ${lines.length.toLocaleString()} rows, exceeding the 10,000 row limit. Please trim the dataset.`,
      rowCount: lines.length,
      fileSizeBytes: fileSize,
    };
  }

  const classCounts: Record<string, number> = {};
  const sampleRows: Array<{ text: string; label: string }> = [];

  for (let i = 0; i < lines.length; i++) {
    const lineNum = i + 1;
    const rawLine = lines[i];

    let parsed: any;
    try {
      parsed = JSON.parse(rawLine);
    } catch (e: any) {
      return {
        valid: false,
        error: `JSON syntax error on line ${lineNum}: ${e.message}. Each line must be a valid JSON object.`,
        rowCount: lines.length,
        fileSizeBytes: fileSize,
      };
    }

    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return {
        valid: false,
        error: `Invalid row structure on line ${lineNum}. Each line must be a JSON object, e.g. {"text": "...", "label": "..."}.`,
        rowCount: lines.length,
        fileSizeBytes: fileSize,
      };
    }

    // Support 'text' or 'sentence' or 'query'
    const textVal = parsed.text ?? parsed.sentence ?? parsed.query;
    // Support 'label' or 'intent' or 'category' or 'class'
    const labelVal = parsed.label ?? parsed.intent ?? parsed.category ?? parsed.class;

    if (!textVal || typeof textVal !== 'string' || textVal.trim().length === 0) {
      return {
        valid: false,
        error: `Missing or invalid 'text' field on line ${lineNum}. Every record must have a non-empty string in "text" (or "sentence"/"query").`,
        rowCount: lines.length,
        fileSizeBytes: fileSize,
      };
    }

    if (labelVal === undefined || labelVal === null || String(labelVal).trim().length === 0) {
      return {
        valid: false,
        error: `Missing or invalid 'label' field on line ${lineNum}. Every record must have a non-empty string in "label" (or "intent"/"category").`,
        rowCount: lines.length,
        fileSizeBytes: fileSize,
      };
    }

    const cleanLabel = String(labelVal).trim();
    classCounts[cleanLabel] = (classCounts[cleanLabel] || 0) + 1;

    if (sampleRows.length < 10) {
      sampleRows.push({
        text: textVal.trim(),
        label: cleanLabel,
      });
    }
  }

  const uniqueClasses = Object.keys(classCounts);
  if (uniqueClasses.length < 2) {
    return {
      valid: false,
      error: `Classification requires at least 2 distinct classes. Found only 1 class: "${uniqueClasses[0]}".`,
      rowCount: lines.length,
      fileSizeBytes: fileSize,
    };
  }

  return {
    valid: true,
    rowCount: lines.length,
    fileSizeBytes: fileSize,
    sampleClasses: uniqueClasses,
    classCounts,
    sampleRows,
  };
}
