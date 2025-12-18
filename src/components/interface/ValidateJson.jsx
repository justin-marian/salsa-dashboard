/**
 * @file ValidateJson.jsx
 * @description
 * Streaming JSON validator component.
 *
 * Purpose:
 *  - Validate large JSON graph exports before normalization
 *  - Detect non-JSON (HTML error pages, truncated streams)
 *  - Provide progress + size feedback
 *
 * Key design choices:
 *  - Streaming when available (ReadableStream)
 *  - Early HTML detection
 *  - Never blocks the main thread
 */

import { useCallback, useEffect, useState } from "react";
import "./style_validatejson.css";

/** Remove UTF-8 BOM if present */
const stripBOM = (s) => { return s.charCodeAt(0) === 0xfeff ? s.slice(1) : s; }

/** Find first non-whitespace character */
const firstNonWsChar = (s) => {
  for (let i = 0; i < s.length; i++) {
    if (s[i] > " ") return s[i];
  }
  return "";
}

/**
 * formatBytesToMB
 *
 * Human-readable file size helper.
 */
const formatBytesToMB = (bytes) => {
  if (!bytes) return "0 MB";
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/**
 * validateJsonStream
 *
 * Fetches and validates JSON from a URL.
 *
 * Responsibilities:
 *  - Detect HTTP errors
 *  - Stream chunks when supported
 *  - Detect HTML masquerading as JSON
 *  - Validate JSON.parse at the end
 *
 * @param {string} url
 * @param {(p:number)=>void} onProgress
 * @param {(chunk:number,total:number)=>void} onChunk
 */
async function validateJsonStream(url, onProgress, onChunk) {
  let chunks = "";

  try {
    const response = await fetch(url, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });

    if (!response.ok) {
      const preview = (await response.text()).slice(0, 300);
      return {
        valid: false,
        error: `HTTP ${response.status} ${response.statusText}\n${preview}`,
      };
    }

    const ct = (response.headers.get("Content-Type") || "").toLowerCase();
    const total = +response.headers.get("Content-Length");

    // No streaming support → fallback to full read
    if (!response.body) {
      chunks = await response.text();

      if (
        !ct.includes("json") ||
        firstNonWsChar(chunks) === "<"
      ) {
        return {
          valid: false,
          error: "Expected JSON but received non-JSON content.",
          size: chunks.length,
        };
      }

      JSON.parse(stripBOM(chunks));
      onProgress?.(100);
      return { valid: true, size: chunks.length };
    }

    // Streaming path
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      received += value.length;
      const piece = decoder.decode(value, { stream: true });
      chunks += piece;

      // Early HTML detection
      if (
        !ct.includes("json") &&
        firstNonWsChar(chunks) === "<"
      ) {
        return {
          valid: false,
          error: "Non-JSON stream detected.",
          size: chunks.length,
        };
      }

      if (onProgress && total) {
        onProgress((received / total) * 100);
      }
      onChunk?.(piece.length, chunks.length);
    }

    JSON.parse(stripBOM(chunks));
    return { valid: true, size: chunks.length };

  } catch (error) {
    return {
      valid: false,
      error: String(error?.message || error),
      size: chunks.length || 0,
    };
  }
}

/**
 * ValidateJson
 *
 * UI wrapper around validateJsonStream.
 *
 * Automatically runs on mount and exposes a manual revalidate button.
 */
export default function ValidateJson({
  url = "/exports/wiki_vote_regraph.json",
}) {
  const [stats, setStats] = useState({ chunks: 0, totalSize: 0 });
  const [result, setResult] = useState(null);
  const [progress, setProgress] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  /**
   * validateJson
   *
   * Resets UI state and triggers validation.
   */
  const validateJson = useCallback(async () => {
    setIsLoading(true);
    setStats({ chunks: 0, totalSize: 0 });
    setResult(null);
    setProgress(0);

    const r = await validateJsonStream(
      url,
      (p) => setProgress(p),
      (_chunk, total) =>
        setStats((prev) => ({
          chunks: prev.chunks + 1,
          totalSize: total,
        }))
    );

    setResult(r);
    setIsLoading(false);
  }, [url]);

  /**
   * Auto-run on mount / URL change
   */
  useEffect(() => {
    validateJson();
  }, [validateJson]);

  const isValid = result?.valid ?? null;

  return (
    <div className="validate-json-container">
      <div className="validate-json-header">
        <h3 className="validate-json-title">JSON Validator</h3>
        {isValid !== null && (
          <span
            className={`validate-json-badge ${
              isValid ? "valid" : "invalid"
            }`}
          >
            {isValid ? "OK" : "NOK"}
          </span>
        )}
      </div>

      <div className="validate-json-url-row">
        <span className="validate-json-url-label">URL:</span>
        <code className="validate-json-url-code">{url}</code>
      </div>

      <div className="validate-json-progress-row">
        <div className="validate-json-progress-bar">
          <div
            className="validate-json-progress-fill"
            style={{ width: `${progress}%` }}
          />
        </div>
        <span className="validate-json-progress-text">
          {progress.toFixed(1)}%
        </span>
      </div>

      {result && (
        <>
          <div className="validate-json-stats-block">
            {result.size != null && (
              <div className="validate-json-stats-row">
                <span className="validate-json-stats-label">
                  File Size
                </span>
                <span className="validate-json-stats-value">
                  {formatBytesToMB(result.size)}
                  <span className="size-detail">
                    ({result.size.toLocaleString()} bytes)
                  </span>
                </span>
              </div>
            )}

            {stats.totalSize > 0 && (
              <div className="validate-json-stats-row">
                <span className="validate-json-stats-label">
                  Streamed
                </span>
                <span className="validate-json-stats-value">
                  {formatBytesToMB(stats.totalSize)}
                  <span className="size-detail">
                    (in {stats.chunks} chunks)
                  </span>
                </span>
              </div>
            )}
          </div>

          {result.error && (
            <div className="validate-json-error-block">
              <pre className="validate-json-error-pre">
                {result.error}
              </pre>
            </div>
          )}
        </>
      )}

      <button
        type="button"
        onClick={validateJson}
        className={`validate-json-button ${
          isLoading ? "validate-json-loading" : ""
        }`}
        disabled={isLoading}
      >
        {isLoading ? "Validating..." : "Re-validate"}
      </button>
    </div>
  );
}
