"use client";

import React from "react";

interface Metadata {
  gene?: string;
  variant?: string;
  transcript?: string;
  proteinChange?: string;
}

interface FormattedClinicalSummaryProps {
  summary: string;
  className?: string;
  metadata?: Metadata;
  showExecutive?: boolean;
}

/**
 * Formats the clinical summary from multimodal RAG with proper markdown rendering.
 * Handles:
 *   - §1-§7 section headings
 *   - **bold text**
 *   - [PMID: 12345] and [PubMed:12345] citations
 *   - Line breaks within paragraphs
 */
export function FormattedClinicalSummary({ summary, className = "", metadata, showExecutive = true }: FormattedClinicalSummaryProps) {
  // Parse markdown-style formatting and convert to JSX
  const formatText = (text: string): React.ReactNode[] => {
    const parts: React.ReactNode[] = [];
    let currentIndex = 0;
    let partKey = 0;

    // Match **bold text**, [PMID: 12345], [PubMed:12345]
    const regex = /(\*\*[^*]+\*\*|\[PMID:\s*\d+\]|\[PubMed:\d+\])/g;
    let match;

    while ((match = regex.exec(text)) !== null) {
      // Add text before the match
      if (match.index > currentIndex) {
        const beforeText = text.slice(currentIndex, match.index);
        // Convert single newlines to <br />
        if (beforeText.includes("\n")) {
          const lines = beforeText.split("\n");
          lines.forEach((line, i) => {
            parts.push(
              <span key={`text-${partKey++}`}>
                {line}
              </span>
            );
            if (i < lines.length - 1) {
              parts.push(<br key={`br-${partKey++}`} />);
            }
          });
        } else {
          parts.push(
            <span key={`text-${partKey++}`}>
              {beforeText}
            </span>
          );
        }
      }

      const matchedText = match[0];

      // Handle bold text
      if (matchedText.startsWith("**") && matchedText.endsWith("**")) {
        const boldContent = matchedText.slice(2, -2);
        parts.push(
          <strong key={`bold-${partKey++}`} className="font-semibold text-[#2d3e2e]">
            {boldContent}
          </strong>
        );
      }
      // Handle PMID citations
      else if (matchedText.startsWith("[PMID:")) {
        const pmid = matchedText.match(/\d+/)?.[0];
        if (pmid) {
          parts.push(
            <a
              key={`pmid-${partKey++}`}
              href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#4a7c59] hover:text-[#3c4f3d] underline decoration-dotted underline-offset-2 transition-colors"
            >
              [PMID: {pmid}]
            </a>
          );
        } else {
          parts.push(<span key={`pmid-${partKey++}`}>{matchedText}</span>);
        }
      }
      // Handle PubMed citations
      else if (matchedText.startsWith("[PubMed:")) {
        const pmid = matchedText.match(/\d+/)?.[0];
        if (pmid) {
          parts.push(
            <a
              key={`pubmed-${partKey++}`}
              href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#4a7c59] hover:text-[#3c4f3d] underline decoration-dotted underline-offset-2 transition-colors"
            >
              [PubMed:{pmid}]
            </a>
          );
        } else {
          parts.push(<span key={`pubmed-${partKey++}`}>{matchedText}</span>);
        }
      }

      currentIndex = match.index + matchedText.length;
    }

    // Add remaining text
    if (currentIndex < text.length) {
      const remaining = text.slice(currentIndex);
      if (remaining.includes("\n")) {
        const lines = remaining.split("\n");
        lines.forEach((line, i) => {
          parts.push(
            <span key={`text-${partKey++}`}>
              {line}
            </span>
          );
          if (i < lines.length - 1) {
            parts.push(<br key={`br-${partKey++}`} />);
          }
        });
      } else {
        parts.push(
          <span key={`text-${partKey++}`}>
            {remaining}
          </span>
        );
      }
    }

    return parts;
  };

  // Split into blocks by double newlines, but also detect § headings
  const blocks = summary.split(/\n\n+/).filter((p) => p.trim());

  // Executive summary: first paragraph or first sentence fallback
  const executive = (() => {
    if (!summary) return null;
    const firstPara = summary.split(/\n\n+/)[0] || "";
    // If paragraph starts with a § heading strip it
    const stripped = firstPara.replace(/^§\d+\s+[^\n]+/i, "").trim();
    const firstSentenceMatch = stripped.match(/^(.+?\.)\s/);
    return (firstSentenceMatch && firstSentenceMatch[1]) || (stripped.length > 300 ? stripped.slice(0, 300) + "…" : stripped);
  })();

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Executive summary header for quick clinician reading */}
      {showExecutive && executive && (
        <div className="rounded-lg border border-[#e6e6e6] bg-white p-4 shadow-sm">
          <div className="flex items-start gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#0f766e]/10 text-[#0f766e]">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
              </svg>
            </div>
            <div>
              <div className="text-xs font-semibold text-[#0f766e]">Executive Summary</div>
              <div className="mt-1 text-sm text-[#0f2933] leading-relaxed">{formatText(String(executive))}</div>
              {metadata && (
                <div className="mt-3 text-[11px] text-[#55606a]">
                  {metadata.gene && <span className="mr-3"><strong>Gene:</strong> {metadata.gene}</span>}
                  {metadata.variant && <span className="mr-3"><strong>Variant:</strong> {metadata.variant}</span>}
                  {metadata.transcript && <span className="mr-3"><strong>Transcript:</strong> {metadata.transcript}</span>}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Sections */}
      <div className="space-y-3">
        {blocks.map((block, idx) => {
          // Check if this block starts with a § heading
          const sectionMatch = block.match(/^§(\d+)\s+(.+)$/m);
          if (sectionMatch) {
            const sectionNum = sectionMatch[1];
            const sectionTitle = sectionMatch[2];
            const restOfBlock = block.slice(sectionMatch[0].length).trim();
            return (
              <div key={idx} className="rounded-md border border-[#e6e6e6] bg-white p-4">
                <div className="flex items-center gap-3">
                  <div className="flex h-7 w-7 items-center justify-center rounded-full bg-[#de8246]/5 text-[#de8246] font-semibold">{sectionNum}</div>
                  <div className="text-xs font-semibold text-[#374151] uppercase tracking-wide">{sectionTitle}</div>
                </div>
                {restOfBlock && (
                  <div className="mt-2 text-sm text-[#0f2933] leading-relaxed">
                    {formatText(restOfBlock)}
                  </div>
                )}
              </div>
            );
          }

          // Regular paragraph
          return (
            <p key={idx} className="text-sm text-[#0f2933] leading-relaxed">
              {formatText(block)}
            </p>
          );
        })}
      </div>
    </div>
  );
}
