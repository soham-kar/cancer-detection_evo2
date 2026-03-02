"use client";

import React from "react";

interface FormattedClinicalSummaryProps {
  summary: string;
  className?: string;
}

/**
 * Formats the clinical summary from multimodal RAG with proper markdown rendering.
 * Handles **bold text**, PMID citations, and structured sections.
 */
export function FormattedClinicalSummary({ summary, className = "" }: FormattedClinicalSummaryProps) {
  // Parse markdown-style formatting and convert to JSX
  const formatText = (text: string): React.ReactNode[] => {
    const parts: React.ReactNode[] = [];
    let currentIndex = 0;
    let partKey = 0;

    // Match **bold text** and [PMID: 12345]
    const regex = /(\*\*[^*]+\*\*|\[PMID:\s*\d+\])/g;
    let match;

    while ((match = regex.exec(text)) !== null) {
      // Add text before the match
      if (match.index > currentIndex) {
        parts.push(
          <span key={`text-${partKey++}`}>
            {text.slice(currentIndex, match.index)}
          </span>
        );
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

      currentIndex = match.index + matchedText.length;
    }

    // Add remaining text
    if (currentIndex < text.length) {
      parts.push(
        <span key={`text-${partKey++}`}>
          {text.slice(currentIndex)}
        </span>
      );
    }

    return parts;
  };

  // Split into paragraphs and format each
  const paragraphs = summary.split(/\n\n+/).filter(p => p.trim());

  return (
    <div className={`space-y-4 ${className}`}>
      {paragraphs.map((paragraph, idx) => (
        <p key={idx} className="text-sm text-[#3c4f3d] leading-relaxed">
          {formatText(paragraph)}
        </p>
      ))}
    </div>
  );
}
