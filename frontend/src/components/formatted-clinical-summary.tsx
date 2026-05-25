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

const SECTION_CONFIG: Record<number, { bg: string; border: string; text: string; iconBg: string; icon: string }> = {
  1: {
    bg: "bg-indigo-50/60",
    border: "border-indigo-200",
    text: "text-indigo-900",
    iconBg: "bg-indigo-100 text-indigo-600",
    icon: "M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z",
  },
  2: {
    bg: "bg-sky-50/60",
    border: "border-sky-200",
    text: "text-sky-900",
    iconBg: "bg-sky-100 text-sky-600",
    icon: "M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z",
  },
  3: {
    bg: "bg-emerald-50/60",
    border: "border-emerald-200",
    text: "text-emerald-900",
    iconBg: "bg-emerald-100 text-emerald-600",
    icon: "M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z",
  },
  4: {
    bg: "bg-amber-50/60",
    border: "border-amber-200",
    text: "text-amber-900",
    iconBg: "bg-amber-100 text-amber-600",
    icon: "M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z",
  },
  5: {
    bg: "bg-violet-50/60",
    border: "border-violet-200",
    text: "text-violet-900",
    iconBg: "bg-violet-100 text-violet-600",
    icon: "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z",
  },
  6: {
    bg: "bg-rose-50/60",
    border: "border-rose-200",
    text: "text-rose-900",
    iconBg: "bg-rose-100 text-rose-600",
    icon: "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z",
  },
};

const SOURCE_TAGS: Record<string, { bg: string; text: string; border: string; label: string }> = {
  "[VEP]": { bg: "bg-purple-50", text: "text-purple-700", border: "border-purple-200", label: "VEP" },
  "[Evo2]": { bg: "bg-orange-50", text: "text-orange-700", border: "border-orange-200", label: "Evo2" },
  "[ClinVar]": { bg: "bg-indigo-50", text: "text-indigo-700", border: "border-indigo-200", label: "ClinVar" },
  "[gnomAD]": { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200", label: "gnomAD" },
  "[UniProt]": { bg: "bg-pink-50", text: "text-pink-700", border: "border-pink-200", label: "UniProt" },
};

function getSectionConfig(num: number) {
  return SECTION_CONFIG[num] || {
    bg: "bg-slate-50/60",
    border: "border-slate-200",
    text: "text-slate-900",
    iconBg: "bg-slate-100 text-slate-600",
    icon: "M9 12h6m-6-4h6m2 5.5c.5.3 1.2.2 1.4-.3.3-.5.2-1.2-.3-1.4-.5-.3-1.2-.2-1.4.3-.3.5-.2 1.2.3 1.4zM7 20h10v-2H7v2zM9 2h6v4H9V2z",
  };
}

export function FormattedClinicalSummary({ summary, className = "", metadata, showExecutive = true }: FormattedClinicalSummaryProps) {
  if (!summary) return null;

  const formatText = (text: string): React.ReactNode[] => {
    const parts: React.ReactNode[] = [];
    let currentIndex = 0;
    let partKey = 0;

    const regex = /(\*\*[^*]+\*\*|\[PMID:\s*\d+\]|\[PubMed:\d+\]|\[VEP\]|\[Evo2\]|\[ClinVar\]|\[gnomAD\]|\[UniProt\])/g;
    let match;

    while ((match = regex.exec(text)) !== null) {
      if (match.index > currentIndex) {
        const beforeText = text.slice(currentIndex, match.index);
        if (beforeText.includes("\n")) {
          const lines = beforeText.split("\n");
          lines.forEach((line, i) => {
            parts.push(<span key={`t-${partKey++}`}>{line}</span>);
            if (i < lines.length - 1) parts.push(<br key={`b-${partKey++}`} />);
          });
        } else {
          parts.push(<span key={`t-${partKey++}`}>{beforeText}</span>);
        }
      }

      const m = match[0];

      if (m.startsWith("**") && m.endsWith("**")) {
        parts.push(<strong key={`bd-${partKey++}`} className="font-semibold text-slate-800">{m.slice(2, -2)}</strong>);
      } else if (m.startsWith("[PMID:")) {
        const pmid = m.match(/\d+/)?.[0];
        if (pmid) {
          parts.push(
            <a key={`pm-${partKey++}`} href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`} target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-blue-50 text-blue-700 text-[11px] font-semibold border border-blue-100 hover:bg-blue-100 transition-colors"
            >
              <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
              PMID:{pmid}
            </a>
          );
        }
      } else if (m.startsWith("[PubMed:")) {
        const pmid = m.match(/\d+/)?.[0];
        if (pmid) {
          parts.push(
            <a key={`pu-${partKey++}`} href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`} target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-blue-50 text-blue-700 text-[11px] font-semibold border border-blue-100 hover:bg-blue-100 transition-colors"
            >
              <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
              PubMed:{pmid}
            </a>
          );
        }
      } else if (SOURCE_TAGS[m]) {
        const tag = SOURCE_TAGS[m];
        parts.push(
          <span key={`tg-${partKey++}`} className={`inline-flex items-center px-1.5 py-0.5 rounded-md ${tag.bg} ${tag.text} text-[10px] font-bold border ${tag.border}`}>
            {tag.label}
          </span>
        );
      }

      currentIndex = match.index + m.length;
    }

    if (currentIndex < text.length) {
      const remaining = text.slice(currentIndex);
      if (remaining.includes("\n")) {
        const lines = remaining.split("\n");
        lines.forEach((line, i) => {
          parts.push(<span key={`t-${partKey++}`}>{line}</span>);
          if (i < lines.length - 1) parts.push(<br key={`b-${partKey++}`} />);
        });
      } else {
        parts.push(<span key={`t-${partKey++}`}>{remaining}</span>);
      }
    }

    return parts;
  };

  // Parse ## 1 TITLE sections
  const sections: { num: number; title: string; content: string }[] = [];
  const lines = summary.split("\n");
  let current: { num: number; title: string; content: string } | null = null;
  const contentLines: string[] = [];

  for (const line of lines) {
    const headingMatch = line.match(/^##\s*(\d*)\s*(.+)$/i);
    if (headingMatch) {
      if (current) {
        current.content = contentLines.join("\n").trim();
        sections.push(current);
      }
      const num = headingMatch[1] ? parseInt(headingMatch[1]) : sections.length + 1;
      const title = headingMatch[2] ? headingMatch[2].trim() : "Untitled";
      current = { num, title, content: "" };
      contentLines.length = 0;
    } else if (line.trim() || contentLines.length > 0) {
      contentLines.push(line);
    }
  }
  if (current) {
    current.content = contentLines.join("\n").trim();
    sections.push(current);
  }

  if (sections.length === 0) {
    sections.push({ num: 1, title: "Summary", content: summary });
  }

  const firstSection = sections[0];
  const executive = firstSection?.content
    ? firstSection.content.match(/^(.+?\.(?:\s|$))/)?.[1] || firstSection.content.slice(0, 200) + "…"
    : null;

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Executive Summary */}
      {showExecutive && executive && (
        <div className="rounded-xl border border-slate-200 bg-gradient-to-br from-white to-slate-50/50 p-5 shadow-sm">
          <div className="flex items-start gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-teal-500 to-emerald-600 shadow-sm flex-shrink-0">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6-4h6m2 5.5c.5.3 1.2.2 1.4-.3.3-.5.2-1.2-.3-1.4-.5-.3-1.2-.2-1.4.3-.3.5-.2 1.2.3 1.4zM7 20h10v-2H7v2zM9 2h6v4H9V2z" />
              </svg>
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-bold text-teal-700 uppercase tracking-wider">Executive Summary</span>
                <span className="text-[9px] text-slate-400">— {sections.length} sections</span>
              </div>
              <div className="text-sm text-slate-700 leading-relaxed">{formatText(executive)}</div>
              {metadata && (
                <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
                  {metadata.gene && <span><strong className="text-slate-700">Gene:</strong> {metadata.gene}</span>}
                  {metadata.variant && <span><strong className="text-slate-700">Variant:</strong> {metadata.variant}</span>}
                  {metadata.transcript && <span><strong className="text-slate-700">Transcript:</strong> {metadata.transcript}</span>}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Section Cards */}
      <div className="space-y-3">
        {sections.map((section) => {
          const cfg = getSectionConfig(section.num);
          return (
            <div key={section.num} className={`rounded-xl border ${cfg.border} ${cfg.bg} overflow-hidden transition-all hover:shadow-sm`}>
              <div className="px-4 py-3 flex items-center gap-3">
                <div className={`flex h-7 w-7 items-center justify-center rounded-lg bg-white shadow-sm ${cfg.iconBg}`}>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d={cfg.icon} />
                  </svg>
                </div>
                <span className={`text-xs font-bold ${cfg.text}`}>
                  {section.num}. {section.title}
                </span>
              </div>
              {section.content && (
                <div className="px-4 pb-4">
                  <div className="pl-10 text-sm text-slate-700 leading-relaxed">
                    {formatText(section.content)}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
