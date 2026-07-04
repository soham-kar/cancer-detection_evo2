"use client";

import { Sparkles, Dna, Microscope, FlaskConical, FileText, HelpCircle } from "lucide-react";
import { cn } from "~/lib/utils";
import type { ActiveVariant } from "~/contexts/active-variant";

// =============================================================================
// ChatSuggestions - Context-aware quick-start chips
// =============================================================================

export interface Suggestion {
  id: string;
  label: string;
  icon?: React.ReactNode;
}

interface ChatSuggestionsProps {
  activeVariant: ActiveVariant | null;
  onSelect: (suggestion: string) => void;
  disabled?: boolean;
}

export function ChatSuggestions({
  activeVariant,
  onSelect,
  disabled = false,
}: ChatSuggestionsProps) {
  const suggestions: Suggestion[] = activeVariant
    ? getReportSuggestions(activeVariant)
    : getGeneralSuggestions();

  return (
    <div className="flex flex-wrap gap-2">
      {suggestions.map((s) => (
        <button
          key={s.id}
          disabled={disabled}
          onClick={() => onSelect(s.label)}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors",
            "border-[#3c4f3d]/20 bg-white text-[#3c4f3d] hover:border-[#de8246] hover:text-[#de8246]",
            "disabled:cursor-not-allowed disabled:opacity-50",
          )}
        >
          {s.icon && <span className="text-[#de8246]">{s.icon}</span>}
          {s.label}
        </button>
      ))}
    </div>
  );
}

function getGeneralSuggestions(): Suggestion[] {
  return [
    {
      id: "what-can-you-do",
      label: "What can HelixMind do?",
      icon: <Sparkles className="h-3 w-3" />,
    },
    {
      id: "interpret-vus",
      label: "How do I interpret a VUS?",
      icon: <HelpCircle className="h-3 w-3" />,
    },
    {
      id: "evo2-scores",
      label: "Explain Evo2 scores",
      icon: <Dna className="h-3 w-3" />,
    },
    {
      id: "therapeutic-strategies",
      label: "What therapeutic strategies are supported?",
      icon: <FlaskConical className="h-3 w-3" />,
    },
  ];
}

function getReportSuggestions(activeVariant: ActiveVariant): Suggestion[] {
  const report = activeVariant.reportData;
  const suggestions: Suggestion[] = [];

  const prediction = String(report.prediction ?? "").toLowerCase();
  if (prediction.includes("uncertain") || prediction.includes("vus")) {
    suggestions.push({
      id: "why-vus",
      label: "Why is this variant classified as VUS?",
      icon: <HelpCircle className="h-3 w-3" />,
    });
  }

  const extScores = report.externalScores as Record<string, unknown> | undefined;
  if (extScores?.alphamissense) {
    suggestions.push({
      id: "alphamissense",
      label: "What does AlphaMissense predict?",
      icon: <Microscope className="h-3 w-3" />,
    });
  }

  const vep = report.vepAnnotation as Record<string, unknown> | undefined;
  if (vep?.domains || vep?.transcriptId) {
    suggestions.push({
      id: "domains",
      label: "Which protein domains are affected?",
      icon: <Dna className="h-3 w-3" />,
    });
  }

  if (report.ismScanData) {
    suggestions.push({
      id: "ism-scan",
      label: "What does the ISM scan reveal?",
      icon: <Microscope className="h-3 w-3" />,
    });
  }

  suggestions.push(
    {
      id: "therapeutic-fit",
      label: "What therapeutic strategy fits this variant?",
      icon: <FlaskConical className="h-3 w-3" />,
    },
    {
      id: "experiments-next",
      label: "What experiments should I run next?",
      icon: <FileText className="h-3 w-3" />,
    },
  );

  return suggestions.slice(0, 4);
}
