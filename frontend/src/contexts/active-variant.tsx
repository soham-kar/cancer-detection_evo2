"use client";

import React, { createContext, useContext, useState, useCallback } from "react";

// =============================================================================
// ActiveVariantContext
// =============================================================================
// Tracks the currently active variant/report so the floating global chatbot
// can operate in two modes:
//   - Report Mode: a report is active → full report context is injected
//   - General Mode: no report active → free-form genomics Q&A
// =============================================================================

export interface ActiveVariant {
  reportId: string;
  geneSymbol: string;
  chromosome: string;
  position: number;
  reference: string;
  alternative: string;
  genomeId: string;
  prediction: string;
  deltaScore: number;
  classificationConfidence: number;
  clinvarClassification?: string | null;
  variationType?: string | null;
  clinvarId?: string | null;
  // Full report data is kept as a generic record so we can pass it straight
  // into the chat API / system prompt builder without re-mapping every field.
  reportData: Record<string, unknown>;
}

interface ActiveVariantContextValue {
  activeVariant: ActiveVariant | null;
  setActiveVariant: (variant: ActiveVariant | null) => void;
  clearActiveVariant: () => void;
  isChatPanelOpen: boolean;
  setChatPanelOpen: (open: boolean) => void;
  isChatExpanded: boolean;
  setChatExpanded: (expanded: boolean) => void;
}

const ActiveVariantContext = createContext<ActiveVariantContextValue | null>(
  null,
);

export function ActiveVariantProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [activeVariant, setActiveVariantState] = useState<ActiveVariant | null>(
    null,
  );
  const [isChatPanelOpen, setChatPanelOpenState] = useState(false);
  const [isChatExpanded, setChatExpandedState] = useState(false);

  const setActiveVariant = useCallback((variant: ActiveVariant | null) => {
    setActiveVariantState(variant);
  }, []);

  const clearActiveVariant = useCallback(() => {
    setActiveVariantState(null);
  }, []);

  const setChatPanelOpen = useCallback((open: boolean) => {
    setChatPanelOpenState(open);
  }, []);

  const setChatExpanded = useCallback((expanded: boolean) => {
    setChatExpandedState(expanded);
  }, []);

  return (
    <ActiveVariantContext.Provider
      value={{ activeVariant, setActiveVariant, clearActiveVariant, isChatPanelOpen, setChatPanelOpen, isChatExpanded, setChatExpanded }}
    >
      {children}
    </ActiveVariantContext.Provider>
  );
}

export function useActiveVariantContext() {
  const context = useContext(ActiveVariantContext);
  if (!context) {
    throw new Error(
      "useActiveVariantContext must be used within an ActiveVariantProvider",
    );
  }
  return context;
}
