"use client";

import { useState } from "react";
import { ChatFloatingButton } from "./chat-floating-button";
import { ChatSidePanel } from "./chat-side-panel";
import { useActiveVariant } from "~/hooks/use-active-variant";

// =============================================================================
// ChatLayer - Wrapper that renders the floating button + side panel
// =============================================================================

export function ChatLayer() {
  const [isOpen, setIsOpen] = useState(false);
  const { activeVariant } = useActiveVariant();

  // Keep panel closed by default. It opens via the floating button.

  return (
    <>
      <ChatFloatingButton
        isOpen={isOpen}
        onClick={() => setIsOpen((v) => !v)}
        hasActiveVariant={!!activeVariant}
      />
      <ChatSidePanel isOpen={isOpen} onClose={() => setIsOpen(false)} />
    </>
  );
}
