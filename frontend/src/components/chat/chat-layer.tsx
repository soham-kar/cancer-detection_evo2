"use client";

import { ChatFloatingButton } from "./chat-floating-button";
import { ChatSidePanel } from "./chat-side-panel";
import { useActiveVariant } from "~/hooks/use-active-variant";

// =============================================================================
// ChatLayer - Wrapper that renders the floating button + side panel
// =============================================================================

export function ChatLayer() {
  const { activeVariant, isChatPanelOpen, setChatPanelOpen, isChatExpanded, setChatExpanded } = useActiveVariant();

  return (
    <>
      <ChatFloatingButton
        isOpen={isChatPanelOpen}
        onClick={() => setChatPanelOpen(!isChatPanelOpen)}
        hasActiveVariant={!!activeVariant}
        isExpanded={isChatExpanded}
      />
      <ChatSidePanel
        isOpen={isChatPanelOpen}
        onClose={() => setChatPanelOpen(false)}
        isExpanded={isChatExpanded}
        onToggleExpand={() => setChatExpanded(!isChatExpanded)}
      />
    </>
  );
}
