"use client";

import { useEffect, useRef } from "react";
import { Bot } from "lucide-react";
import { ScrollArea } from "~/components/ui/scroll-area";
import { ChatMessage, type ChatMessageData } from "./chat-message";
import { ChatSuggestions } from "./chat-suggestions";
import type { ActiveVariant } from "~/contexts/active-variant";

// =============================================================================
// ChatMessageList - Scrollable conversation + suggestions
// =============================================================================

interface ChatMessageListProps {
  messages: ChatMessageData[];
  activeVariant: ActiveVariant | null;
  isLoading: boolean;
  onSuggestionSelect: (suggestion: string) => void;
  emptyTitle?: string;
  emptySubtitle?: string;
}

export function ChatMessageList({
  messages,
  activeVariant,
  isLoading,
  onSuggestionSelect,
  emptyTitle = "HelixMind AI Assistant",
  emptySubtitle = "Ask me anything about variant interpretation, therapeutic design, or genomic research.",
}: ChatMessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const showEmpty = messages.length === 0;

  return (
    <ScrollArea className="flex-1">
      <div className="flex min-h-full flex-col px-4 py-4">
        {showEmpty ? (
          <div className="flex flex-1 flex-col items-center justify-center text-center">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-[#de8246]/10">
              <Bot className="h-6 w-6 text-[#de8246]" />
            </div>
            <h3 className="mb-1 text-sm font-semibold text-[#3c4f3d]">{emptyTitle}</h3>
            <p className="mb-5 max-w-[260px] text-xs text-[#3c4f3d]/60">{emptySubtitle}</p>
            <ChatSuggestions
              activeVariant={activeVariant}
              onSelect={onSuggestionSelect}
              disabled={isLoading}
            />
          </div>
        ) : (
          <div className="flex flex-col gap-5">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>
    </ScrollArea>
  );
}
