"use client";

import { Bot, User, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "~/lib/utils";
import { Avatar, AvatarFallback } from "~/components/ui/avatar";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "~/components/ui/collapsible";
import { useState } from "react";

// =============================================================================
// ChatMessage - Individual message bubble with optional reasoning
// =============================================================================

export interface ChatMessageData {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  reasoning?: string;
  createdAt: string;
  isStreaming?: boolean;
}

interface ChatMessageProps {
  message: ChatMessageData;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";
  const [reasoningOpen, setReasoningOpen] = useState(false);

  return (
    <div
      className={cn(
        "flex w-full gap-3",
        isUser ? "flex-row-reverse" : "flex-row",
      )}
    >
      <Avatar className={cn("h-8 w-8 shrink-0", isUser ? "bg-[#3c4f3d]" : "bg-[#de8246]")}>
        <AvatarFallback className={cn("text-white", isUser ? "bg-[#3c4f3d]" : "bg-[#de8246]")}>
          {isUser ? (
            <User className="h-4 w-4" />
          ) : (
            <Bot className="h-4 w-4" />
          )}
        </AvatarFallback>
      </Avatar>

      <div
        className={cn(
          "flex max-w-[85%] flex-col gap-1.5 rounded-2xl px-4 py-2.5 text-sm",
          isUser
            ? "bg-[#3c4f3d] text-white"
            : "border border-[#3c4f3d]/10 bg-white text-[#3c4f3d]",
        )}
      >
        {!isUser && message.reasoning && (
          <Collapsible open={reasoningOpen} onOpenChange={setReasoningOpen}>
            <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium text-[#de8246] hover:underline">
              {reasoningOpen ? (
                <ChevronUp className="h-3 w-3" />
              ) : (
                <ChevronDown className="h-3 w-3" />
              )}
              {message.isStreaming ? "Thinking..." : "Chain of thought"}
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="mt-1.5 rounded-lg bg-[#f4f7f5] p-2.5 text-xs italic text-[#3c4f3d]/80">
                {message.reasoning}
                {message.isStreaming && (
                  <span className="ml-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-[#de8246]" />
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        <div className="whitespace-pre-wrap leading-relaxed">
          {message.content}
          {message.isStreaming && !message.reasoning && (
            <span className="ml-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-[#de8246]" />
          )}
        </div>
      </div>
    </div>
  );
}
