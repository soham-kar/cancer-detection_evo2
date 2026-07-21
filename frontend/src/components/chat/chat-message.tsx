"use client";

import { useDeferredValue } from "react";
import { Bot, User, Brain } from "lucide-react";
import { cn } from "~/lib/utils";
import { Avatar, AvatarFallback } from "~/components/ui/avatar";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChatToolCallBlock, type ToolCallData } from "./chat-tool-call-block";

// =============================================================================
// StreamingCursor — blinking cursor at end of streaming text
// =============================================================================

function StreamingCursor() {
  return (
    <span className="animate-blink inline-block h-[1.1em] w-[2px] bg-[#de8246] align-text-bottom ml-0.5" />
  );
}

// =============================================================================
// ChatMessage - Individual message bubble with always-visible reasoning
// =============================================================================

export interface ChatMessageData {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  reasoning?: string;
  toolCalls?: ToolCallData[];
  createdAt: string;
  isStreaming?: boolean;
}

interface ChatMessageProps {
  message: ChatMessageData;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  // Defer content rendering during streaming to prevent jank from rapid SSE updates
  const deferredContent = useDeferredValue(message.content);
  const deferredReasoning = useDeferredValue(message.reasoning);
  const isStale = message.isStreaming && (
    deferredContent !== message.content || deferredReasoning !== message.reasoning
  );

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
          "flex max-w-[85%] flex-col gap-1.5 rounded-2xl px-4 py-2.5 text-sm transition-opacity duration-150",
          isStale ? "opacity-90" : "opacity-100",
          isUser
            ? "bg-[#3c4f3d] text-white"
            : "border border-[#3c4f3d]/10 bg-white text-[#3c4f3d]",
        )}
      >
        {/* Chain of thought — always visible for assistant messages with reasoning */}
        {!isUser && message.reasoning && (
          <div className="rounded-lg border border-[#de8246]/20 bg-[#de8246]/5 p-2.5">
            <div className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold tracking-wide text-[#de8246] uppercase">
              <Brain className="h-3 w-3" />
              {message.isStreaming ? "Thinking..." : "Chain of Thought"}
            </div>
            <div className="text-xs leading-relaxed text-[#3c4f3d]/70">
              {deferredReasoning}
              {message.isStreaming && <StreamingCursor />}
            </div>
          </div>
        )}

        {/* Tool calls — rendered between chain-of-thought and answer */}
        {!isUser && message.toolCalls && message.toolCalls.length > 0 && (
          <div className="flex flex-col gap-1.5">
            {message.toolCalls.map((tc) => (
              <ChatToolCallBlock key={tc.toolCallId} toolCall={tc} />
            ))}
          </div>
        )}

        {isUser ? (
          <div className="whitespace-pre-wrap leading-relaxed">
            {message.content}
          </div>
        ) : (
          <div className="chat-markdown leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {deferredContent || ""}
            </ReactMarkdown>
            {message.isStreaming && !message.reasoning && <StreamingCursor />}
          </div>
        )}
      </div>
    </div>
  );
}
