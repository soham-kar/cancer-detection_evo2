"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Bot, Sparkles, X } from "lucide-react";
import { cn } from "~/lib/utils";
import { Button } from "~/components/ui/button";
import { Badge } from "~/components/ui/badge";
import { ChatMessageList } from "./chat-message-list";
import { ChatInput } from "./chat-input";
import { ChatSessionList, type ChatSessionItem } from "./chat-session-list";
import type { ChatMessageData } from "./chat-message";
import { useActiveVariant } from "~/hooks/use-active-variant";

// =============================================================================
// ChatSidePanel - Right-side push panel for the floating chatbot
// =============================================================================

interface ChatSidePanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ChatSidePanel({ isOpen, onClose }: ChatSidePanelProps) {
  const { activeVariant, clearActiveVariant } = useActiveVariant();
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [sessions, setSessions] = useState<ChatSessionItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const mode = activeVariant ? "report" : "general";

  // Load sessions whenever the panel opens or active variant changes
  useEffect(() => {
    if (!isOpen) return;

    let cancelled = false;
    async function loadSessions() {
      try {
        setIsLoadingHistory(true);
        const params = activeVariant
          ? `?reportId=${activeVariant.reportId}`
          : "";
        const res = await fetch(`/api/chat${params}`);
        if (!res.ok) throw new Error("Failed to load sessions");
        const data = (await res.json()) as { sessions: ChatSessionItem[] };
        if (cancelled) return;

        setSessions(data.sessions || []);

        // Auto-select most recent session if none active
        if (data.sessions?.length && !activeSessionId) {
          const latest = data.sessions[0]!;
          setActiveSessionId(latest.id);
          await loadSessionMessages(latest.id);
        }
      } catch (e) {
        if (!cancelled) console.error("[ChatSidePanel] loadSessions error:", e);
      } finally {
        if (!cancelled) setIsLoadingHistory(false);
      }
    }

    async function loadSessionMessages(sessionId: string) {
      try {
        const res = await fetch(`/api/chat?sessionId=${sessionId}`);
        if (!res.ok) throw new Error("Failed to load session messages");
        const data = (await res.json()) as {
          session: { messages: ChatMessageData[] };
        };
        if (cancelled) return;
        setMessages(data.session?.messages || []);
      } catch (e) {
        if (!cancelled) console.error("[ChatSidePanel] loadSessionMessages error:", e);
      }
    }

    loadSessions();

    return () => {
      cancelled = true;
    };
  }, [isOpen, activeVariant?.reportId]);

  const startNewChat = useCallback(() => {
    setActiveSessionId(null);
    setMessages([]);
  }, []);

  const selectSession = useCallback(async (sessionId: string) => {
    setActiveSessionId(sessionId);
    try {
      const res = await fetch(`/api/chat?sessionId=${sessionId}`);
      if (!res.ok) throw new Error("Failed to load session");
      const data = (await res.json()) as {
        session: { messages: ChatMessageData[] };
      };
      setMessages(data.session?.messages || []);
    } catch (e) {
      console.error("[ChatSidePanel] selectSession error:", e);
    }
  }, []);

  const stopGeneration = useCallback(() => {
    abortControllerRef.current?.abort();
    setIsLoading(false);
  }, []);

  const sendMessage = useCallback(
    async (messageText: string) => {
      if (isLoading) return;

      const userMessage: ChatMessageData = {
        id: `user-${Date.now()}`,
        role: "user",
        content: messageText,
        createdAt: new Date().toISOString(),
      };

      const assistantPlaceholder: ChatMessageData = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: "",
        reasoning: "",
        createdAt: new Date().toISOString(),
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
      setIsLoading(true);

      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: messageText,
            sessionId: activeSessionId,
            mode,
            reportId: activeVariant?.reportId,
          }),
          signal: abortController.signal,
        });

        if (!res.ok) {
          const err = (await res.json()) as { error?: string };
          throw new Error(err.error || `Request failed: ${res.status}`);
        }

        if (!res.body) throw new Error("No response body");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (let i = 0; i < lines.length; i++) {
            const trimmed = lines[i]!.trim();
            if (!trimmed) continue;

            // Parse SSE lines: "event: xxx" then "data: {...}"
            const eventMatch = trimmed.match(/^event:\s*(\S+)$/);
            if (eventMatch) {
              const eventName = eventMatch[1];
              const dataLine = lines[i + 1];
              if (!dataLine?.startsWith("data:")) continue;

              const dataStr = dataLine.slice(5).trim();
              try {
                const data = JSON.parse(dataStr) as Record<string, unknown>;

                if (eventName === "reasoning_delta") {
                  const delta = String(data.delta || "");
                  setMessages((prev) => {
                    const last = prev[prev.length - 1];
                    if (!last || last.role !== "assistant") return prev;
                    const updated = { ...last };
                    updated.reasoning = (updated.reasoning || "") + delta;
                    return [...prev.slice(0, -1), updated];
                  });
                } else if (eventName === "content_delta") {
                  const delta = String(data.delta || "");
                  setMessages((prev) => {
                    const last = prev[prev.length - 1];
                    if (!last || last.role !== "assistant") return prev;
                    const updated = { ...last };
                    updated.content = (updated.content || "") + delta;
                    return [...prev.slice(0, -1), updated];
                  });
                } else if (eventName === "done") {
                  const doneData = data as {
                    sessionId?: string;
                    latencyMs?: number;
                    tokens?: unknown;
                  };
                  if (doneData.sessionId) {
                    setActiveSessionId(doneData.sessionId);
                    setSessions((prev) => {
                      const exists = prev.some((s) => s.id === doneData.sessionId);
                      if (exists) return prev;
                      return [
                        {
                          id: doneData.sessionId!,
                          title: activeVariant
                            ? `Chat about ${activeVariant.geneSymbol}`
                            : "General genomics chat",
                          updatedAt: new Date().toISOString(),
                          reportGeneSymbol: activeVariant?.geneSymbol || null,
                        },
                        ...prev,
                      ];
                    });
                  }
                } else if (eventName === "error") {
                  throw new Error(String(data.error || "Streaming error"));
                }
              } catch {
                // Ignore malformed chunks
              }
            }
          }
        }

        // Mark streaming complete
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (!last || last.role !== "assistant") return prev;
          return [...prev.slice(0, -1), { ...last, isStreaming: false }];
        });
      } catch (e) {
        const errMsg = e instanceof Error ? e.message : "Failed to send message";
        console.error("[ChatSidePanel] sendMessage error:", errMsg);

        // Replace placeholder with error message
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last?.role === "assistant") {
            return [
              ...prev.slice(0, -1),
              {
                ...last,
                content: `Sorry, I encountered an error: ${errMsg}`,
                isStreaming: false,
              },
            ];
          }
          return prev;
        });
      } finally {
        setIsLoading(false);
        abortControllerRef.current = null;
      }
    },
    [isLoading, activeSessionId, mode, activeVariant],
  );

  const handleSuggestionSelect = useCallback(
    (suggestion: string) => {
      sendMessage(suggestion);
    },
    [sendMessage],
  );

  return (
    <div
      className={cn(
        "fixed top-0 right-0 z-[60] flex h-screen w-[400px] flex-col border-l border-[#3c4f3d]/10 bg-white shadow-2xl transition-transform duration-300 ease-in-out",
        isOpen ? "translate-x-0" : "translate-x-full",
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#3c4f3d]/10 px-4 py-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#de8246]/10">
            <Bot className="h-4 w-4 text-[#de8246]" />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold text-[#3c4f3d]">HelixMind AI</span>
            <span className="text-[10px] text-[#3c4f3d]/60">Nemotron-3 Ultra 550B</span>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {activeVariant ? (
            <Badge
              variant="outline"
              className="border-[#de8246]/30 bg-[#de8246]/10 text-[#de8246]"
            >
              {activeVariant.geneSymbol} {activeVariant.reference}&gt;{activeVariant.alternative}
            </Badge>
          ) : (
            <Badge
              variant="outline"
              className="border-[#3c4f3d]/20 bg-[#f4f7f5] text-[#3c4f3d]/70"
            >
              General
            </Badge>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-[#3c4f3d]/60 hover:text-[#3c4f3d]"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Active variant banner */}
      {activeVariant && (
        <div className="flex items-center justify-between border-b border-[#3c4f3d]/10 bg-[#f4f7f5] px-3 py-2">
          <div className="flex items-center gap-2">
            <Sparkles className="h-3 w-3 text-[#de8246]" />
            <span className="text-xs text-[#3c4f3d]">
              Context: {activeVariant.geneSymbol} @ {activeVariant.position}
            </span>
          </div>
          <button
            onClick={clearActiveVariant}
            className="text-[10px] text-[#3c4f3d]/60 hover:text-[#de8246]"
          >
            Clear
          </button>
        </div>
      )}

      {/* Session list */}
      <ChatSessionList
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelect={selectSession}
        onNew={startNewChat}
        onClose={onClose}
      />

      {/* Messages */}
      <ChatMessageList
        messages={messages}
        activeVariant={activeVariant}
        isLoading={isLoading || isLoadingHistory}
        onSuggestionSelect={handleSuggestionSelect}
        emptySubtitle={
          activeVariant
            ? `Ask me anything about ${activeVariant.geneSymbol} ${activeVariant.reference}>${activeVariant.alternative}. I have the full report context.`
            : "Ask me anything about variant interpretation, therapeutic design, or genomic research."
        }
      />

      {/* Input */}
      <ChatInput
        onSend={sendMessage}
        onStop={stopGeneration}
        isLoading={isLoading}
        placeholder={
          activeVariant
            ? `Ask about ${activeVariant.geneSymbol}...`
            : "Ask a genomics question..."
        }
      />
    </div>
  );
}
