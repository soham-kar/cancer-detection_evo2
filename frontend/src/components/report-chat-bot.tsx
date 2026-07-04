"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "./ui/button";
import { Send, Bot, User, Loader2, Sparkles, Trash2 } from "lucide-react";

// =============================================================================
// ReportChatBot - AI Chatbot with Full Report Context
// =============================================================================
// Embedded chat interface that has access to the complete variant analysis
// report. Powered by NVIDIA Nemotron-3 550B.
//
// Features:
//   - Full report context (VEP, Evo2, gnomAD, ClinVar, UniProt, PubMed,
//     ACMG, AlphaMissense, CADD, REVEL, ISM, XAI, counterfactuals, etc.)
//   - Persistent chat sessions per report
//   - Message history across page reloads
//   - Streaming-like UX with loading states
// =============================================================================

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: string;
}

interface ChatSession {
  id: string;
  title: string;
  updatedAt: string;
  messages: ChatMessage[];
}

interface ReportChatBotProps {
  reportId: string;
  geneSymbol: string;
  variantLabel?: string;
}

export function ReportChatBot({ reportId, geneSymbol, variantLabel }: ReportChatBotProps) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load existing sessions on mount
  useEffect(() => {
    let cancelled = false;
    async function fetchSessions() {
      try {
        setIsLoadingHistory(true);
        const res = await fetch(`/api/chat?reportId=${reportId}`);
        if (!res.ok) throw new Error("Failed to load chat history");
        const data = (await res.json()) as { sessions: ChatSession[] };
        if (cancelled) return;
        setSessions(data.sessions || []);

        // Auto-select the most recent session
        if (data.sessions && data.sessions.length > 0) {
          const latest = data.sessions[0]!;
          setActiveSessionId(latest.id);
          setMessages(latest.messages || []);
        }
      } catch (e) {
        if (!cancelled) console.error("Failed to load chat sessions:", e);
      } finally {
        if (!cancelled) setIsLoadingHistory(false);
      }
    }
    fetchSessions();
    return () => { cancelled = true; };
  }, [reportId]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function loadSessions() {
    try {
      const res = await fetch(`/api/chat?reportId=${reportId}`);
      if (!res.ok) throw new Error("Failed to load chat history");
      const data = (await res.json()) as { sessions: ChatSession[] };
      setSessions(data.sessions || []);

      // Auto-select the most recent session
      if (data.sessions && data.sessions.length > 0) {
        const latest = data.sessions[0]!;
        setActiveSessionId(latest.id);
        setMessages(latest.messages || []);
      }
    } catch (e) {
      console.error("Failed to load chat sessions:", e);
    } finally {
      setIsLoadingHistory(false);
    }
  }

  async function loadSession(sessionId: string) {
    try {
      const res = await fetch(`/api/chat?sessionId=${sessionId}`);
      if (!res.ok) throw new Error("Failed to load session");
      const data = (await res.json()) as { session: ChatSession };
      setActiveSessionId(sessionId);
      setMessages(data.session.messages || []);
    } catch (e) {
      console.error("Failed to load session:", e);
    }
  }

  async function sendMessage() {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMessage = trimmed;
    setInput("");
    setError(null);
    setIsLoading(true);

    // Optimistically add user message
    const tempUserMsg: ChatMessage = {
      id: `temp-${Date.now()}`,
      role: "user",
      content: userMessage,
      createdAt: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUserMsg]);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reportId,
          message: userMessage,
          sessionId: activeSessionId,
        }),
      });

      if (!res.ok) {
        const errData = (await res.json()) as { error?: string };
        throw new Error(errData.error || "Failed to send message");
      }

      const data = (await res.json()) as {
        sessionId: string;
        message: string;
        tokens: { total_tokens: number };
        latency_ms: number;
      };

      // Update active session
      if (!activeSessionId) {
        setActiveSessionId(data.sessionId);
      }

      // Add assistant response
      const assistantMsg: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: data.message,
        createdAt: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);

      // Update session title in background without overwriting messages
      setSessions((prev) =>
        prev.map((s) =>
          s.id === data.sessionId
            ? { ...s, title: s.title || `Chat about ${geneSymbol}` }
            : s,
        ),
      );
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : "Failed to send message";
      setError(errMsg);
      // Remove the optimistic user message on error
      setMessages((prev) => prev.filter((m) => m.id !== tempUserMsg.id));
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function startNewChat() {
    setActiveSessionId(null);
    setMessages([]);
    setError(null);
    inputRef.current?.focus();
  }

  return (
    <div className="flex h-[500px] flex-col rounded-lg border border-slate-700 bg-slate-900/50">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-700 px-4 py-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-emerald-400" />
          <span className="text-sm font-medium text-slate-200">
            AI Chat — {geneSymbol} {variantLabel || ""}
          </span>
          <span className="rounded bg-emerald-900/50 px-1.5 py-0.5 text-[10px] text-emerald-400">
            Nemotron 550B
          </span>
        </div>
        <div className="flex items-center gap-2">
          {/* Session selector */}
          {sessions.length > 0 && (
            <select
              value={activeSessionId || ""}
              onChange={(e) => {
                if (e.target.value === "new") {
                  startNewChat();
                } else if (e.target.value) {
                  loadSession(e.target.value);
                }
              }}
              className="rounded border border-slate-600 bg-slate-800 px-2 py-1 text-xs text-slate-300"
            >
              <option value="new">+ New Chat</option>
              {sessions.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.title || `Chat ${s.id.slice(0, 8)}`}
                </option>
              ))}
            </select>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={startNewChat}
            className="h-7 text-xs text-slate-400 hover:text-slate-200"
          >
            <Sparkles className="mr-1 h-3 w-3" />
            New
          </Button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        {isLoadingHistory ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-5 w-5 animate-spin text-slate-500" />
          </div>
        ) : messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Bot className="mb-3 h-8 w-8 text-slate-600" />
            <p className="text-sm text-slate-400">
              Ask me anything about this variant analysis
            </p>
            <p className="mt-1 text-xs text-slate-600">
              I have full context of VEP, Evo2, gnomAD, ClinVar, UniProt,
              PubMed, ACMG, AlphaMissense, CADD, and more
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {[
                "Explain the Evo2 prediction",
                "What does the ACMG classification mean?",
                "Compare with ClinVar",
                "Is this variant pathogenic?",
                "What are the therapeutic options?",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => {
                    setInput(q);
                    inputRef.current?.focus();
                  }}
                  className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-400 hover:border-slate-500 hover:text-slate-300"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-3 ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {msg.role === "assistant" && (
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-emerald-900/50">
                    <Bot className="h-4 w-4 text-emerald-400" />
                  </div>
                )}
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-2.5 text-sm ${
                    msg.role === "user"
                      ? "bg-emerald-600/20 text-emerald-100"
                      : "bg-slate-800 text-slate-200"
                  }`}
                >
                  <div className="prose prose-sm prose-invert max-w-none whitespace-pre-wrap">
                    {msg.content}
                  </div>
                </div>
                {msg.role === "user" && (
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-700">
                    <User className="h-4 w-4 text-slate-300" />
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="flex items-center gap-2 text-slate-500">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-xs">Nemotron is thinking...</span>
              </div>
            )}
            {error && (
              <div className="rounded-lg bg-red-900/30 px-4 py-2 text-sm text-red-300">
                {error}
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-slate-700 px-4 py-3">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about this variant..."
            disabled={isLoading}
            className="flex-1 rounded-lg border border-slate-600 bg-slate-800 px-4 py-2 text-sm text-slate-200 placeholder:text-slate-500 focus:border-emerald-500 focus:outline-none disabled:opacity-50"
          />
          <Button
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            size="sm"
            className="bg-emerald-600 hover:bg-emerald-500"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
        <p className="mt-1.5 text-[10px] text-slate-600">
          Powered by NVIDIA Nemotron-3 550B. Full report context: VEP, Evo2,
          gnomAD, ClinVar, UniProt, PubMed, ACMG, AlphaMissense, CADD, REVEL.
          Research use only.
        </p>
      </div>
    </div>
  );
}
