"use client";

import { useRef, useState } from "react";
import { Send, Square } from "lucide-react";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";

// =============================================================================
// ChatInput - Text input with send and stop controls
// =============================================================================

interface ChatInputProps {
  onSend: (message: string) => void;
  onStop?: () => void;
  isLoading: boolean;
  placeholder?: string;
  disabled?: boolean;
}

export function ChatInput({
  onSend,
  onStop,
  isLoading,
  placeholder = "Ask a question...",
  disabled = false,
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  function handleSend() {
    const trimmed = input.trim();
    if (!trimmed || isLoading || disabled) return;
    setInput("");
    onSend(trimmed);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="flex items-end gap-2 border-t border-[#3c4f3d]/10 bg-white p-3">
      <Input
        ref={inputRef}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled || isLoading}
        className="min-h-[40px] flex-1 resize-none border-[#3c4f3d]/20 bg-[#f4f7f5] text-sm text-[#3c4f3d] placeholder:text-[#3c4f3d]/40 focus-visible:border-[#de8246] focus-visible:ring-[#de8246]/20"
      />
      {isLoading ? (
        <Button
          type="button"
          size="icon"
          variant="outline"
          onClick={onStop}
          className="h-9 w-9 shrink-0 border-red-200 text-red-600 hover:bg-red-50 hover:text-red-700"
        >
          <Square className="h-4 w-4 fill-current" />
        </Button>
      ) : (
        <Button
          type="button"
          size="icon"
          onClick={handleSend}
          disabled={!input.trim() || disabled}
          className="h-9 w-9 shrink-0 bg-[#de8246] hover:bg-[#c97340] disabled:opacity-50"
        >
          <Send className="h-4 w-4" />
        </Button>
      )}
    </div>
  );
}
