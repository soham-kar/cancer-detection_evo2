"use client";

import { useState } from "react";
import { History, MessageSquare, X } from "lucide-react";
import { cn } from "~/lib/utils";
import { Button } from "~/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

// =============================================================================
// ChatSessionList - Quick switcher for chat sessions
// =============================================================================

export interface ChatSessionItem {
  id: string;
  title: string | null;
  updatedAt: string;
  reportGeneSymbol?: string | null;
}

interface ChatSessionListProps {
  sessions: ChatSessionItem[];
  activeSessionId: string | null;
  onSelect: (sessionId: string) => void;
  onNew: () => void;
  onClose: () => void;
}

export function ChatSessionList({
  sessions,
  activeSessionId,
  onSelect,
  onNew,
  onClose,
}: ChatSessionListProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="border-b border-[#3c4f3d]/10 bg-[#f9fafb]">
      <div className="flex items-center justify-between px-3 py-2">
        <div className="flex items-center gap-2">
          <History className="h-3.5 w-3.5 text-[#3c4f3d]/60" />
          <span className="text-xs font-medium text-[#3c4f3d]/80">Sessions</span>
        </div>
        <div className="flex items-center gap-1">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 text-[#3c4f3d]/60 hover:text-[#de8246]"
                  onClick={onNew}
                >
                  <MessageSquare className="h-3.5 w-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p>New chat</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 text-[#3c4f3d]/60 hover:text-[#3c4f3d]"
            onClick={onClose}
          >
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {isExpanded && sessions.length > 0 && (
        <div className="max-h-40 overflow-y-auto border-t border-[#3c4f3d]/10 px-2 py-2">
          {sessions.map((session) => (
            <button
              key={session.id}
              onClick={() => onSelect(session.id)}
              className={cn(
                "w-full rounded-md px-2 py-1.5 text-left text-xs transition-colors",
                activeSessionId === session.id
                  ? "bg-[#de8246]/10 text-[#de8246]"
                  : "text-[#3c4f3d]/70 hover:bg-[#3c4f3d]/5",
              )}
            >
              <div className="truncate font-medium">
                {session.title || `Chat ${session.id.slice(0, 8)}`}
              </div>
              <div className="text-[10px] text-[#3c4f3d]/50">
                {formatDate(session.updatedAt)}
              </div>
            </button>
          ))}
        </div>
      )}

      {sessions.length > 0 && (
        <button
          onClick={() => setIsExpanded((v) => !v)}
          className="w-full border-t border-[#3c4f3d]/10 py-1 text-center text-[10px] text-[#3c4f3d]/50 hover:bg-[#3c4f3d]/5"
        >
          {isExpanded ? "Hide sessions" : `Show ${sessions.length} sessions`}
        </button>
      )}
    </div>
  );
}

function formatDate(dateString: string) {
  const date = new Date(dateString);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
