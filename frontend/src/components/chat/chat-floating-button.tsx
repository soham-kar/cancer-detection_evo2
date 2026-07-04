"use client";

import { MessageCircle } from "lucide-react";
import { cn } from "~/lib/utils";
import { Button } from "~/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

// =============================================================================
// ChatFloatingButton - Circular FAB that opens the side panel
// =============================================================================

interface ChatFloatingButtonProps {
  isOpen: boolean;
  onClick: () => void;
  hasActiveVariant: boolean;
}

export function ChatFloatingButton({
  isOpen,
  onClick,
  hasActiveVariant,
}: ChatFloatingButtonProps) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            onClick={onClick}
            size="icon"
            className={cn(
              "fixed bottom-6 right-6 z-50 h-14 w-14 rounded-full shadow-xl transition-transform hover:scale-105 active:scale-95",
              isOpen ? "translate-x-[400px]" : "translate-x-0",
              hasActiveVariant
                ? "bg-[#de8246] hover:bg-[#c97340]"
                : "bg-[#3c4f3d] hover:bg-[#2a3a2b]",
            )}
          >
            <MessageCircle className="h-6 w-6 text-white" />
            {hasActiveVariant && (
              <span className="absolute top-1 right-1 h-3 w-3 rounded-full border-2 border-white bg-emerald-500" />
            )}
          </Button>
        </TooltipTrigger>
        <TooltipContent side="left">
          <p>{isOpen ? "Close AI assistant" : "Open AI assistant"}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
