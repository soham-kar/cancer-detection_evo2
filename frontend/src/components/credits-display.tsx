"use client";

import { useEffect, useState } from "react";
import { Coins, Loader2, ShoppingCart } from "lucide-react";

interface CreditStatus {
    credits: number;
    freeRunsToday: number;
    freeRunsRemaining: number;
    cooldownUntil: string | null;
    isOnCooldown: boolean;
}

export function CreditsDisplay() {
    const [creditStatus, setCreditStatus] = useState<CreditStatus | null>(null);
    const [loading, setLoading] = useState(true);
    const [purchasing, setPurchasing] = useState(false);

    const fetchCredits = async () => {
        try {
            const response = await fetch("/api/analyze");
            if (response.ok) {
                const data = await response.json();
                setCreditStatus(data);
            }
        } catch (error) {
            console.error("Failed to fetch credits:", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchCredits();

        // Listen for credits update events (dispatched after analysis)
        const handleCreditsUpdate = () => {
            fetchCredits();
        };
        window.addEventListener('credits-updated', handleCreditsUpdate);

        return () => {
            window.removeEventListener('credits-updated', handleCreditsUpdate);
        };
    }, []);

    const handlePurchase = async (packageId: string = "credits_5") => {
        setPurchasing(true);
        try {
            const response = await fetch("/api/stripe/checkout", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ packageId }),
            });

            const data = await response.json();
            if (data.url) {
                window.location.href = data.url;
            }
        } catch (error) {
            console.error("Checkout error:", error);
        } finally {
            setPurchasing(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center gap-2 bg-white px-3 py-1.5 rounded-lg border border-slate-200 shadow-sm">
                <Loader2 className="h-4 w-4 animate-spin text-slate-400" />
            </div>
        );
    }

    if (!creditStatus) return null;

    return (
        <div className="flex items-center gap-2">
            {/* Credits Display */}
            <div className="flex items-center gap-2 bg-white px-3 py-1.5 rounded-lg border border-slate-200 shadow-sm">
                <Coins className="h-4 w-4 text-[#de8246]" />
                <span className="text-xs font-mono font-bold text-slate-700">
                    {creditStatus.credits}
                </span>
                <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                    Credits
                </span>
            </div>

            {/* Buy Button */}
            <button
                onClick={() => handlePurchase("credits_5")}
                disabled={purchasing}
                className="flex items-center gap-1.5 bg-[#de8246] text-white px-3 py-1.5 rounded-lg text-xs font-medium hover:bg-[#c97340] transition-colors disabled:opacity-50"
            >
                {purchasing ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                    <ShoppingCart className="h-3.5 w-3.5" />
                )}
                Buy
            </button>

            {/* Cooldown Warning */}
            {creditStatus.isOnCooldown && (
                <div className="text-xs text-red-500 font-medium">
                    Cooldown Active
                </div>
            )}
        </div>
    );
}
