"use client";

import { SignUp } from "@clerk/nextjs";
import { Dna, Sparkles, ArrowRight } from "lucide-react";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export default function SignUpPage() {
    const router = useRouter();
    const [phase, setPhase] = useState<"splash" | "transition" | "login">("splash");
    const [demoLoading, setDemoLoading] = useState(false);

    useEffect(() => {
        // Phase 1: Show splash for 2 seconds
        const timer1 = setTimeout(() => setPhase("transition"), 2000);
        // Phase 2: Transition to signup after animation
        const timer2 = setTimeout(() => setPhase("login"), 2800);

        return () => {
            clearTimeout(timer1);
            clearTimeout(timer2);
        };
    }, []);

    const handleTryDemo = () => {
        setDemoLoading(true);
        router.push("/demo");  // Go to demo variant selection
    };

    return (
        <div className="flex min-h-screen items-center justify-center overflow-hidden relative">
            {/* Static beautiful background image */}
            <div
                className="absolute inset-0 bg-cover bg-center bg-no-repeat"
                style={{
                    backgroundImage: 'url("/Gemini_Generated_Image_953ram953ram953r.png")',
                    opacity: 0.85,
                }}
            />
            {/* Gradient overlay for readability */}
            <div className="absolute inset-0 bg-gradient-to-br from-[#e9eeea]/35 via-[#d4ddd6]/30 to-[#c5d1c7]/35" />

            <div className="relative w-full max-w-md px-4">
                {/* SPLASH SCREEN - Big Logo */}
                <div
                    className={`absolute inset-0 flex flex-col items-center justify-center transition-all duration-800 ease-out ${phase === "splash"
                        ? 'opacity-100 scale-100'
                        : 'opacity-0 scale-50 pointer-events-none'
                        }`}
                    style={{ transitionDuration: '800ms' }}
                >
                    <div className="flex items-center gap-4 mb-6">
                        <div className="relative">
                            <Dna
                                className="h-20 w-20 text-[#de8246]"
                                style={{
                                    animation: 'breathe 2s ease-in-out infinite',
                                }}
                            />
                            <style jsx global>{`
                                @keyframes breathe {
                                    0%, 100% {
                                        transform: scale(1);
                                        filter: drop-shadow(0 0 8px rgba(222, 130, 70, 0.4));
                                    }
                                    50% {
                                        transform: scale(1.1);
                                        filter: drop-shadow(0 0 20px rgba(222, 130, 70, 0.6));
                                    }
                                }
                            `}</style>
                            <div className="absolute inset-0 h-20 w-20 rounded-full bg-[#de8246]/20 blur-2xl animate-pulse" />
                        </div>
                        <h1 className="text-6xl font-bold text-[#3c4f3d]">
                            HelixMind
                        </h1>
                    </div>
                    <p className="text-xl text-[#3c4f3d]/70 animate-pulse">
                        AI-Powered Genomic Variant Analysis
                    </p>
                    <div className="mt-8 flex gap-2">
                        <div className="h-2 w-2 rounded-full bg-[#de8246] animate-bounce" style={{ animationDelay: '0ms' }} />
                        <div className="h-2 w-2 rounded-full bg-[#de8246] animate-bounce" style={{ animationDelay: '150ms' }} />
                        <div className="h-2 w-2 rounded-full bg-[#de8246] animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                </div>

                {/* SIGNUP FORM - Small Logo + Form + Try Demo */}
                <div
                    className={`transition-all duration-700 ease-out ${phase === "login"
                        ? 'opacity-100 translate-y-0'
                        : 'opacity-0 translate-y-8'
                        }`}
                >
                    {/* Small Logo Header */}
                    <div className="mb-6 text-center">
                        <div className="flex items-center justify-center gap-2 mb-2">
                            <Dna className="h-8 w-8 text-[#de8246]" />
                            <h1 className="text-2xl font-bold text-[#3c4f3d]">
                                HelixMind
                            </h1>
                        </div>
                        <p className="text-sm text-[#3c4f3d]/70">
                            Clinical-grade variant interpretation powered by AI
                        </p>
                    </div>

                    {/* TRY DEMO BUTTON - Bright glowing attention grabber */}
                    <button
                        onClick={handleTryDemo}
                        disabled={demoLoading}
                        className="w-full mb-4 py-4 px-6 rounded-xl bg-gradient-to-r from-violet-500 via-fuchsia-500 to-pink-500 text-white font-bold text-lg flex items-center justify-center gap-2 shadow-[0_0_30px_rgba(168,85,247,0.5)] hover:shadow-[0_0_40px_rgba(236,72,153,0.6)] hover:scale-[1.03] active:scale-[0.98] transition-all duration-300 disabled:opacity-70 disabled:scale-100 animate-pulse"
                        style={{ animationDuration: '2s' }}
                    >
                        <Sparkles className="h-5 w-5" />
                        {demoLoading ? "Loading..." : "Try Demo"}
                        <ArrowRight className="h-5 w-5" />
                    </button>

                    {/* Divider */}
                    <div className="flex items-center gap-3 mb-4">
                        <div className="flex-1 h-px bg-[#3c4f3d]/20" />
                        <span className="text-xs text-[#3c4f3d]/50 uppercase tracking-wider">or sign up for full access</span>
                        <div className="flex-1 h-px bg-[#3c4f3d]/20" />
                    </div>

                    {/* Sign Up Form */}
                    <SignUp
                        appearance={{
                            elements: {
                                rootBox: "w-full",
                                card: "shadow-xl border border-[#3c4f3d]/10 rounded-xl",
                                formButtonPrimary: "bg-[#3c4f3d] hover:bg-[#3c4f3d]/90 transition-colors",
                                headerTitle: "hidden",
                                headerSubtitle: "hidden",
                                socialButtonsBlockButton: "border-[#3c4f3d]/20 hover:bg-[#e9eeea]/50",
                                formFieldInput: "border-[#3c4f3d]/20 focus:border-[#de8246] focus:ring-[#de8246]/20",
                                footerActionLink: "text-[#de8246] hover:text-[#de8246]/80",
                            },
                        }}
                    />

                    {/* Features hint */}
                    <div className="mt-4 text-center">
                        <p className="text-xs text-[#3c4f3d]/50">
                            Free accounts get 10 analyses/day • Full ClinVar + gnomAD + PubMed
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
