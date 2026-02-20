"use client";

import { AlertTriangle, CheckCircle } from 'lucide-react';

export default function ScoreGauge({
    score,
    confidence,
    prediction
}: {
    score: number,
    confidence: number,
    prediction: string
}) {
    const isPathogenic = prediction.toLowerCase().includes("pathogenic");

    // LOGIC: Map confidence to a 0-100% position on the spectrum
    // Negative scores (Pathogenic) → LEFT side
    // Positive scores (Benign) → RIGHT side
    let position = 50;
    if (isPathogenic) {
        position = 50 - (confidence * 45); // e.g., 90% conf -> 5% position (left)
    } else {
        position = 50 + (confidence * 45); // e.g., 90% conf -> 95% position (right)
    }

    return (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            {/* Header Text */}
            <div className="flex justify-between items-end mb-4">
                <div className="flex flex-col">
                    <span className="text-slate-500 font-bold text-xs uppercase tracking-wider">Evo-2 Pathogenicity Score</span>
                    <div className="flex items-center gap-2 mt-1">
                        {isPathogenic ? (
                            <AlertTriangle className="w-5 h-5 text-rose-500" />
                        ) : (
                            <CheckCircle className="w-5 h-5 text-emerald-500" />
                        )}
                        <span className="text-2xl font-bold text-slate-900">
                            {score.toFixed(4)}
                        </span>
                    </div>
                </div>
                <div className="text-right">
                    <span className={`text-sm font-bold px-3 py-1 rounded-full ${isPathogenic ? "bg-rose-100 text-rose-700" : "bg-emerald-100 text-emerald-700"}`}>
                        {prediction}
                    </span>
                    <p className="text-xs text-slate-400 mt-1">{Math.round(confidence * 100)}% Confidence</p>
                </div>
            </div>

            {/* --- THE SPECTRUM GAUGE --- */}
            {/* Negative (Pathogenic) on LEFT → Positive (Benign) on RIGHT */}
            <div className="relative pt-2 pb-6 px-2">

                {/* 1. The Gradient Track: Red (left/negative) → Green (right/positive) */}
                <div
                    className="h-4 w-full rounded-full shadow-inner"
                    style={{
                        background: 'linear-gradient(90deg, #f43f5e 0%, #e5e7eb 50%, #10b981 100%)'
                    }}
                />

                {/* 2. The Marker (Triangle + Dot) */}
                <div
                    className="absolute top-0 transition-all duration-1000 ease-out flex flex-col items-center"
                    style={{ left: `${position}%`, transform: 'translateX(-50%)' }}
                >
                    {/* Triangle Indicator */}
                    <div className="w-0 h-0 border-l-[8px] border-l-transparent border-r-[8px] border-r-transparent border-t-[10px] border-t-slate-800 mb-1" />

                    {/* Small Dot on the bar */}
                    <div className="w-5 h-5 bg-white border-4 border-slate-800 rounded-full shadow-md mt-[-6px]" />
                </div>

                {/* 3. Labels: Negative (left) → Positive (right) */}
                <div className="flex justify-between text-xs font-bold text-slate-400 mt-3 uppercase tracking-widest">
                    <span className="text-rose-600">Pathogenic</span>
                    <span className="text-slate-300">Uncertain</span>
                    <span className="text-emerald-600">Benign</span>
                </div>
            </div>
        </div>
    );
}
