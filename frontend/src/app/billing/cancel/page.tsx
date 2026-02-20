"use client";

import { useRouter } from "next/navigation";
import { XCircle } from "lucide-react";

export default function BillingCancelPage() {
    const router = useRouter();

    return (
        <div className="min-h-screen bg-[#f4f7f5] flex items-center justify-center p-6">
            <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md w-full text-center">
                <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <XCircle className="h-10 w-10 text-slate-500" />
                </div>
                <h1 className="text-2xl font-semibold text-[#3c4f3d] mb-2">
                    Payment Cancelled
                </h1>
                <p className="text-[#3c4f3d]/70 mb-6">
                    Your payment was cancelled. No charges were made to your account.
                </p>
                <button
                    onClick={() => router.push("/")}
                    className="px-6 py-2 bg-[#de8246] text-white rounded-lg hover:bg-[#c97340] transition-colors"
                >
                    Return to Dashboard
                </button>
            </div>
        </div>
    );
}
