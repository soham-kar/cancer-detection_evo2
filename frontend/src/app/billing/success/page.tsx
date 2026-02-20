"use client";

import { useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { CheckCircle, Loader2, XCircle } from "lucide-react";

export default function BillingSuccessPage() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
    const [message, setMessage] = useState("");
    const [creditsGranted, setCreditsGranted] = useState(0);

    useEffect(() => {
        const sessionId = searchParams.get("session_id");

        if (!sessionId) {
            setStatus("error");
            setMessage("No session ID found. Please try again.");
            return;
        }

        // Call the grant API to process the payment
        const grantCredits = async () => {
            try {
                const response = await fetch("/api/billing/grant", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ sessionId }),
                });

                const data = await response.json();

                if (response.ok) {
                    setStatus("success");
                    setCreditsGranted(data.creditsGranted);
                    setMessage(
                        data.alreadyProcessed
                            ? "Your credits were already added to your account."
                            : `Successfully added ${data.creditsGranted} credits to your account!`
                    );

                    // Redirect to home after 3 seconds
                    setTimeout(() => {
                        router.push("/");
                    }, 3000);
                } else {
                    setStatus("error");
                    setMessage(data.error || "Failed to process payment.");
                }
            } catch (error) {
                setStatus("error");
                setMessage("An error occurred. Please contact support.");
                console.error("Grant credits error:", error);
            }
        };

        grantCredits();
    }, [searchParams, router]);

    return (
        <div className="min-h-screen bg-[#f4f7f5] flex items-center justify-center p-6">
            <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md w-full text-center">
                {status === "loading" && (
                    <>
                        <Loader2 className="h-16 w-16 text-[#de8246] animate-spin mx-auto mb-4" />
                        <h1 className="text-2xl font-semibold text-[#3c4f3d] mb-2">
                            Processing Payment...
                        </h1>
                        <p className="text-[#3c4f3d]/70">
                            Please wait while we add credits to your account.
                        </p>
                    </>
                )}

                {status === "success" && (
                    <>
                        <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-4">
                            <CheckCircle className="h-10 w-10 text-emerald-600" />
                        </div>
                        <h1 className="text-2xl font-semibold text-[#3c4f3d] mb-2">
                            Payment Successful!
                        </h1>
                        <p className="text-[#3c4f3d]/70 mb-4">{message}</p>
                        <div className="bg-[#e9eeea] rounded-lg p-4 mb-6">
                            <p className="text-sm text-[#3c4f3d]/60">Credits Added</p>
                            <p className="text-3xl font-bold text-[#3c4f3d]">
                                +{creditsGranted}
                            </p>
                        </div>
                        <p className="text-sm text-[#3c4f3d]/50">
                            Redirecting to dashboard in 3 seconds...
                        </p>
                    </>
                )}

                {status === "error" && (
                    <>
                        <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                            <XCircle className="h-10 w-10 text-red-600" />
                        </div>
                        <h1 className="text-2xl font-semibold text-[#3c4f3d] mb-2">
                            Payment Error
                        </h1>
                        <p className="text-[#3c4f3d]/70 mb-6">{message}</p>
                        <button
                            onClick={() => router.push("/")}
                            className="px-6 py-2 bg-[#de8246] text-white rounded-lg hover:bg-[#c97340] transition-colors"
                        >
                            Return to Dashboard
                        </button>
                    </>
                )}
            </div>
        </div>
    );
}
