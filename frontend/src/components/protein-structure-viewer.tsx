"use client";

import { useEffect, useRef, useState } from "react";
import { Dna, Loader2, AlertTriangle, RotateCcw, ZoomIn, ZoomOut, Eye, EyeOff } from "lucide-react";

interface ProteinStructureViewerProps {
    uniprotId: string;
    geneSymbol: string;
    variantAA?: number | null;
    className?: string;
}

// Fetch PDB file via our proxy (bypasses CORS)
function getPdbProxyUrl(uniprotId: string): string {
    return `/api/pdb-proxy?uniprotId=${uniprotId}`;
}

export function ProteinStructureViewer({ uniprotId, geneSymbol, variantAA, className = "" }: ProteinStructureViewerProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const viewerRef = useRef<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [pdbData, setPdbData] = useState<string | null>(null);
    const [showVariant, setShowVariant] = useState(true);

    // Fetch PDB file from AlphaFold
    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        setError(null);

        async function fetchPdb() {
            try {
                const url = getPdbProxyUrl(uniprotId);
                const response = await fetch(url, { signal: AbortSignal.timeout(15000) });
                if (!response.ok) {
                    const err = await response.json().catch(() => ({}));
                    throw new Error(err.error || `PDB not found for ${uniprotId}`);
                }
                const text = await response.text();
                if (cancelled) return;
                setPdbData(text);
            } catch (err) {
                if (cancelled) return;
                setError(err instanceof Error ? err.message : "Failed to load structure");
                setLoading(false);
            }
        }

        fetchPdb();
        return () => { cancelled = true; };
    }, [uniprotId]);

    // Initialize 3Dmol viewer when PDB data is ready
    useEffect(() => {
        if (!pdbData || !containerRef.current) return;

        let cancelled = false;

        async function initViewer() {
            try {
                // Dynamic import 3Dmol (it's heavy, so load on demand)
                const $3Dmol = await import("3dmol");
                if (cancelled || !containerRef.current) return;

                // Clear previous viewer
                if (viewerRef.current) {
                    viewerRef.current.removeAllModels();
                }

                const element = containerRef.current;
                const config = { backgroundColor: "white" };
                const viewer = $3Dmol.createViewer(element, config);
                if (!viewer) throw new Error("Failed to create 3Dmol viewer");

                viewerRef.current = viewer;

                // Add model from PDB data
                viewer.addModel(pdbData, "pdb");

                // Style: cartoon with secondary structure coloring
                viewer.setStyle({}, { cartoon: { color: "spectrum" } });

                // If variant position is known, highlight it
                if (variantAA && showVariant) {
                    // Highlight the variant residue as spheres
                    const selection = { resi: variantAA };
                    viewer.addStyle(selection, { sphere: { radius: 0.8, color: "red" } });
                    viewer.addStyle(selection, { stick: { radius: 0.3, color: "red" } });

                    // Label the residue
                    viewer.addLabel(`${geneSymbol} ${variantAA}`, {
                        backgroundColor: "rgba(255,255,255,0.9)",
                        fontColor: "red",
                        fontSize: 12,
                        borderThickness: 1,
                        borderColor: "red",
                        inFront: true,
                    }, selection);
                }

                // Zoom to fit
                viewer.zoomTo();
                viewer.render();

                setLoading(false);
            } catch (err) {
                if (cancelled) return;
                setError(err instanceof Error ? err.message : "Viewer initialization failed");
                setLoading(false);
            }
        }

        initViewer();
        return () => { cancelled = true; };
    }, [pdbData, variantAA, geneSymbol, showVariant]);

    // Handle zoom controls
    const handleZoomIn = () => {
        if (viewerRef.current) {
            viewerRef.current.zoom(1.2);
            viewerRef.current.render();
        }
    };

    const handleZoomOut = () => {
        if (viewerRef.current) {
            viewerRef.current.zoom(0.8);
            viewerRef.current.render();
        }
    };

    const handleReset = () => {
        if (viewerRef.current) {
            viewerRef.current.zoomTo();
            viewerRef.current.render();
        }
    };

    const handleToggleVariant = () => {
        setShowVariant((v) => !v);
    };

    if (error) {
        return (
            <div className={`rounded-xl border border-slate-200 bg-white p-4 ${className}`}>
                <div className="flex items-center gap-2 mb-2">
                    <Dna className="h-4 w-4 text-indigo-600" />
                    <span className="text-xs font-semibold text-slate-700">3D Protein Structure</span>
                </div>
                <div className="rounded-lg bg-amber-50 border border-amber-200 p-3 flex items-start gap-2">
                    <AlertTriangle className="h-4 w-4 text-amber-600 flex-shrink-0 mt-0.5" />
                    <div className="text-xs text-amber-700">
                        <strong>Structure unavailable:</strong> {error}
                        <div className="mt-1">
                            <a
                                href={`https://alphafold.ebi.ac.uk/entry/${uniprotId}`}
                                target="_blank" rel="noopener noreferrer"
                                className="text-indigo-600 hover:text-indigo-800 underline"
                            >
                                View on AlphaFold DB →
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className={`rounded-xl border border-slate-200 bg-white overflow-hidden ${className}`}>
            {/* Header */}
            <div className="border-b border-slate-100 px-4 py-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Dna className="h-4 w-4 text-indigo-600" />
                    <div>
                        <span className="text-xs font-semibold text-slate-700">3D Protein Structure</span>
                        <span className="text-[10px] text-slate-400 ml-1.5">AlphaFold DB v4 · {geneSymbol}</span>
                    </div>
                </div>
                <div className="flex items-center gap-1">
                    {variantAA && (
                        <button
                            onClick={handleToggleVariant}
                            className={`p-1.5 rounded-md transition-colors ${showVariant ? "bg-red-50 text-red-600" : "bg-slate-50 text-slate-400"}`}
                            title={showVariant ? "Hide variant highlight" : "Show variant highlight"}
                        >
                            {showVariant ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
                        </button>
                    )}
                    <button onClick={handleZoomIn} className="p-1.5 rounded-md hover:bg-slate-100 text-slate-500" title="Zoom in">
                        <ZoomIn className="h-3.5 w-3.5" />
                    </button>
                    <button onClick={handleZoomOut} className="p-1.5 rounded-md hover:bg-slate-100 text-slate-500" title="Zoom out">
                        <ZoomOut className="h-3.5 w-3.5" />
                    </button>
                    <button onClick={handleReset} className="p-1.5 rounded-md hover:bg-slate-100 text-slate-500" title="Reset view">
                        <RotateCcw className="h-3.5 w-3.5" />
                    </button>
                </div>
            </div>

            {/* Viewer */}
            <div className="relative">
                {loading && (
                    <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-white/90">
                        <Loader2 className="h-8 w-8 text-indigo-500 animate-spin mb-2" />
                        <span className="text-xs text-slate-500">Loading AlphaFold structure…</span>
                    </div>
                )}
                <div
                    ref={containerRef}
                    style={{ width: "100%", height: "380px", position: "relative" }}
                />
            </div>

            {/* Footer info */}
            <div className="border-t border-slate-100 px-4 py-2.5 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    {variantAA && showVariant && (
                        <span className="inline-flex items-center gap-1 text-[10px] text-red-600">
                            <span className="h-2 w-2 rounded-full bg-red-500" />
                            Variant position: aa {variantAA}
                        </span>
                    )}
                    <span className="text-[10px] text-slate-400">
                        Coloring: N-term (blue) → C-term (red)
                    </span>
                </div>
                <a
                    href={`https://alphafold.ebi.ac.uk/entry/${uniprotId}`}
                    target="_blank" rel="noopener noreferrer"
                    className="text-[10px] text-indigo-500 hover:text-indigo-700 underline"
                >
                    AlphaFold:{uniprotId} ↗
                </a>
            </div>
        </div>
    );
}
