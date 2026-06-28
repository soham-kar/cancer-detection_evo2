"use client";

import { useEffect, useRef, useState } from "react";
import {
  Dna,
  Loader2,
  AlertTriangle,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Eye,
  EyeOff,
} from "lucide-react";

interface ProteinStructureViewerProps {
  uniprotId: string;
  geneSymbol: string;
  variantAA?: number | null;
  className?: string;
}

interface WindowWithPdb extends Window {
  __pdbFormat?: string;
  __pdbConfidence?: string | null;
}

// Fetch PDB file via our proxy (bypasses CORS)
function getPdbProxyUrl(uniprotId: string): string {
  return `/api/pdb-proxy?uniprotId=${uniprotId}`;
}

export function ProteinStructureViewer({
  uniprotId,
  geneSymbol,
  variantAA,
  className = "",
}: ProteinStructureViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pdbData, setPdbData] = useState<string | null>(null);
  const [showVariant, setShowVariant] = useState(true);
  const [variantPlddt, setVariantPlddt] = useState<number | null>(null);

  // Fetch PDB file from AlphaFold
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    async function fetchPdb() {
      try {
        const url = getPdbProxyUrl(uniprotId);
        const response = await fetch(url, {
          signal: AbortSignal.timeout(15000),
        });
        if (!response.ok) {
          const err = await response.json().catch(() => ({}));
          throw new Error(err.error || `Structure not found for ${uniprotId}`);
        }

        // Get format from headers
        const format = response.headers.get("X-Model-Format") || "PDB";
        const confidence = response.headers.get("X-Confidence-Score");

        const text = await response.text();
        if (cancelled) return;
        setPdbData(text);

        // Store format for viewer initialization
        (window as WindowWithPdb).__pdbFormat =
          format.toLowerCase() === "mmcif" ? "mmcif" : "pdb";
        (window as WindowWithPdb).__pdbConfidence = confidence;
      } catch (err) {
        if (cancelled) return;
        setError(
          err instanceof Error ? err.message : "Failed to load structure",
        );
        setLoading(false);
      }
    }

    fetchPdb();
    return () => {
      cancelled = true;
    };
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
        const previousViewer = viewerRef.current as {
          removeAllModels?: () => void;
        } | null;
        if (previousViewer?.removeAllModels) {
          previousViewer.removeAllModels();
        }

        const element = containerRef.current;
        const config = { backgroundColor: "white" };
        const viewer = $3Dmol.createViewer(element, config);
        if (!viewer) throw new Error("Failed to create 3Dmol viewer");

        viewerRef.current = viewer;

        // Add model from structure data (auto-detect format)
        const format = (window as WindowWithPdb).__pdbFormat || "pdb";
        const model = viewer.addModel(pdbData, format);

        // ── pLDDT Confidence Coloring ──
        // AlphaFold stores per-residue confidence (pLDDT) in the B-factor column
        // Color scheme: Very high (>90) = dark blue, High (70-90) = light blue,
        // Low (50-70) = yellow, Very low (<50) = orange/red
        viewer.setStyle(
          {},
          {
            cartoon: {
              colorscheme: {
                prop: "b",
                gradient: "roygb",
                min: 50,
                max: 100,
              },
            },
          },
        );

        // Alternative: custom color function for discrete pLDDT bins
        // This gives clinicians the standard AlphaFold color scheme
        const plddtColors: Record<number, string> = {};
        const atoms = model.selectedAtoms({});
        for (const atom of atoms) {
          const plddt = atom.b || 0; // B-factor = pLDDT
          let color: string;
          if (plddt >= 90)
            color = "#0053D9"; // Very high (dark blue)
          else if (plddt >= 70)
            color = "#65CBF3"; // High (light blue)
          else if (plddt >= 50)
            color = "#FFDB13"; // Low (yellow)
          else color = "#FF7D45"; // Very low (orange)

          if (atom.serial != null) {
            plddtColors[atom.serial] = color;
          }
        }

        // Apply pLDDT coloring
        viewer.setStyle(
          {},
          {
            cartoon: {
              colorfunc: (atom: { b?: number }) => {
                const plddt = atom.b || 0;
                if (plddt >= 90) return "#0053D9";
                if (plddt >= 70) return "#65CBF3";
                if (plddt >= 50) return "#FFDB13";
                return "#FF7D45";
              },
            },
          },
        );

        // If variant position is known, highlight it and extract pLDDT
        if (variantAA && showVariant) {
          // Find the variant atom to get its pLDDT
          const variantAtoms = model.selectedAtoms({ resi: variantAA });
          if (variantAtoms.length > 0) {
            const plddt = variantAtoms[0]?.b || 0;
            setVariantPlddt(Math.round(plddt));
          }

          // Highlight the variant residue as spheres
          const selection = { resi: variantAA };
          viewer.addStyle(selection, { sphere: { radius: 0.8, color: "red" } });
          viewer.addStyle(selection, { stick: { radius: 0.3, color: "red" } });

          // Label the residue
          viewer.addLabel(
            `${geneSymbol} ${variantAA}`,
            {
              backgroundColor: "rgba(255,255,255,0.9)",
              fontColor: "red",
              fontSize: 12,
              borderThickness: 1,
              borderColor: "red",
              inFront: true,
            },
            selection,
          );
        }

        // Zoom to fit
        viewer.zoomTo();
        viewer.render();

        setLoading(false);
      } catch (err) {
        if (cancelled) return;
        setError(
          err instanceof Error ? err.message : "Viewer initialization failed",
        );
        setLoading(false);
      }
    }

    initViewer();
    return () => {
      cancelled = true;
    };
  }, [pdbData, variantAA, geneSymbol, showVariant]);

  // Handle zoom controls
  const handleZoomIn = () => {
    const viewer = viewerRef.current as {
      zoom?: (factor: number) => void;
      render?: () => void;
    } | null;
    if (viewer?.zoom && viewer?.render) {
      viewer.zoom(1.2);
      viewer.render();
    }
  };

  const handleZoomOut = () => {
    const viewer = viewerRef.current as {
      zoom?: (factor: number) => void;
      render?: () => void;
    } | null;
    if (viewer?.zoom && viewer?.render) {
      viewer.zoom(0.8);
      viewer.render();
    }
  };

  const handleReset = () => {
    const viewer = viewerRef.current as {
      zoomTo?: () => void;
      render?: () => void;
    } | null;
    if (viewer?.zoomTo && viewer?.render) {
      viewer.zoomTo();
      viewer.render();
    }
  };

  const handleToggleVariant = () => {
    setShowVariant((v) => !v);
  };

  if (error) {
    return (
      <div
        className={`rounded-xl border border-slate-200 bg-white p-4 ${className}`}
      >
        <div className="mb-2 flex items-center gap-2">
          <Dna className="h-4 w-4 text-indigo-600" />
          <span className="text-xs font-semibold text-slate-700">
            3D Protein Structure
          </span>
        </div>
        <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3">
          <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-600" />
          <div className="text-xs text-amber-700">
            <strong>Structure unavailable:</strong> {error}
            <div className="mt-1.5 flex items-center gap-2">
              <button
                onClick={() => {
                  setError(null);
                  setLoading(true);
                  setPdbData(null);
                }}
                className="rounded border border-indigo-200 bg-indigo-50 px-2 py-1 text-[10px] text-indigo-600 transition-colors hover:bg-indigo-100"
              >
                Retry
              </button>
              <a
                href={`https://alphafold.ebi.ac.uk/entry/${uniprotId}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-indigo-600 underline hover:text-indigo-800"
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
    <div
      className={`overflow-hidden rounded-xl border border-slate-200 bg-white ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <div className="flex items-center gap-2">
          <Dna className="h-4 w-4 text-indigo-600" />
          <div>
            <span className="text-xs font-semibold text-slate-700">
              3D Protein Structure
            </span>
            <span className="ml-1.5 text-[10px] text-slate-400">
              AlphaFold DB v4 · {geneSymbol}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {variantAA && (
            <button
              onClick={handleToggleVariant}
              className={`rounded-md p-1.5 transition-colors ${showVariant ? "bg-red-50 text-red-600" : "bg-slate-50 text-slate-400"}`}
              title={
                showVariant
                  ? "Hide variant highlight"
                  : "Show variant highlight"
              }
            >
              {showVariant ? (
                <Eye className="h-3.5 w-3.5" />
              ) : (
                <EyeOff className="h-3.5 w-3.5" />
              )}
            </button>
          )}
          <button
            onClick={handleZoomIn}
            className="rounded-md p-1.5 text-slate-500 hover:bg-slate-100"
            title="Zoom in"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={handleZoomOut}
            className="rounded-md p-1.5 text-slate-500 hover:bg-slate-100"
            title="Zoom out"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={handleReset}
            className="rounded-md p-1.5 text-slate-500 hover:bg-slate-100"
            title="Reset view"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Viewer */}
      <div className="relative">
        {loading && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-white/90">
            <Loader2 className="mb-2 h-8 w-8 animate-spin text-indigo-500" />
            <span className="text-xs text-slate-500">
              Loading AlphaFold structure…
            </span>
          </div>
        )}
        <div
          ref={containerRef}
          style={{ width: "100%", height: "380px", position: "relative" }}
        />
      </div>

      {/* Footer info */}
      <div className="space-y-2 border-t border-slate-100 px-4 py-2.5">
        {/* pLDDT Legend */}
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-[10px] font-medium text-slate-500">
            pLDDT Confidence:
          </span>
          <span className="inline-flex items-center gap-1 text-[10px]">
            <span
              className="h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: "#0053D9" }}
            />
            <span className="text-slate-600">Very high (&gt;90)</span>
          </span>
          <span className="inline-flex items-center gap-1 text-[10px]">
            <span
              className="h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: "#65CBF3" }}
            />
            <span className="text-slate-600">High (70–90)</span>
          </span>
          <span className="inline-flex items-center gap-1 text-[10px]">
            <span
              className="h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: "#FFDB13" }}
            />
            <span className="text-slate-600">Low (50–70)</span>
          </span>
          <span className="inline-flex items-center gap-1 text-[10px]">
            <span
              className="h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: "#FF7D45" }}
            />
            <span className="text-slate-600">Very low (&lt;50)</span>
          </span>
        </div>

        {/* Variant-specific pLDDT */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {variantAA && showVariant && variantPlddt !== null && (
              <span
                className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-medium ${variantPlddt >= 90 ? "border border-blue-200 bg-blue-50 text-blue-700" : variantPlddt >= 70 ? "border border-sky-200 bg-sky-50 text-sky-700" : variantPlddt >= 50 ? "border border-yellow-200 bg-yellow-50 text-yellow-700" : "border border-orange-200 bg-orange-50 text-orange-700"}`}
              >
                <span
                  className="h-2 w-2 rounded-full"
                  style={{
                    backgroundColor:
                      variantPlddt >= 90
                        ? "#0053D9"
                        : variantPlddt >= 70
                          ? "#65CBF3"
                          : variantPlddt >= 50
                            ? "#FFDB13"
                            : "#FF7D45",
                  }}
                />
                Variant pLDDT: {variantPlddt}
                {variantPlddt >= 90
                  ? " (Very high — reliable)"
                  : variantPlddt >= 70
                    ? " (High — interpretable)"
                    : variantPlddt >= 50
                      ? " (Low — caution)"
                      : " (Very low — speculative)"}
              </span>
            )}
            {variantAA && showVariant && variantPlddt === null && (
              <span className="text-[10px] text-slate-400">
                Variant position: aa {variantAA}
              </span>
            )}
          </div>
          <a
            href={`https://alphafold.ebi.ac.uk/entry/${uniprotId}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[10px] text-indigo-500 underline hover:text-indigo-700"
          >
            AlphaFold:{uniprotId} ↗
          </a>
        </div>
      </div>
    </div>
  );
}
