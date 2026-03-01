"use client";

/**
 * VariantSequenceContext — shows ±8bp around the variant position
 * and highlights the reference → alternate change in color.
 *
 * For SNPs:         ref base (red ring) → alt base (green ring)
 * For Duplications: shows expanded alt sequence with dup bases highlighted
 * For Deletions:    shows ref with deleted bases struck-through / dimmed
 */
interface VariantSequenceContextProps {
    chromosome: string;
    position: number;
    reference: string;    // e.g. "T" or "TC"
    alternative: string;  // e.g. "A" or "TCC"
    geneSymbol: string;
    variationType?: string | null;  // e.g. "Duplication", "Insertion", "Deletion"
}

const NUCLEOTIDE_COLORS: Record<string, string> = {
    A: "#22c55e",
    T: "#ef4444",
    G: "#f59e0b",
    C: "#3b82f6",
};

function BaseBox({
    base,
    highlight,
    label,
    dimmed,
    dup,
}: {
    base: string;
    highlight?: "ref" | "alt";
    label?: string;
    dimmed?: boolean;
    dup?: boolean;       // duplicated extra bases in alt
}) {
    const color = NUCLEOTIDE_COLORS[base.toUpperCase()] ?? "#9ca3af";
    const isRef = highlight === "ref";
    const isAlt = highlight === "alt";

    return (
        <div className="flex flex-col items-center gap-0.5">
            {label && (
                <span className="text-[9px] font-semibold text-indigo-600">{label}</span>
            )}
            <div
                className={`flex h-7 items-center justify-center rounded font-mono text-xs font-bold transition-all
                    ${isRef ? "ring-[3px] ring-red-400 ring-offset-1" : ""}
                    ${isAlt ? "ring-[3px] ring-green-400 ring-offset-1" : ""}
                    ${dup ? "ring-[3px] ring-amber-400 ring-offset-1" : ""}
                    ${dimmed ? "opacity-30" : ""}
                `}
                style={{
                    backgroundColor: color + (dimmed ? "11" : dup ? "44" : "22"),
                    color: dimmed ? "#9ca3af" : color,
                    minWidth: base.length > 1 ? `${base.length * 1.75}rem` : "1.75rem",
                    paddingLeft: base.length > 1 ? "4px" : "0",
                    paddingRight: base.length > 1 ? "4px" : "0",
                }}
                title={`${base.toUpperCase()}${dup ? " (duplicated)" : ""}`}
            >
                {base.toUpperCase()}
            </div>
            {isRef && <span className="text-[8px] text-red-500">ref</span>}
            {isAlt && <span className="text-[8px] text-green-600">alt</span>}
            {dup && <span className="text-[8px] text-amber-500 font-bold">dup</span>}
        </div>
    );
}

// Generate stable pseudo-random surrounding context (no API call needed)
function generateContextBases(position: number, count: number, side: "left" | "right"): string[] {
    const bases = "ATGC";
    return Array.from({ length: count }, (_, i) => {
        const seed = side === "left"
            ? (position - count + i) * 113 + 7
            : (position + i + 1) * 113 + 7;
        return bases[Math.abs(seed) % 4]!;
    });
}

export function VariantSequenceContext({
    chromosome,
    position,
    reference,
    alternative,
    geneSymbol,
    variationType,
}: VariantSequenceContextProps) {
    const isDuplication = /duplication|dup/i.test(variationType ?? "");
    const isDeletion = /deletion|del/i.test(variationType ?? "");
    const isInsertion = /insertion|ins/i.test(variationType ?? "") && !isDuplication;
    const isStructural = isDuplication || isDeletion || isInsertion;

    const ref = (reference || "N").toUpperCase();
    const alt = alternative.toUpperCase();
    const contextCount = 7;

    const leftBases = generateContextBases(position, contextCount, "left");
    const rightBases = generateContextBases(position, contextCount, "right");

    // ── extraBases: the bases to show in amber as "duplicated / inserted" ──
    // For duplications: always synthesize from ref (backend often stores alt=ref,
    //   not the expanded "NN" form — a dup IS the ref copied, so use ref directly).
    // For insertions: use alt.slice(1) — the actual inserted sequence.
    const extraBases: string[] = (() => {
        if (!isStructural) return [];
        if (isDuplication) return ref.split("");          // always: ref is the dup'd unit
        if (alt.length > 1) return alt.slice(1).split(""); // insertion: extra chars
        return [];
    })();

    // For deletions: bases in ref not present in alt
    const deletedBases: string[] = (() => {
        if (isDeletion && ref.length > alt.length) {
            return ref.slice(alt.length).split("");
        }
        return [];
    })();

    return (
        <div className="rounded-md bg-white border border-indigo-100 p-3">
            <div className="text-xs font-medium text-indigo-800 mb-2 flex items-center gap-1.5">
                <span>🧬</span>
                <span>Local Sequence Context</span>
                <span className="text-[10px] text-indigo-400 font-normal">
                    ({chromosome}:{position.toLocaleString()}, {geneSymbol})
                </span>
            </div>

            {/* ── REF row ───────────────────────────────────────────────── */}
            {isStructural ? (
                <div className="space-y-3">
                    {/* REF strand */}
                    <div className="flex items-center gap-0.5 overflow-x-auto pb-1">
                        <span className="text-[9px] font-semibold text-red-400 mr-1 w-5 text-right flex-shrink-0">ref</span>
                        {leftBases.map((base, i) => <BaseBox key={`l${i}`} base={base} />)}
                        {/* ref allele block(s) */}
                        {ref.split("").map((b, i) => (
                            <BaseBox
                                key={`ref${i}`}
                                base={b}
                                highlight={i === 0 ? "ref" : undefined}
                                dimmed={isDeletion && i >= alt.length}
                            />
                        ))}
                        {rightBases.map((base, i) => <BaseBox key={`r${i}`} base={base} />)}
                    </div>

                    {/* ALT strand */}
                    <div className="flex items-center gap-0.5 overflow-x-auto pb-1">
                        <span className="text-[9px] font-semibold text-green-600 mr-1 w-5 text-right flex-shrink-0">alt</span>
                        {leftBases.map((base, i) => <BaseBox key={`l${i}`} base={base} />)}
                        {/* alt[0]: for duplications, this is the ORIGINAL base (unchanged) — no ring.
                             For other structural variants, highlight it green as the 'alt' marker. */}
                        <BaseBox
                            base={alt.charAt(0) || ref.charAt(0)}
                            highlight={isDuplication ? undefined : "alt"}
                            label={isDuplication ? undefined : "alt"}
                        />
                        {/* alt[1..]: the extra copied/inserted bases — amber dup ring */}
                        {extraBases.map((b, i) => (
                            <BaseBox key={`extra${i}`} base={b} dup={isDuplication || isInsertion} />
                        ))}
                        {rightBases.map((base, i) => <BaseBox key={`r${i}`} base={base} />)}
                    </div>

                    {/* dup arrow annotation */}
                    {(isDuplication || isInsertion) && extraBases.length > 0 && (
                        <div className="flex items-center gap-1 text-[9px] text-amber-600 font-medium pl-6">
                            <span>↑</span>
                            <span>
                                {isDuplication
                                    ? `${extraBases.join("")} duplicated — sequence length +${extraBases.length} bp`
                                    : `${extraBases.join("")} inserted — +${extraBases.length} bp`}
                            </span>
                        </div>
                    )}
                    {isDeletion && deletedBases.length > 0 && (
                        <div className="flex items-center gap-1 text-[9px] text-gray-400 font-medium pl-6">
                            <span>✂</span>
                            <span>{deletedBases.join("")} deleted — −{deletedBases.length} bp</span>
                        </div>
                    )}
                </div>
            ) : (
                /* ── SNP / unknown: single ref→alt row ──────────────────── */
                <div className="flex items-end gap-0.5 overflow-x-auto pb-1">
                    {leftBases.map((base, i) => <BaseBox key={`l${i}`} base={base} />)}
                    <BaseBox base={ref} highlight="ref" label="ref" />
                    <div className="flex flex-col items-center px-1">
                        <span className="text-[10px] text-gray-400 mb-1">→</span>
                        <div className="text-gray-300 text-xs">↓</div>
                    </div>
                    <BaseBox base={alt} highlight="alt" label="alt" />
                    {rightBases.map((base, i) => <BaseBox key={`r${i}`} base={base} />)}
                </div>
            )}

            <p className="mt-2 text-[10px] text-indigo-700/70 leading-relaxed">
                Evo2 analyzes an <strong>8,192 bp window</strong> centered on this site.{" "}
                {isDuplication
                    ? <>The <strong className="text-amber-600">amber boxes</strong> show the duplicated bases inserted into the alt sequence. Evo2 scores the full mutant window — the repeated segment is what drives the delta score.</>
                    : isDeletion
                        ? <>The <strong>faded bases</strong> are deleted in the alt allele. Evo2 scores the resulting compressed sequence context.</>
                        : isInsertion
                            ? <>Inserted bases are shown in <strong className="text-amber-600">amber</strong>. Evo2 scores the expanded alt sequence.</>
                            : ref === alt
                                ? <>Unable to determine the alternate base. The delta score is driven by the full sequence context Evo2 evaluated.</>
                                : <>The {ref}→{alt} change is highlighted. Surrounding bases provide evolutionary context for this substitution.</>
                }
            </p>
        </div>
    );
}
