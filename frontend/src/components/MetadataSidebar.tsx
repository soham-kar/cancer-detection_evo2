"use client";

interface MetadataSidebarProps {
    gene: string;
    variant: string;
    chromosome: string;
    position: number;
    reference?: string;
    alternative?: string;
    genomeBuild?: string;
    transcript?: string;
    dbSnpId?: string;
}

export default function MetadataSidebar({
    gene,
    variant,
    chromosome,
    position,
    reference,
    alternative,
    genomeBuild = "GRCh38 / hg38",
    transcript,
    dbSnpId,
}: MetadataSidebarProps) {
    return (
        <div className="w-full space-y-4">
            {/* Variant Details Card */}
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">
                    Variant Details
                </h4>

                <div className="space-y-4">
                    <div>
                        <p className="text-xs text-slate-500 mb-0.5">Gene Symbol</p>
                        <p className="text-sm font-bold text-slate-900">{gene}</p>
                    </div>

                    <div>
                        <p className="text-xs text-slate-500 mb-0.5">HGVS Notation</p>
                        <p className="text-sm font-medium text-slate-900 font-mono">{variant}</p>
                    </div>

                    <div>
                        <p className="text-xs text-slate-500 mb-0.5">Genome Build</p>
                        <p className="text-sm font-medium text-slate-900 font-mono">{genomeBuild}</p>
                    </div>

                    <div>
                        <p className="text-xs text-slate-500 mb-0.5">Genomic Position</p>
                        <p className="text-sm font-medium text-slate-900 font-mono">{chromosome}:{position}</p>
                    </div>

                    {reference && alternative && (
                        <div>
                            <p className="text-xs text-slate-500 mb-0.5">Allele Change</p>
                            <p className="text-sm font-medium text-slate-900 font-mono">
                                <span className="text-emerald-600">{reference}</span>
                                <span className="text-slate-400 mx-1">→</span>
                                <span className="text-rose-600">{alternative}</span>
                            </p>
                        </div>
                    )}

                    {transcript && (
                        <div>
                            <p className="text-xs text-slate-500 mb-0.5">Transcript</p>
                            <p className="text-sm font-medium text-slate-900 font-mono">{transcript}</p>
                        </div>
                    )}

                    {dbSnpId && (
                        <div>
                            <p className="text-xs text-slate-500 mb-0.5">dbSNP ID</p>
                            <a
                                href={`https://www.ncbi.nlm.nih.gov/snp/${dbSnpId}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-sm font-medium text-indigo-600 hover:underline font-mono"
                            >
                                {dbSnpId}
                            </a>
                        </div>
                    )}
                </div>
            </div>

            {/* Analysis Info Card */}
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 p-4 rounded-xl text-white">
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
                    Analysis Pipeline
                </h4>
                <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                        <span className="text-slate-400">AI Model</span>
                        <span className="font-mono text-slate-200">Evo2-7B</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-slate-400">RAG Engine</span>
                        <span className="font-mono text-slate-200">Llama-3.3-70B</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-slate-400">Sources</span>
                        <span className="font-mono text-slate-200">Tri-Modal</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
