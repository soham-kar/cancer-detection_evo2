"use client";

import { UserButton, useAuth } from "@clerk/nextjs";
import { Dna, Search, Zap, Database, Cpu, Building2, ChevronDown, ChevronUp, Activity } from "lucide-react";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import GeneViewer from "~/components/gene-viewer";
import { CreditsDisplay } from "~/components/credits-display";
import { AnalysisHistory } from "~/components/analysis-history";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import {
  type ChromosomeFromSeach,
  type GeneFromSearch,
  type GenomeAssemblyFromSearch,
  getAvailableGenomes,
  getGenomeChromosomes,
  searchGenes,
} from "~/utils/genome-api";

export default function HomePage() {
  const { isSignedIn, isLoaded } = useAuth();
  const router = useRouter();
  const [genomes, setGenomes] = useState<GenomeAssemblyFromSearch[]>([]);
  const [selectedGenome, setSelectedGenome] = useState<string>("hg38");
  const [chromosomes, setChromosomes] = useState<ChromosomeFromSeach[]>([]);
  const [selectedGene, setSelectedGene] = useState<GeneFromSearch | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<GeneFromSearch[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isHydrated, setIsHydrated] = useState(false);
  const [showBrowseMore, setShowBrowseMore] = useState(false);
  const [demoLoading, setDemoLoading] = useState(false);

  // Run demo analysis (for guests)
  const runDemoAnalysis = async (variantId: string = "brca1") => {
    setDemoLoading(true);
    router.push(`/demo?variant=${variantId}`);
  };

  // Restore state from sessionStorage on mount
  useEffect(() => {
    setIsHydrated(true);
    try {
      const savedGene = sessionStorage.getItem('selectedGene');
      const savedGenome = sessionStorage.getItem('selectedGenome');

      if (savedGene) {
        setSelectedGene(JSON.parse(savedGene));
      }
      if (savedGenome) {
        setSelectedGenome(savedGenome);
      }
    } catch (e) {
      // Ignore sessionStorage errors
    }
  }, []);

  // Save gene to sessionStorage when it changes
  useEffect(() => {
    if (!isHydrated) return;
    try {
      if (selectedGene) {
        sessionStorage.setItem('selectedGene', JSON.stringify(selectedGene));
      } else {
        sessionStorage.removeItem('selectedGene');
      }
    } catch (e) {
      // Ignore sessionStorage errors
    }
  }, [selectedGene, isHydrated]);

  // Save genome to sessionStorage when it changes
  useEffect(() => {
    if (!isHydrated) return;
    try {
      sessionStorage.setItem('selectedGenome', selectedGenome);
    } catch (e) {
      // Ignore sessionStorage errors
    }
  }, [selectedGenome, isHydrated]);

  useEffect(() => {
    const fetchGenomes = async () => {
      try {
        setIsLoading(true);
        const data = await getAvailableGenomes();
        if (data.genomes && data.genomes["Human"]) {
          setGenomes(data.genomes["Human"]);
        }
      } catch (err) {
        console.error("Error fetching genomes:", err);
        setError("Failed to load genome assemblies");
      } finally {
        setIsLoading(false);
      }
    };

    fetchGenomes();
  }, []);

  useEffect(() => {
    const fetchChromosomes = async () => {
      if (!selectedGenome) return;

      try {
        const data = await getGenomeChromosomes(selectedGenome);
        if (data.chromosomes) {
          setChromosomes(data.chromosomes);
        }
      } catch (err) {
        console.error("Error fetching chromosomes:", err);
      }
    };

    fetchChromosomes();
  }, [selectedGenome]);

  const performGeneSearch = async (query: string, genomeId: string) => {
    try {
      setIsLoading(true);
      setError(null);
      console.log(`Searching for "${query}" in genome "${genomeId}"...`);
      const data = await searchGenes(query, genomeId);
      console.log("Search response:", data);
      if (data.results && data.results.length > 0) {
        setSearchResults(data.results);
      } else {
        setSearchResults([]);
      }
    } catch (err) {
      console.error("Error searching genes:", err);
      setError("Failed to search genes. Please try again.");
      setSearchResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenomeChange = (value: string) => {
    setSelectedGenome(value);
    if (searchQuery.trim()) {
      performGeneSearch(searchQuery, value);
    }
  };

  const handleSearch = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) return;

    // Directly open gene viewer (skip results table)
    directGeneNavigation(searchQuery);
  };

  // For search bar - shows results table
  const loadGeneExample = (gene: string) => {
    setSearchQuery(gene);
    performGeneSearch(gene, selectedGenome);
  };

  // For Popular chips - directly open gene viewer (skip table)
  const directGeneNavigation = async (geneSymbol: string) => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await searchGenes(geneSymbol, selectedGenome);

      if (data.results && data.results.length > 0) {
        const query = geneSymbol.toUpperCase();

        // STEP 1: Filter out noise (LOC entries)
        const cleanResults = data.results.filter(gene => {
          // Hide weird LOC entries
          if (gene.symbol.startsWith('LOC')) return false;
          return true;
        });

        // STEP 2: Smart sorting (exact match first, then starts-with, then alphabetical)
        const sortedResults = cleanResults.sort((a, b) => {
          const symbolA = a.symbol.toUpperCase();
          const symbolB = b.symbol.toUpperCase();

          // RULE 1: Exact Match ALWAYS goes first
          if (symbolA === query && symbolB !== query) return -1;
          if (symbolB === query && symbolA !== query) return 1;

          // RULE 2: "Starts With" goes second (e.g., BRCA1P1 after BRCA1)
          const startsA = symbolA.startsWith(query);
          const startsB = symbolB.startsWith(query);
          if (startsA && !startsB) return -1;
          if (!startsA && startsB) return 1;

          // RULE 3: Alphabetical fallback
          return symbolA.localeCompare(symbolB);
        });

        // Use the best match (first after sorting)
        if (sortedResults.length > 0) {
          setSelectedGene(sortedResults[0]!);
        } else if (data.results.length > 0) {
          // Fallback to unfiltered first result if all were filtered
          setSelectedGene(data.results[0]!);
        } else {
          setError(`No genes found for "${geneSymbol}"`);
        }
      } else {
        setError(`No genes found for "${geneSymbol}". Try a different gene symbol.`);
      }
    } catch (err) {
      console.error("Error navigating to gene:", err);
      setError("Failed to load gene. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Quick Start gene examples
  const quickStartGenes = [
    { symbol: "BRCA1", description: "Breast cancer susceptibility" },
    { symbol: "TP53", description: "Tumor suppressor" },
    { symbol: "MSH2", description: "DNA mismatch repair" },
  ];

  return (
    <div className="flex min-h-screen flex-col bg-[#f4f7f5] animate-in fade-in duration-500">
      {/* Header */}
      <header className="border-b border-[#3c4f3d]/10 bg-white">
        <div className="container mx-auto flex items-center justify-between px-6 py-3">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <Dna className="h-7 w-7 text-[#de8246]" />
            <h1 className="text-xl font-semibold text-[#3c4f3d]">
              HelixMind
            </h1>
          </div>

          {/* Right side - Conditional based on auth */}
          <div className="flex items-center gap-4">
            {isLoaded && isSignedIn ? (
              <>
                {/* Signed-in user: Show genome selector + credits + user button */}
                <div className="flex items-center gap-2 bg-white px-3 py-1.5 rounded-lg border border-slate-200 shadow-sm">
                  <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                    Assembly
                  </span>
                  <Select
                    value={selectedGenome}
                    onValueChange={handleGenomeChange}
                    disabled={isLoading}
                  >
                    <SelectTrigger className="h-auto w-auto p-0 border-0 bg-transparent focus:ring-0 shadow-none gap-1.5">
                      <span className="text-xs font-mono font-bold text-slate-700">
                        {selectedGenome === "hg38" ? "GRCh38" : "GRCh37"}
                      </span>
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="hg38" className="font-mono text-xs">GRCh38</SelectItem>
                      <SelectItem value="hg19" className="font-mono text-xs">GRCh37 (hg19)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <CreditsDisplay />
                <UserButton
                  afterSignOutUrl="/sign-in"
                  appearance={{
                    elements: {
                      avatarBox: "h-8 w-8",
                    },
                  }}
                />
              </>
            ) : (
              <>
                {/* Guest: Show demo button + sign in */}
                <Button
                  onClick={() => runDemoAnalysis("brca1")}
                  disabled={demoLoading}
                  className="bg-[#de8246] hover:bg-[#c97339] text-white font-medium px-4"
                >
                  {demoLoading ? "Loading..." : "⚡ Try Demo"}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => router.push("/sign-in")}
                  className="border-slate-300"
                >
                  Sign In
                </Button>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1">
        {selectedGene ? (
          <div className="container mx-auto px-6 py-6">
            <GeneViewer
              gene={selectedGene}
              genomeId={selectedGenome}
              onClose={() => setSelectedGene(null)}
            />
          </div>
        ) : (
          <>
            {/* Hero Section - Minimal padding for small screens */}
            <div className="relative flex flex-col items-center justify-center px-4 py-2 sm:py-4 md:py-8 lg:py-10 overflow-hidden">
              {/* Decorative background elements for larger screens */}
              <div className="absolute inset-0 overflow-hidden pointer-events-none">
                {/* Gradient orbs */}
                <div className="absolute top-10 left-[10%] w-64 h-64 bg-gradient-to-br from-[#de8246]/10 to-transparent rounded-full blur-3xl hidden md:block" />
                <div className="absolute bottom-10 right-[10%] w-80 h-80 bg-gradient-to-tr from-[#3c4f3d]/10 to-transparent rounded-full blur-3xl hidden md:block" />
                <div className="absolute top-1/2 left-[5%] w-48 h-48 bg-gradient-to-r from-teal-500/5 to-transparent rounded-full blur-2xl hidden lg:block" />
                <div className="absolute top-20 right-[15%] w-32 h-32 bg-gradient-to-bl from-amber-400/10 to-transparent rounded-full blur-2xl hidden lg:block" />
                <div className="absolute bottom-1/4 left-[20%] w-40 h-40 bg-gradient-to-t from-emerald-500/5 to-transparent rounded-full blur-2xl hidden lg:block" />
                <div className="absolute top-1/3 right-[25%] w-56 h-56 bg-gradient-to-bl from-[#de8246]/5 to-transparent rounded-full blur-3xl hidden lg:block" />

                {/* Subtle grid pattern */}
                <div className="absolute inset-0 opacity-[0.02] hidden md:block" style={{
                  backgroundImage: 'radial-gradient(circle, #3c4f3d 1px, transparent 1px)',
                  backgroundSize: '50px 50px'
                }} />
              </div>
              {/* Hero Title - Responsive text sizing */}
              <h2 className={`mb-1 text-center text-xl font-semibold tracking-tight text-[#3c4f3d] sm:text-2xl md:text-3xl lg:text-4xl sm:mb-2 transition-all duration-700 ease-out ${isHydrated ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'}`}>
                Precision Genomic Variant Analysis
              </h2>
              <p className={`mb-4 max-w-2xl text-center text-xs text-[#3c4f3d]/70 leading-relaxed sm:text-sm md:text-base sm:mb-6 md:mb-8 transition-all duration-700 delay-100 ease-out ${isHydrated ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'}`}>
                Unifying <span className="font-semibold text-[#3c4f3d]">Evo-2</span> sequence modeling
                and <span className="font-semibold text-[#3c4f3d]">Llama-3</span> clinical reasoning
                for real-time variant interpretation.
              </p>

              {/* Centered Search Bar */}
              <form
                onSubmit={handleSearch}
                className={`w-full max-w-xl transition-all duration-700 delay-200 ease-out ${isHydrated ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'}`}
              >
                <div className="relative">
                  <Input
                    type="text"
                    placeholder="Search gene (e.g. BRCA1, TP53)"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="h-10 sm:h-12 md:h-14 rounded-full border-[#3c4f3d]/20 bg-white pl-4 pr-10 sm:pl-5 sm:pr-12 text-xs sm:text-sm md:text-base font-mono shadow-lg focus:border-[#3c4f3d]/40 focus:ring-2 focus:ring-[#3c4f3d]/10"
                  />
                  <Button
                    type="submit"
                    className="absolute top-1/2 right-2 h-10 w-10 -translate-y-1/2 cursor-pointer rounded-full bg-[#de8246] text-white hover:bg-[#de8246]/90"
                    size="icon"
                    disabled={isLoading || !searchQuery.trim()}
                  >
                    <Search className="h-5 w-5" />
                    <span className="sr-only">Search</span>
                  </Button>
                </div>
              </form>

              {/* Quick Start Chips */}
              <div className="mt-2 sm:mt-4 flex flex-wrap items-center justify-center gap-1.5 sm:gap-2">
                <span className="text-xs sm:text-sm text-[#3c4f3d]/60">Popular:</span>
                {quickStartGenes.map((gene) => (
                  <Button
                    key={gene.symbol}
                    variant="outline"
                    size="sm"
                    onClick={() => directGeneNavigation(gene.symbol)}
                    className="cursor-pointer rounded-full border-[#3c4f3d]/20 bg-white px-4 font-mono text-[#3c4f3d] hover:border-[#de8246]/50 hover:bg-[#de8246]/5"
                  >
                    <Dna className="mr-1.5 h-3.5 w-3.5 text-[#de8246]" />
                    {gene.symbol}
                  </Button>
                ))}
              </div>

              {/* Browse More - Collapsible */}
              <div className="mt-2 sm:mt-4 text-center">
                <button
                  onClick={() => {
                    const newState = !showBrowseMore;
                    setShowBrowseMore(newState);
                    if (!newState) {
                      // Hiding browser - also clear results
                      setSearchResults([]);
                      setSearchQuery("");
                    }
                  }}
                  className="inline-flex items-center gap-1 text-sm text-[#3c4f3d]/60 hover:text-[#3c4f3d] transition-colors"
                >
                  {showBrowseMore ? (
                    <>
                      <ChevronUp className="h-4 w-4" />
                      Hide chromosome browser
                    </>
                  ) : (
                    <>
                      <ChevronDown className="h-4 w-4" />
                      Browse by chromosome
                    </>
                  )}
                </button>

                {/* Animated container for chromosome browser */}
                <div
                  className={`grid transition-all duration-300 ease-out ${showBrowseMore ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}
                >
                  <div className="overflow-hidden">
                    <div className="mt-4 mx-auto max-w-2xl rounded-xl border border-[#3c4f3d]/10 bg-white p-4 shadow-sm">
                      <p className="mb-3 text-xs text-[#3c4f3d]/60">Select a chromosome to explore genes:</p>
                      <div className="flex flex-wrap justify-center gap-2">
                        {chromosomes.slice(0, 25).map((chrom) => (
                          <Button
                            key={chrom.name}
                            variant="outline"
                            size="sm"
                            className="h-8 cursor-pointer border-[#3c4f3d]/10 font-mono text-xs hover:bg-[#e9eeea] hover:text-[#3c4f3d]"
                            onClick={() => {
                              // Directly navigate to first gene on this chromosome
                              directGeneNavigation(chrom.name.replace('chr', 'chromosome '));
                            }}
                          >
                            {chrom.name.replace('chr', '')}
                          </Button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Loading and Error States */}
            <div className="container mx-auto px-6 pb-8">
              {isLoading && (
                <div className="flex flex-col items-center justify-center py-12">
                  <div className="h-10 w-10 animate-spin rounded-full border-2 border-[#3c4f3d]/20 border-t-[#de8246]"></div>
                  <p className="mt-4 text-sm text-[#3c4f3d]/60">Loading gene analysis...</p>
                </div>
              )}

              {error && (
                <div className="mx-auto max-w-xl rounded-lg border border-red-200 bg-red-50 p-4 text-center text-sm text-red-700">
                  {error}
                </div>
              )}
            </div>
          </>
        )
        }
      </main >

      {/* Feature Grid - Only show on landing page (no gene selected) */}
      {
        !selectedGene && (
          <section className="py-2 sm:py-4 md:py-8 lg:py-10 bg-gradient-to-b from-white to-slate-50/50">
            <div className="container mx-auto px-3 sm:px-4 md:px-6">
              <div className={`grid grid-cols-3 gap-2 sm:gap-3 md:gap-5 max-w-4xl mx-auto transition-all duration-700 delay-500 ease-out ${isHydrated ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>

                {/* Column 1: The AI Model */}
                <div className="flex flex-col items-center text-center p-3 sm:p-4 md:p-5 bg-white border border-slate-100 rounded-xl shadow-sm hover:shadow-lg hover:-translate-y-1 hover:border-orange-200 transition-all duration-300 cursor-default">
                  <div className="w-10 h-10 sm:w-12 sm:h-12 bg-orange-50 text-orange-600 rounded-full flex items-center justify-center mb-2 sm:mb-3">
                    <Zap className="w-5 h-5 sm:w-6 sm:h-6" />
                  </div>
                  <h3 className="text-slate-900 font-semibold text-sm sm:text-base mb-1 sm:mb-2">Evo-2 Powered</h3>
                  <p className="text-xs sm:text-sm text-slate-500 leading-relaxed">
                    Zero-shot pathogenicity prediction using the 7B parameter evolutionary model via Modal H100s.
                  </p>
                </div>

                {/* Column 2: The Data */}
                <div className="flex flex-col items-center text-center p-3 sm:p-4 md:p-5 bg-white border border-slate-100 rounded-xl shadow-sm hover:shadow-lg hover:-translate-y-1 hover:border-blue-200 transition-all duration-300 cursor-default">
                  <div className="w-10 h-10 sm:w-12 sm:h-12 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center mb-2 sm:mb-3">
                    <Database className="w-5 h-5 sm:w-6 sm:h-6" />
                  </div>
                  <h3 className="text-slate-900 font-semibold text-sm sm:text-base mb-1 sm:mb-2">Clinical Context</h3>
                  <p className="text-xs sm:text-sm text-slate-500 leading-relaxed">
                    Real-time cross-referencing with ClinVar, PubMed, and 800k+ individuals from gnomAD v4.
                  </p>
                </div>

                {/* Column 3: The Speed */}
                <div className="flex flex-col items-center text-center p-3 sm:p-4 md:p-5 bg-white border border-slate-100 rounded-xl shadow-sm hover:shadow-lg hover:-translate-y-1 hover:border-emerald-200 transition-all duration-300 cursor-default">
                  <div className="w-10 h-10 sm:w-12 sm:h-12 bg-emerald-50 text-emerald-600 rounded-full flex items-center justify-center mb-2 sm:mb-3">
                    <Activity className="w-5 h-5 sm:w-6 sm:h-6" />
                  </div>
                  <h3 className="text-slate-900 font-semibold text-sm sm:text-base mb-1 sm:mb-2">Instant Analysis</h3>
                  <p className="text-xs sm:text-sm text-slate-500 leading-relaxed">
                    Analyze coding and non-coding variants in milliseconds with our optimized inference pipeline.
                  </p>
                </div>

              </div>
            </div>
          </section>
        )
      }

      {/* Tech Stack Footer */}
      <footer className="border-t border-[#3c4f3d]/10 bg-white py-3 sm:py-4">
        <div className="container mx-auto flex flex-wrap items-center justify-center gap-6 px-6 text-xs text-[#3c4f3d]/50">
          <div className="flex items-center gap-1.5">
            <Zap className="h-3.5 w-3.5" />
            <span>Modal H100</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Cpu className="h-3.5 w-3.5" />
            <span>Llama-3 via Groq</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Database className="h-3.5 w-3.5" />
            <span>gnomAD v4.1</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Building2 className="h-3.5 w-3.5" />
            <span>ClinVar</span>
          </div>
        </div>
      </footer>
    </div >
  );
}
