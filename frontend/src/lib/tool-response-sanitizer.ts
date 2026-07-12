// =============================================================================
// Tool Response Sanitizer — Prevents Context Poisoning
// =============================================================================
// Raw tool results (PDB files, BLAST XML, per-residue arrays) are too large
// and unstructured for the LLM to reason over. This module intercepts tool
// results BEFORE they are pushed into nvidiaMessages and extracts only the
// semantically meaningful summary that the LLM needs.
//
// Without this, a fetch_alphafold_db result (10K+ lines of ATOM records) gets
// truncated at 50K chars and fed to Nemotron, which cannot parse 3D coordinates
// from text — it hallucinates or says "data unavailable."
// =============================================================================

/**
 * Sanitize a raw tool response into a compact, LLM-readable string.
 *
 * @param toolName - The Nemotron function name (e.g. "fetch_alphafold_db")
 * @param rawResult - The raw tool result from Modal (or error object)
 * @returns A JSON string safe to push into nvidiaMessages as role: "tool"
 */
export function sanitizeToolResponse(
  toolName: string,
  rawResult: Record<string, unknown> | undefined,
): string {
  // If the tool failed, just pass the error — Nemotron needs to see it for self-healing
  if (rawResult?.error) {
    return JSON.stringify(rawResult);
  }

  if (!rawResult) {
    return JSON.stringify({ status: "failed", error: "No result returned from tool" });
  }

  switch (toolName) {
    // ── AlphaFold DB: Strip raw PDB, keep only metadata + pLDDT summary ──
    case "fetch_alphafold_db": {
      const raw = rawResult as Record<string, unknown>;
      // Extract pLDDT statistics without the full per-residue array
      const plddtArray = raw.plddt_scores as number[] | undefined;
      const plddtSummary = plddtArray
        ? {
            mean: Number((plddtArray.reduce((a, b) => a + b, 0) / plddtArray.length).toFixed(1)),
            min: Math.min(...plddtArray),
            max: Math.max(...plddtArray),
            residues_above_70: plddtArray.filter((p) => p > 70).length,
            residues_below_50: plddtArray.filter((p) => p < 50).length,
            total: plddtArray.length,
          }
        : undefined;

      return JSON.stringify({
        status: "success",
        source: "AlphaFold Protein Structure Database",
        uniprot_id: raw.uniprot_id,
        global_plddt: raw.global_plddt ?? plddtSummary?.mean,
        plddt_summary: plddtSummary,
        pdb_available: !!raw.pdb_string || !!raw.pdb_url,
        note: "Raw PDB coordinates are available in the system but excluded from this context to save tokens. The structure can be visualized in the 3D viewer.",
      });
    }

    // ── BLAST: Keep only top 5 hits, strip raw alignments ──
    case "run_blast_search": {
      const raw = rawResult as Record<string, unknown>;
      const hits = (raw.hits || raw.results || []) as Array<Record<string, unknown>>;
      const topHits = hits.slice(0, 5).map((h) => ({
        accession: h.accession || h.id,
        description: String(h.description || h.title || "").substring(0, 100),
        e_value: h.e_value ?? h.evalue,
        percent_identity: h.percent_identity ?? h.identity,
        bit_score: h.bit_score ?? h.score,
        coverage: h.coverage ?? h.query_coverage,
      }));

      return JSON.stringify({
        status: "success",
        source: "NCBI BLAST",
        total_hits: hits.length,
        database: raw.database,
        program: raw.program,
        top_hits: topHits,
        note: hits.length > 5 ? `Showing top 5 of ${hits.length} hits. Full alignments excluded to save tokens.` : undefined,
      });
    }

    // ── DSSP: Keep aggregate stats, strip per-residue arrays ──
    case "run_dssp_secondary_structure": {
      const raw = rawResult as Record<string, unknown>;
      return JSON.stringify({
        status: "success",
        source: "DSSP",
        total_residues: raw.total_residues ?? raw.length,
        helix_percentage: raw.helix_percentage ?? raw.helix_percent,
        sheet_percentage: raw.sheet_percentage ?? raw.sheet_percent,
        loop_percentage: raw.loop_percentage ?? raw.loop_percent,
        coil_percentage: raw.coil_percentage ?? raw.coil_percent,
        longest_helix: raw.longest_helix,
        longest_sheet: raw.longest_sheet,
        note: "Per-residue secondary structure assignments excluded to save tokens.",
      });
    }

    // ── Structure Metrics: Already compact, pass through ──
    case "run_structure_metrics": {
      const raw = rawResult as Record<string, unknown>;
      return JSON.stringify({
        status: "success",
        source: "Structure Metrics",
        radius_of_gyration: raw.radius_of_gyration ?? raw.gyration_radius,
        longest_helix: raw.longest_helix,
        longest_sheet: raw.longest_sheet,
        sasa_total: raw.sasa_total ?? raw.total_sasa,
        sasa_mean: raw.sasa_mean ?? raw.mean_sasa,
      });
    }

    // ── UniProt: Strip bloated cross-references and features arrays ──
    case "fetch_uniprot": {
      const raw = rawResult as Record<string, unknown>;
      // Keep only the essential fields, strip xrefs, features (can be 1000+ entries)
      const { xrefs, features, comments, ...cleanUniProt } = raw;
      // If there are domains in features, extract just the domain names and positions
      const domains = (features as Array<Record<string, unknown>> | undefined)
        ?.filter((f) => f.type === "Domain" || f.type === "Region")
        .map((f) => ({
          name: f.description || f.name,
          start: f.start,
          end: f.end,
        }))
        .slice(0, 20); // Cap at 20 domains

      return JSON.stringify({
        ...cleanUniProt,
        domains: domains || cleanUniProt.domains,
        note: "Cross-references and full feature lists excluded to save tokens.",
      });
    }

    // ── ESMFold: Strip raw PDB, keep pLDDT summary ──
    case "run_esmfold_prediction": {
      const raw = rawResult as Record<string, unknown>;
      const results = (raw.results || raw.predictions || []) as Array<Record<string, unknown>>;
      const summary = results.map((r) => {
        const plddtArray = r.plddt as number[] | undefined;
        const plddtMean = plddtArray
          ? Number((plddtArray.reduce((a, b) => a + b, 0) / plddtArray.length).toFixed(1))
          : r.mean_plddt;
        return {
          sequence_length: r.sequence_length ?? r.length,
          mean_plddt: plddtMean,
          pdb_available: !!r.pdb_string || !!r.pdb,
          note: "Raw PDB coordinates excluded. Structure available for 3D viewer.",
        };
      });

      return JSON.stringify({
        status: "success",
        source: "ESMFold",
        predictions: summary,
      });
    }

    // ── ESM2 Score: Already compact (just scores), pass through with cap ──
    case "run_esm2_score": {
      const str = JSON.stringify(rawResult);
      return str.length > 8000 ? str.substring(0, 8000) + '...[TRUNCATED]' : str;
    }

    // ── ProteinMPNN: Keep scores, strip generated sequences if too many ──
    case "run_proteinmpnn_sample": {
      const raw = rawResult as Record<string, unknown>;
      const results = (raw.results || raw.samples || []) as Array<Record<string, unknown>>;
      const summary = results.slice(0, 4).map((r) => ({
        sequence: r.sequence ? String(r.sequence).substring(0, 200) + "..." : undefined,
        perplexity: r.perplexity,
        sequence_recovery: r.sequence_recovery,
        score: r.score,
      }));

      return JSON.stringify({
        status: "success",
        source: "ProteinMPNN",
        total_samples: results.length,
        top_samples: summary,
        note: results.length > 4 ? `Showing top 4 of ${results.length} samples.` : undefined,
      });
    }

    case "run_proteinmpnn_score": {
      const str = JSON.stringify(rawResult);
      return str.length > 8000 ? str.substring(0, 8000) + '...[TRUNCATED]' : str;
    }

    // ── Boltz2: Strip raw PDB, keep affinity + confidence ──
    case "run_boltz2_affinity": {
      const raw = rawResult as Record<string, unknown>;
      const results = (raw.results || raw.predictions || []) as Array<Record<string, unknown>>;
      const summary = results.map((r) => ({
        protein_length: r.protein_length ?? r.sequence_length,
        ligand_smiles: r.ligand_smiles ? String(r.ligand_smiles).substring(0, 80) : undefined,
        predicted_ic50: r.predicted_ic50 ?? r.ic50,
        log10_ic50: r.log10_ic50,
        binder_probability: r.binder_probability ?? r.probability,
        pdb_available: !!r.pdb_string || !!r.pdb,
      }));

      return JSON.stringify({
        status: "success",
        source: "Boltz2 Affinity",
        predictions: summary,
        note: "Raw PDB coordinates excluded. Structure available for 3D viewer.",
      });
    }

    case "run_boltz2_prediction": {
      const raw = rawResult as Record<string, unknown>;
      const results = (raw.results || raw.predictions || []) as Array<Record<string, unknown>>;
      const summary = results.map((r) => ({
        complex_type: r.complex_type,
        confidence: r.confidence ?? r.mean_plddt,
        pdb_available: !!r.pdb_string || !!r.pdb,
        note: "Raw PDB coordinates excluded. Structure available for 3D viewer.",
      }));

      return JSON.stringify({
        status: "success",
        source: "Boltz2 Structure",
        predictions: summary,
      });
    }

    // ── PyMOL RMSD: Already compact, pass through ──
    case "run_pymol_rmsd_alignment": {
      const str = JSON.stringify(rawResult);
      return str.length > 8000 ? str.substring(0, 8000) + '...[TRUNCATED]' : str;
    }

    // ── Foldseek: Keep top hits, strip raw alignments ──
    case "run_foldseek_search": {
      const raw = rawResult as Record<string, unknown>;
      const hits = (raw.hits || raw.results || []) as Array<Record<string, unknown>>;
      const topHits = hits.slice(0, 5).map((h) => ({
        target: h.target ?? h.accession,
        description: String(h.description || h.title || "").substring(0, 100),
        tm_score: h.tm_score ?? h.tmscore,
        e_value: h.e_value ?? h.evalue,
        rmsd: h.rmsd,
        coverage: h.coverage,
      }));

      return JSON.stringify({
        status: "success",
        source: "Foldseek",
        total_hits: hits.length,
        database: raw.database,
        top_hits: topHits,
        note: hits.length > 5 ? `Showing top 5 of ${hits.length} structural hits.` : undefined,
      });
    }

    // ── MMseqs2: Keep top hits, strip raw alignments ──
    case "run_mmseqs2_search_proteins": {
      const raw = rawResult as Record<string, unknown>;
      const results = (raw.results || raw.hits || []) as Array<Record<string, unknown>>;
      const topHits = results.slice(0, 5).map((h) => ({
        target: h.target ?? h.accession,
        e_value: h.e_value ?? h.evalue,
        percent_identity: h.percent_identity ?? h.identity,
        coverage: h.coverage,
      }));

      return JSON.stringify({
        status: "success",
        source: "MMseqs2",
        total_hits: results.length,
        top_hits: topHits,
      });
    }

    // ── MAFFT: Keep alignment summary, strip full alignment ──
    case "run_mafft_align": {
      const raw = rawResult as Record<string, unknown>;
      const aligned = (raw.aligned_sequences || raw.alignment || []) as Array<Record<string, unknown>>;
      return JSON.stringify({
        status: "success",
        source: "MAFFT",
        num_sequences: aligned.length,
        alignment_length: raw.alignment_length ?? raw.length,
        method: raw.method,
        note: "Full alignment excluded to save tokens. Alignment is available for visualization.",
      });
    }

    // ── InterProScan: Keep domain hits, strip raw HMMER output ──
    case "run_interproscan_fetch": {
      const raw = rawResult as Record<string, unknown>;
      const matches = (raw.matches || raw.results || []) as Array<Record<string, unknown>>;
      const domains = matches.slice(0, 20).map((m) => ({
        database: m.database ?? m.source,
        signature: m.signature ?? m.accession,
        name: m.name ?? m.description,
        start: m.start ?? m.start_pos,
        end: m.end ?? m.end_pos,
        e_value: m.evalue ?? m.e_value,
      }));

      return JSON.stringify({
        status: "success",
        source: "InterProScan",
        total_matches: matches.length,
        domains: domains,
        note: matches.length > 20 ? `Showing top 20 of ${matches.length} domain matches.` : undefined,
      });
    }

    // ── SpliceAI: Keep delta scores, strip per-position arrays if too long ──
    case "run_spliceai_predict": {
      const str = JSON.stringify(rawResult);
      return str.length > 10000 ? str.substring(0, 10000) + '...[TRUNCATED]' : str;
    }

    // ── Pangolin: Same as SpliceAI ──
    case "run_pangolin_predict":
    case "run_pangolin_score_variants": {
      const str = JSON.stringify(rawResult);
      return str.length > 10000 ? str.substring(0, 10000) + '...[TRUNCATED]' : str;
    }

    // ── ViennaRNA: Keep MFE + structure, strip partition function if large ──
    case "run_viennarna_prediction": {
      const raw = rawResult as Record<string, unknown>;
      return JSON.stringify({
        status: "success",
        source: "ViennaRNA",
        mfe: raw.mfe ?? raw.minimum_free_energy,
        mfe_structure: raw.mfe_structure ?? raw.structure,
        sequence_length: raw.sequence_length ?? raw.length,
        note: "Per-base pairing probabilities excluded to save tokens.",
      });
    }

    // ── Segmasker: Already compact, pass through ──
    case "run_segmasker_score": {
      const str = JSON.stringify(rawResult);
      return str.length > 8000 ? str.substring(0, 8000) + '...[TRUNCATED]' : str;
    }

    // ── Default: Standard truncation for safe tools (VEP, ClinVar, PubChem, etc.) ──
    default: {
      const str = JSON.stringify(rawResult);
      return str.length > 8000 ? str.substring(0, 8000) + '...[TRUNCATED]' : str;
    }
  }
}