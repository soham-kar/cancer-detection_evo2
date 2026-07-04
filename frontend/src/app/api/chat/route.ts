// =============================================================================
// Chat API Route - Report-Aware AI Chatbot
// =============================================================================
// This endpoint powers a chatbot that has full context of the variant analysis
// report, including:
//   - VEP molecular consequence annotation
//   - Evo2-7B prediction, delta score, confidence
//   - gnomAD population frequency data
//   - ClinVar clinical classification
//   - UniProt protein structure & function
//   - PubMed literature evidence
//   - ACMG/AMP criteria (rule-based + LLM-refined)
//   - AlphaMissense, CADD, REVEL external scores
//   - Multi-tool concordance analysis
//   - ISM scan, XAI factors, counterfactuals
//   - Knowledge graph (gene-disease-drug)
//   - Clinical summary (multi-modal RAG)
//
// Powered by NVIDIA Nemotron-3 550B via NVIDIA API.
// =============================================================================

import { NextRequest, NextResponse } from "next/server";
import { currentUser } from "@clerk/nextjs/server";
import { db } from "~/lib/db";

// =============================================================================
// Configuration
// =============================================================================

const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;
const NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions";
const MODEL = "nvidia/nemotron-3-ultra-550b-a55b";

// =============================================================================
// Types
// =============================================================================

interface ChatRequestBody {
  reportId: string;
  message: string;
  sessionId?: string; // existing session to continue
}

// =============================================================================
// System Prompt Builder
// =============================================================================

function buildSystemPrompt(report: Record<string, unknown>): string {
  // Extract all report sections for context
  const vep = report.vepAnnotation as Record<string, unknown> | null;
  const popFreq = report.populationFrequency as Record<string, unknown> | null;
  const acmg = report.acmgEvidence as Record<string, unknown> | null;
  const lit = report.literatureContext as Record<string, unknown> | null;
  const evidence = report.evidenceConfidence as Record<string, unknown> | null;
  const acmgCriteria = report.acmgCriteria as Record<string, unknown> | null;
  const acmgRefined = report.acmgCriteriaRefined as Record<string, unknown> | null;
  const externalScores = report.externalScores as Record<string, unknown> | null;
  const xai = report.xaiFactors as Record<string, unknown> | null;
  const counterfactuals = report.counterfactuals as Record<string, unknown> | null;
  const kg = report.knowledgeGraph as Record<string, unknown> | null;
  const ism = report.ismScanData as Record<string, unknown> | null;

  return `You are HelixMind Chat, an expert AI assistant for clinical variant interpretation. You have access to the complete analysis report for a genetic variant and can answer questions about any aspect of it.

## YOUR CAPABILITIES
- Explain variant pathogenicity predictions in plain language
- Interpret ACMG/AMP criteria and classification rationale
- Discuss population frequency data (gnomAD) and its clinical significance
- Explain protein structural consequences using UniProt domain data
- Summarize literature evidence from PubMed
- Compare Evo2 AI predictions with ClinVar, AlphaMissense, CADD, and REVEL
- Explain VEP molecular consequences (missense, frameshift, splice, etc.)
- Discuss therapeutic strategies and clinical actionability
- Help researchers understand confidence metrics and evidence quality
- Explain ISM (In-Silico Mutagenesis) scan results and XAI factors

## THE VARIANT REPORT

### Basic Information
- Gene: ${report.geneSymbol}
- Chromosome: ${report.chromosome}
- Position: ${report.position}
- Reference Allele: ${report.reference}
- Alternative Allele: ${report.alternative}
- Genome Build: ${report.genomeId}

### Evo2-7B AI Prediction
- Prediction: ${report.prediction}
- Delta Likelihood Score: ${report.deltaScore}
- Classification Confidence: ${((report.classificationConfidence as number) * 100).toFixed(1)}%
- Classification Source: ${report.classificationSource || "Evo2_AI"}

### VEP Molecular Consequence
${vep ? `- Consequence: ${vep.consequence || "N/A"}
- Impact: ${vep.impact || "N/A"}
- Amino Acid Change: ${vep.aaChange || "N/A"} (${vep.aminoAcids || "N/A"})
- Codons: ${vep.codons || "N/A"}
- Transcript: ${vep.transcriptId || "N/A"}
- Is Synonymous: ${vep.isSynonymous}
- Is Frameshift: ${vep.isFrameshift}
- Is Nonsense: ${vep.isNonsense}` : "- VEP annotation not available"}

### Population Frequency (gnomAD)
${popFreq ? `- gnomAD Allele Frequency: ${popFreq.gnomad_af ?? "Not found"}
- Max Population AF: ${popFreq.gnomad_max_pop_af ?? "N/A"}
- Source: ${popFreq.source || "N/A"}
- Is Common Variant: ${popFreq.is_common_variant}` : "- gnomAD data not available"}

### ClinVar Classification
- ClinVar Status: ${report.clinvarClassification || "Not curated"}
- Variation Type: ${report.variationType || "N/A"}
- ClinVar ID: ${report.clinvarId || "N/A"}

### ACMG Evidence
${acmg ? `- Code: ${acmg.code || "None"}
- Strength: ${acmg.strength || "None"}
- Description: ${acmg.description}
- Clinical Note: ${acmg.clinical_note}` : "- ACMG evidence not available"}

### Literature Context (PubMed)
${lit ? `- Summary: ${lit.summary || "N/A"}
- Articles Found: ${lit.articles_found || 0}
- PubMed IDs: ${Array.isArray(lit.pubmed_ids) ? (lit.pubmed_ids as string[]).join(", ") : "N/A"}
- Gene Function: ${lit.gene_function || "N/A"}` : "- Literature data not available"}

### Evidence Confidence
${evidence ? `- VEP: ${(evidence.vep as Record<string,unknown>)?.confidence || "N/A"} (${(evidence.vep as Record<string,unknown>)?.available ? "Available" : "Missing"})
- Evo2: ${(evidence.evo2 as Record<string,unknown>)?.confidence || "N/A"} (${(evidence.evo2 as Record<string,unknown>)?.available ? "Available" : "Missing"})
- gnomAD: ${(evidence.gnomad as Record<string,unknown>)?.confidence || "N/A"} (${(evidence.gnomad as Record<string,unknown>)?.available ? "Available" : "Missing"})
- ClinVar: ${(evidence.clinvar as Record<string,unknown>)?.confidence || "N/A"} (${(evidence.clinvar as Record<string,unknown>)?.available ? "Available" : "Missing"})
- UniProt: ${(evidence.uniprot as Record<string,unknown>)?.confidence || "N/A"} (${(evidence.uniprot as Record<string,unknown>)?.available ? "Available" : "Missing"})
- PubMed: ${(evidence.pubmed as Record<string,unknown>)?.confidence || "N/A"} (${(evidence.pubmed as Record<string,unknown>)?.available ? "Available" : "Missing"})
- Overall: ${(evidence.overall as Record<string,unknown>)?.level || "N/A"} (${(evidence.overall as Record<string,unknown>)?.sources_available || "N/A"})` : "- Evidence confidence not available"}

### ACMG/AMP Criteria (Rule-Based)
${acmgCriteria ? formatACMGCriteria(acmgCriteria) : "- ACMG criteria not available"}

### LLM-Refined ACMG Classification
${acmgRefined ? `- ACMG Classification: ${acmgRefined.acmg_classification || "N/A"}
- Confidence: ${acmgRefined.classification_confidence || "N/A"}
- Narrative: ${acmgRefined.narrative || "N/A"}
${formatRefinedCriteria(acmgRefined)}` : "- Refined ACMG not available"}

### External Scores
${externalScores ? formatExternalScores(externalScores) : "- External scores not available"}

### Multi-Tool Concordance
${formatConcordance(report)}

### Clinical Summary (Multi-Modal RAG)
${report.clinicalSummary || "Not available"}

### XAI Confidence Factors
${xai ? formatXAI(xai) : "- XAI factors not available"}

### Counterfactual Analysis
${counterfactuals ? formatCounterfactuals(counterfactuals) : "- Counterfactuals not available"}

### Knowledge Graph
${kg ? formatKnowledgeGraph(kg) : "- Knowledge graph not available"}

### ISM Scan
${ism ? formatISM(ism) : "- ISM scan not available"}

## GUIDELINES
1. Answer questions using ONLY the data provided above. Do not fabricate information.
2. If asked about data not in the report, say "This information is not available in the current analysis report."
3. Use plain language when explaining complex concepts. Assume the user is a researcher or clinician.
4. Always include a disclaimer when discussing clinical implications: "This is a computational prediction for research purposes. Clinical decisions require professional genetic counseling."
5. Be concise but thorough. Use bullet points for clarity when listing multiple items.
6. When discussing pathogenicity, always reference the specific evidence sources (Evo2 score, ClinVar status, ACMG criteria, population data).
7. For therapeutic questions, note that recommendations are computational hypotheses requiring experimental validation.`;
}

// =============================================================================
// Formatting Helpers
// =============================================================================

function formatACMGCriteria(acmg: Record<string, unknown>): string {
  const criteria = acmg.criteria as Record<string, Record<string, unknown>> | undefined;
  if (!criteria) return "- ACMG criteria not available";
  
  const lines = Object.entries(criteria).map(([code, detail]) => {
    return `  - ${code}: ${detail.met ? "MET" : "Not Met"} (Strength: ${detail.strength || "N/A"}) — ${detail.rationale || ""}`;
  });
  
  return `- Met Count: ${acmg.met_count}/${acmg.total_evaluated}
- Classification: ${acmg.acmg_classification || "N/A"}
- Criteria:\n${lines.join("\n")}`;
}

function formatRefinedCriteria(acmg: Record<string, unknown>): string {
  const criteria = acmg.criteria as Record<string, Record<string, unknown>> | undefined;
  if (!criteria) return "";
  
  const lines = Object.entries(criteria).map(([code, detail]) => {
    return `  - ${code}: ${detail.met ? "MET" : "Not Met"} (Strength: ${detail.strength || "N/A"}) — ${detail.justification || ""}`;
  });
  
  return `- Refined Criteria:\n${lines.join("\n")}`;
}

function formatExternalScores(scores: Record<string, unknown>): string {
  const parts: string[] = [];
  
  const cadd = scores.cadd as Record<string, unknown> | null;
  if (cadd) {
    parts.push(`- CADD: PHRED ${cadd.phred} (Raw: ${cadd.raw}) — ${cadd.interpretation || "N/A"}`);
  }
  
  const revel = scores.revel as Record<string, unknown> | null;
  if (revel) {
    parts.push(`- REVEL: ${revel.score} — ${revel.interpretation || "N/A"}`);
  }
  
  const am = scores.alphamissense as Record<string, unknown> | null;
  if (am) {
    parts.push(`- AlphaMissense: ${am.score} (${am.classification || "N/A"}) — Confidence: ${am.confidence || "N/A"}`);
  }
  
  if (scores.concordance_note) {
    parts.push(`- Concordance Note: ${scores.concordance_note}`);
  }
  
  return parts.join("\n") || "- External scores not available";
}

function formatConcordance(report: Record<string, unknown>): string {
  const parts: string[] = [];
  parts.push(`- Evo2-7B: ${report.prediction} (Δ = ${report.deltaScore})`);
  
  const extScores = report.externalScores as Record<string, unknown> | null;
  if (extScores) {
    const am = extScores.alphamissense as Record<string, unknown> | null;
    if (am) parts.push(`- AlphaMissense: ${am.classification || "N/A"} (Score: ${am.score})`);
    
    const cadd = extScores.cadd as Record<string, unknown> | null;
    if (cadd) parts.push(`- CADD: ${cadd.interpretation || "N/A"} (PHRED: ${cadd.phred})`);
  }
  
  parts.push(`- ClinVar: ${report.clinvarClassification || "Not curated"}`);
  
  return parts.join("\n");
}

function formatXAI(xai: Record<string, unknown>): string {
  const factors = xai.factors as Array<Record<string, unknown>> | undefined;
  if (!factors) return "- XAI factors not available";
  
  const lines = factors.map(f => 
    `  - ${f.label}: Contribution ${f.contribution} — ${f.detail || ""}`
  );
  return `- Total: ${xai.total}\n- Factors:\n${lines.join("\n")}`;
}

function formatCounterfactuals(cf: Record<string, unknown>): string {
  return `- Reference: ${cf.reference}
- Tolerated Alleles: ${Array.isArray(cf.tolerated_alleles) ? (cf.tolerated_alleles as string[]).join(", ") : "N/A"}
- Pathogenic Alleles: ${Array.isArray(cf.pathogenic_alleles) ? (cf.pathogenic_alleles as string[]).join(", ") : "N/A"}
- Summary: ${cf.summary || "N/A"}`;
}

function formatKnowledgeGraph(kg: Record<string, unknown>): string {
  const diseases = kg.diseases as Array<Record<string, unknown>> | undefined;
  const drugs = kg.drugs as Array<Record<string, unknown>> | undefined;
  
  const diseaseLines = diseases?.map(d => `  - ${d.name} (Score: ${d.score})`).join("\n") || "  None";
  const drugLines = drugs?.map(d => `  - ${d.name} (${d.type}, Phase ${d.phase}): ${d.mechanism}`).join("\n") || "  None";
  
  return `- Gene: ${kg.gene}
- Clinical Actionability: ${kg.clinical_actionability || "N/A"}
- Associated Diseases:\n${diseaseLines}
- Relevant Drugs:\n${drugLines}`;
}

function formatISM(ism: Record<string, unknown>): string {
  const summary = ism.summary as Record<string, unknown> | undefined;
  if (!summary) return "- ISM scan data not available";
  
  return `- Constrained Positions: ${summary.constrained_positions}
- Total Scanned: ${summary.total_positions_scanned}
- Constraint Zone: ${summary.constraint_zone}
- Peak Constraint Position: ${summary.peak_constraint_position || "N/A"}
- Peak Magnitude: ${summary.peak_constraint_magnitude || "N/A"}`;
}

// =============================================================================
// POST /api/chat
// =============================================================================

export async function POST(request: NextRequest) {
  try {
    // --- Auth ---
    const user = await currentUser();
    if (!user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = (await request.json()) as ChatRequestBody;
    const { reportId, message, sessionId } = body;

    if (!reportId || !message) {
      return NextResponse.json(
        { error: "reportId and message are required" },
        { status: 400 },
      );
    }

    // --- Load the full report ---
    const report = await db.analysisReport.findFirst({
      where: { id: reportId, clerkUserId: user.id },
    });

    if (!report) {
      return NextResponse.json(
        { error: "Report not found or access denied" },
        { status: 404 },
      );
    }

    // --- Get or create chat session ---
    let chatSession;
    if (sessionId) {
      chatSession = await db.chatSession.findFirst({
        where: { id: sessionId, clerkUserId: user.id },
      });
      if (!chatSession) {
        return NextResponse.json(
          { error: "Chat session not found" },
          { status: 404 },
        );
      }
    } else {
      chatSession = await db.chatSession.create({
        data: {
          clerkUserId: user.id,
          analysisReportId: reportId,
          title: `Chat about ${report.geneSymbol} ${report.reference}>${report.alternative}`,
        },
      });
    }

    // --- Load previous messages for context ---
    const previousMessages = await db.chatMessage.findMany({
      where: { sessionId: chatSession.id },
      orderBy: { createdAt: "asc" },
      take: 20, // last 20 messages for context window
    });

    // --- Build the system prompt with full report context ---
    const systemPrompt = buildSystemPrompt(report as unknown as Record<string, unknown>);

    // --- Build messages array for NVIDIA API ---
    const messages: Array<{ role: string; content: string }> = [
      { role: "system", content: systemPrompt },
    ];

    // Add previous conversation history
    for (const msg of previousMessages) {
      messages.push({ role: msg.role, content: msg.content });
    }

    // Add the new user message
    messages.push({ role: "user", content: message });

    // --- Save user message ---
    await db.chatMessage.create({
      data: {
        sessionId: chatSession.id,
        role: "user",
        content: message,
      },
    });

    // --- Call NVIDIA Nemotron API ---
    if (!NVIDIA_API_KEY) {
      throw new Error("NVIDIA_API_KEY is not configured");
    }

    console.log("[Chat] Calling NVIDIA Nemotron API...");
    const startTime = Date.now();

    const nvidiaResponse = await fetch(NVIDIA_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${NVIDIA_API_KEY}`,
      },
      body: JSON.stringify({
        model: MODEL,
        messages,
        temperature: 0.3,
        top_p: 0.9,
        max_tokens: 1024,
        stream: false,
      }),
      signal: AbortSignal.timeout(180_000),
    });

    if (!nvidiaResponse.ok) {
      const errorText = await nvidiaResponse.text();
      console.error("[Chat] NVIDIA API error:", errorText);
      throw new Error(`NVIDIA API error: ${nvidiaResponse.status}`);
    }

    const nvidiaData = (await nvidiaResponse.json()) as {
      choices?: Array<{ message?: { content?: string } }>;
      usage?: { total_tokens?: number; prompt_tokens?: number; completion_tokens?: number };
      id?: string;
    };

    const latency = Date.now() - startTime;
    const assistantContent = nvidiaData.choices?.[0]?.message?.content || "";

    if (!assistantContent) {
      console.error("[Chat] Empty response from NVIDIA:", JSON.stringify(nvidiaData));
      throw new Error("NVIDIA returned an empty response");
    }

    console.log(`[Chat] Response received in ${latency}ms (${nvidiaData.usage?.total_tokens || 0} tokens)`);

    // --- Save assistant message ---
    await db.chatMessage.create({
      data: {
        sessionId: chatSession.id,
        role: "assistant",
        content: assistantContent,
        metadata: {
          latency_ms: latency,
          model: MODEL,
          tokens: nvidiaData.usage,
        },
      },
    });

    // --- Update session timestamp ---
    await db.chatSession.update({
      where: { id: chatSession.id },
      data: { updatedAt: new Date() },
    });

    return NextResponse.json({
      sessionId: chatSession.id,
      message: assistantContent,
      tokens: nvidiaData.usage,
      latency_ms: latency,
    });
  } catch (error) {
    console.error("[Chat] Error:", error);
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Internal server error",
      },
      { status: 500 },
    );
  }
}

// =============================================================================
// GET /api/chat — Load chat history for a session
// =============================================================================

export async function GET(request: NextRequest) {
  try {
    const user = await currentUser();
    if (!user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { searchParams } = new URL(request.url);
    const sessionId = searchParams.get("sessionId");
    const reportId = searchParams.get("reportId");

    if (sessionId) {
      // Load specific session
      const session = await db.chatSession.findFirst({
        where: { id: sessionId, clerkUserId: user.id },
        include: {
          messages: { orderBy: { createdAt: "asc" } },
        },
      });

      if (!session) {
        return NextResponse.json({ error: "Session not found" }, { status: 404 });
      }

      return NextResponse.json({ session });
    }

    if (reportId) {
      // Load all sessions for a report
      const sessions = await db.chatSession.findMany({
        where: { analysisReportId: reportId, clerkUserId: user.id },
        orderBy: { updatedAt: "desc" },
        include: {
          messages: { orderBy: { createdAt: "asc" }, take: 1 },
        },
      });

      return NextResponse.json({ sessions });
    }

    // Load all user's chat sessions
    const sessions = await db.chatSession.findMany({
      where: { clerkUserId: user.id },
      orderBy: { updatedAt: "desc" },
      take: 20,
      include: {
        messages: { orderBy: { createdAt: "asc" }, take: 1 },
      },
    });

    return NextResponse.json({ sessions });
  } catch (error) {
    console.error("[Chat] Error loading history:", error);
    return NextResponse.json(
      { error: "Failed to load chat history" },
      { status: 500 },
    );
  }
}
