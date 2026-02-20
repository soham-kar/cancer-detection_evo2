"""
Multi-Modal RAG - Tri-Source Evidence Synthesis

Novel Architecture for Publication:
"Unlike existing RAG systems that rely solely on literature, HelixMind implements 
a Tri-Modal Retrieval Architecture integrating:
1. Clinical Consensus: Real-time status from ClinVar
2. Biological Context: Protein function data from UniProtKB
3. Literature Evidence: Abstracts from PubMed

These three streams are synthesized by a Large Language Model (Llama-3-70B) 
to produce a hallucination-resistant clinical narrative."
"""

import modal
import os
import json
import time
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

rag_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "biopython", "groq", "redis", "requests", "fpdf2", "fastapi"
)

app = modal.App("multimodal-rag", image=rag_image)


@dataclass
class MultiModalResult:
    found: bool
    summary: str
    sources: Dict
    pmids: List[str]
    clinvar_status: Optional[str]
    protein_function: Optional[str]


@app.cls(
    cpu=1.0,
    memory=1024,
    secrets=[
        modal.Secret.from_name("groq-config"),
        modal.Secret.from_name("redis-credentials"),
    ],
    scaledown_window=120,
)
class MultiModalRAG:
    CACHE_TTL = 2592000  # 30 days

    @modal.enter()
    def initialize(self):
        from Bio import Entrez
        from groq import Groq

        Entrez.email = os.getenv("ENTREZ_EMAIL", "helixmind@example.com")
        Entrez.api_key = os.getenv("NCBI_API_KEY")
        self.entrez = Entrez
        self.groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Connect Redis for caching
        self.redis = None
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis
                self.redis = redis.from_url(redis_url, decode_responses=True)
                print("✅ Redis connected")
            except Exception as e:
                print(f"⚠️ Redis error: {e}")

    # =========================================================================
    # ROBUST CLINVAR SENSOR (with validate=False fix)
    # =========================================================================
    def _get_clinvar_status(self, gene: str, variant: str) -> Dict:
        try:
            # STRATEGY: Broad Search
            query = f"{gene} AND {variant}"

            handle = self.entrez.esearch(db="clinvar", term=query, retmax=3)
            record = self.entrez.read(handle, validate=False)  # FIX: validate=False
            handle.close()
            uids = record.get("IdList", [])

            if not uids:
                return {
                    "status": "Not Found",
                    "text": f"Variant '{variant}' not found in ClinVar (Novel VUS).",
                    "id": None,
                }

            # Rate limiting
            time.sleep(0.35)

            # Fetch details
            handle = self.entrez.esummary(db="clinvar", id=uids[0])
            summary = self.entrez.read(handle, validate=False)  # FIX: validate=False
            handle.close()
            doc = summary["DocumentSummarySet"]["DocumentSummary"][0]

            # Check fields (handle both dict and str types)
            classification = "Unknown"
            if "germline_classification" in doc:
                gc = doc["germline_classification"]
                if isinstance(gc, dict):
                    classification = gc.get("description", "Unknown")
                elif isinstance(gc, str):
                    classification = gc
            elif "clinical_significance" in doc:
                cs = doc["clinical_significance"]
                if isinstance(cs, dict):
                    classification = cs.get("description", "Unknown")
                elif isinstance(cs, str):
                    classification = cs

            review_status = doc.get("review_status", "Unknown")
            return {
                "status": classification,
                "text": f"ClinVar Classification: {classification} (Review Status: {review_status})",
                "id": uids[0],
            }

        except Exception as e:
            return {"status": "Error", "text": f"ClinVar Error: {str(e)}", "id": None}

    # =========================================================================
    # UNIPROT SENSOR
    # =========================================================================
    def _get_protein_function(self, gene: str) -> Dict:
        import requests

        try:
            url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene}+AND+organism_id:9606&fields=accession,cc_function&format=json"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("results"):
                return {"function": None, "text": "Protein function not found.", "id": None}

            result = data["results"][0]
            acc = result.get("primaryAccession", "Unknown")
            func = "Function unknown."
            for c in result.get("comments", []):
                if c.get("commentType") == "FUNCTION":
                    texts = c.get("texts", [])
                    if texts:
                        func = texts[0].get("value", "")[:600]
                        break

            return {"function": func, "text": f"Protein Function ({acc}): {func}", "id": acc}
        except Exception as e:
            return {"function": None, "text": f"UniProt Error: {str(e)}", "id": None}

    # =========================================================================
    # PUBMED SENSOR (Hierarchical Search)
    # =========================================================================
    def _get_pubmed_evidence(self, gene: str, variant: str) -> Dict:
        try:
            # L1: Exact Variant
            query = f'"{gene}"[Gene] AND "{variant}"[Title/Abstract]'
            handle = self.entrez.esearch(db="pubmed", term=query, retmax=3, sort="relevance")
            record = self.entrez.read(handle, validate=False)
            handle.close()
            pmids = record.get("IdList", [])
            level = "Exact Variant"

            # L2: Gene Context (Fallback)
            if not pmids:
                level = "Gene Context"
                query = f'"{gene}"[Gene] AND (pathogenic[Title] OR disease[Title])'
                handle = self.entrez.esearch(db="pubmed", term=query, retmax=3, sort="relevance")
                record = self.entrez.read(handle, validate=False)
                handle.close()
                pmids = record.get("IdList", [])

            if not pmids:
                return {"text": "No literature found.", "pmids": [], "level": "None"}

            # Rate limiting
            time.sleep(0.35)

            handle = self.entrez.efetch(db="pubmed", id=pmids, rettype="medline", retmode="text")
            text = handle.read()
            handle.close()

            return {
                "text": f"Found {len(pmids)} papers ({level}):\n{text[:3000]}",
                "pmids": pmids,
                "level": level,
            }
        except Exception as e:
            return {"text": f"PubMed Error: {str(e)}", "pmids": [], "level": "Error"}

    # =========================================================================
    # MAIN SYNTHESIS (STRUCTURED "Top 1%" Format)
    # =========================================================================
    @modal.method()
    def search_and_synthesize(self, gene: str, variant: str) -> Dict:
        print(f"🚀 Tri-Modal Search: {gene} {variant}")

        cache_key = f"multimodal:{gene}:{variant}"

        # Check cache first
        if self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    print("✅ Cache HIT")
                    result = json.loads(cached)
                    result["cached"] = True
                    return result
            except Exception as e:
                print(f"⚠️ Cache read error: {e}")

        # Fetch from all sources
        print("🏥 Fetching ClinVar...")
        clinvar = self._get_clinvar_status(gene, variant)
        print(f"   → ClinVar: {clinvar['status']}")

        print("🧬 Fetching UniProt...")
        uniprot = self._get_protein_function(gene)

        print("📚 Fetching PubMed...")
        pubmed = self._get_pubmed_evidence(gene, variant)
        print(f"   → PubMed: {len(pubmed['pmids'])} papers")

        # --- THE STRUCTURED "TOP 1%" PROMPT ---
        print("⚡ Synthesizing with Llama-3.3-70B...")

        system_prompt = """You are an expert clinical geneticist writing a structured report for a doctor.
You have access to ClinVar (Clinical Truth), UniProt (Mechanism), and PubMed (Latest Evidence).

Structure your response EXACTLY like this with markdown headers:

**1. Clinical Classification:**
Start with the ClinVar status. State clearly if it is Pathogenic, Benign, or VUS (Variant of Uncertain Significance).

**2. Biological Mechanism:**
Explain the molecular impact using UniProt data. Does it disrupt DNA binding, enzyme activity, or structural stability?

**3. Evidence & Implications:**
Cite specific findings from the PubMed abstracts. Mention associated diseases, drug resistance, or specific phenotypes. Use [PMID:XXXX] citations.

Keep the tone professional, objective, and evidence-based. Be thorough but concise."""

        user_prompt = f"""GENE: {gene}
VARIANT: {variant}

--- CLINVAR (Official Status) ---
{clinvar['text']}

--- UNIPROT (Mechanism) ---
{uniprot['text']}

--- PUBMED (Literature) ---
{pubmed['text']}

Synthesize a structured clinical report:"""

        try:
            completion = self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=600,
                temperature=0.3,
            )
            summary = completion.choices[0].message.content.strip()
        except Exception as e:
            summary = f"LLM Synthesis Error: {e}"

        print("✅ Tri-Modal Search Complete")

        result = {
            "found": True,
            "summary": summary,
            "clinvar_status": clinvar["status"],
            "clinvar_id": clinvar.get("id"),
            "protein_function": uniprot.get("function"),
            "uniprot_id": uniprot.get("id"),
            "pmids": pubmed["pmids"],
            "pubmed_level": pubmed["level"],
            "sources": {
                "clinvar": {"status": clinvar["status"], "id": clinvar.get("id")},
                "uniprot": {"id": uniprot.get("id"), "has_function": uniprot.get("function") is not None},
                "pubmed": {"count": len(pubmed["pmids"]), "level": pubmed["level"]},
            },
            "cached": False,
        }

        # Cache result
        if self.redis:
            try:
                self.redis.set(cache_key, json.dumps(result), ex=self.CACHE_TTL)
                print("💾 Cached result")
            except Exception as e:
                print(f"⚠️ Cache write error: {e}")

        return result

    # =========================================================================
    # PDF REPORT GENERATOR
    # =========================================================================
    @modal.method()
    def generate_pdf_report(self, variant_data: Dict, rag_data: Dict) -> bytes:
        from fpdf import FPDF
        from datetime import datetime

        pdf = FPDF()
        pdf.add_page()

        # Title
        pdf.set_font("Arial", "B", 18)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 12, "HelixMind Clinical Variant Report", ln=True, align="C")
        pdf.set_font("Arial", "I", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        pdf.cell(0, 5, "Tri-Modal Evidence Synthesis", ln=True, align="C")
        pdf.ln(8)

        # Variant Info
        gene = variant_data.get("gene", "Unknown")
        variant = variant_data.get("variant", "Unknown")
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"Variant: {gene} {variant}", ln=True)
        pdf.ln(3)

        # AI Prediction (if available)
        pred = variant_data.get("prediction", "")
        score = variant_data.get("score", 0)
        if pred:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(50, 8, "AI Prediction:", ln=False)
            if "pathogenic" in pred.lower():
                pdf.set_text_color(180, 0, 0)
            else:
                pdf.set_text_color(0, 130, 0)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, f"{pred} (Score: {score:.4f})", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)

        # ClinVar Status
        clinvar_status = rag_data.get("clinvar_status", "Unknown")
        pdf.set_font("Arial", "B", 12)
        pdf.cell(50, 8, "ClinVar Status:", ln=False)
        if "pathogenic" in clinvar_status.lower():
            pdf.set_text_color(180, 0, 0)
        elif "benign" in clinvar_status.lower():
            pdf.set_text_color(0, 130, 0)
        else:
            pdf.set_text_color(150, 100, 0)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, clinvar_status, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # Structured Summary
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Tri-Modal Evidence Analysis", ln=True)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        pdf.set_font("Arial", "", 11)
        summary = rag_data.get("summary", "No data").encode("latin-1", "replace").decode("latin-1")
        # Convert markdown bold to regular text for PDF
        summary = summary.replace("**", "")
        pdf.multi_cell(0, 6, summary)
        pdf.ln(5)

        # References
        pmids = rag_data.get("pmids", [])
        if pmids:
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, "References:", ln=True)
            pdf.set_font("Arial", "", 10)
            for pmid in pmids[:5]:
                pdf.cell(0, 6, f"  PMID: {pmid} (https://pubmed.ncbi.nlm.nih.gov/{pmid})", ln=True)

        # Footer
        pdf.set_y(-30)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.set_font("Arial", "I", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(
            0,
            4,
            "Generated by HelixMind | Evo2-7B + Llama-3.3-70B\n"
            "Sources: ClinVar, UniProtKB, PubMed\n"
            "FOR RESEARCH USE ONLY",
        )

        return pdf.output(dest="S").encode("latin-1")


# =============================================================================
# WEB ENDPOINT (for HTTP access from frontend)
# =============================================================================
@app.function(
    cpu=1.0,
    memory=1024,
    secrets=[
        modal.Secret.from_name("groq-config"),
        modal.Secret.from_name("redis-credentials"),
    ],
)
@modal.web_endpoint(method="POST")
def search_and_synthesize_web(data: dict):
    """
    HTTP endpoint for the Tri-Modal RAG.
    Called from the frontend demo-analyze API.
    
    Request body: {"gene": "BRCA1", "variant": "c.68_69delAG"}
    """
    gene = data.get("gene", "")
    variant = data.get("variant", "")
    
    if not gene or not variant:
        return {"error": "Missing gene or variant", "found": False}
    
    # Use the MultiModalRAG class
    rag = MultiModalRAG()
    result = rag.search_and_synthesize.remote(gene, variant)
    
    return result


# =============================================================================
# TEST ENTRYPOINT
# =============================================================================
@app.local_entrypoint()
def main():
    rag = MultiModalRAG()

    test_cases = [
        ("BRCA1", "c.68_69delAG"),
        ("TP53", "R248W"),
    ]

    for gene, var in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {gene} {var}")
        print("=" * 60)

        res = rag.search_and_synthesize.remote(gene, var)

        print(f"\n📊 RESULTS:")
        print(f"ClinVar Status: {res['clinvar_status']}")
        print(f"PMIDs: {res['pmids']}")
        print(f"\n📝 STRUCTURED SYNTHESIS:\n{res['summary']}")
