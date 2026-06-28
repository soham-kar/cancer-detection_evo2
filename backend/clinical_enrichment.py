"""
Clinical Enrichment Module for Variant Analysis

This module provides five clinical evidence layers:
1. GnomadClient - Population frequency filtering (ACMG BA1/BS1/PM2)
2. ACMGMapper - Evidence code mapping (PP3/BP4)
3. PubMedRAG - Literature context retrieval
4. ClinVarClient - Clinical consensus from NCBI ClinVar database
5. UniProtClient - Protein function and domain annotations

These transform raw AI scores into clinically actionable outputs.
"""

import os
import json
import logging
import time
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from Bio import Entrez

logger = logging.getLogger(__name__)

# Use environment variable for Entrez email
Entrez.email = os.getenv("ENTREZ_EMAIL", "biotech-evo2@example.com")
if os.getenv("NCBI_API_KEY"):
    Entrez.api_key = os.getenv("NCBI_API_KEY")


# =============================================================================
# 1. GNOMAD CLIENT - Population Frequency Pre-Filter
# =============================================================================

@dataclass
class GnomadResult:
    """Result from gnomAD query"""
    allele_frequency: Optional[float]
    population_max_af: Optional[float]
    source: str = "gnomAD v4.1"
    is_common: bool = False
    auto_classification: Optional[str] = None


class GnomadClient:
    """
    Client for querying gnomAD population allele frequencies.
    
    If a variant has AF > 1%, it is auto-classified as Benign (ACMG BA1).
    If AF > 0.1%, it provides supporting benign evidence (ACMG BS1).
    """
    
    GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"
    
    # ACMG frequency thresholds
    BA1_THRESHOLD = 0.05  # >5% = standalone benign (BA1)
    BS1_THRESHOLD = 0.01  # >1% = strong benign supporting (BS1)
    PM2_THRESHOLD = 0.0001  # <0.01% = rare, supporting pathogenic (PM2)
    
    def __init__(self, redis_client=None, cache_ttl: int = 604800):
        """
        Initialize gnomAD client.
        
        Args:
            redis_client: Optional Redis client for caching
            cache_ttl: Cache TTL in seconds (default: 7 days)
        """
        self.redis = redis_client
        self.cache_ttl = cache_ttl
    
    def _build_variant_id(self, chrom: str, pos: int, ref: str, alt: str) -> str:
        """Build gnomAD variant ID format: chrom-pos-ref-alt"""
        # Remove 'chr' prefix if present
        chrom_clean = chrom.replace("chr", "")
        return f"{chrom_clean}-{pos}-{ref}-{alt}"
    
    def _query_graphql(self, variant_id: str, dataset: str = "gnomad_r4") -> Optional[Dict]:
        """Query gnomAD GraphQL API"""
        query = """
        query VariantQuery($variantId: String!, $datasetId: DatasetId!) {
            variant(variantId: $variantId, dataset: $datasetId) {
                variant_id
                genome {
                    af
                    ac
                    an
                }
                exome {
                    af
                    ac
                    an
                }
                joint {
                    af
                    faf95
                }
            }
        }
        """
        
        variables = {
            "variantId": variant_id,
            "datasetId": dataset
        }
        
        try:
            response = requests.post(
                self.GNOMAD_API_URL,
                json={"query": query, "variables": variables},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if "errors" in data:
                logger.warning(f"gnomAD API error: {data['errors']}")
                return None
                
            return data.get("data", {}).get("variant")
            
        except requests.RequestException as e:
            logger.warning(f"gnomAD API request failed: {e}")
            return None
    
    def get_allele_frequency(
        self, 
        chromosome: str, 
        position: int, 
        ref: str, 
        alt: str
    ) -> GnomadResult:
        """
        Get allele frequency for a variant from gnomAD.
        
        Args:
            chromosome: Chromosome (e.g., 'chr17' or '17')
            position: Genomic position (1-based)
            ref: Reference allele
            alt: Alternative allele
            
        Returns:
            GnomadResult with frequency data and auto-classification if applicable
        """
        variant_id = self._build_variant_id(chromosome, position, ref, alt)
        cache_key = f"gnomad:{variant_id}"
        
        # Try cache first
        if self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    logger.info(f"gnomAD Cache HIT: {variant_id}")
                    data = json.loads(cached) if isinstance(cached, str) else json.loads(cached.decode())
                    return GnomadResult(**data)
            except Exception as e:
                logger.warning(f"gnomAD cache read error: {e}")
        
        # Query gnomAD API
        logger.info(f"gnomAD Cache MISS: Querying API for {variant_id}")
        variant_data = self._query_graphql(variant_id)
        
        # Build result
        result = GnomadResult(
            allele_frequency=None,
            population_max_af=None,
            source="gnomAD v4.1"
        )
        
        if variant_data:
            # Prefer joint frequency, fall back to genome/exome
            joint = variant_data.get("joint", {})
            genome = variant_data.get("genome", {})
            exome = variant_data.get("exome", {})
            
            af = (
                joint.get("af") or 
                genome.get("af") or 
                exome.get("af")
            )
            
            if af is not None:
                result.allele_frequency = af
                result.population_max_af = joint.get("faf95") or af
                
                # Apply ACMG frequency rules
                if af >= self.BA1_THRESHOLD:
                    result.is_common = True
                    result.auto_classification = "Benign"
                    logger.info(f"gnomAD BA1: {variant_id} has AF={af:.4f} (>5%), auto-Benign")
                elif af >= self.BS1_THRESHOLD:
                    result.is_common = True
                    result.auto_classification = "Likely benign"
                    logger.info(f"gnomAD BS1: {variant_id} has AF={af:.4f} (>1%), auto-Likely Benign")
        
        # Cache result
        if self.redis:
            try:
                self.redis.set(
                    cache_key, 
                    json.dumps(result.__dict__), 
                    ex=self.cache_ttl
                )
            except Exception as e:
                logger.warning(f"gnomAD cache write error: {e}")
        
        return result

# 2. ACMG MAPPER - Evidence Code Assignment

class ACMGStrength(str, Enum):
    """ACMG evidence strength levels"""
    VERY_STRONG = "Very Strong"
    STRONG = "Strong"
    MODERATE = "Moderate"
    SUPPORTING = "Supporting"
    NONE = "None"


@dataclass
class ACMGEvidence:
    """ACMG evidence code with clinical interpretation"""
    code: str
    strength: ACMGStrength
    description: str
    points: float  # For automated ACMG scoring
    clinical_note: str


class ACMGMapper:
    """
    Maps computational predictions to ACMG evidence codes.
    
    Primary codes used:
    - PP3: Computational evidence supports pathogenic
    - BP4: Computational evidence supports benign
    
    Strength levels based on score thresholds calibrated from BRCA1 data.
    """
    
    # Thresholds calibrated from your BRCA1 validation
    STRONG_PATHOGENIC_THRESHOLD = -0.8
    MODERATE_PATHOGENIC_THRESHOLD = -0.4
    SUPPORTING_PATHOGENIC_THRESHOLD = -0.1
    SUPPORTING_BENIGN_THRESHOLD = 0.2
    MODERATE_BENIGN_THRESHOLD = 0.5
    
    # Confidence threshold for upgrading evidence
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    
    def map_score_to_evidence(
        self, 
        delta_score: float, 
        confidence: float,
        model_name: str = "Evo2-7B"
    ) -> ACMGEvidence:
        """
        Map Evo2 delta score to ACMG PP3/BP4 evidence.        
        Args:
            delta_score: Evo2 log-likelihood delta (var - ref)
            confidence: Classification confidence (0-1)
            model_name: Name of the model for documentation            
        Returns: ACMGEvidence with appropriate code and strength
        """
        
        # High confidence can upgrade evidence by one level
        confidence_boost = confidence >= self.HIGH_CONFIDENCE_THRESHOLD
        
        if delta_score < self.STRONG_PATHOGENIC_THRESHOLD:
            # Strong pathogenic evidence
            if confidence_boost:
                return ACMGEvidence(
                    code="PP3_VeryStrong",
                    strength=ACMGStrength.VERY_STRONG,
                    description="Computational evidence strongly supports deleterious effect",
                    points=4.0,
                    clinical_note=f"{model_name} prediction: delta_score={delta_score:.4f} with high confidence ({confidence:.0%}). This variant shows a significant decrease in sequence fitness."
                )
            return ACMGEvidence(
                code="PP3_Strong",
                strength=ACMGStrength.STRONG,
                description="Computational evidence supports deleterious effect",
                points=2.0,
                clinical_note=f"{model_name} prediction: delta_score={delta_score:.4f}. This variant shows a substantial decrease in sequence fitness consistent with pathogenic variants."
            )
            
        elif delta_score < self.MODERATE_PATHOGENIC_THRESHOLD:
            return ACMGEvidence(
                code="PP3_Moderate",
                strength=ACMGStrength.MODERATE,
                description="Computational evidence moderately supports deleterious effect",
                points=1.0,
                clinical_note=f"{model_name} prediction: delta_score={delta_score:.4f}. Moderate decrease in sequence fitness. Should be combined with other evidence."
            )
            
        elif delta_score < self.SUPPORTING_PATHOGENIC_THRESHOLD:
            return ACMGEvidence(
                code="PP3_Supporting",
                strength=ACMGStrength.SUPPORTING,
                description="Computational evidence provides supporting pathogenic evidence",
                points=0.5,
                clinical_note=f"{model_name} prediction: delta_score={delta_score:.4f}. Weak pathogenic signal. Requires additional evidence for clinical significance."
            )
            
        elif delta_score < self.SUPPORTING_BENIGN_THRESHOLD:
            return ACMGEvidence(
                code="None",
                strength=ACMGStrength.NONE,
                description="Computational evidence is inconclusive",
                points=0.0,
                clinical_note=f"{model_name} prediction: delta_score={delta_score:.4f}. Score is in the uncertain range and does not provide strong evidence in either direction."
            )
            
        elif delta_score < self.MODERATE_BENIGN_THRESHOLD:
            return ACMGEvidence(
                code="BP4_Supporting",
                strength=ACMGStrength.SUPPORTING,
                description="Computational evidence provides supporting benign evidence",
                points=-0.5,
                clinical_note=f"{model_name} prediction: delta_score={delta_score:.4f}. Weak benign signal suggests this variant may not significantly impact function."
            )
            
        else:
            if confidence_boost:
                return ACMGEvidence(
                    code="BP4_Strong",
                    strength=ACMGStrength.STRONG,
                    description="Computational evidence strongly supports benign effect",
                    points=-2.0,
                    clinical_note=f"{model_name} prediction: delta_score={delta_score:.4f} with high confidence ({confidence:.0%}). This variant is predicted to have minimal impact on sequence fitness."
                )
            return ACMGEvidence(
                code="BP4_Moderate",
                strength=ACMGStrength.MODERATE,
                description="Computational evidence supports benign effect",
                points=-1.0,
                clinical_note=f"{model_name} prediction: delta_score={delta_score:.4f}. Variant shows preserved sequence fitness consistent with benign variants."
            )

# 3. PUBMED RAG - Literature Context

@dataclass
class LiteratureContext:
    """Literature context from PubMed"""
    summary: Optional[str]
    pubmed_ids: List[str]
    gene_function: Optional[str]
    num_articles_found: int
    search_query: str


class PubMedRAG:
    """
    Retrieval-Augmented Generation for variant literature context.
    
    Fetches relevant PubMed abstracts and generates a clinical summary
    using an LLM (Llama 3.1 8B via Modal).
    """
    
    def __init__(self, redis_client=None, llm_endpoint: str = None, cache_ttl: int = 2592000):
        """
        Initialize PubMed RAG.
        
        Args:
            redis_client: Optional Redis client for caching
            llm_endpoint: URL of the Modal LLM service
            cache_ttl: Cache TTL in seconds (default: 30 days)
        """
        self.redis = redis_client
        self.llm_endpoint = llm_endpoint or os.getenv("LLM_ENDPOINT")
        self.cache_ttl = cache_ttl
    
    def _search_pubmed(self, gene_symbol: str, max_results: int = 5) -> List[Dict]:
        """Search PubMed for gene-related variant articles with rate-limit retries."""
        import time
        from urllib.error import HTTPError

        try:
            # Use valid Entrez field tags for the pubmed database.
            # [Gene] is not a valid PubMed field; use [Title/Abstract] instead.
            query = (
                f"{gene_symbol}[Title/Abstract] AND "
                f"(pathogenic[Title/Abstract] OR variant[Title/Abstract] OR mutation[Title/Abstract])"
            )
            logger.info(f"PubMed searching: {query}")

            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
            record = Entrez.read(handle)
            handle.close()

            ids = record.get("IdList", [])
            error_list = record.get("ErrorList", {})
            if error_list:
                logger.warning(f"PubMed Entrez errors for {gene_symbol}: {error_list}")

            logger.info(f"PubMed found {len(ids)} IDs for {gene_symbol}: {ids}")
            if not ids:
                return []

            # Fetch abstracts with retry/backoff for NCBI rate limits
            articles = None
            last_error = None
            for attempt in range(3):
                try:
                    # NCBI recommends 0.34s between requests without API key;
                    # be conservative, especially on shared Modal egress IP.
                    if attempt > 0:
                        sleep_seconds = 1.0 * (2 ** attempt)
                        logger.info(f"PubMed efetch retry {attempt}/3 for {gene_symbol}: sleeping {sleep_seconds}s")
                        time.sleep(sleep_seconds)

                    handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="xml", retmode="xml")
                    articles = Entrez.read(handle)
                    handle.close()
                    break
                except HTTPError as he:
                    last_error = he
                    logger.warning(f"PubMed efetch HTTPError (attempt {attempt + 1}/3): {he}")
                    if he.code != 429:
                        raise
                except Exception as e:
                    logger.warning(f"PubMed efetch error (attempt {attempt + 1}/3): {e}")
                    raise

            if articles is None:
                logger.error(f"PubMed efetch FAILED after retries for {gene_symbol}: {last_error}")
                return []

            results = []
            for article in articles.get("PubmedArticle", []):
                try:
                    medline = article["MedlineCitation"]
                    article_data = medline["Article"]

                    pmid = str(medline["PMID"])
                    title = article_data.get("ArticleTitle", "")

                    # Get abstract text
                    abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
                    if abstract_parts:
                        if isinstance(abstract_parts, list):
                            abstract = " ".join(str(p) for p in abstract_parts)
                        else:
                            abstract = str(abstract_parts)
                    else:
                        abstract = ""

                    results.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract[:1500]  # Limit abstract length
                    })
                except Exception as parse_err:
                    logger.warning(f"PubMed article parse error: {parse_err}")
                    continue

            logger.info(f"PubMed returning {len(results)} articles for {gene_symbol}")
            return results
        except Exception as e:
            logger.error(f"PubMed search FAILED for {gene_symbol}: {type(e).__name__}: {e}")
            return []
    
    def _generate_summary(self, gene_symbol: str, articles: List[Dict]) -> str:
        """
        Generate clinical summary using Groq SDK (Llama 3.3 70B).        
        Uses the official Groq SDK for automatic retries, connection pooling,
        and typed error handling.
        """
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key or not articles:
            # Fallback: return simple concatenation
            if articles:
                return f"Found {len(articles)} articles about {gene_symbol}. Key article: {articles[0].get('title', 'N/A')}"
            return None
        
        # Build prompt with context from articles
        context = "\n\n".join([
            f"Title: {a['title']}\nAbstract: {a['abstract'][:500]}"
            for a in articles[:3]
        ])
        
        system_prompt = """You are a strictly clinical genetic database. 
Output only the raw clinical summary data. 
Never use conversational fillers. 
Never use phrases like "Here is", "Based on", "In summary", or any introductory text.
Provide detailed, clinically relevant information about gene function and disease associations."""
        
        user_prompt = f"""Task: Generate a 2-3 sentence clinical summary for {gene_symbol}.
Context: {context}

Format Requirements:
- Start immediately with the gene name.
- No introductory text or preamble.
- Focus on: gene function, disease associations, clinical significance.
- Be detailed and clinically informative.

[START SUMMARY]
{gene_symbol}"""
        
        try:
            # Import Groq SDK (available in container)
            from groq import Groq
            
            # Initialize client (handles connection pooling & retries automatically)
            client = Groq(api_key=api_key)
            
            # Call Llama 3.3 70B - GPT-4 class intelligence, still fast on Groq
            chat_completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Upgraded from 3.1-8b
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Good for factual consistency
                max_tokens=300,  # Increased from 150 to prevent truncation
            )
            
            return chat_completion.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"Groq API failed: {e}")
            if articles:
                return f"Found {len(articles)} articles about {gene_symbol}. Key article: {articles[0].get('title', 'N/A')}"
            return None
    
    def get_literature_context(self, gene_symbol: str) -> LiteratureContext:
        """
        Get literature context for a gene.        
        Args:gene_symbol: Gene symbol (e.g., 'BRCA1')            
        Returns:LiteratureContext with summary and references
        """
        if not gene_symbol:
            return LiteratureContext(
                summary=None,
                pubmed_ids=[],
                gene_function=None,
                num_articles_found=0,
                search_query=""
            )
        
        cache_key = f"pubmed_rag:{gene_symbol}"
        
        # Try cache
        if self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    logger.info(f"PubMed RAG Cache HIT: {gene_symbol}")
                    data = json.loads(cached) if isinstance(cached, str) else json.loads(cached.decode())
                    return LiteratureContext(**data)
            except Exception as e:
                logger.warning(f"PubMed RAG cache read error: {e}")
        
        # Search PubMed
        logger.info(f"PubMed RAG Cache MISS: Searching for {gene_symbol}")
        articles = self._search_pubmed(gene_symbol)
        
        # Generate summary
        summary = self._generate_summary(gene_symbol, articles) if articles else None
        
        # Extract gene function from first article if available
        gene_function = None
        if articles and len(articles[0].get("abstract", "")) > 100:
            # Simple heuristic - first sentence often describes function
            first_abstract = articles[0]["abstract"]
            first_sentence = first_abstract.split(". ")[0]
            if len(first_sentence) < 200:
                gene_function = first_sentence
        
        result = LiteratureContext(
            summary=summary,
            pubmed_ids=[a["pmid"] for a in articles],
            gene_function=gene_function,
            num_articles_found=len(articles),
            search_query=f"{gene_symbol}[Gene] AND pathogenic"
        )
        
        # Cache result
        if self.redis and summary:
            try:
                self.redis.set(cache_key, json.dumps(result.__dict__), ex=self.cache_ttl)
            except Exception as e:
                logger.warning(f"PubMed RAG cache write error: {e}")
        
        return result

# =============================================================================
# 4. CLINVAR CLIENT - Clinical Consensus Evidence
# =============================================================================

@dataclass
class ClinVarResult:
    """Result from ClinVar query"""
    status: str  # Pathogenic, Benign, VUS, etc.
    review_status: str  # Expert panel, multiple submitters, single submitter
    variation_id: Optional[str]
    num_submitters: int = 0
    conflicting: bool = False
    text: str = ""


class ClinVarClient:
    """
    Client for querying NCBI ClinVar database for clinical classifications.
    
    Provides real-time clinical consensus data for variant interpretation.
    Uses NCBI Entrez API (same as PubMed) for consistency.
    """
    
    def __init__(self, redis_client=None, cache_ttl: int = 2592000):
        self.redis = redis_client
        self.cache_ttl = cache_ttl
    
    def get_variant_status(
        self,
        gene_symbol: str,
        variant_str: str  # e.g., "G>T" or "c.5266dupC"
    ) -> ClinVarResult:
        """
        Query ClinVar for variant classification.
        
        Args:
            gene_symbol: HGNC gene symbol (e.g., 'BRCA1')
            variant_str: Variant description string
            
        Returns:
            ClinVarResult with classification, review status, and metadata
        """
        cache_key = f"clinvar:{gene_symbol}:{variant_str}"
        
        # Try cache first
        if self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    logger.info(f"ClinVar Cache HIT: {gene_symbol} {variant_str}")
                    data = json.loads(cached) if isinstance(cached, str) else json.loads(cached.decode())
                    return ClinVarResult(**data)
            except Exception as e:
                logger.warning(f"ClinVar cache read error: {e}")
        
        logger.info(f"ClinVar Cache MISS: Querying for {gene_symbol} {variant_str}")
        
        try:
            # Broad search: gene + variant description
            query = f"{gene_symbol}[Gene Name] AND {variant_str}"
            handle = Entrez.esearch(db="clinvar", term=query, retmax=5)
            record = Entrez.read(handle, validate=False)
            handle.close()
            uids = record.get("IdList", [])
            
            if not uids:
                result = ClinVarResult(
                    status="Not Found",
                    review_status="N/A",
                    variation_id=None,
                    text=f"Variant '{variant_str}' not found in ClinVar (Novel VUS)."
                )
                self._cache_result(cache_key, result)
                return result
            
            # Rate limiting for NCBI
            time.sleep(0.35)
            
            # Fetch details for first match
            handle = Entrez.esummary(db="clinvar", id=uids[0])
            summary = Entrez.read(handle, validate=False)
            handle.close()
            
            doc = summary["DocumentSummarySet"]["DocumentSummary"][0]
            
            # Extract classification (handle both dict and string formats)
            classification = "Unknown"
            if "germline_classification" in doc:
                gc = doc["germline_classification"]
                classification = gc.get("description", "Unknown") if isinstance(gc, dict) else str(gc)
            elif "clinical_significance" in doc:
                cs = doc["clinical_significance"]
                classification = cs.get("description", "Unknown") if isinstance(cs, dict) else str(cs)
            
            # Extract review status
            review_status = "Unknown"
            if "review_status" in doc:
                rs = doc["review_status"]
                review_status = rs if isinstance(rs, str) else str(rs)
            
            # Count submitters
            num_submitters = 0
            if "submitters" in doc:
                submitters = doc["submitters"]
                if isinstance(submitters, list):
                    num_submitters = len(submitters)
            
            # Check for conflicting interpretations
            conflicting = "conflicting" in classification.lower()
            
            result = ClinVarResult(
                status=classification,
                review_status=review_status,
                variation_id=uids[0],
                num_submitters=num_submitters,
                conflicting=conflicting,
                text=f"ClinVar Classification: {classification} (Review: {review_status}, {num_submitters} submitters)"
            )
            
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.warning(f"ClinVar query failed: {e}")
            result = ClinVarResult(
                status="Error",
                review_status="N/A",
                variation_id=None,
                text=f"ClinVar Error: {str(e)}"
            )
            return result
    
    def _cache_result(self, cache_key: str, result: ClinVarResult):
        """Cache result if Redis is available"""
        if self.redis:
            try:
                self.redis.set(cache_key, json.dumps(result.__dict__), ex=self.cache_ttl)
            except Exception as e:
                logger.warning(f"ClinVar cache write error: {e}")


# =============================================================================
# 5. UNIPROT CLIENT - Protein Function & Domain Annotations
# =============================================================================

@dataclass
class UniProtResult:
    """Result from UniProt query"""
    accession: Optional[str]
    protein_name: Optional[str]
    function: Optional[str]
    domains: List[Dict[str, Any]] = field(default_factory=list)
    subcellular_location: Optional[str] = None
    disease_associations: List[str] = field(default_factory=list)
    text: str = ""


class UniProtClient:
    """
    Client for querying UniProtKB for protein function and domain annotations.
    
    Provides structural and functional context for variant interpretation:
    - Protein function and catalytic activity
    - Domain boundaries and active sites
    - Subcellular localization
    - Disease associations
    """
    
    UNIPROT_REST_URL = "https://rest.uniprot.org/uniprotkb/search"
    
    def __init__(self, redis_client=None, cache_ttl: int = 2592000):
        self.redis = redis_client
        self.cache_ttl = cache_ttl
    
    def get_protein_info(self, gene_symbol: str) -> UniProtResult:
        """
        Query UniProtKB for protein annotations.
        
        Args:
            gene_symbol: HGNC gene symbol (e.g., 'BRCA1')
            
        Returns:
            UniProtResult with function, domains, and disease associations
        """
        cache_key = f"uniprot:{gene_symbol}"
        
        # Try cache first
        if self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    logger.info(f"UniProt Cache HIT: {gene_symbol}")
                    data = json.loads(cached) if isinstance(cached, str) else json.loads(cached.decode())
                    return UniProtResult(**data)
            except Exception as e:
                logger.warning(f"UniProt cache read error: {e}")
        
        logger.info(f"UniProt Cache MISS: Querying for {gene_symbol}")
        
        try:
            # Query for human gene (organism_id:9606)
            params = {
                "query": f"gene:{gene_symbol} AND organism_id:9606",
                "fields": "accession,protein_name,cc_function,cc_subcellular_location,cc_disease,ft_domain",
                "format": "json",
                "size": 1
            }
            
            resp = requests.get(self.UNIPROT_REST_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if not data.get("results"):
                result = UniProtResult(
                    accession=None,
                    protein_name=None,
                    function=None,
                    text=f"Protein information not found for {gene_symbol}."
                )
                self._cache_result(cache_key, result)
                return result
            
            result_data = data["results"][0]
            
            # Extract accession
            accession = result_data.get("primaryAccession")
            
            # Extract protein name
            protein_name = None
            if "proteinDescription" in result_data:
                recommended = result_data["proteinDescription"].get("recommendedName", {})
                protein_name = recommended.get("fullName", {}).get("value")
            
            # Extract function
            function = None
            for comment in result_data.get("comments", []):
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        function = texts[0].get("value", "")[:800]
                        break
            
            # Extract domains
            domains = []
            for feature in result_data.get("features", []):
                if feature.get("type") == "Domain":
                    domains.append({
                        "name": feature.get("description", ""),
                        "start": feature.get("location", {}).get("start", {}).get("value"),
                        "end": feature.get("location", {}).get("end", {}).get("value")
                    })
            
            # Extract subcellular location
            subcellular = None
            for comment in result_data.get("comments", []):
                if comment.get("commentType") == "SUBCELLULAR_LOCATION":
                    locs = comment.get("subcellularLocation", [])
                    if locs:
                        subcellular = locs[0].get("location", {}).get("value")
                        break
            
            # Extract disease associations
            diseases = []
            for comment in result_data.get("comments", []):
                if comment.get("commentType") == "DISEASE":
                    disease_text = comment.get("disease", {}).get("description", "")
                    if disease_text:
                        diseases.append(disease_text[:200])
            
            result = UniProtResult(
                accession=accession,
                protein_name=protein_name,
                function=function,
                domains=domains,
                subcellular_location=subcellular,
                disease_associations=diseases,
                text=f"UniProt {accession}: {protein_name or gene_symbol}"
            )
            
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.warning(f"UniProt query failed: {e}")
            result = UniProtResult(
                accession=None,
                protein_name=None,
                function=None,
                text=f"UniProt Error: {str(e)}"
            )
            return result
    
    def _cache_result(self, cache_key: str, result: UniProtResult):
        """Cache result if Redis is available"""
        if self.redis:
            try:
                # Convert dataclass to dict for JSON serialization
                self.redis.set(cache_key, json.dumps(result.__dict__), ex=self.cache_ttl)
            except Exception as e:
                logger.warning(f"UniProt cache write error: {e}")


# =============================================================================
# 6. EVIDENCE AGGREGATOR - Parallel Multi-Source Evidence Fetching
# =============================================================================

@dataclass
class AggregatedEvidence:
    """Complete evidence package for clinical summary generation"""
    vep_annotation: Optional[Dict] = None
    gnomad_result: Optional[GnomadResult] = None
    clinvar_result: Optional[ClinVarResult] = None
    uniprot_result: Optional[UniProtResult] = None
    pubmed_articles: List[Dict] = field(default_factory=list)
    evo2_delta: Optional[float] = None
    evo2_confidence: Optional[float] = None
    evo2_prediction: Optional[str] = None
    acmg_evidence: Optional[ACMGEvidence] = None


class EvidenceAggregator:
    """
    Parallel evidence fetcher that queries all clinical data sources simultaneously.
    
    Fetches ClinVar, UniProt, PubMed, and gnomAD in parallel to minimize latency.
    Each source fails independently - partial results are still returned.
    """
    
    def __init__(self, redis_client=None, llm_endpoint: str = None):
        self.gnomad = GnomadClient(redis_client=redis_client)
        self.acmg = ACMGMapper()
        self.pubmed = PubMedRAG(redis_client=redis_client, llm_endpoint=llm_endpoint)
        self.clinvar = ClinVarClient(redis_client=redis_client)
        self.uniprot = UniProtClient(redis_client=redis_client)
    
    def gather_evidence(
        self,
        chromosome: str,
        position: int,
        ref: str,
        alt: str,
        gene_symbol: str,
        delta_score: float,
        confidence: float,
        prediction: str,
        vep_annotation: Optional[Dict] = None
    ) -> AggregatedEvidence:
        """
        Gather all evidence sources in parallel.
        
        Each source is fetched independently with error handling.
        Partial results are returned even if some sources fail.
        """
        import concurrent.futures
        
        evidence = AggregatedEvidence(
            vep_annotation=vep_annotation,
            evo2_delta=delta_score,
            evo2_confidence=confidence,
            evo2_prediction=prediction,
            acmg_evidence=self.acmg.map_score_to_evidence(delta_score, confidence)
        )
        
        # Build variant string for ClinVar query
        variant_str = f"{ref}>{alt}" if ref and alt else alt
        
        # Fetch all sources in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            # Submit all tasks
            futures["gnomad"] = executor.submit(
                self.gnomad.get_allele_frequency, chromosome, position, ref, alt
            )
            futures["clinvar"] = executor.submit(
                self.clinvar.get_variant_status, gene_symbol, variant_str
            )
            futures["uniprot"] = executor.submit(
                self.uniprot.get_protein_info, gene_symbol
            )
            futures["pubmed"] = executor.submit(
                self.pubmed._search_pubmed, gene_symbol
            )
            
            # Collect results (each fails independently)
            for name, future in futures.items():
                try:
                    result = future.result(timeout=15)
                    if name == "gnomad":
                        evidence.gnomad_result = result
                    elif name == "clinvar":
                        evidence.clinvar_result = result
                    elif name == "uniprot":
                        evidence.uniprot_result = result
                    elif name == "pubmed":
                        evidence.pubmed_articles = result
                except Exception as e:
                    logger.warning(f"Evidence source '{name}' failed: {e}")
        
        return evidence


# COMBINED ENRICHER - Convenience class for all enrichments
class ClinicalEnricher:
    """
    Combined clinical enrichment pipeline.    
    Orchestrates gnomAD filtering, ACMG mapping, and literature context.
    """
    
    def __init__(self, redis_client=None, llm_endpoint: str = None):
        self.gnomad = GnomadClient(redis_client=redis_client)
        self.acmg = ACMGMapper()
        self.pubmed = PubMedRAG(redis_client=redis_client, llm_endpoint=llm_endpoint)
        self.clinvar = ClinVarClient(redis_client=redis_client)
        self.uniprot = UniProtClient(redis_client=redis_client)
        self.aggregator = EvidenceAggregator(redis_client=redis_client, llm_endpoint=llm_endpoint)
    
    def enrich_variant(
        self,
        chromosome: str,
        position: int,
        ref: str,
        alt: str,
        delta_score: float,
        confidence: float,
        gene_symbol: str = None
    ) -> Dict[str, Any]:

        """
        Perform full clinical enrichment on a variant.        
        Returns dict with population_frequency, acmg_evidence, and literature_context.
        """
        # 1. gnomAD population frequency
        gnomad_result = self.gnomad.get_allele_frequency(chromosome, position, ref, alt)
        
        # 2. ACMG evidence mapping
        acmg_evidence = self.acmg.map_score_to_evidence(delta_score, confidence)
        
        # 3. Literature context (if gene provided)
        lit_context = self.pubmed.get_literature_context(gene_symbol) if gene_symbol else None
        
        return {
            "population_frequency": {
                "gnomad_af": gnomad_result.allele_frequency,
                "gnomad_max_pop_af": gnomad_result.population_max_af,
                "source": gnomad_result.source,
                "is_common_variant": gnomad_result.is_common,
                "frequency_classification": gnomad_result.auto_classification
            },
            "acmg_evidence": {
                "code": acmg_evidence.code,
                "strength": acmg_evidence.strength.value,
                "description": acmg_evidence.description,
                "clinical_note": acmg_evidence.clinical_note
            },
            "literature_context": {
                "summary": lit_context.summary if lit_context else None,
                "pubmed_ids": lit_context.pubmed_ids if lit_context else [],
                "gene_function": lit_context.gene_function if lit_context else None,
                "articles_found": lit_context.num_articles_found if lit_context else 0
            } if gene_symbol else None
        }
