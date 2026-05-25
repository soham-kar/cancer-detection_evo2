import { NextRequest, NextResponse } from "next/server";

/**
 * PDB Proxy API Route
 * 
 * Fetches AlphaFold structures server-side to bypass browser CORS restrictions.
 * Uses the AlphaFold API to discover the correct model URL dynamically,
 * then fetches the structure file (MMCIF or PDB format).
 * 
 * GET /api/pdb-proxy?uniprotId=P38398
 * Returns: Structure file content as text
 */

interface AlphaFoldApiResponse {
    uniprot_entry: {
        ac: string;
        id: string;
        sequence_length: number;
    };
    structures: Array<{
        summary: {
            model_identifier: string;
            model_url: string;
            model_format: string;
            model_page_url: string;
            confidence_avg_local_score: number;
        };
    }>;
}

export async function GET(request: NextRequest) {
    const { searchParams } = new URL(request.url);
    const uniprotId = searchParams.get("uniprotId");

    if (!uniprotId) {
        return NextResponse.json(
            { error: "Missing uniprotId parameter" },
            { status: 400 }
        );
    }

    // Validate UniProt ID format
    if (!/^[A-NR-Z][0-9][A-Z]{2}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$/.test(uniprotId)) {
        return NextResponse.json(
            { error: "Invalid UniProt ID format" },
            { status: 400 }
        );
    }

    try {
        // Step 1: Query AlphaFold API to get the correct model URL
        const apiUrl = `https://alphafold.ebi.ac.uk/api/uniprot/summary/${uniprotId}.json`;
        console.log(`[PDB-Proxy] Querying AlphaFold API: ${apiUrl}`);

        const apiResponse = await fetch(apiUrl, {
            headers: {
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            },
            signal: AbortSignal.timeout(10000),
        });

        if (!apiResponse.ok) {
            console.warn(`[PDB-Proxy] AlphaFold API returned ${apiResponse.status}`);
            return NextResponse.json(
                { error: `AlphaFold has no structure for ${uniprotId}` },
                { status: 404 }
            );
        }

        const apiData: AlphaFoldApiResponse = await apiResponse.json();
        
        if (!apiData.structures || apiData.structures.length === 0) {
            return NextResponse.json(
                { error: `No AlphaFold structures available for ${uniprotId}` },
                { status: 404 }
            );
        }

        // Get the best structure (first one, usually the highest quality)
        const structure = apiData.structures[0]!;
        const modelUrl = structure.summary.model_url;
        const modelFormat = structure.summary.model_format; // "MMCIF" or "PDB"
        const confidence = structure.summary.confidence_avg_local_score;

        console.log(`[PDB-Proxy] Found ${modelFormat} structure for ${uniprotId}: ${modelUrl} (confidence: ${confidence})`);

        // Step 2: Fetch the actual structure file
        const fileResponse = await fetch(modelUrl, {
            headers: {
                "Accept": "*/*",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            },
            signal: AbortSignal.timeout(15000),
        });

        if (!fileResponse.ok) {
            return NextResponse.json(
                { error: `Failed to fetch structure file: ${fileResponse.status}` },
                { status: 502 }
            );
        }

        const structureData = await fileResponse.text();

        // Validate content
        const hasAtoms = structureData.includes("ATOM");
        const hasHeader = structureData.includes("HEADER") || structureData.includes("data_");
        
        if (!hasAtoms && !hasHeader) {
            return NextResponse.json(
                { error: "Invalid structure file content" },
                { status: 502 }
            );
        }

        console.log(`[PDB-Proxy] Successfully fetched ${structureData.length} bytes (${modelFormat}) for ${uniprotId}`);

        // Return with appropriate content type
        const contentType = modelFormat === "MMCIF" ? "chemical/x-mmcif" : "chemical/x-pdb";

        return new NextResponse(structureData, {
            headers: {
                "Content-Type": contentType,
                "X-Model-Format": modelFormat,
                "X-Confidence-Score": String(confidence),
                "Cache-Control": "public, max-age=86400",
            },
        });

    } catch (error) {
        console.error("[PDB-Proxy] Error:", error);
        return NextResponse.json(
            { error: "Failed to fetch structure from AlphaFold" },
            { status: 502 }
        );
    }
}
