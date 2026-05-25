import { NextRequest, NextResponse } from "next/server";

/**
 * PDB Proxy API Route
 * 
 * Fetches AlphaFold PDB files server-side to bypass browser CORS restrictions.
 * The frontend calls this endpoint instead of fetching AlphaFold directly.
 * 
 * GET /api/pdb-proxy?uniprotId=P38398
 * Returns: PDB file content as text
 */

export async function GET(request: NextRequest) {
    const { searchParams } = new URL(request.url);
    const uniprotId = searchParams.get("uniprotId");

    if (!uniprotId) {
        return NextResponse.json(
            { error: "Missing uniprotId parameter" },
            { status: 400 }
        );
    }

    // Validate UniProt ID format (e.g., P38398, Q9Y6I9)
    if (!/^[A-NR-Z][0-9][A-Z]{2}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$/.test(uniprotId)) {
        return NextResponse.json(
            { error: "Invalid UniProt ID format" },
            { status: 400 }
        );
    }

    const pdbUrl = `https://alphafold.ebi.ac.uk/files/AF-${uniprotId}-F1-model_v4.pdb`;

    try {
        console.log(`[PDB-Proxy] Fetching: ${pdbUrl}`);
        
        const response = await fetch(pdbUrl, {
            headers: {
                "Accept": "text/plain",
                "User-Agent": "HelixMind-Variant-Analysis/1.0",
            },
            signal: AbortSignal.timeout(15000),
        });

        if (!response.ok) {
            console.warn(`[PDB-Proxy] AlphaFold returned ${response.status}: ${response.statusText}`);
            return NextResponse.json(
                { error: `AlphaFold PDB not found for ${uniprotId}`, status: response.status },
                { status: 404 }
            );
        }

        const pdbData = await response.text();
        
        // Validate it's actually a PDB file
        if (!pdbData.includes("ATOM") && !pdbData.includes("HEADER")) {
            return NextResponse.json(
                { error: "Invalid PDB file content" },
                { status: 502 }
            );
        }

        console.log(`[PDB-Proxy] Successfully fetched ${pdbData.length} bytes for ${uniprotId}`);

        return new NextResponse(pdbData, {
            headers: {
                "Content-Type": "chemical/x-pdb",
                "Cache-Control": "public, max-age=86400", // Cache for 24 hours
            },
        });

    } catch (error) {
        console.error("[PDB-Proxy] Error fetching PDB:", error);
        return NextResponse.json(
            { error: "Failed to fetch PDB file from AlphaFold" },
            { status: 502 }
        );
    }
}
