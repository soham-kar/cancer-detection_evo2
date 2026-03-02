import modal

try:
    rag_cls = modal.Cls.from_name('pubmed-rag', 'SmartPubMedRAG')
    print("✅ pubmed-rag IS deployed")
    
    result = rag_cls().search_and_summarize.remote('BRCA1', 'c.5266dupC')
    print(f"\n✅ Test successful!")
    print(f"Level: {result.get('level')}")
    print(f"Found: {result.get('found')}")
    print(f"PMIDs: {result.get('pmids', [])}")
    print(f"Summary length: {len(result.get('summary', ''))}")
    
except Exception as e:
    print(f"❌ pubmed-rag NOT deployed or not accessible")
    print(f"Error: {type(e).__name__}: {str(e)}")
