"""Debug consensus engine on Modal."""
import modal

app = modal.App("consensus-debug")
image = modal.Image.debian_slim().pip_install("requests")
image = image.add_local_file("backend/phase1_implementation/consensus_engine.py", remote_path="/root/consensus_engine.py")

@app.function(image=image, timeout=60)
def test_consensus():
    import sys
    sys.path.insert(0, "/root")
    try:
        from consensus_engine import ConsensusEngine
        engine = ConsensusEngine()
        result = engine.compute_consensus(
            evo2_prediction="Uncertain significance",
            evo2_confidence=0.15,
            alphamissense_score=0.1359,
            alphamissense_confidence="benign",
            cadd_phred=None,
            gene_symbol="BRCA1",
            variant_str="T>C"
        )
        d = engine.to_dict(result)
        print(f"Consensus: {d['consensus_classification']}")
        print(f"Models: {d['models_agree']}/{d['models_total']}")
        print(f"Predictions: {d['predictions']}")
        return d
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        return {"error": str(e)}

@app.local_entrypoint()
def main():
    result = test_consensus.remote()
    print(result)
