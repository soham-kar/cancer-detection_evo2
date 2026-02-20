"""
Clinical Explanation Module

Generates human-readable clinical reports from model predictions.
Designed for integration with clinical decision support systems.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


# Biological interpretation mapping
PATHWAY_INTERPRETATIONS = {
    'HALLMARK_G2M_CHECKPOINT': {
        'short': 'Cell cycle dysregulation',
        'long': 'High cell cycle activity suggests rapidly dividing tumor cells with aggressive proliferation',
        'clinical': 'Associated with poor prognosis, may benefit from cell cycle inhibitors'
    },
    'HALLMARK_MYC_TARGETS_V1': {
        'short': 'MYC oncogene activation',
        'long': 'MYC transcription factor drives uncontrolled cell growth and metabolism',
        'clinical': 'Aggressive phenotype, consider intensified treatment'
    },
    'HALLMARK_MYC_TARGETS_V2': {
        'short': 'MYC pathway (secondary)',
        'long': 'Additional MYC-regulated genes contributing to proliferation',
        'clinical': 'Supports MYC-driven tumor biology'
    },
    'HALLMARK_E2F_TARGETS': {
        'short': 'E2F cell cycle activation',
        'long': 'E2F transcription factors drive G1/S transition and DNA replication',
        'clinical': 'Indicates active cell division, poor prognosis marker'
    },
    'HALLMARK_DNA_REPAIR': {
        'short': 'DNA repair deficiency',
        'long': 'Genomic instability from DNA repair pathway alterations',
        'clinical': 'May respond to PARP inhibitors or platinum-based chemotherapy'
    },
    'HALLMARK_HYPOXIA': {
        'short': 'Hypoxic tumor',
        'long': 'Poor blood supply creates aggressive hypoxic microenvironment',
        'clinical': 'Associated with radiation resistance, consider hypoxia-targeted therapy'
    },
    'HALLMARK_APOPTOSIS': {
        'short': 'Apoptosis resistance',
        'long': 'Tumor cells evade programmed cell death',
        'clinical': 'May benefit from BH3 mimetics or combination therapy'
    },
    'HALLMARK_INFLAMMATORY_RESPONSE': {
        'short': 'Inflammatory signature',
        'long': 'Active immune/inflammatory response in tumor microenvironment',
        'clinical': 'May indicate immunotherapy response potential'
    },
    'HALLMARK_XENOBIOTIC_METABOLISM': {
        'short': 'Drug/toxin metabolism',
        'long': 'Active xenobiotic metabolism pathway (tobacco/alcohol exposure)',
        'clinical': 'May affect drug metabolism, consider dosing adjustments'
    },
    'HALLMARK_MTORC1_SIGNALING': {
        'short': 'mTOR pathway activation',
        'long': 'Nutrient sensing and growth signaling through mTOR',
        'clinical': 'May respond to mTOR inhibitors (everolimus, temsirolimus)'
    },
    'HALLMARK_GLYCOLYSIS': {
        'short': 'Warburg effect',
        'long': 'Aerobic glycolysis provides energy for rapid tumor growth',
        'clinical': 'Metabolic reprogramming, aggressive phenotype'
    }
}


def get_pathway_interpretation(pathway_name: str) -> Dict[str, str]:
    """Get biological interpretation for a pathway."""
    for key in PATHWAY_INTERPRETATIONS:
        if key in pathway_name:
            return PATHWAY_INTERPRETATIONS[key]
    return {
        'short': 'Unknown pathway',
        'long': f'Pathway {pathway_name} detected',
        'clinical': 'Consult oncology literature for interpretation'
    }


def generate_clinical_report(
    model,
    patient_expression: np.ndarray,
    pathway_names: List[str],
    patient_id: Optional[str] = None,
    gene_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate human-readable clinical report for a single patient.
    
    Args:
        model: Trained HybridPathwayMLP model
        patient_expression: Gene expression vector (n_genes,)
        pathway_names: List of pathway names
        patient_id: Optional patient identifier
        gene_names: Optional list of gene names
        
    Returns:
        Dictionary with structured clinical report
    """
    model.eval()
    x = torch.from_numpy(patient_expression).float().unsqueeze(0)
    
    with torch.no_grad():
        output = model(x, return_interpretation=True)
        
        # Risk score (convert to probability range)
        raw_risk = output['risk_score'].item()
        risk_prob = 1 / (1 + np.exp(-raw_risk))  # Sigmoid
        
        # Pathway importance
        pathway_imp = output['pathway_importance'][0].numpy()
        
        # Risk stratification
        if risk_prob > 0.7:
            risk_category = "HIGH"
            risk_color = "🔴"
            recommendation = "Consider aggressive treatment approach"
            actions = [
                "Recommend multidisciplinary tumor board review",
                "Consider adjuvant chemotherapy",
                "Schedule 3-month surveillance imaging"
            ]
        elif risk_prob > 0.4:
            risk_category = "INTERMEDIATE"
            risk_color = "🟡"
            recommendation = "Standard care with enhanced surveillance"
            actions = [
                "Follow standard treatment guidelines",
                "Schedule 6-month surveillance",
                "Consider clinical trial eligibility"
            ]
        else:
            risk_category = "LOW"
            risk_color = "🟢"
            recommendation = "Standard protocol"
            actions = [
                "Follow standard care pathway",
                "Annual surveillance sufficient"
            ]
        
        # Top pathways analysis
        top_idx = np.argsort(pathway_imp)[-5:][::-1]
        
        primary_pathway = pathway_names[top_idx[0]]
        primary_interp = get_pathway_interpretation(primary_pathway)
        
        secondary_pathways = []
        for idx in top_idx[1:4]:
            if pathway_imp[idx] > 0.001:  # Only include meaningful pathways
                interp = get_pathway_interpretation(pathway_names[idx])
                secondary_pathways.append({
                    'name': pathway_names[idx].replace('HALLMARK_', ''),
                    'importance': f"{pathway_imp[idx]:.1%}",
                    'interpretation': interp['short']
                })
        
        report = {
            'patient_id': patient_id or 'Unknown',
            'risk_assessment': {
                'score': f"{risk_prob:.3f}",
                'category': risk_category,
                'indicator': risk_color,
                'percentile': f"{int(risk_prob * 100)}th percentile"
            },
            'primary_driver': {
                'pathway': primary_pathway.replace('HALLMARK_', ''),
                'importance': f"{pathway_imp[top_idx[0]]:.1%}",
                'biological_meaning': primary_interp['long'],
                'clinical_significance': primary_interp['clinical']
            },
            'secondary_factors': secondary_pathways,
            'clinical_recommendation': recommendation,
            'suggested_actions': actions,
            'model_confidence': 'High' if pathway_imp[top_idx[0]] > 0.5 else 'Medium',
            'disclaimer': 'This report is for research purposes only. Clinical decisions should be made by qualified healthcare providers.'
        }
        
        return report


def format_report_text(report: Dict[str, Any]) -> str:
    """Format report as human-readable text."""
    lines = [
        "=" * 60,
        "ORAL CANCER RISK ASSESSMENT REPORT",
        "=" * 60,
        "",
        f"Patient ID: {report['patient_id']}",
        "",
        "--- RISK ASSESSMENT ---",
        f"Risk Score: {report['risk_assessment']['score']} {report['risk_assessment']['indicator']}",
        f"Category: {report['risk_assessment']['category']}",
        f"Percentile: {report['risk_assessment']['percentile']}",
        "",
        "--- PRIMARY BIOLOGICAL DRIVER ---",
        f"Pathway: {report['primary_driver']['pathway']}",
        f"Confidence: {report['primary_driver']['importance']}",
        f"Meaning: {report['primary_driver']['biological_meaning']}",
        f"Clinical: {report['primary_driver']['clinical_significance']}",
        "",
    ]
    
    if report['secondary_factors']:
        lines.append("--- SECONDARY FACTORS ---")
        for factor in report['secondary_factors']:
            lines.append(f"  • {factor['name']} ({factor['importance']}): {factor['interpretation']}")
        lines.append("")
    
    lines.extend([
        "--- RECOMMENDATIONS ---",
        f"{report['clinical_recommendation']}",
        "",
        "Suggested Actions:",
    ])
    
    for action in report['suggested_actions']:
        lines.append(f"  • {action}")
    
    lines.extend([
        "",
        "-" * 60,
        f"Model Confidence: {report['model_confidence']}",
        "",
        f"⚠️ {report['disclaimer']}",
        "=" * 60
    ])
    
    return "\n".join(lines)


def batch_predict(
    model,
    X: np.ndarray,
    pathway_names: List[str],
    patient_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Generate reports for multiple patients."""
    reports = []
    
    for i in range(len(X)):
        patient_id = patient_ids[i] if patient_ids else f"Patient_{i+1}"
        report = generate_clinical_report(
            model, X[i], pathway_names, patient_id
        )
        reports.append(report)
    
    return reports
