from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_druglikeness(molecule):
    """
    Calculate drug-likeness metrics for a molecule.
    
    Args:
        molecule (rdkit.Chem.rdchem.Mol): The molecule to analyze
        
    Returns:
        dict: Dictionary containing drug-likeness metrics
    """
    try:
        # Calculate Lipinski's Rule of Five violations
        mol_weight = Descriptors.MolWt(molecule)
        logp = Descriptors.MolLogP(molecule)
        h_donors = Descriptors.NumHDonors(molecule)
        h_acceptors = Descriptors.NumHAcceptors(molecule)
        
        violations = 0
        
        if mol_weight > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if h_donors > 5:
            violations += 1
        if h_acceptors > 10:
            violations += 1
        
        # Calculate Quantitative Estimate of Drug-likeness (QED)
        qed_value = QED.qed(molecule)
        
        # Calculate Veber rule compliance
        rotatable_bonds = Descriptors.NumRotatableBonds(molecule)
        tpsa = Descriptors.TPSA(molecule)
        veber_compliant = rotatable_bonds <= 10 and tpsa <= 140
        
        # Calculate Ghose filter compliance
        ghose_compliant = (
            160 <= mol_weight <= 480 and
            -0.4 <= logp <= 5.6 and
            20 <= molecule.GetNumAtoms() <= 70
        )
        
        # Calculate Muegge filter compliance
        muegge_compliant = (
            200 <= mol_weight <= 600 and
            -2 <= logp <= 5 and
            h_donors <= 5 and
            h_acceptors <= 10 and
            rotatable_bonds <= 15 and
            molecule.GetNumAtoms() <= 100
        )
        
        # Calculate custom drug-likeness score (0-1 scale)
        # This is a simplified score combining several factors
        rule_of_five_score = max(0, 1 - (violations * 0.25))
        
        # Normalize properties to 0-1 scale
        mw_score = max(0, min(1, 1 - abs(mol_weight - 350) / 250))  # Optimal around 350
        logp_score = max(0, min(1, 1 - abs(logp - 2.5) / 4))        # Optimal around 2.5
        h_donors_score = max(0, min(1, 1 - h_donors / 6))
        h_acceptors_score = max(0, min(1, 1 - h_acceptors / 11))
        rotatable_score = max(0, min(1, 1 - rotatable_bonds / 11))
        
        # Combined score with weights
        druglikeness_score = (
            0.25 * rule_of_five_score +
            0.25 * qed_value +
            0.15 * mw_score +
            0.15 * logp_score +
            0.10 * (h_donors_score + h_acceptors_score) / 2 +
            0.10 * rotatable_score
        )
        
        return {
            "lipinski_violations": violations,
            "druglikeness_score": druglikeness_score,
            "qed_value": qed_value,
            "veber_compliant": veber_compliant,
            "ghose_compliant": ghose_compliant,
            "muegge_compliant": muegge_compliant
        }
    
    except Exception as e:
        logger.error(f"Error calculating drug-likeness: {str(e)}")
        return {
            "lipinski_violations": 0,
            "druglikeness_score": 0.0,
            "qed_value": 0.0,
            "veber_compliant": False,
            "ghose_compliant": False,
            "muegge_compliant": False
        }

def check_lead_likeness(molecule):
    """
    Check if a molecule is lead-like according to standard criteria.
    
    Args:
        molecule (rdkit.Chem.rdchem.Mol): The molecule to analyze
        
    Returns:
        bool: True if the molecule is lead-like, False otherwise
    """
    try:
        # Calculate relevant properties
        mol_weight = Descriptors.MolWt(molecule)
        logp = Descriptors.MolLogP(molecule)
        rotatable_bonds = Descriptors.NumRotatableBonds(molecule)
        
        # Lead-likeness criteria:
        # - Molecular weight between 200 and 350 Da
        # - LogP between -1 and 3
        # - Rotatable bonds <= 7
        lead_like = (
            200 <= mol_weight <= 350 and
            -1 <= logp <= 3 and
            rotatable_bonds <= 7
        )
        
        return lead_like
    
    except Exception as e:
        logger.error(f"Error checking lead-likeness: {str(e)}")
        return False

def check_fragment_likeness(molecule):
    """
    Check if a molecule is fragment-like according to the Rule of Three.
    
    Args:
        molecule (rdkit.Chem.rdchem.Mol): The molecule to analyze
        
    Returns:
        bool: True if the molecule is fragment-like, False otherwise
    """
    try:
        # Calculate relevant properties
        mol_weight = Descriptors.MolWt(molecule)
        logp = Descriptors.MolLogP(molecule)
        h_donors = Descriptors.NumHDonors(molecule)
        h_acceptors = Descriptors.NumHAcceptors(molecule)
        rotatable_bonds = Descriptors.NumRotatableBonds(molecule)
        tpsa = Descriptors.TPSA(molecule)
        
        # Rule of Three criteria:
        # - Molecular weight <= 300 Da
        # - LogP <= 3
        # - H-bond donors <= 3
        # - H-bond acceptors <= 3
        # - Rotatable bonds <= 3
        # - TPSA <= 60 Å²
        fragment_like = (
            mol_weight <= 300 and
            logp <= 3 and
            h_donors <= 3 and
            h_acceptors <= 3 and
            rotatable_bonds <= 3 and
            tpsa <= 60
        )
        
        return fragment_like
    
    except Exception as e:
        logger.error(f"Error checking fragment-likeness: {str(e)}")
        return False
