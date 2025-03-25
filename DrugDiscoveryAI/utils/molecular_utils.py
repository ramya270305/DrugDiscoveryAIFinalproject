from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import base64
import io
import logging
import traceback
import numpy as np
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import drawing capabilities, but provide fallbacks
try:
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    DRAWING_AVAILABLE = True
except ImportError:
    logger.warning("RDKit drawing capabilities not available. Using fallback methods.")
    DRAWING_AVAILABLE = False

def validate_smiles(smiles_string):
    """
    Validate a SMILES string by attempting to convert it to an RDKit molecule.
    
    Args:
        smiles_string (str): The SMILES string to validate
        
    Returns:
        tuple: (is_valid, message) where is_valid is a boolean and message is a string
    """
    if not smiles_string or not isinstance(smiles_string, str):
        return False, "SMILES string must be a non-empty string"
    
    mol = Chem.MolFromSmiles(smiles_string)
    
    if mol is None:
        return False, "Invalid SMILES string format"
    
    return True, "Valid SMILES string"

def generate_molecule_from_smiles(smiles_string):
    """
    Generate an RDKit molecule object from a SMILES string.
    
    Args:
        smiles_string (str): The SMILES string
        
    Returns:
        rdkit.Chem.rdchem.Mol: The molecule object
    """
    mol = Chem.MolFromSmiles(smiles_string)
    
    if mol is None:
        raise ValueError("Failed to generate molecule from SMILES string")
    
    # Add hydrogens and generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100)
    AllChem.MMFFOptimizeMolecule(mol)
    
    return mol

def get_molecule_svg(mol, width=500, height=300):
    """
    Generate an SVG representation of a molecule.
    
    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule object
        width (int): The width of the SVG
        height (int): The height of the SVG
        
    Returns:
        str: The SVG as a string that can be embedded in HTML
    """
    try:
        if not DRAWING_AVAILABLE:
            return '<div style="text-align: center; padding: 20px; border: 1px solid #ccc;">Molecule visualization not available</div>'
            
        # Remove hydrogens for cleaner visualization
        mol = Chem.RemoveHs(mol)
        
        # Generate drawing
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        
        # Convert to HTML-embeddable format
        svg = svg.replace('svg:', '')
        svg_html = f'<div style="text-align: center; margin: 20px;">{svg}</div>'
        
        return svg_html
    except Exception as e:
        logger.error(f"Error generating molecule SVG: {str(e)}")
        logger.error(traceback.format_exc())
        return f'<div style="text-align: center; padding: 20px; border: 1px solid #ccc;">Error visualizing molecule: {str(e)}</div>'

def get_molecule_3d_data(mol):
    """
    Generate 3D coordinates and atom information for interactive 3D visualization.
    
    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule object with 3D coordinates
        
    Returns:
        dict: Dictionary with atom coordinates, elements, and bond information
    """
    try:
        # Make sure the molecule has 3D coordinates
        if not mol.GetNumConformers() or mol.GetConformer().Is3D() == False:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100)
            AllChem.MMFFOptimizeMolecule(mol)
            
        # Get atoms and coordinates
        atoms = []
        positions = []
        
        # Color mapping for common elements
        element_colors = {
            'C': '#909090',  # Gray
            'H': '#FFFFFF',  # White
            'O': '#FF0D0D',  # Red
            'N': '#3050F8',  # Blue
            'S': '#FFFF30',  # Yellow
            'Cl': '#1FF01F', # Green
            'F': '#90E050',  # Light green
            'Br': '#A62929', # Brown
            'P': '#FF8000',  # Orange
        }
        
        # Create atom data
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            element = atom.GetSymbol()
            atom_idx = atom.GetIdx()
            pos = conf.GetAtomPosition(atom_idx)
            
            # Get atom color
            color = element_colors.get(element, '#FFFFFF')
            
            atoms.append({
                'element': element,
                'index': atom_idx,
                'color': color,
                'size': 0.4 if element == 'H' else 0.6
            })
            
            positions.append([pos.x, pos.y, pos.z])
        
        # Create bond data
        bonds = []
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()
            
            bonds.append({
                'begin_atom_idx': begin_idx,
                'end_atom_idx': end_idx,
                'bond_type': bond_type
            })
        
        return {
            'atoms': atoms,
            'positions': positions,
            'bonds': bonds
        }
        
    except Exception as e:
        logger.error(f"Error generating 3D data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def get_molecule_properties(mol):
    """
    Calculate basic molecular properties using RDKit.
    
    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule object
        
    Returns:
        dict: A dictionary of molecular properties
    """
    # Remove hydrogens for consistent property calculation
    mol_no_h = Chem.RemoveHs(mol)
    
    properties = {
        "molecular_weight": Descriptors.MolWt(mol_no_h),
        "logP": Descriptors.MolLogP(mol_no_h),
        "num_hydrogen_donors": Descriptors.NumHDonors(mol_no_h),
        "num_hydrogen_acceptors": Descriptors.NumHAcceptors(mol_no_h),
        "molar_refractivity": Descriptors.MolMR(mol_no_h),
        "topological_polar_surface_area": Descriptors.TPSA(mol_no_h),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol_no_h),
        "num_atoms": mol_no_h.GetNumAtoms(),
        "num_rings": Descriptors.RingCount(mol_no_h),
        "num_aromatic_rings": Descriptors.NumAromaticRings(mol_no_h)
    }
    
    return properties

def process_batch_smiles(smiles_list):
    """
    Process a batch of SMILES strings to calculate properties and generate molecules.
    
    Args:
        smiles_list (list): List of SMILES strings to process
        
    Returns:
        list: List of dictionaries with molecule data and properties
    """
    results = []
    
    for smiles in smiles_list:
        try:
            # Validate SMILES
            is_valid, message = validate_smiles(smiles)
            
            if not is_valid:
                results.append({
                    "smiles": smiles,
                    "valid": False,
                    "error": message
                })
                continue
                
            # Generate molecule
            mol = generate_molecule_from_smiles(smiles)
            
            # Get properties
            properties = get_molecule_properties(mol)
            
            # Get toxicity
            from models.toxicity import predict_toxicity
            toxicity = predict_toxicity(mol)
            
            # Get druglikeness
            from models.druglikeness import calculate_druglikeness
            druglikeness = calculate_druglikeness(mol)
            
            # Combine results
            results.append({
                "smiles": smiles,
                "valid": True,
                "properties": properties,
                "toxicity": toxicity,
                "druglikeness": druglikeness
            })
            
        except Exception as e:
            results.append({
                "smiles": smiles,
                "valid": False,
                "error": str(e)
            })
    
    return results
