from rdkit import Chem

def smiles_to_formula(smiles):
    """Преобразует SMILES в молекулярную формулу"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.rdMolDescriptors.CalcMolFormula(mol)
    except:
        return None