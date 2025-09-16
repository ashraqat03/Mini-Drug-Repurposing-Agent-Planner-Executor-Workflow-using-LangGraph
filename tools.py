import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Molecular descriptor setup
descriptor_list = [
    'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
    'NumRotatableBonds', 'TPSA', 'HeavyAtomCount',
    'RingCount', 'FractionCSP3'
]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_list)

def compute_descriptors_pro(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptors = calculator.CalcDescriptors(mol)
    return np.array(descriptors)

def find_targets(disease_name):
    try:
        df = pd.read_csv('data/gene_disease.csv')
        results_df = df[df['disease_name'].str.contains(disease_name, case=False, na=False)]
        return {"targets": results_df.to_dict('records')} if not results_df.empty else {"error": "No targets found"}
    except Exception as e:
        return {"error": f"Failed to read gene-disease data: {str(e)}"}

def find_compounds(target_genes):
    try:
        df = pd.read_csv('data/drug_target.csv')
        if isinstance(target_genes, str):
            target_genes = [target_genes]
        results_df = df[df['target_gene'].isin(target_genes)]
        return {"compounds": results_df.to_dict('records')} if not results_df.empty else {"error": "No compounds found"}
    except Exception as e:
        return {"error": f"Failed to read drug-target data: {str(e)}"}

def predict_activity(smiles_list):
    try:
        model = joblib.load('models/qsar_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        predictions = []
        for smiles in smiles_list:
            desc_vector = compute_descriptors_pro(smiles)
            if desc_vector is not None:
                desc_vector_scaled = scaler.transform(desc_vector.reshape(1, -1))
                proba = model.predict_proba(desc_vector_scaled)[0][1]
                predictions.append(float(round(proba, 2)))
            else:
                predictions.append(0.0)
        return {"predictions": predictions}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
