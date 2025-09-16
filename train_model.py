import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import re

def compute_descriptors_pro(smiles):
    """
    Simple descriptor calculation without RDKit.
    Uses basic string-based features that correlate with molecular properties.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    
    try:
        # Simple features that don't require chemical parsing
        features = [
            len(smiles),                    # Rough correlate of molecular size
            smiles.count('C'),              # Number of carbon atoms (approx)
            smiles.count('O'),              # Number of oxygen atoms (approx)  
            smiles.count('N'),              # Number of nitrogen atoms (approx)
            smiles.count('='),              # Double bonds (approx)
            smiles.count('#'),              # Triple bonds (approx)
            smiles.count('1'),              # Ring closures (approx)
            len(re.findall(r'\[.*?\]', smiles)),  # Complex atoms/groups
            smiles.count('('),              # Branching (approx)
        ]
        return np.array(features)
    except:
        return None

def find_targets(disease_name):
    """Looks up a disease in gene_disease.csv and returns associated targets."""
    try:
        df = pd.read_csv('data/gene_disease.csv')
        results_df = df[df['disease_name'].str.contains(disease_name, case=False, na=False)]
        if results_df.empty:
            return {"error": f"No targets found for disease: '{disease_name}'."}
        return {"targets": results_df.sort_values('association_score', ascending=False).to_dict('records')}
    except Exception as e:
        return {"error": f"Failed to read gene-disease data: {str(e)}"}

def find_compounds(target_genes):
    """Looks up target genes in drug_target.csv and returns associated drugs."""
    try:
        df = pd.read_csv('data/drug_target.csv')
        if isinstance(target_genes, str):
            target_genes = [target_genes]
        results_df = df[df['target_gene'].isin(target_genes)]
        if results_df.empty:
            return {"error": f"No drugs found for target gene(s): {target_genes}."}
        return {"compounds": results_df.to_dict('records')}
    except Exception as e:
        return {"error": f"Failed to read drug-target data: {str(e)}"}

def predict_activity(smiles_list):
    """Uses pre-trained QSAR model to predict activity probabilities."""
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
