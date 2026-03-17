import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import csv
import random
import time # Import the time module
from rdkit import Chem

# Assuming the model definitions are in a 'models' subdirectory
# Make sure these files are in ./models/
from models.fusion_model import DrugTargetFusionModel
from models.gatv2 import GATv2Encoder
from models.cnn_protein import ProteinCNN

# --- 1. CONVERSION LOGIC (from preprocessing script) ---

# Helper function to get atom features
def get_atom_features(atom):
    """
    Creates a feature vector for a single atom.
    This must match the features used during training.
    Total features = 78
    """
    # Features are:
    # 1. One-hot encoding of atom symbol (44)
    # 2. One-hot encoding of atom degree (11)
    # 3. One-hot encoding of formal charge (11)
    # 4. One-hot encoding of hybridization (7)
    # 5. Aromaticity flag (1)
    # 6. Number of hydrogens (4)
    possible_atom = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    atom_symbol = [0] * len(possible_atom)
    try:
        atom_symbol[possible_atom.index(atom.GetSymbol())] = 1
    except:
        atom_symbol[-1] = 1 # Unknown

    atom_degree = [0] * 11
    atom_degree[atom.GetDegree()] = 1

    formal_charge = [0] * 11
    formal_charge[atom.GetFormalCharge() + 5] = 1 # Shift to be non-negative

    hybridization_type = [0] * 7
    hybridization_type[int(atom.GetHybridization())] = 1

    is_aromatic = [1] if atom.GetIsAromatic() else [0]
    
    num_hydrogens = [0] * 4
    num_hydrogens[atom.GetTotalNumHs()] = 1


    return torch.tensor(atom_symbol + atom_degree + formal_charge + hybridization_type + is_aromatic + num_hydrogens, dtype=torch.float)

def smiles_to_graph(smiles_string):
    """Converts a SMILES string to a PyG Data object."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    # Get atom features
    atom_features_list = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_features_list)

    # Get bond connectivity
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append((i, j))
        edge_indices.append((j, i))

    # Handle molecules with no bonds
    if not edge_indices:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    graph = Data(x=x, edge_index=edge_index)
    return graph


# Protein sequence conversion
VOCAB = "ACDEFGHIKLMNPQRSTVWY?"
CHAR_TO_INT = {char: i for i, char in enumerate(VOCAB)}

def protein_to_tensor(sequence, max_len=1000):
    """Converts a protein sequence to a tensor of integer indices."""
    sequence = sequence[:max_len] # Truncate if necessary
    indices = [CHAR_TO_INT.get(char, len(VOCAB) - 1) for char in sequence]
    # Pad if necessary
    while len(indices) < max_len:
        indices.append(len(VOCAB) -1) # Using '?' as padding token
    
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)


# --- 2. PREDICTION SCRIPT ---

def predict_affinity(model, device, drug_smiles, protein_sequence):
    """
    Takes a trained model and raw data, performs conversion, and returns a prediction.
    """
    model.eval() # Set the model to evaluation mode

    # Convert raw data to tensors
    drug_graph = smiles_to_graph(drug_smiles)
    if drug_graph is None:
        print("Error: Invalid drug SMILES string.")
        return None
    # The model expects a batch, so we create a batch of size 1
    drug_batch = Batch.from_data_list([drug_graph]).to(device)

    protein_tensor = protein_to_tensor(protein_sequence).to(device)

    # Perform prediction
    with torch.no_grad():
        prediction = model(drug_batch, protein_tensor)
        
    return prediction.item()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    MODEL_PATH = 'model_DrugTargetFusionModel_kiba.model'
    CSV_FILE = 'kiba_test.csv'
    
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model architecture
    model = DrugTargetFusionModel().to(device)

    # Load saved model weights
    try:
        print(f"Loading model weights from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{MODEL_PATH}'. Please ensure the file is in the correct directory.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit()
        
    # Get a random sample from the CSV file
    try:
        # *** THIS IS THE MODIFIED PART ***
        # Explicitly seed the random generator with the current time
        # to ensure a different random choice on each run.
        random.seed(time.time())

        with open(CSV_FILE, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            all_rows = list(reader)
            if not all_rows:
                print("ERROR: CSV file is empty.")
                exit()
            
            random_row = random.choice(all_rows) # Select one random row
            
            sample_smiles = random_row[0]
            sample_protein = random_row[1]
            actual_affinity = float(random_row[2])

    except FileNotFoundError:
        print(f"ERROR: Test data file not found at '{CSV_FILE}'.")
        exit()
    except (IndexError, ValueError) as e:
        print(f"ERROR: Problem reading the data from the chosen row: {e}")
        exit()


    print("\n--- Running Prediction on a Random Sample ---")
    print(f"Drug SMILES: {sample_smiles[:50]}...")
    print(f"Protein Sequence: {sample_protein[:50]}...")
    
    # Get the prediction
    predicted_affinity = predict_affinity(model, device, sample_smiles, sample_protein)

    if predicted_affinity is not None:
        print(f"\nPredicted Affinity Score: {predicted_affinity:.4f}")
        print(f"Actual Affinity Score:    {actual_affinity:.4f}")

