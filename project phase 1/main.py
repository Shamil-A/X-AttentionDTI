import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import pandas as pd
import torch.nn.functional as F

# Import V2 modules
from models.fusion_model import DrugTargetFusionModel
from models.gatv2 import GATv2Encoder
from models.cnn_protein import ProteinCNN
from models.cross_attention import CrossModalAttention
from utils import TestbedDataset, rmse, mse, pearson, spearman, ci

loss_fn = nn.MSELoss()  # Global loss function

# --- TRAINING FUNCTION ---
def train_epoch(model, device, data_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        prot_seq = data.target.to(device).long()
        optimizer.zero_grad()
        output = model(data, prot_seq)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(data_loader)}] Loss: {loss.item():.6f}")

# --- EVALUATION FUNCTION ---
def evaluate_model(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            prot_seq = data.target.to(device)
            output = model(data, prot_seq)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

# --- MODULE CHECKER FUNCTION ---
def module_checker():
    print("🔍 Checking all submodules...")
    B, N, F = 4, 10, 78
    dummy_graph = type("Dummy", (), {})()
    dummy_graph.x = torch.rand(B*N, F)
    dummy_graph.edge_index = torch.tensor([[0,1],[1,2]], dtype=torch.long).t()
    dummy_graph.batch = torch.tensor([0,0,1,1]* (B*N//4))

    dummy_seq = torch.randint(0, 26, (B, 100))

    gat = GATv2Encoder()
    prot = CNN = ProteinCNN()
    attn = CrossModalAttention()

    try:
        out_gat = gat(dummy_graph)

        out_prot = prot(dummy_seq)
        out_attn, _ = attn(out_gat, out_prot)
        print("📐 out_gat shape:", out_gat.shape)
        print("📐 out_prot shape:", out_prot.shape)
        print("📐 out_attn shape:", out_attn.shape)

        print("✅ All modules work correctly")
        print("✅ All modules work correctly")
    except Exception as e:
        print("❌ Module check failed:", e)

# --- MAIN FUNCTION ---
def main():
    datasets = ['kiba']
    model_class = DrugTargetFusionModel
    model_name = model_class.__name__
    cuda_device = "cuda:0"

    train_batch_size = 64
    test_batch_size = 128
    learning_rate = 1e-3
    num_epochs = 100
    patience = 10
    log_interval = 100


    print("Running module check...")
    module_checker()  # Validate modules before training
    print(num_epochs)

    for dataset in datasets:
        print(f"\nRunning on {model_name}_{dataset}")
        train_file = f'data/processed/{dataset}_train.pt'
        test_file = f'data/processed/{dataset}_test.pt'
        if not (os.path.isfile(train_file) and os.path.isfile(test_file)):
            print("❌ Preprocessed data not found. Run create_data.py first!")
            continue

        full_train_data = TestbedDataset(root='data', dataset=f'{dataset}_train')
        test_data = TestbedDataset(root='data', dataset=f'{dataset}_test')

        train_size = int(0.8 * len(full_train_data))
        valid_size = len(full_train_data) - train_size
        train_data, valid_data = random_split(full_train_data, [train_size, valid_size])

        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

        device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_valid_mse = float('inf')
        best_epoch = -1
        epochs_no_improve = 0
        model_file = f'model_{model_name}_{dataset}.model'
        result_file = f'result_{model_name}_{dataset}.csv'
        results_history = []

        for epoch in range(1, num_epochs+1):
            train_epoch(model, device, train_loader, optimizer, epoch, log_interval)
            valid_labels, valid_preds = evaluate_model(model, device, valid_loader)
            valid_mse = mse(valid_labels, valid_preds)

            if valid_mse < best_valid_mse:
                best_valid_mse = valid_mse
                best_epoch = epoch
                torch.save(model.state_dict(), model_file)
                epochs_no_improve = 0
                test_labels, test_preds = evaluate_model(model, device, test_loader)
                test_metrics = [rmse(test_labels, test_preds), mse(test_labels, test_preds),
                                pearson(test_labels, test_preds), spearman(test_labels, test_preds),
                                ci(test_labels, test_preds)]
                with open(result_file, 'w') as f:
                    f.write('rmse,mse,pearson,spearman,ci\n')
                    f.write(','.join(map(str, test_metrics)))
                epoch_results = {'epoch': epoch, 'valid_mse': valid_mse}
                epoch_results.update({'test_rmse': test_metrics[0], 'test_mse': test_metrics[1],
                                      'test_pearson': test_metrics[2], 'test_spearman': test_metrics[3],
                                      'test_ci': test_metrics[4]})
                results_history.append(epoch_results)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        history_df = pd.DataFrame(results_history)
        history_df.to_csv(f'full_results_history_{model_name}_{dataset}.csv', index=False)
        print(f"✅ Training finished. Results saved.")
        
if __name__ == "__main__":
    main()
