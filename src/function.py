import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from builder_utils import MILDataset
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np

def validate(model, loader, criterion, device="cuda", model_name="model"):
    
    model.eval()
    
    total_loss = 0
    all_probs = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for features, label in loader:
            
            features = features.to(device)
            label = label.to(device)
            
            output = model(features)
            loss = criterion(output, label)
            
            total_loss += loss.item()
            
            probs = torch.softmax(output, dim=1)
            
            positive_probs = probs[:, 1]
            preds = torch.argmax(probs, dim=1)
            
            all_probs.extend(positive_probs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    cm.figure_.savefig(f'confusion_matrix_{model_name}.png')
    
    return avg_loss, auc, cm


def train_model(model, dataset, device="cuda", epochs=80, model_name="model", output_path="results"):
    
    # 🔵 Split train / val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Vérifiez un échantillon de votre dataset
    sample_features, sample_label = train_dataset[0]
    print(f"Sample features shape: {sample_features.shape if hasattr(sample_features, 'shape') else type(sample_features)}")
    print(f"Sample label: {sample_label}")

    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    train_loader = MILDataset(train_dataset)
    val_loader = MILDataset(val_dataset)

    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    df_model = pd.DataFrame(columns=["Epoch", "Train Loss", "Train AUC", "Val Loss", "Val AUC"])

    # 🔵 Historique
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []
    
    for epoch in range(epochs):
        
        # =======================
        # TRAIN
        # =======================
        model.train()
        
        total_loss = 0
        all_probs = []
        all_labels = []
        

        for features, label in train_loader:
            
            features = features.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            
            output = model(features)
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            probs = torch.softmax(output, dim=1)[:, 1]
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(label.detach().cpu().numpy())
        
        train_loss = total_loss / len(train_loader)
        
        try:
            train_auc = roc_auc_score(all_labels, all_probs)
        except:
            train_auc = 0.0
        
        # =======================
        # VALIDATION
        # =======================
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        
        # 🔵 Sauvegarde métriques
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   AUC: {val_auc:.4f}")
        print("-"*40)

        df_model = df_model.append({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train AUC": train_auc,
            "Val Loss": val_loss,
            "Val AUC": val_auc
        }, ignore_index=True)
    
    # 🔵 Courbes
    str_chain = output_path + "/" + model_name #Output path + model name
    plot_learning_curves(train_losses, val_losses, train_aucs, val_aucs, model_name=str_chain)
    
    df_model.to_csv(f"{str_chain}_metrics.csv", index=False)
    return model


def plot_learning_curves(train_losses, val_losses, train_aucs, val_aucs, model_name):
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure()
    
    # 🔵 Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    
    # 🔵 AUC
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_aucs, label="Train AUC")
    plt.plot(epochs, val_aucs, label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("AUC Curve")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_learning_curves.png")
    plt.show()



def plot_confusion_matrix(cm):
    
    plt.figure()
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Neg", "Pos"],
        yticklabels=["Neg", "Pos"]
    )
    
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
