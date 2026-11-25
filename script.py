import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import time
import random
import os
import csv
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

SEED = 42
weights = models.MobileNet_V3_Small_Weights.DEFAULT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
num_epochs = 20

# Fix all random seeds and force deterministic cuDNN, for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)      
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Copy the first correct and the first wrong test image of a method; the filename tells true and predicted breed
def save_examples(paths, class_names, res, key, out_dir):
    y_true, y_pred = res["y_true"], res["y_pred"]
    for tag, k in (("correct", np.where(y_true == y_pred)[0][0]), ("wrong", np.where(y_true != y_pred)[0][0])):
        t, p = class_names[y_true[k]].replace(" ", "_"), class_names[y_pred[k]].replace(" ", "_")
        with open(paths[k], "rb") as src, open(f"{out_dir}/{key}_{tag}_true-{t}_pred-{p}{paths[k].suffix}", "wb") as dst:
            dst.write(src.read())

# Plot the confusion matrix
def plot_cm(cm, class_names, title, path):
    cm_norm = cm/cm.sum(axis=1, keepdims = True)
    fig, ax = plt.subplots(figsize = (12,12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# Return the most frequent classes that were incorrectly predicted
def top_confusions(cm, class_names, k=8):
    off = cm.copy()
    np.fill_diagonal(off, 0)                                            # ignore correct predictions
    out = []
    for flat in np.argsort(off.ravel())[::-1][:k]:
        i, j = np.unravel_index(flat, off.shape)
        out.append((class_names[i], class_names[j], int(off[i, j]),
                    round(off[i, j] / cm[i].sum(), 3)))   
    return out

# Build the model
def build_model():
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features,37)         # Substitute the last layer with a new one that outputs 37 scores
    return model.to(device)

# Evaluate a model on a loader: returns macro-F1, accuracy, confusion matrix and predictions
def evaluation(model,loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()               

            all_preds.append(predicted.cpu())                                   # Collect predictions and labels of all batches: macro-F1 must be computed on the whole set, not per batch
            all_labels.append(labels.cpu())
    
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    
    accuracy = correct/total
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(37))

    return {"f1": macro_f1, "acc": accuracy, "cm": cm, "y_true": y_true, "y_pred": y_pred}

# Train one epoch
def trainer(model, loader, optimizer, freeze=False):
    model.train()                                                       # Set the model in training mode
    if freeze:                                                          # Check to distinguish between head training and full finetune
        model.features.eval()                                           # Keep the backbone BatchNorm statistics fixed
    running_loss = 0.0                                                  
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()                                           # Reset the gradient every cycle (otherwise the new gradient is added to the old one)
        loss = criterion(model(images), labels)                         # Average of the loss over the entire batch
        loss.backward()                                                 # Compute the gradient of the loss for every trainable parameter (stored in .grad)
        optimizer.step()                                                # New weights
        running_loss += loss.item()                                     
    return running_loss / len(loader)                                   # Average loss

# Train for num_epochs, validate after each epoch and keep the checkpoint with the best validation macro-F1
def fit(model, train_loader, val_loader, optimizer, freeze=False):
    best_f1, best_epoch, best_state = -1, -1, None

    # Compute the number of trainable parameters
    train_param = 0
    for p in model.parameters():
        if p.requires_grad:
            train_param += p.numel()

    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Training
    for epoch in range(num_epochs):
        train_loss = trainer(model, train_loader, optimizer, freeze)
        res = evaluation(model, val_loader)
        f1 = res["f1"]
        print(f"Epoch {epoch+1}/{num_epochs} -- Train loss: {train_loss:.4f} -- Macro averaged F1 score: {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if device.type == "cuda": torch.cuda.synchronize()
    train_time = time.perf_counter() - t0

    model.load_state_dict(best_state)                                                                                   # Update the model with the best checkpoint
    return model, {"train_param": train_param, "time": train_time, "val_f1": best_f1, "best_epoch": best_epoch}

# Pick the best learning rate (the one that has the best f1 result)
def lr_search(train_loader, val_loader, lrs, freeze=False):
    best_model, best_info = None, None
    for lr in lrs:
        set_seed(SEED)
        model = build_model()
        if freeze:
            for p in model.features.parameters():
                p.requires_grad = False
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        print(f"--- lr = {lr:g} ---")
        model, info = fit(model, train_loader, val_loader, optimizer, freeze)
        info["lr"] = lr
        info["hparams"] = f"Adam lr={lr:g}, batch=32, epochs={num_epochs}"
        print(f"lr = {lr:g}: best val macro-F1 {info['val_f1']:.4f} (epoch {info['best_epoch']})\n")
        if best_info is None or info["val_f1"] > best_info["val_f1"]:
            best_model, best_info = model, info
    print(f"Selected lr = {best_info['lr']:g}\n")
    return best_model, best_info

# Random forest classifier
def rfc(train_eval_loader, val_loader, test_loader):
    set_seed(SEED)
    model = models.mobilenet_v3_small(weights=weights)
    t0 = time.perf_counter()
    f_train, l_train = extractor(model, train_eval_loader)
    f_val, l_val = extractor(model, val_loader)
    extract_time = time.perf_counter() - t0
    print(f_train.shape, f_val.shape)

    best_f1, best_p, best_rf = -1, None, None
    select_time = 0.0
    # Fit the classifier with four different parameters setting and pick the one with the best f1
    for max_features in ["sqrt", 0.1]:
        for min_leaf in [1, 3]:
            rf = RandomForestClassifier(n_estimators=300, max_features=max_features, min_samples_leaf=min_leaf, random_state=SEED, n_jobs=-1)
            t1 = time.perf_counter()
            rf.fit(f_train, l_train)
            p_val = rf.predict(f_val)
            select_time += time.perf_counter() - t1

            f1 = f1_score(l_val, p_val, average="macro", zero_division=0)
            print(max_features, min_leaf, round(f1, 4))
            if f1 > best_f1:
                best_f1, best_rf, best_p = f1, rf, (max_features, min_leaf)

    f_test, l_test = extractor(model, test_loader)

    info = {"train_param": "-", "time": extract_time + select_time, "val_f1": best_f1, "best_epoch": "-", "best_params": {"n_estimators": 300, "max_features": best_p[0], "min_samples_leaf": best_p[1]}, "hparams": f"RandomForest n_estimators=300, max_features={best_p[0]}, min_samples_leaf={best_p[1]}"}

    return info, metrics_rfc(f_train, l_train, best_rf), metrics_rfc(f_test, l_test, best_rf)

# Extract the features from model
def extractor(model, loader):
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)          # Remove the classfier
    feature_extractor.eval()                                                            # Set the model in evaluation mode

    all_feats, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            f = feature_extractor(images.to(device))
            f = torch.flatten(f, 1)
            all_feats.append(f.cpu())
            all_labels.append(labels)

    return torch.cat(all_feats).numpy(), torch.cat(all_labels).numpy()

# Predict with the Random Forest and return macro-F1, accuracy, confusion matrix and predictions
def metrics_rfc(f, l, rf):
    p = rf.predict(f)
    return {"f1": f1_score(l, p, average="macro", zero_division=0), "acc": (l == p).mean(), "cm": confusion_matrix(l, p, labels=np.arange(37)), "y_true": l, "y_pred": p}

# Create a row with all the info associated to the model
def make_row(name, info, train_res, test_res):
    return {
        "Method": name,
        "Trainable params": info["train_param"],
        "Training time (s)": round(info["time"], 1),
        "Best epoch": info["best_epoch"],
        "Train acc": round(train_res["acc"], 4),
        "Test acc": round(test_res["acc"], 4),
        "Gap (train - test)": round(train_res["acc"] - test_res["acc"], 4),
        "Val macro-F1": round(info["val_f1"], 4),
        "Test macro-F1": round(test_res["f1"], 4),
        "Selected hyperparameters": info["hparams"],
    }

def main():

    print(f"Using device: {device}\n")
    
    # Standard preprocessing (for validation and test)
    eval_trans = weights.transforms()

    # Preprocessing used for the training (augmentation)
    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = eval_trans.mean, std = eval_trans.std)
    ])

    # Load the datasets
    trainval_aug = datasets.OxfordIIITPet(root="./data", split="trainval", target_types="category", download=True, transform=train_trans)
    trainval_eval = datasets.OxfordIIITPet(root="./data", split="trainval", target_types="category", download=True, transform=eval_trans)
    test_dataset = datasets.OxfordIIITPet(root="./data", split="test", target_types="category", download=True, transform=eval_trans)

    trainval_labels = np.array(trainval_aug._labels)

    # Split partition (80/20)
    idx_train, idx_val = train_test_split(np.arange(len(trainval_labels)), test_size=0.2, stratify=trainval_labels, random_state=SEED)

    # Save the split
    os.makedirs("splits", exist_ok=True)
    np.save("splits/split_train_idx.npy", idx_train)        
    np.save("splits/split_val_idx.npy", idx_val)

    # Subdivisions of the datasets
    train_dataset = Subset(trainval_aug, idx_train)
    val_dataset = Subset(trainval_eval, idx_val)
    train_eval_dataset = Subset(trainval_eval, idx_train)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)                   # Augmentation + shuffle (training)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)                      # Standard preprocessing
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)                    # Standard preprocessing
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=32, shuffle=False)        # Training without augmentation (train metrics, feature extraction)

    # Full fine-tuning: all layers are trained, learning rate chosen on validation
    finetune_mod, info_finetune = lr_search(train_loader, val_loader, lrs=[1e-3, 1e-4, 1e-5])

    # Head training: frozen backbone, only the classifier is trained
    head_mod, info_head = lr_search(train_loader, val_loader, lrs=[1e-2, 1e-3, 1e-4], freeze=True)

    # Evaluation of the models
    train_finetune = evaluation(finetune_mod, train_eval_loader)
    test_finetune = evaluation(finetune_mod, test_loader)
    
    train_head = evaluation(head_mod, train_eval_loader)
    test_head = evaluation(head_mod, test_loader)
    
    # Random forest
    info_rf, train_rf, test_rf = rfc(train_eval_loader, val_loader, test_loader)

    # Table containing the parameters
    rows = [
        make_row("Full fine-tuning", info_finetune, train_finetune, test_finetune),
        make_row("Head training", info_head, train_head, test_head),
        make_row("Random Forest", info_rf, train_rf, test_rf)
    ]

    # Count number of images associated with each breed
    counts = {
        "train": np.bincount(trainval_labels[idx_train], minlength=37),
        "val":   np.bincount(trainval_labels[idx_val], minlength=37),
        "test":  np.bincount(np.array(test_dataset._labels), minlength=37)
    }

    # Create folders in results
    for d in ("tables", "figures", "checkpoints", "examples"):
        os.makedirs(f"results/{d}", exist_ok=True)

    # Save the selected checkpoints and the random forest hyperparameters
    torch.save({
        "finetune": {"state_dict": finetune_mod.state_dict(), "best_epoch": info_finetune["best_epoch"], "lr": info_finetune["lr"]},
        "head":     {"state_dict": head_mod.state_dict(),     "best_epoch": info_head["best_epoch"],     "lr": info_head["lr"]},
        "rf":       {"best_params": info_rf["best_params"]},
        "seed": SEED,
    }, "results/checkpoints/selected_models.pt")

    class_names = test_dataset.classes
    
    # Test results for each approach, used for the confusion matrices and the most confused classes
    cms = {
        "finetune": ("Full fine-tuning", test_finetune),
        "head":     ("Head training",    test_head),
        "rf":       ("Random Forest",    test_rf),
    }

    # Save confusion matrix as images
    for key, (name, res) in cms.items():
        plot_cm(res["cm"], class_names, f"{name} - test set", f"results/figures/cm_{key}.png")
        save_examples(test_dataset._images, class_names, res, key, "results/examples")

    # Most confused pairs of breeds for each approach
    for key, (name, res) in cms.items():
        conf = top_confusions(res["cm"], class_names)
        with open(f"results/tables/top_confusions_{key}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true", "predicted", "count", "fraction_of_true_class"])
            writer.writerows(conf)

    # Report the model's parameters in results.csv
    with open("results/tables/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()       
        writer.writerows(rows)     
    
    # Report in class_count.csv the number of images per class
    with open("results/tables/class_counts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "train", "val", "test"])
        for i, name in enumerate(test_dataset.classes):
            writer.writerow([name, counts["train"][i], counts["val"][i], counts["test"][i]])

if __name__ == "__main__":
    main()