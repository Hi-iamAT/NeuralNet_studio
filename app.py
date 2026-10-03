import io
import json
import base64
import threading

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, random_split, DataLoader
from flask import Flask, Response, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

# ──────────────────────────────────────────────
#  Model definitions (mirrored from minimodel.py)
# ──────────────────────────────────────────────

def my_activation(x):
    return torch.where(x >= 0, x, 0.01 * x)   # leaky relu


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.01):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(self.linear.weight) * self.sigma
            noisy_weight = self.linear.weight + noise
            return torch.nn.functional.linear(x, noisy_weight, self.linear.bias)
        else:
            return self.linear(x)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1  = NoisyLinear(5, 16, sigma=0.01)
        self.layer2  = nn.Linear(16, 8)
        self.output  = nn.Linear(8, 1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = my_activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


# ──────────────────────────────────────────────
#  Global training state
# ──────────────────────────────────────────────

state = {
    "status": "idle",          # idle | training | done | error
    "train_losses": [],
    "val_losses":   [],
    "model":        None,
    "plot_b64":     None,
    "error":        None,
}
_lock = threading.Lock()


def build_dataset():
    torch.manual_seed(123)
    n_samples, n_features = 1000, 5
    X = torch.randn(n_samples, n_features)
    true_W = torch.tensor([[2.0], [-1.5], [1.0], [0.5], [-2.0]])
    true_b = 0.3
    score = X @ true_W + true_b + 0.5 * torch.randn(n_samples, 1)
    y = (score > 0).float()
    dataset = TensorDataset(X, y)
    N = len(dataset)
    n_train = int(0.70 * N)
    n_val   = int(0.15 * N)
    n_test  = N - n_train - n_val
    generator = torch.Generator().manual_seed(123)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


def make_plot(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.plot(train_losses, color="#38bdf8", linewidth=2, label="Train loss")
    ax.plot(val_losses,   color="#f472b6", linewidth=2, label="Val loss")
    ax.set_xlabel("Epoch", color="#94a3b8")
    ax.set_ylabel("Loss",  color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.legend(facecolor="#1e293b", labelcolor="#e2e8f0")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _train_worker(max_epoch, lr, batch_size):
    try:
        with _lock:
            state["status"]       = "training"
            state["train_losses"] = []
            state["val_losses"]   = []
            state["plot_b64"]     = None
            state["error"]        = None

        train_ds, val_ds, _ = build_dataset()
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        model     = NeuralNetwork()
        loss_fn   = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     betas=(0.9, 0.999), amsgrad=True, eps=1e-8)

        for epoch in range(max_epoch):
            # ── train ──
            model.train()
            total_train = 0.0
            for Xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(Xb)
                loss   = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                total_train += loss.item() * Xb.size(0)
            train_loss = total_train / len(train_ds)

            # ── validate ──
            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    logits = model(Xb)
                    loss   = loss_fn(logits, yb)
                    total_val += loss.item() * Xb.size(0)
            val_loss = total_val / len(val_ds)

            with _lock:
                state["train_losses"].append(round(train_loss, 6))
                state["val_losses"].append(round(val_loss, 6))

        # final plot + save model
        with _lock:
            state["plot_b64"] = make_plot(state["train_losses"], state["val_losses"])
            state["model"]    = model
            state["status"]   = "done"

    except Exception as exc:
        with _lock:
            state["status"] = "error"
            state["error"]  = str(exc)


# ──────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/train", methods=["POST"])
def train():
    with _lock:
        if state["status"] == "training":
            return jsonify({"error": "Already training"}), 409
    body      = request.get_json(silent=True) or {}
    max_epoch = int(body.get("epochs", 100))
    lr        = float(body.get("lr", 0.01))
    batch_size= int(body.get("batch_size", 32))
    t = threading.Thread(target=_train_worker, args=(max_epoch, lr, batch_size), daemon=True)
    t.start()
    return jsonify({"message": "Training started"})


@app.route("/api/status")
def status():
    with _lock:
        return jsonify({
            "status":       state["status"],
            "train_losses": state["train_losses"],
            "val_losses":   state["val_losses"],
            "plot_b64":     state["plot_b64"],
            "error":        state["error"],
        })


@app.route("/api/predict", methods=["POST"])
def predict():
    with _lock:
        model = state["model"]
    if model is None:
        return jsonify({"error": "Model not trained yet"}), 400
    body = request.get_json(silent=True) or {}
    features = body.get("features", [])
    if len(features) != 5:
        return jsonify({"error": "Need exactly 5 feature values"}), 400
    x = torch.tensor([features], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logit = model(x).item()
    prob  = torch.sigmoid(torch.tensor(logit)).item()
    label = 1 if prob >= 0.5 else 0
    return jsonify({"logit": round(logit, 4), "probability": round(prob, 4), "class": label})


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)
