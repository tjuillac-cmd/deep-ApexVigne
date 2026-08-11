import torch
import torch.nn as nn
import copy


# ==============================================================================
# MODÈLES
# ==============================================================================

class deepmaxent_model(nn.Module):
    """
    MLP avec connexions résiduelles pour DeepMaxEnt.
    Sortie : logits bruts (shape : n_cells, output_size).
    """
    def __init__(self, input_size, hidden_size, output_size, hidden_nbr):
        super(deepmaxent_model, self).__init__()
        self.fc1_lambda = nn.Linear(input_size, hidden_size)
        self.hidden_layers_lambda = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_nbr)]
        )
        self.fc3_lambda = nn.Linear(hidden_size, output_size)

    def forward(self, xinput):
        x = self.fc1_lambda(xinput).relu()
        for layer in self.hidden_layers_lambda:
            x = layer(x).relu() + x          # connexion résiduelle
        x = self.fc3_lambda(x)
        return x


class mlp_model(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_nbr, dropout_p=0.2):
        super(mlp_model, self).__init__()
        self.dropout_p = dropout_p

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.drop1 = nn.Dropout(p=dropout_p)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_nbr - 1)]
        )
        self.hidden_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(hidden_nbr - 1)]
        )
        self.hidden_dropouts = nn.ModuleList(
            [nn.Dropout(p=dropout_p) for _ in range(hidden_nbr - 1)]
        )

        self.fc_out = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.drop1(self.ln1(self.fc1(x)).relu())

        for layer, ln, drop in zip(self.hidden_layers, self.hidden_norms, self.hidden_dropouts):
            x = drop(ln(layer(x)).relu()) + x

        return self.fc_out(x)


# ==============================================================================
# INFÉRENCE
# ==============================================================================

def predict_with_uncertainty(model, X_tensor, device, n_samples=100):
    """
    Prédit les scores DeepMaxEnt de présence et d'absence avec incertitude
    via Monte Carlo Dropout, en figeant le BatchNorm.

    Returns:
        mean_pred     : (n_cells, 2) — scores moyens [présence, absence]
        std_pred      : (n_cells, 2) — écart-types   [présence, absence]
        mean_presence : (n_cells,)   — score moyen de présence
        std_presence  : (n_cells,)   — incertitude sur la présence
        mean_absence  : (n_cells,)   — score moyen d'absence
        std_absence   : (n_cells,)   — incertitude sur l'absence
    """
    model.eval()  # fige BatchNorm

    # Force uniquement le Dropout en mode train
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    preds = []
    X_tensor = X_tensor.to(device)

    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(X_tensor)
            probs  = torch.softmax(logits, dim=0)                           # (n_cells, 2)
            preds.append(probs.cpu())  # normalise sur les cellules

    # (n_samples, n_cells, 2)
    preds = torch.stack(preds)

    mean_pred = preds.mean(dim=0)   # (n_cells, 2)
    std_pred  = preds.std(dim=0)    # (n_cells, 2)

    mean_presence = mean_pred[:, 0]
    std_presence  = std_pred[:, 0]
    mean_absence  = mean_pred[:, 1]
    std_absence   = std_pred[:, 1]

    return mean_pred, std_pred, mean_presence, std_presence, mean_absence, std_absence


def make_predictions(model, X_tensor):
    """
    Inférence simple sans MC-Dropout (pas d'incertitude).
    Retourne les logits bruts sur CPU.
    """
    model.eval()
    model     = model.to("cpu")
    X_tensor  = X_tensor.to("cpu")
    with torch.no_grad():
        predictions = model(X_tensor)
    return predictions


# ==============================================================================
# SAUVEGARDE / CHARGEMENT
# ==============================================================================

def save_mlp_model(model, filepath):
    """
    Sauvegarde le state_dict du modèle.
    Args:
        model    : nn.Module
        filepath : str — chemin complet du fichier .pth
    """
    torch.save(model.state_dict(), filepath)


def load_mlp_model(model, filepath):
    """
    Charge le state_dict dans un modèle existant.
    Args:
        model    : nn.Module — instance déjà instanciée avec la bonne architecture
        filepath : str — chemin complet du fichier .pth
    Returns:
        model chargé
    """
    model.load_state_dict(torch.load(filepath))
    return model