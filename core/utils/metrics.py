"""Evaluation metrics"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np
import pandas as pd


class ModelMetrics:
    """Compute and display model metrics"""
    
    @staticmethod
    def classification_metrics(y_true, y_pred, y_pred_proba=None):
        """
        Compute classification metrics
        
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                pass
        
        return metrics
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        """
        Compute regression metrics
        
        Returns:
            Dictionary with metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    @staticmethod
    def print_classification_report(y_true, y_pred):
        """Print detailed classification report"""
        print(classification_report(y_true, y_pred))
    
    @staticmethod
    def get_confusion_matrix(y_true, y_pred):
        """Get confusion matrix"""
        return confusion_matrix(y_true, y_pred)

    # ---- Métricas de forecasting probabilístico (fútbol 1X2) -----------
    # Estándar en la literatura para evaluar pronósticos V/E/D con su
    # probabilidad. Habilitan el backtest multi-Mundial (TIER 1 del plan).

    @staticmethod
    def multiclass_logloss(y_true, proba, classes, eps: float = 1e-15):
        """Log-loss multiclase, respetando el orden de columnas de `proba`
        dado por `classes` (no asume orden lexicográfico como sklearn).
        Menor = mejor."""
        classes = list(classes)
        col = {c: i for i, c in enumerate(classes)}
        idx = np.array([col[v] for v in y_true])
        p = np.clip(np.asarray(proba, dtype=float), eps, 1.0)
        return float(-np.mean(np.log(p[np.arange(len(idx)), idx])))

    @staticmethod
    def rps(y_true_idx, proba):
        """Rank Probability Score — métrica estándar en forecasting de
        fútbol: respeta el ORDEN V/E/D (penaliza menos confundir V-E que
        V-D). `y_true_idx`: índice (0..k-1) de la clase real en el mismo
        orden que las columnas de `proba` (n, k). Menor = mejor."""
        proba = np.asarray(proba, dtype=float)
        n_classes = proba.shape[1]
        cum_proba = np.cumsum(proba, axis=1)
        cum_true = np.cumsum(np.eye(n_classes)[np.asarray(y_true_idx)], axis=1)
        return float(np.mean(np.sum((cum_proba - cum_true) ** 2, axis=1)
                             / (n_classes - 1)))

    @staticmethod
    def brier_multiclass(y_true_idx, proba):
        """Brier score multiclase (suma de errores cuadráticos por clase),
        promediado por partido. Menor = mejor."""
        proba = np.asarray(proba, dtype=float)
        onehot = np.eye(proba.shape[1])[np.asarray(y_true_idx)]
        return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))

    @staticmethod
    def plot_reliability_curve(y_true_idx, proba, path,
                               model_name: str = "modelo", n_bins: int = 10):
        """Guarda un reliability plot (calibración) por clase 1/X/2 en `path`.
        Un modelo calibrado cae sobre la diagonal. Devuelve la ruta escrita."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        proba = np.asarray(proba, dtype=float)
        y = np.asarray(y_true_idx)
        names = ["1 (gana local)", "X (empate)", "2 (gana visita)"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for c, (ax, name) in enumerate(zip(axes, names)):
            tab = ModelMetrics.reliability_table(
                (y == c).astype(float), proba[:, c], n_bins=n_bins)
            ax.plot(tab["p_pred_mean"], tab["p_obs"], "o-", label=model_name)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfecto")
            ax.set_xlabel("probabilidad predicha")
            ax.set_ylabel("frecuencia real")
            ax.set_title(f"Calibración: {name}")
            ax.legend(); ax.grid(True, alpha=0.3)
        fig.suptitle(f"Reliability curves — {model_name}")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    @staticmethod
    def reliability_table(y_true_binary, proba, n_bins: int = 10):
        """Tabla de calibración: agrupa por bins de probabilidad predicha
        y compara contra la frecuencia real observada. Un modelo calibrado
        tiene p_pred_mean ≈ p_obs en cada bin."""
        y = np.asarray(y_true_binary, dtype=float)
        p = np.asarray(proba, dtype=float)
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
        rows = []
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() == 0:
                continue
            rows.append({"bin": b, "p_pred_mean": float(p[mask].mean()),
                         "p_obs": float(y[mask].mean()), "n": int(mask.sum())})
        return pd.DataFrame(rows)


if __name__ == "__main__":
    pass
