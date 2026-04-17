import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, MAPE, and Directional Accuracy.

    Parameters
    ----------
    y_true : array-like, shape (n,) or (n, 1)
        Ground-truth prices in original (unscaled) units.
    y_pred : array-like, shape (n,) or (n, 1)
        Model predictions in original (unscaled) units.

    Returns
    -------
    dict with keys: MAE, RMSE, MAPE (%), Directional Acc
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

    # Directional accuracy: fraction of steps where predicted direction matches actual
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    dir_acc = (actual_dir == pred_dir).mean() if len(actual_dir) > 0 else np.nan

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE (%)": round(float(mape), 4),
        "Directional Acc": round(float(dir_acc), 4),
    }
