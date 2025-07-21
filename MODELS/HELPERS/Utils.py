import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from vmdpy import VMD



def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    abs_errors = np.abs(y_true - y_pred)
    percent_errors = abs_errors / y_true
    mape = np.mean(percent_errors)
    return mape


def mean_squared_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    abs_errors = np.abs(y_true - y_pred)
    percent_errors = abs_errors / y_true
    mspe = np.mean(percent_errors**2)
    return mspe


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        # "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "RMSE": sqrt(mean_squared_error(y_true, y_pred)),
        # "RMSPE": np.sqrt(mean_squared_percentage_error(y_true, y_pred)),
        # "R-squared": r2_score(y_true, y_pred),
    }
    return metrics

def list_of_dicts_to_dict_of_lists(list_of_dicts):
  """Converts a list of dictionaries to a dictionary where values are lists.

  Args:
    list_of_dicts: A list of dictionaries.

  Returns:
    A dictionary where keys are the unique keys from the input dictionaries and
    values are lists containing the corresponding values from the input dictionaries.
  """

  result = {}
  for dictionary in list_of_dicts:
    for key, value in dictionary.items():
      if key not in result:
        result[key] = []
      result[key].append(value)
  return result


def apply_vmd_to_node_series(signal, alpha=2000, tau=0., K=5, DC=0, init=1, tol=1e-7):
    """
    Applies VMD to a single 1D signal (e.g., one node's time series).
    Returns all decomposed IMFs.
    """
    # Ensure signal is float64
    signal = signal.astype(np.float64)
    
    # Apply VMD
    u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)  # u shape: (K, T)
    print(f"[VMD] Applied VMD to node with shape: {u.shape}")
    return u  # return IMFs