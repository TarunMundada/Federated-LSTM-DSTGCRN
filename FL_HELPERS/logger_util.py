# logger_utils.py
import json
import pickle
import time
import torch

training_log = []  # global master log list

def log_round(round_num, y_true, y_pred, loss_value,
              start_time, module_replacement=None,
              gatn_attention=None, gpu_memory=None):
    """
    Logs all important metrics and artifacts for a given FL round.
    """
    round_runtime = time.time() - start_time

    round_log = {
        "round": round_num,
        "loss": loss_value,
        "runtime_sec": round_runtime,
        "gpu_memory_mb": gpu_memory,
        "y_true": y_true.tolist() if hasattr(y_true, "tolist") else list(y_true),
        "y_pred": y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred),
        "module_replacement": module_replacement,
        "gatn_attention": gatn_attention.tolist() if gatn_attention is not None else None
    }

    training_log.append(round_log)

def save_log_json(path="training_log.json"):
    with open(path, "w") as f:
        json.dump(training_log, f)

def save_log_pickle(path="training_log.pkl"):
    with open(path, "wb") as f:
        pickle.dump(training_log, f)