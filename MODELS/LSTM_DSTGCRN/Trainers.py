import torch
import math
import os
import time
import copy
import numpy as np
from datetime import datetime
from FL_HELPERS.FL_constants import *
from MODELS.HELPERS.Utils import evaluate_metrics, list_of_dicts_to_dict_of_lists
import logging
import os


class Trainer(object):
    def __init__(
        self,
        model,
        loss,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        scaler,
        args,
        lr_scheduler=None,
        verbose=True,
        name=None,
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.verbose = verbose
        self.name = name
        self.train_loss_list = []
        self.val_loss_list = []
        self.val_metrics_list = []
        self.lr_scheduler = lr_scheduler
        self.best_path = os.path.join(self.args.log_dir, f"best_model_{self.name}.pth")
        self.logger = logging.getLogger("LSTM-DSTGCRN")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


    def val_epoch(self):
        self.model.eval()
        total_val_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for _, (source, label) in enumerate(self.val_loader):
                source = source[..., : self.args.input_dim]
                label = label[..., : self.args.output_dim]
                
                output, _ = self.model(source)

                loss = self.loss(output, label)
                total_val_loss += loss.item()

                all_labels.append(label)
                all_predictions.append(output)

        val_loss = total_val_loss / len(self.val_loader)

        # all_labels = np.concatenate(all_labels, axis=0)
        # all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)


        # all_labels = self.scaler.inverse_transform(all_labels)
        # all_predictions = self.scaler.inverse_transform(all_predictions)
        mean = torch.tensor(self.scaler.mean, device=all_labels.device, dtype=torch.float32)
        std = torch.tensor(self.scaler.std, device=all_labels.device, dtype=torch.float32)
        all_labels = all_labels * std + mean
        all_predictions = all_predictions * std + mean
        #----------------
        all_predictions = all_predictions.cpu().numpy()
        all_labels = all_labels.cpu().numpy()


        # Evaluate metrics
        avg_metrics = {"MAE": 0, "MAPE": 0, "RMSE": 0, "RMSPE": 0, "R-squared": 0}

        for t in range(all_predictions.shape[1]):
            metrics = evaluate_metrics(
                all_labels[:, t, ...].reshape(all_labels.shape[0], -1),
                all_predictions[:, t, ...].reshape(all_predictions.shape[0], -1),
            )
            for metric, value in metrics.items():
                avg_metrics[metric] += value

        for metric, value in avg_metrics.items():
            avg_metrics[metric] /= all_predictions.shape[1]

        if self.verbose>1:
            for metric, value in avg_metrics.items():
                print(f"{CLIENT_INFO_VALIDATE} {self.name} Val 👉 {metric}: {value}")

        return val_loss, avg_metrics


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (source, label) in enumerate(self.train_loader):
            source = source[..., : self.args.input_dim]
            label = label[..., : self.args.output_dim]
            self.optimizer.zero_grad()
            output, _ = self.model(source)
            loss = self.loss(output, label)
            loss.backward()

            self.optimizer.step()
            total_loss += loss.item()

            # Log training information
            if self.verbose>1:
                if batch_idx % self.args.log_step == 0:
                    print(f"{CLIENT_INFO_TRAINING} {self.name} Train Epoch {epoch}: {batch_idx}/{len(self.train_loader)} Loss: {loss.item():.6f}")

        train_epoch_loss = total_loss / len(self.train_loader)
        train_epoch_rmse = math.sqrt(train_epoch_loss)
        # if self.verbose:
        #     print(f"{CLIENT_INFO_TRAINING} {self.name} Epoch {epoch} - Train Loss: {train_epoch_loss:8.4f}", end='')
        

        return train_epoch_loss, train_epoch_rmse

    def train(self):
        best_model = None
        best_train_rmse = torch.tensor(float("inf"), dtype=torch.float32)
        best_loss = torch.tensor(float("inf"), dtype=torch.float32)
        not_improved_count = 0
        val_metrics_list = []
        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss, train_epoch_rmse = self.train_epoch(epoch)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            val_epoch_loss, val_epoch_metrics = self.val_epoch()

            self.train_loss_list.append(train_epoch_loss)
            self.val_loss_list.append(val_epoch_loss)
            val_metrics_list.append(val_epoch_metrics)
            if train_epoch_loss > 1e6:
                print(f"{CLIENT_INFO_TRAINING} {self.name} Warning: Gradient explosion detected. Ending...")
                break

            if self.verbose:
                print(f"{CLIENT_INFO_TRAINING} {self.name} Epoch {epoch} - Train Loss: {train_epoch_loss:8.4f} - Valid loss: {val_epoch_loss:8.4f}")
        
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_train_rmse = train_epoch_rmse
                not_improved_count = 0
                best_state = True
                best_epoch_number = epoch
            else:
                not_improved_count += 1
                best_state = False
            
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    print(f"{CLIENT_INFO_TRAINING} {self.name} Validation performance didn't improve for {self.args.early_stop_patience} epochs. Training stops.")
                    break
            
            # save the best state
            if best_state == True:
                best_model = copy.deepcopy(self.model.state_dict())
                if self.verbose>1:
                    print(f"{CLIENT_INFO_TRAINING} {self.name} 💾 Current best model saved!")

        self.val_metrics_dict = list_of_dicts_to_dict_of_lists(val_metrics_list)
        training_time = time.time() - start_time
        if self.verbose>1:
            print(f"{CLIENT_INFO_TRAINING} {self.name} Training time: {(training_time / 60):.2f} minutes")

        # save the best model to file
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        torch.save(best_model, self.best_path)
        if self.verbose>1:
            print(f"{CLIENT_INFO_TRAINING} {self.name} Saving best model to " + self.best_path)

        # test
        self.model.load_state_dict(best_model)


    # @staticmethod
    def test(self, results_path=None):
        model, args, test_loader, scaler, name = self.model, self.args, self.test_loader, self.scaler, self.name
        model.eval()
        y_pred = []
        y_true = []
        adjs = []
        

        
        with torch.no_grad():
            for _, (source, label) in enumerate(test_loader):
                source = source[..., : args.input_dim]
                label = label[..., : args.output_dim]
                output, adjmatrices = model(source)
                if self.model.model_name == "LSTM-DSTGCRN":
                    adjs.append(adjmatrices.cpu().numpy())
                y_true.append(label.cpu().numpy())  
                y_pred.append(output.cpu().numpy())   

        # Apply the inverse transformation after converting tensors to numpy and concatenating
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)
        y_true = np.maximum(np.round(y_true), 0) #This is not good, but not too bad. Can be overcomed by using a better approach for test_loader
        y_pred = np.maximum(np.round(y_pred), 0)
        true_pred_y = np.concatenate([y_true[:,0,:,0], y_pred[:,0,:,0]], axis=1)
        
        # Evaluate metrics
        avg_metrics = {"MAE": 0, "MAPE": 0, "RMSE": 0, "RMSPE": 0, "R-squared": 0}

        for t in range(y_pred.shape[1]):
            metrics = evaluate_metrics(
                y_true[:, t, ...].reshape(y_pred.shape[0], -1),
                y_pred[:, t, ...].reshape(y_pred.shape[0], -1),
            )
            for metric, value in metrics.items():
                avg_metrics[metric] += value

        for metric, value in avg_metrics.items():
            avg_metrics[metric] /= y_pred.shape[1]

        for metric, value in avg_metrics.items():
            print(f"{CLIENT_INFO_TRAINING} {name} 📊 Test {metric}: {value:.4f}")

        if self.verbose:
            print(f"{CLIENT_INFO_TRAINING} {name} Lookback: {args.lookback}, Lookahead: {args.lookahead}")


        if results_path is not None:
        
            if False:
                if not os.path.exists(results_path):
                    # If the file doesn't exist, add column names
                    with open(results_path, "w") as f:
                        f.write("| Time | Model | Client | Metrics | Args |\n")
                        f.write("|---|---|---|---|---|\n")

                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                metrics_str = ", ".join(f"{metric}: {value:.3f}" for metric, value in avg_metrics.items())
                args_str = ", ".join(f"{key}: {value}" for key, value in self.args.__dict__.items())

                with open(results_path, "a") as f:
                    f.write(f"| {current_time} | LSTM-DSTGCRN | {self.name} | {metrics_str} | {args_str} |\n")

        return avg_metrics, true_pred_y