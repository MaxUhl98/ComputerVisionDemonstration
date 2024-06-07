"""
Contains functions for training and testing PyTorch models.
Partially based on https://github.com/mrdbourke/pytorch-deep-learning
"""
import logging
import os
from pathlib import Path
import pandas as pd

from demo_configuration import DemonstrationConfig
import re
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Union
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from classification.data.image_data_class import ImageDataset


class AverageMeter(object):
    """Computes and stores the average and current value"""

    val: float
    avg: float
    sum: float
    count: int

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all the metrics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the metrics with the new value.

        :param val: The new value to add.
        :param n: The number of instances the value represents (default is 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device, lr_scheduling: bool, logger: logging.Logger, cfg: DemonstrationConfig,
               log_batch_loss: bool = False) -> tuple[float, float]:
    """
    Trains a PyTorch model for a single epoch.

    :param model: A PyTorch model to be trained.
    :param dataloader: A DataLoader instance for the model to be trained on.
    :param loss_fn: A PyTorch loss function to minimize.
    :param optimizer: A PyTorch optimizer to help minimize the loss function.
    :param device: A target device to compute on (e.g. "cuda" or "cpu").
    :param lr_scheduling: A boolean indicating whether learning rate scheduling is used.
    :param logger: A logger instance for logging training information.
    :param cfg: A configuration object containing various training settings.
    :param log_batch_loss: A boolean indicating whether to log the loss for each batch.
    :return: A tuple of training loss and training accuracy metrics (train_loss, train_accuracy).
    """
    # Put model in train mode
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
    avg_meter = AverageMeter()

    correct_predictions = 0
    total_datapoints = 0

    train_loss = 0
    logger.info('Starting training...')
    for batch in dataloader:
        if lr_scheduling:
            optimizer.optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        X, y = batch
        X = X.to(device)
        y = y.to(device)

        with torch.cuda.amp.autocast(enabled=cfg.apex):
            y_pred = F.softmax(model(X), 1)
            correct_predictions += (y_pred.argmax(dim=1) == y).sum()
            total_datapoints += y_pred.shape[0]

        with torch.cuda.amp.autocast(enabled=cfg.apex):
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

        scaler.scale(loss).backward()

        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        avg_meter.update(loss.item())

        if lr_scheduling:
            scaler.step(optimizer.optimizer)
            optimizer.step()
            scaler.update()
        else:
            scaler.step(optimizer)
            scaler.update()
        if log_batch_loss:
            logger.info(f'Loss in batch {batch}: {loss:.5f}')
            logger.info(f'Accuracy in batch {batch}: {correct_predictions / total_datapoints:.5f}')

    return avg_meter.avg, correct_predictions / total_datapoints


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> tuple[float, float]:
    """
    Tests a PyTorch model for a single epoch.

    :param model: A PyTorch model to be tested.
    :param dataloader: A DataLoader instance for the model to be tested on.
    :param loss_fn: A PyTorch loss function to calculate loss on the test data.
    :param device: A target device to compute on (e.g. "cuda" or "cpu").
    :return: A tuple of testing loss and testing accuracy metrics (val_loss, val_accuracy).
    """
    model.eval()
    val_loss = 0

    loss_avg_meter = AverageMeter()
    accuracy_average_meter = AverageMeter()
    total_test_datapoints = 0
    correct_predictions = 0

    with torch.inference_mode():
        for batch in dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            output_chances = F.softmax(model(X), 1)
            correct_predictions += (output_chances.argmax(dim=1) == y).sum().item()
            total_test_datapoints += output_chances.shape[0]
            loss = loss_fn(output_chances, y)
            loss_avg_meter.update(loss.item())

    return loss_avg_meter.avg, correct_predictions / total_test_datapoints


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: Union[torch.device, str], logger: logging.Logger, cfg: DemonstrationConfig,
          lr_scheduling: bool = False,
          patience: int = 10000,
          save_best: bool = True, save_path: str = 'model_weights/default_name.pth',
          log_batch_loss: bool = False) -> Dict[str, List]:
    """
    Trains and tests a PyTorch model.

    :param model: A PyTorch model to be trained and tested.
    :param train_dataloader: A DataLoader instance for the model to be trained on.
    :param test_dataloader: A DataLoader instance for the model to be tested on.
    :param optimizer: A PyTorch optimizer to help minimize the loss function.
    :param loss_fn: A PyTorch loss function to calculate loss on both datasets.
    :param epochs: An integer indicating how many epochs to train for.
    :param device: A target device to compute on (e.g. "cuda" or "cpu").
    :param logger: A logger instance for logging training information.
    :param cfg: A configuration object containing various training settings.
    :param lr_scheduling: A boolean indicating whether learning rate scheduling is used (default is False).
    :param patience: An integer indicating the number of epochs to wait before early stopping (default is 10000).
    :param save_best: A boolean indicating whether to save the best model based on validation loss (default is True).
    :param save_path: A string indicating the path to save the best model weights (default is 'model_weights/default_name.pth').
    :param log_batch_loss: A boolean indicating whether to log the loss for each batch (default is False).
    :return: A dictionary of training and testing loss as well as training and testing accuracy metrics.
             Each metric has a value in a list for each epoch.
    """
    results = {"train_loss": [],
               "val_loss": [],
               'train_acc': [],
               'val_acc': []
               }

    model.to(device)

    best_val_loss = 10 ** 6
    patience_count = 0

    best_model = model.state_dict()

    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(model=model,
                                                dataloader=train_dataloader,
                                                loss_fn=loss_fn,
                                                optimizer=optimizer,
                                                device=device, lr_scheduling=lr_scheduling, logger=logger,
                                                log_batch_loss=log_batch_loss, cfg=cfg)
        val_loss, val_accuracy = test_step(model=model,
                                           dataloader=test_dataloader,
                                           loss_fn=loss_fn,
                                           device=device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            if save_best:
                torch.save(model.state_dict(), save_path + '.pth')
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(
                    f"Epoch: {epoch + 1} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Accuracy: {train_accuracy:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Accuracy: {val_accuracy:.4f} | "
                )

                return results

        logger.info(
            f"Epoch: {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
        )

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["train_acc"].append(train_accuracy)
        results["val_acc"].append(val_accuracy)

    return results


def k_fold_train(models: List[torch.nn.Module], paths_to_data: List[Union[os.PathLike, str]],
                 optimizers: List[torch.optim.Optimizer],
                 loss_fn: torch.nn.Module,
                 device: Union[torch.device, str], logger: logging.Logger, cfg: DemonstrationConfig):
    """
    Trains multiple models using k-fold cross-validation.

    :param models: A list of PyTorch models to be trained.
    :param paths_to_data: A list of paths to the datasets.
    :param optimizers: A list of PyTorch optimizers corresponding to each model.
    :param loss_fn: A PyTorch loss function to calculate loss.
    :param device: A target device to compute on (e.g. "cuda" or "cpu").
    :param logger: A logger instance for logging training information.
    :param cfg: A configuration object containing various training settings.
    :return: A dictionary containing the training results for each fold.
    """
    folder = StratifiedKFold(n_splits=len(models), shuffle=cfg.shuffle_folds)
    paths = sum([list(Path(dir_path).glob('**/*/*.jpg')) for dir_path in paths_to_data], [])
    paths += sum([list(Path(dir_path).glob('**/*/*.png')) for dir_path in paths_to_data], [])
    data = pd.DataFrame(
        {'path': paths, 'target': [cfg.class_mappings.get(data_path.parent.name) for data_path in paths]})
    data = ImageDataset(data)
    folds = folder.split([num for num in range(len(data))], data.targets.cpu().numpy())
    fold_results = {}
    base_save_path = cfg.model_save_path.split('.pth')[0]
    save_dir = '/'.join(re.split(r'[\\/]', base_save_path)[:-1])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for num, (train_indexes, validation_indexes) in enumerate(folds):
        train_indexes, validation_indexes = torch.from_numpy(train_indexes), torch.from_numpy(validation_indexes)
        train_data, validation_data = torch.utils.data.Subset(data, train_indexes), torch.utils.data.Subset(data,
                                                                                                            validation_indexes)
        train_loader, validation_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                                                     generator=torch.Generator('cuda')), DataLoader(validation_data,
                                                                                                    batch_size=cfg.batch_size,
                                                                                                    shuffle=True,
                                                                                                    generator=torch.Generator(
                                                                                                        'cuda'))
        fold_save_path = base_save_path + f'_fold_{num}'
        logger.info(f'Starting training of fold {num}')
        fold_results[f'fold_{num}'] = train(model=models[num], train_dataloader=train_loader,
                                            test_dataloader=validation_loader,
                                            optimizer=optimizers[num], loss_fn=loss_fn,
                                            lr_scheduling=cfg.lr_scheduling,
                                            epochs=cfg.num_epochs,
                                            device=device, patience=cfg.patience,
                                            save_path=fold_save_path,
                                            save_best=cfg.save_best, logger=logger,
                                            log_batch_loss=cfg.log_batch_loss, cfg=cfg)
        best_val_acc = max(fold_results[f"fold_{num}"]["val_acc"])
        fold_results[f'fold_{num}']['best_accuracy'] = best_val_acc
        logger.info(f'Fold {num} Best Accuracy: {best_val_acc:.5f}')
    logger.info(
        f'Total Validation Accuracy: {sum([fold_results[f"fold_{num}"]["best_accuracy"] for num in range(len(models))])}')
    return fold_results
