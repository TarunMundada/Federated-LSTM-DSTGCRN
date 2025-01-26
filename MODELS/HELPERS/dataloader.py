import torch
import torch.utils.data
from FL_HELPERS.FL_constants import *
from MODELS.HELPERS.add_window import Add_Window_Horizon
from MODELS.HELPERS.load_dataset import load_and_transform_data, load_and_transform_data_OD
from MODELS.HELPERS.normalization import (
    NScaler,
    MinMax01Scaler,
    MinMax11Scaler,
    StandardScaler,
    ColumnMinMaxScaler,
)
import numpy as np


def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == "max01":
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print(f"{CLIENT_INFO_TRAINING} Normalize the dataset by MinMax01 Normalization")
    elif normalizer == "max11":
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print(f"{CLIENT_INFO_TRAINING} Normalize the dataset by MinMax11 Normalization")
    elif normalizer == "std":
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        # print(f"{CLIENT_INFO_TRAINING} Normalize the dataset by Standard Normalization")
    elif normalizer == None:
        scaler = NScaler()
        data = scaler.transform(data)
        # print("Does not normalize the dataset")
    elif normalizer == "cmax":
        # column min max, to be depressed
        # note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print(f"{CLIENT_INFO_TRAINING} Normalize the dataset by Column Min-Max Normalization")
    else:
        raise ValueError
    return data, scaler


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio) :]
    val_data = data[
        -int(data_len * (test_ratio + val_ratio)) : -int(data_len * test_ratio)
    ]
    train_data = data[: -int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data



def split_data_by_ratio_OD(data, val_ratio, test_ratio):
    
    dates = data[:, :, -1]
    unique_months = np.unique(dates.astype('datetime64[M]'))
    
    train_data = []
    val_data = []
    test_data = []
    
    for month in unique_months:
        month_mask = (dates.astype('datetime64[M]') == month)
        month_data = data[month_mask[:, 0], :, :]
        data_len = month_data.shape[0]
        
        test_split = int(data_len * test_ratio)
        val_split = int(data_len * val_ratio)
        
        test_data.append(month_data[-test_split:])
        val_data.append(month_data[-(test_split + val_split):-test_split])
        train_data.append(month_data[:-(test_split + val_split)])
    
    train_data = np.concatenate(train_data, axis=0)
    val_data = np.concatenate(val_data, axis=0)
    test_data = np.concatenate(test_data, axis=0)
    
    sep_train_size = data_len - test_split - val_split
    sep_val_size   = val_split
    sep_test_size  = test_split
    
    return train_data, val_data, test_data, (sep_train_size, sep_val_size, sep_test_size)


def data_loader(X, Y, batch_size, shuffle=False, drop_last=False, device=None):
    # TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=drop_last,
        # num_workers=2,#DEBUG
        # pin_memory=True#DEBUG
    )
    return dataloader


def get_dataloader(trips_data_path, args, normalizer="std", single=True, name=None, device=None):


    data = load_and_transform_data(trips_data_path)  # B, N, D

    data_train, data_val, data_test = split_data_by_ratio(
        data, args.val_ratio, args.test_ratio
    )

    # normalize st data
    data_train[:, :, : args.normalized_col], scaler = normalize_dataset(
        data_train[:, :, : args.normalized_col], normalizer, args.column_wise
    )
    data_val[:, :, : args.normalized_col], _ = normalize_dataset(
        data_val[:, :, : args.normalized_col], normalizer, args.column_wise
    )
    data_test[:, :, : args.normalized_col], _ = normalize_dataset(
        data_test[:, :, : args.normalized_col], normalizer, args.column_wise
    )

    # add time window
    if args.lookahead == 1:
        x_train, y_train = Add_Window_Horizon(data_train, args.lookback, args.lookahead, single)
        x_val, y_val = Add_Window_Horizon(data_val, args.lookback, args.lookahead, single)
        x_test, y_test = Add_Window_Horizon(data_test, args.lookback, args.lookahead, single)
    else:
        x_train, y_train = Add_Window_Horizon(
            data_train, args.lookback, args.lookahead, single=False
        )
        x_val, y_val = Add_Window_Horizon(
            data_val, args.lookback, args.lookahead, single=False
        )
        x_test, y_test = Add_Window_Horizon(
            data_test, args.lookback, args.lookahead, single=False
        )
    if DATA_INFO_VERBOSE:
        print(f"{CLIENT_INFO_MODEL} {name} X_train: {x_train.shape} Y_train: {y_train.shape}")
        print(f"{CLIENT_INFO_MODEL} {name} X_val: {x_val.shape} Y_val: {y_val.shape}")
        print(f"{CLIENT_INFO_MODEL} {name} X_test: {x_test.shape} Y_test: {y_test.shape}")

    if not args.TNE:
        train_dataloader = data_loader(
            x_train, y_train, args.batch_size, shuffle=False, drop_last=True, device=device
        )
        val_dataloader = data_loader(
            x_val, y_val, args.batch_size, shuffle=False, drop_last=False, device=device
        )
        test_dataloader = data_loader(
            x_test, y_test, args.batch_size, shuffle=False, drop_last=False, device=device
        )
    else:
        train_dataloader = data_loader(
            x_train, y_train, args.batch_size, shuffle=False, drop_last=True, device=device
        )
        val_dataloader = data_loader(
            x_val, y_val, args.batch_size, shuffle=False, drop_last=True, device=device
        )
        test_dataloader = data_loader(
            x_test, y_test, args.batch_size, shuffle=False, drop_last=True, device=device
        )
    num_nodes=data_train.shape[1]
    num_features=data_train.shape[2]
    num_obsevations=data.shape[0]
    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        scaler,
        num_nodes,
        num_features,
        num_obsevations
    )
    
    
    
    
def get_dataloader_OD(trips_data_path, args, normalizer="std", single=True, name=None, device=None):

    data = load_and_transform_data_OD(trips_data_path)  # B, N, D

    data_train, data_val, data_test, size_info = split_data_by_ratio_OD(
        data, args.val_ratio, args.test_ratio
    )
    
    data_train = data_train[:, :, : -1]
    data_val = data_val[:, :, : -1]
    data_test = data_test[:, :, : -1]

    # normalize st data
    data_train[:, :, : args.normalized_col], scaler = normalize_dataset(
        data_train[:, :, : args.normalized_col], normalizer, args.column_wise
    )
    data_val[:, :, : args.normalized_col], _ = normalize_dataset(
        data_val[:, :, : args.normalized_col], normalizer, args.column_wise
    )
    data_test[:, :, : args.normalized_col], _ = normalize_dataset(
        data_test[:, :, : args.normalized_col], normalizer, args.column_wise
    )
    
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []
    

    if args.lookahead == 1:
        x_train_3, y_train_3 = Add_Window_Horizon(data_train[:-size_info[0],:,:], args.lookback, args.lookahead, single)
        x_val_3, y_val_3 = Add_Window_Horizon(data_val[:-size_info[1],:,:], args.lookback, args.lookahead, single)
        x_test_3, y_test_3 = Add_Window_Horizon(data_test[:-size_info[2],:,:], args.lookback, args.lookahead, single)
        
        x_train_9, y_train_9 = Add_Window_Horizon(data_train[-size_info[0]:,:,:], args.lookback, args.lookahead, single)
        x_val_9, y_val_9 = Add_Window_Horizon(data_val[-size_info[1]:,:,:], args.lookback, args.lookahead, single)
        x_test_9, y_test_9 = Add_Window_Horizon(data_test[-size_info[2]:,:,:], args.lookback, args.lookahead, single)
    else:
        x_train_3, y_train_3 = Add_Window_Horizon(
            data_train[:-size_info[0],:,:], args.lookback, args.lookahead, single=False
        )
        x_val_3, y_val_3 = Add_Window_Horizon(
            data_val[:-size_info[1],:,:], args.lookback, args.lookahead, single=False
        )
        x_test_3, y_test_3 = Add_Window_Horizon(
            data_test[:-size_info[2],:,:], args.lookback, args.lookahead, single=False
        )
        
        x_train_9, y_train_9 = Add_Window_Horizon(
            data_train[-size_info[0]:,:,:], args.lookback, args.lookahead, single=False
        )
        x_val_9, y_val_9 = Add_Window_Horizon(
            data_val[-size_info[1]:,:,:], args.lookback, args.lookahead, single=False
        )
        x_test_9, y_test_9 = Add_Window_Horizon(
            data_test[-size_info[2]:,:,:], args.lookback, args.lookahead, single=False
        )
        
    x_train = np.concatenate((x_train_3, x_train_9), axis=0).astype(np.float32)
    y_train = np.concatenate((y_train_3, y_train_9), axis=0).astype(np.float32)
    x_val = np.concatenate((x_val_3, x_val_9), axis=0).astype(np.float32)
    y_val = np.concatenate((y_val_3, y_val_9), axis=0).astype(np.float32)
    x_test = np.concatenate((x_test_3, x_test_9), axis=0).astype(np.float32)
    y_test = np.concatenate((y_test_3, y_test_9), axis=0).astype(np.float32)
    
    
         
            
    if DATA_INFO_VERBOSE:
        print(f"{CLIENT_INFO_MODEL} {name} X_train: {x_train.shape} Y_train: {y_train.shape}")
        print(f"{CLIENT_INFO_MODEL} {name} X_val: {x_val.shape} Y_val: {y_val.shape}")
        print(f"{CLIENT_INFO_MODEL} {name} X_test: {x_test.shape} Y_test: {y_test.shape}")

    if not args.TNE:
        train_dataloader = data_loader(
            x_train, y_train, args.batch_size, shuffle=False, drop_last=True, device=device
        )
        val_dataloader = data_loader(
            x_val, y_val, args.batch_size, shuffle=False, drop_last=False, device=device
        )
        test_dataloader = data_loader(
            x_test, y_test, args.batch_size, shuffle=False, drop_last=False, device=device
        )
    else:
        train_dataloader = data_loader(
            x_train, y_train, args.batch_size, shuffle=False, drop_last=True, device=device
        )
        val_dataloader = data_loader(
            x_val, y_val, args.batch_size, shuffle=False, drop_last=True, device=device
        )
        test_dataloader = data_loader(
            x_test, y_test, args.batch_size, shuffle=False, drop_last=True, device=device
        )
    num_nodes=data_train.shape[1]
    num_features=data_train.shape[2]
    num_obsevations=data.shape[0]
    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        scaler,
        num_nodes,
        num_features,
        num_obsevations
    )
