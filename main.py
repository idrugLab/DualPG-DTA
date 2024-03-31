import json
import torch
import torch.profiler
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
from models import DTANet
import numpy as np
import random
import time
import socket
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef
import copy
from torch.multiprocessing import spawn
import json
import os
from torch.utils.tensorboard import SummaryWriter
from preprocess import load_data
import warnings
from metrics import R2mIndex, ConcordanceIndex


def seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)


def main(rank, world_size, start_time, master_port, params):
    seed(params["seed"])
    if rank <= 0:
        if os.access("history.json", os.R_OK):
            with open("history.json", "r", encoding="utf-8") as f:
                j = json.load(f)
        else:
            j = {}
        meta = {"Hostname": socket.gethostname(), "Worldsize": world_size}
        j[start_time] = {**meta, **params}
        with open("history.json", "w", encoding="utf-8") as f:
            json.dump(j, f, indent=4, ensure_ascii=False)

        writer = SummaryWriter("./logs/" + start_time)
    else:
        writer = None

    if rank == -1:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = master_port
        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        device = torch.device("cuda", rank)
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    test_metrics = train(rank, device, writer, params, start_time)
    if rank <= 0:
        writer.close()
        with open("history.json", "r", encoding="utf-8") as f:
            j = json.load(f)
        for k in test_metrics.keys():
            j[start_time].update({f"Test{k}": test_metrics[k]})
        with open("history.json", "w", encoding="utf-8") as f:
            json.dump(j, f, indent=4, ensure_ascii=False)


def train(local_rank, device, writer, params, start_time):
    with_valid = params["valid"]
    if with_valid:
        train_data, valid_data, test_data = load_data(params)
    else:
        train_data, test_data = load_data(params)

    model = DTANet()
    model.to(device)
    if local_rank != -1:
        model = DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    if local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data, shuffle=True
        )
        train_loader = DataLoader(
            train_data,
            batch_size=params["batch_size"],
            sampler=train_sampler,
            pin_memory=True,
            num_workers=1,
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
        test_loader = DataLoader(
            test_data,
            batch_size=params["batch_size"],
            sampler=test_sampler,
            pin_memory=True,
            num_workers=1,
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=params["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        )
        test_loader = DataLoader(
            test_data, batch_size=params["batch_size"], pin_memory=True, num_workers=1
        )

    if with_valid:
        valid_mse_obj = MeanSquaredError().to(device)
        if local_rank != -1:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)
            valid_loader = DataLoader(
                valid_data,
                batch_size=params["batch_size"],
                sampler=valid_sampler,
                pin_memory=True,
                num_workers=1,
            )
        else:
            valid_loader = DataLoader(
                valid_data,
                batch_size=params["batch_size"],
                pin_memory=True,
                num_workers=1,
            )

    best_mse = 1000
    best_epoch = -1
    for epoch in range(1, params["max_epoch"] + 1):
        if local_rank != -1:
            train_loader.sampler.set_epoch(epoch)
            if with_valid:
                valid_loader.sampler.set_epoch(epoch)

        model.train()
        loss_list = []
        for _, data in enumerate(train_loader):
            drug_graph = data[0].to(device)
            target_graph = data[2].to(device)
            output = model(drug_graph, target_graph).view(-1)
            Y = data[4].to(device).view(-1)
            loss = loss_fn(output, Y)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if writer:
            writer.add_scalar("Train Loss", np.mean(loss_list), epoch)

        if with_valid:
            if epoch % 10 == 0 and epoch >= 150:
                if local_rank != -1:
                    dist.barrier()
                model.eval()
                with torch.no_grad():
                    for data in valid_loader:
                        drug_graph = data[0].to(device)
                        target_graph = data[2].to(device)
                        output = model(drug_graph, target_graph).view(-1)
                        Y = data[4].to(device).view(-1)
                        valid_mse_obj.update(output, Y)
                if local_rank != -1:
                    dist.barrier()
                valid_mse = float(valid_mse_obj.compute())
                valid_mse_obj.reset()

                if writer:
                    writer.add_scalar("Valid Loss", valid_mse, epoch)
                    writer.flush()
                if valid_mse < best_mse:
                    best_mse = valid_mse
                    best_epoch = epoch
                    # if local_rank == 0:
                    #     torch.save(model.module.state_dict(), f'./pts/{params["dataset"]}_seed_{params["seed"]}_fold_{params["fold"]}.pt')
                    # elif local_rank == -1:
                    #     torch.save(model.state_dict(), f'./pts/{params["dataset"]}_seed_{params["seed"]}_fold_{params["fold"]}.pt')
                    state_dict_for_test = copy.deepcopy(model.state_dict())
                if local_rank != -1:
                    dist.barrier()
                if epoch - best_epoch >= 100:
                    break
    if local_rank != -1:
        dist.barrier()
    warnings.filterwarnings("ignore", category=UserWarning)
    metrics = {
        "RMSE": MeanSquaredError(squared=False).to(device),
        "MSE": MeanSquaredError().to(device),
        "R2m": R2mIndex().to(device),
        "CI": ConcordanceIndex().to(device),
        "Pearson": PearsonCorrCoef().to(device),
        "Spearman": SpearmanCorrCoef().to(device),
    }
    warnings.resetwarnings()
    if "state_dict_for_test" in locals():
        model.load_state_dict(state_dict_for_test)
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            drug_graph = data[0].to(device)
            target_graph = data[2].to(device)
            output = model(drug_graph, target_graph).view(-1)
            Y = data[4].to(device).view(-1)
            for m in metrics.values():
                m.update(output, Y)
    if local_rank != -1:
        dist.barrier()
    vals = {}
    for k, v in metrics.items():
        vals[k] = float(v.compute())
        v.reset()
    if local_rank != -1:
        dist.barrier()
        dist.destroy_process_group()
    return vals


def run(params):
    world_size = torch.cuda.device_count()
    start_time = time.strftime("%m%d-%H%M%S", time.localtime())
    master_port = str(random.randint(30000, 39999))
    if world_size > 1:
        spawn(
            main,
            args=(world_size, start_time, master_port, params),
            nprocs=world_size,
            join=True,
        )
    else:
        main(-1, world_size, start_time, None, params)
    with open("history.json", "r", encoding="utf-8") as f:
        j = json.load(f)
    mse = j[start_time]["TestMSE"]
    r2m = j[start_time]["TestR2m"]
    ci = j[start_time]["TestCI"]
    pearson = j[start_time]["TestPearson"]
    spearman = j[start_time]["TestSpearman"]
    return mse, r2m, ci, pearson, spearman


if __name__ == "__main__":
    params = {
        "dataset": "Davis",
        "valid": True,
        "seed": 0,
        "max_epoch": 2000,
        "batch_size": 64,
        "learning_rate": 0.0001,
    }
    list_mse = []
    list_r2m = []
    list_ci = []
    list_pearson = []
    list_spearman = []
    for idx in range(0, 5):
        params["fold"] = idx
        mse, r2m, ci, pearson, spearman = run(params)
        list_mse.append(mse)
        list_r2m.append(r2m)
        list_ci.append(ci)
        list_pearson.append(pearson)
        list_spearman.append(spearman)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(f"MSE: {np.mean(list_mse):.4f} ({np.std(list_mse):.4f})")
    print(f"R2m: {np.mean(list_r2m):.4f} ({np.std(list_r2m):.4f})")
    print(f"CI: {np.mean(list_ci):.4f} ({np.std(list_ci):.4f})")
    print(f"Pearson: {np.mean(list_pearson):.4f} ({np.std(list_pearson):.4f})")
    print(f"Spearman: {np.mean(list_spearman):.4f} ({np.std(list_spearman):.4f})")
