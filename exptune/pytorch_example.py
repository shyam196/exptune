from dataclasses import dataclass
from os import environ
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.tune.schedulers import AsyncHyperBandScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .exptune import ExperimentConfig, ExperimentSettings, TrialResources
from .hyperparams import LogUniformHyperParam
from .search_strategies import RandomSearchStrategy


def _trim_dataset_for_debug(dataset, prop=0.1):
    return Subset(dataset, [i for i in range(int(prop * len(dataset)))])


class MnistMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.reshape(x, (-1, 784))
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


@dataclass
class Extra:
    device: torch.device
    lr_scheduler: ReduceLROnPlateau


class PytorchMnistMlpConfig(ExperimentConfig):
    def settings(self):
        return ExperimentSettings(exp_name="MnistMlpExample", final_max_iterations=10)

    def configure_seeds(self, seed):
        torch.manual_seed(seed)

    def resource_requirements(self):
        return TrialResources(cpus=4, gpus=0.0)

    def hyperparams(self):
        return {
            "lr": LogUniformHyperParam(1e-4, 1e-1, default=0.01),
            "wd": LogUniformHyperParam(1e-5, 1e-2, default=1e-4),
        }

    def search_strategy(self):
        return RandomSearchStrategy(num_samples=5)

    def trial_scheduler(self):
        metric, mode = self.trial_metric()
        return AsyncHyperBandScheduler(
            metric=metric, mode=mode, max_t=20, grace_period=10
        )

    def trial_metric(self):
        return "val_loss", "min"

    def data(self, pinned_objs, hparams, debug_mode):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = datasets.MNIST(
            environ.get("DATASET_DIRECTORY", "~/datasets"),
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            environ.get("DATASET_DIRECTORY", "~/datasets"),
            train=False,
            download=True,
            transform=transform,
        )

        train_len = int(0.9 * len(train_dataset))
        train_split = Subset(train_dataset, [i for i in range(train_len)])
        val_split = Subset(
            train_dataset, [i for i in range(train_len, len(train_dataset))]
        )

        # if in debug mode, cut the dataset size down to a tiny fraction
        if debug_mode:
            train_split = _trim_dataset_for_debug(train_split)
            val_split = _trim_dataset_for_debug(val_split)
            test_dataset = _trim_dataset_for_debug(train_dataset)

        return {
            "train": DataLoader(train_split, batch_size=32, shuffle=True),
            "val": DataLoader(val_split, batch_size=512, shuffle=False),
            "test": DataLoader(test_dataset, batch_size=512, shuffle=False),
        }

    def model(self, hparams, debug_mode):
        return MnistMlp()

    def optimizer(self, model, hparams, debug_mode):
        return Adam(model.parameters(), lr=hparams["lr"], weight_decay=hparams["wd"])

    def extra_setup(self, model, optimizer, hparams, debug_mode):
        return Extra(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            lr_scheduler=ReduceLROnPlateau(optimizer),
        )

    def __checkpoint_paths(self, checkpoint_dir: Path):
        return [
            checkpoint_dir / "model.pt",
            checkpoint_dir / "optimizer.pt",
            checkpoint_dir / "lr_scheduler.pt",
            checkpoint_dir / "hparams.pickle",
        ]

    def persist_trial(self, checkpoint_dir, model, optimizer, hparams, extra):
        for item, path in zip(
            [model, optimizer, extra.lr_scheduler, hparams],
            self.__checkpoint_paths(checkpoint_dir),
        ):
            torch.save(item, path)

    def restore_trial(self, checkpoint_dir):
        paths = self.__checkpoint_paths(checkpoint_dir)
        model = torch.load(paths[0])
        opt = torch.load(paths[1])
        lr_scheduler = torch.load(paths[2])
        hparams = torch.load(paths[3])

        extra = Extra(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            lr_scheduler=lr_scheduler,
        )

        return (
            model,
            opt,
            hparams,
            extra,
        )

    def trial_update_hparams(self, model, optimizer, extra, new_hparams):
        optim = optimizer.optimizer
        for g in optim.param_groups:
            g["lr"] = new_hparams["lr"]
        return True

    def train(self, model, optimizer, data, extra, debug_mode):
        device = extra.device

        model.train()
        total_loss = 0.0
        correct = 0
        for input, target in data["train"]:
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = F.nll_loss(output, target, reduction="sum")
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        total_items = len(data["train"].dataset)
        return (
            {
                "train_loss": total_loss / total_items,
                "train_accuracy": correct / total_items,
            },
            None,
        )

    def __eval(self, split, model, data, extra, debug_mode):
        device = extra.device

        model.eval()
        eval_loss = 0
        correct = 0
        with torch.no_grad():
            for input, target in data[split]:
                input, target = input.to(device), target.to(device)
                output = model(input)
                eval_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        num_items = len(data[split].dataset)
        return (
            {
                f"{split}_loss": eval_loss / num_items,
                f"{split}_accuracy": correct / num_items,
            },
            None,
        )

    def val(self, model, data, extra, debug_mode):
        metrics, extra_output = self.__eval("val", model, data, extra, debug_mode)
        # Cycle the LR scheduler along
        extra.lr_scheduler.step(metrics[self.trial_metric()[0]])
        return metrics, extra_output

    def test(self, model, data, extra, debug_mode):
        return self.__eval("test", model, data, extra, debug_mode)
