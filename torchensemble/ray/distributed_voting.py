import ray
import torch
import torch.nn as nn
from ray import train
from ray.air import session, Checkpoint
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig, RunConfig, CheckpointConfig

from ._base_parallel import BaseParallel


def _train_loop_per_worker(
        estimator_provider_func,
        optimizer_provider_func,
        loss_provider_func,
        num_epochs,
        batch_size,
        log_interval=100,
        estimator_args=None,
        optimizer_args=None):

    rank = session.get_local_rank()
    dataset_shard = session.get_dataset_shard("train")

    if estimator_args:
        estimator = estimator_provider_func(**estimator_args)
    else:
        estimator = estimator_provider_func()

    loss_fn = loss_provider_func()

    if optimizer_args:
        optimizer = optimizer_provider_func(estimator.parameters(), **optimizer_args)
    else:
        optimizer = optimizer_provider_func(estimator.parameters())

    model = train.torch.prepare_model(model)

    for epoch in range(num_epochs):
        for batch_idx, batches in enumerate(dataset_shard.iter_torch_batches(batch_size=batch_size)):
            inputs = torch.unsqueeze(batches["x"], 1)
            labels = batches["y"]

            output = model(inputs)
            loss = loss_fn(output.squeeze(), labels)
            optimizer.zero_grad()
            loss.backgrand()
            optimizer.step()

            if batch_idx % log_interval == 0:
                msg = (
                    "Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                    " | Loss: {:.5f}"
                )
                print(msg.format(rank, epoch, batch_idx, loss))

        session.report(
            {"loss": loss.item(), "epoch": epoch},
            checkpoint = Checkpoint.from_dict({"epoch": epoch, "model": model.state_dict()})
        )


class DistributedVoting(BaseParallel):

    def __init__(self, estimator_provider_func, n_estimators, use_gpu=True,
                 estimator_args=None):
        super().__init__(estimator_provider_func, n_estimators, use_gpu,
                         estimator_args)

    def fit(self, optimizer_provider_func, loss_provider_func, train_data,
            epochs, batch_size, optimizer_args=None, loss_args=None,
            val_data=None, log_interval=100):
        super().fit(optimizer_provider_func, loss_provider_func, train_data,
                    epochs, batch_size, optimizer_args, loss_args, val_data,
                    log_interval)

        train_loop_config = {
            "estimator_provider_func": self.estimator_provider_func,
            "optimizer_provider_func": optimizer_provider_func,
            "loss_provider_func": loss_provider_func,
            "num_epochs": epochs,
            "batch_size": batch_size,
            "log_interval": log_interval,
            "estimator_args": self.estimator_args,
            "optimizer_args": optimizer_args,
        }
        scaling_config = ScalingConfig(
            num_workers=self.n_estimators,
            use_gpu = self.use_gpu
        )
        run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1))

        trainer = TorchTrainer(
            train_loop_per_worker=_train_loop_per_worker,
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            run_config=run_config,
            datasets={"train": train_data, "val": val_data}
        )

        result = trainer.fit()