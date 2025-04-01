from dacapo.compute_context import create_compute_context
from dacapo.store.create_store import (
    create_array_store,
    create_config_store,
    create_stats_store,
    create_weights_store,
)
from dacapo.experiments import RunConfig
from dacapo.validate import validate_run
from dacapo.experiments.training_iteration_stats import TrainingIterationStats

from dacapo.experiments import ValidationIterationScores

import torch
from tqdm import tqdm

import logging
import time

import numpy as np
from itertools import product

logger = logging.getLogger(__name__)


def train(run_name: str, validate=True):
    """
    Train a run

    Args:
        run_name: Name of the run to train
    Raises:
        ValueError: If run_name is not found in config store
    Examples:
        >>> train("run_name")
    """

    # check config store to see if run is already being trained TODO
    # if ...:
    #     logger.error(f"Run {run_name} is already being trained")
    #     # if compute context runs train in some other process
    #     # we are done here.
    #     return

    logger.info(f"Training run {run_name}")

    # create run

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)

    return train_run(run_config, validate)


def train_run(
    run: RunConfig,
    validate: bool = True,
    save_snapshots: bool = False,
    log_train_stats_callback=None,
    log_validation_scores_callback=None,
):
    """
    Train a run

    Args:
        run: RunConfig object to train
    Raises:
        ValueError: If run_name is not found in config store

    """
    logger.info(f"Starting/resuming training for run {run.name}...")

    if log_train_stats_callback is None:
        assert log_validation_scores_callback is None

        def log_train_stats_callback(run: RunConfig, train_stats: dict[str, float]):
            iteration_stats = TrainingIterationStats(
                loss=train_stats["loss"],
                iteration=train_stats["iteration"],
                time=time.time(),
            )
            run.training_stats.add_iteration_stats(iteration_stats)
            if train_stats["iteration"] % run.validation_interval == 0:
                stats_store = create_stats_store()
                stats_store.store_training_stats(run.name, run.training_stats)

        def log_validation_scores_callback(
            run: RunConfig, iteration, iteration_scores: ValidationIterationScores
        ):
            run.validation_scores.add_iteration_scores(iteration_scores)
            stats_store = create_stats_store()
            stats_store.store_validation_iteration_scores(
                run.name, run.validation_scores
            )

    assert run.num_iterations is not None, (
        "num_iterations must be set in RunConfig to train"
    )

    stats_store = create_stats_store()
    weights_store = create_weights_store()
    array_store = create_array_store()

    start_iteration = run.resume_training(stats_store, weights_store)

    # start/resume training
    # set flag to improve training speeds
    torch.backends.cudnn.benchmark = True

    # make sure model and optimizer are on correct device.
    # loading weights directly from a checkpoint into cuda
    # can allocate twice the memory of loading to cpu before
    # moving to cuda.
    compute_context = create_compute_context()
    run.to(compute_context.device)
    logger.info(f"Training on {compute_context.device}")

    dataloader = run.data_loader()
    snapshot_container = array_store.snapshot_container(run.name)

    for i, batch in (
        bar := tqdm(
            enumerate(dataloader, start=start_iteration),
            total=run.num_iterations,
            initial=start_iteration,
            desc="training",
            postfix={"loss": None},
        )
    ):
        loss, batch_out = run.train_step(batch["raw"], batch["target"], batch["weight"])

        log_train_stats_callback(run, {"iteration": i, "loss": loss})

        bar.set_postfix({"loss": loss})

        if (
            run.snapshot_interval is not None
            and i % run.snapshot_interval == 0
            and save_snapshots
        ):
            # save snapshot. We save the snapshot at the start of every
            # {snapshot_interval} iterations. This is for debugging
            # purposes so you get snapshots quickly.
            run.save_snapshot(i, batch, batch_out, snapshot_container)

        if (
            i % run.validation_interval == run.validation_interval - 1
            or i == run.num_iterations - 1
        ):
            # run "end of epoch steps" such as stepping the learning rate
            # scheduler, storing stats, and writing out weights.
            try:
                run.lr_scheduler.step((i + 1) // run.validation_interval)
            except UserWarning as w:
                # TODO: What is going on here? Definitely calling optimizer.step()
                # before calling lr_scheduler.step(), but still getting a warning.
                logger.warning(w)
                pass

            # Store checkpoint and training stats
            weights_store.store_weights(run, i + 1)

            if validate:
                # VALIDATE
                validation_iteration_scores = validate_run(
                    run,
                    i + 1,
                )
                log_validation_scores_callback(run, i + 1, validation_iteration_scores)

        if i >= run.num_iterations - 1:
            break
