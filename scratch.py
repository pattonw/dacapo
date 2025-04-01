import multiprocessing as mp

mp.set_start_method("fork", force=True)

from dacapo.experiments.logging.tensorboard import TensorboardLogger
from dacapo.experiments.logging.wandb import WandBLogger
from dacapo.experiments.logging.mlflow import MLflowLogger
from dacapo.experiments.logging.comet import CometLogger
from dacapo.experiments.logging.sacred import SacredLogger
from dacapo.experiments.logging.logging_backend import LoggingBackend


from dacapo_toolbox.datasplits.simple_config import SimpleDataSplitConfig
from dacapo_toolbox.architectures.cnnectome_unet import CNNectomeUNetConfig
from dacapo_toolbox.tasks.affinities_task_config import AffinitiesTaskConfig
from dacapo_toolbox.trainers.gunpowder_trainer_config import GunpowderTrainerConfig
from dacapo_toolbox.trainers.gp_augments.elastic_config import ElasticAugmentConfig

from dacapo.experiments import RunConfig
from dacapo.train import train_run

import zarr
import numpy as np


import random
import pprint

import matplotlib.pyplot as plt
import numpy as np
from funlib.geometry import Coordinate
from funlib.persistence import prepare_ds
from scipy.ndimage import label
from skimage import data
from skimage.filters import gaussian

from dacapo.store.create_store import create_stats_store, create_config_store

import shutil

from pathlib import Path

if not Path("cells3d.zarr/raw").exists():
    # Download the data
    cell_data = (data.cells3d().transpose((1, 0, 2, 3)) / 256).astype(np.uint8)

    # Handle metadata
    offset = Coordinate(0, 0, 0)
    voxel_size = Coordinate(290, 260, 260)
    axis_names = ["c^", "z", "y", "x"]
    units = ["nm", "nm", "nm"]

    # Create the zarr array with appropriate metadata
    cell_array = prepare_ds(
        "cells3d.zarr/raw",
        cell_data.shape,
        offset=offset,
        voxel_size=voxel_size,
        axis_names=axis_names,
        units=units,
        mode="w",
        dtype=np.uint8,
    )

    # Save the cell data to the zarr array
    cell_array[:] = cell_data

    cell_mask = np.clip(gaussian(cell_data[1] / 255.0, sigma=1), 0, 255) * 255 > 30
    not_membrane_mask = (
        np.clip(gaussian(cell_data[0] / 255.0, sigma=1), 0, 255) * 255 < 10
    )
    mask = cell_mask * not_membrane_mask

    # Generate labels via connected components
    labels_array = prepare_ds(
        "cells3d.zarr/gt",
        cell_data.shape[1:],
        offset=offset,
        voxel_size=voxel_size,
        axis_names=axis_names[1:],
        units=units,
        mode="w",
        dtype=np.uint8,
    )
    labels_array[:] = label(mask)[0]

# Example usage
if __name__ == "__main__":
    datasplit = SimpleDataSplitConfig(name="cells3d", path=Path("cells3d.zarr"))
    unet = CNNectomeUNetConfig(
        name="Mini-unet",
        input_shape=(24, 24, 24),
        fmaps_out=3,
        fmaps_in=2,
        num_fmaps=4,
        fmap_inc_factor=2,
        downsample_factors=[(2, 2, 2)],
        eval_shape_increase=(48, 48, 48),
    )
    assert unet.eval_shape_increase == (48, 48, 48), unet.eval_shape_increase
    task = AffinitiesTaskConfig(
        name="affs",
        neighborhood=[Coordinate(1, 0, 0), Coordinate(0, 1, 0), Coordinate(0, 0, 1)],
        lsds=True,
    )
    trainer = GunpowderTrainerConfig(
        name="rotations",
        augments=[
            # ElasticAugmentConfig(
            #     control_point_spacing=(4, 4, 4),
            #     control_point_displacement_sigma=[1, 1, 1],
            #     uniform_3d_rotation=True,
            #     rotation_interval=(0, 3.14),
            # )
        ],
    )

    run = RunConfig(
        name="simple_run",
        architecture_config=unet,
        task_config=task,
        trainer_config=trainer,
        datasplit_config=datasplit,
        batch_size=5,
        validation_interval=100,
        snapshot_interval=100,
        num_iterations=500,
        num_workers=10,
    )

    config_store = create_config_store()

    for logger_type in [
        TensorboardLogger,
        WandBLogger,
        "custom",
        None,
    ]:
        if logger_type in [TensorboardLogger, WandBLogger]:
            logger = logger_type()
            log_train_stats = logger.log_training_iteration
            log_validation_scores = logger.log_validation_iteration_scores

        elif logger_type == "custom":

            def log_train_stats(run_name: str, training_stats: dict[str, float]):
                print(
                    f"run({run_name}): {training_stats['iteration']} -- {training_stats['loss']}"
                )

            def log_validation_scores(
                run_name: str, iteration: int, validation_scores: dict[str:float]
            ):
                print(
                    f"run ({run_name}): {iteration} -- "
                    f"{np.array([v for k, v in validation_scores.items() if 'f1' in k]).mean()}"
                )
        elif logger_type is None:
            log_train_stats = None
            log_validation_scores = None
        else:
            raise Exception(logger_type)

        if Path("/Users/pattonw/dacapo").exists():
            shutil.rmtree(Path("/Users/pattonw/dacapo"))

        train_run(
            run,
            validate=True,
            save_snapshots=True,
            log_train_stats_callback=log_train_stats,
            log_validation_scores_callback=log_validation_scores,
        )

        # %% [markdown]
        # ## Visualize
        # Let's visualize the results of the training run. DaCapo saves a few artifacts during training
        # including snapshots, validation results, and the loss.

        # %%
        if logger_type is None:
            stats_store = create_stats_store()
            config_store = create_config_store()
            training_stats = stats_store.retrieve_training_stats(run.name)
            stats = training_stats.to_xarray()
            plt.plot(stats)
            plt.title("Training Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.show()

            # %%
            from dacapo.convenience.plotting.plot import plot_runs

            plot_runs(
                run_config_base_names=[run],
                validation_scores=["voi"],
                plot_losses=[True],
            )

            # # other ways to visualize the training stats
            # stats_store = create_stats_store()
            # training_stats = stats_store.retrieve_training_stats(run.name)
            # stats = training_stats.to_xarray()
            # plt.plot(stats)
            # plt.title("Training Loss")
            # plt.xlabel("Iteration")
            # plt.ylabel("Loss")
            # plt.show()
        elif logger_type in [TensorboardLogger, WandBLogger]:
            # Use the logger to visualize
            logger.visualize(run)
        # %%
        import zarr
        from matplotlib.colors import ListedColormap

        np.random.seed(1)
        colors = [[0, 0, 0]] + [
            list(np.random.choice(range(256), size=3)) for _ in range(254)
        ]
        label_cmap = ListedColormap(colors)

        run_path = config_store.path.parent / run.name

        # BROWSER = False
        num_snapshots = run.num_iterations // run.snapshot_interval
        print(num_snapshots)

        if num_snapshots > 0:
            fig, ax = plt.subplots(num_snapshots, 3, figsize=(10, 2 * num_snapshots))

            # Set column titles
            column_titles = ["Raw", "Target", "Prediction"]
            for col in range(3):
                ax[0, col].set_title(column_titles[col])

            for snapshot in range(num_snapshots):
                snapshot_it = snapshot * run.snapshot_interval
                # break
                raw = zarr.open(f"{run_path}/snapshot.zarr/volumes/raw")[snapshot, 0]
                target = zarr.open(f"{run_path}/snapshot.zarr/volumes/target")[
                    snapshot, 0
                ]
                prediction = zarr.open(f"{run_path}/snapshot.zarr//volumes/prediction")[
                    snapshot, 0
                ]
                c = (raw.shape[2] - target.shape[2]) // 2
                print(raw.shape, target.shape, prediction.shape)
                ax[snapshot, 0].imshow(raw[1, raw.shape[0] // 2, c:-c, c:-c])
                ax[snapshot, 1].imshow(target[0, target.shape[0] // 2])
                ax[snapshot, 2].imshow(prediction[0, prediction.shape[0] // 2])
                ax[snapshot, 0].set_ylabel(f"Snapshot {snapshot_it}")
            plt.show()

        # # %%
        # Visualize validations
        import zarr

        num_validations = run.num_iterations // run.validation_interval
        fig, ax = plt.subplots(num_validations, 4, figsize=(10, 2 * num_validations))

        # Set column titles
        column_titles = ["Raw", "Ground Truth", "Prediction", "Segmentation"]
        for col in range(len(column_titles)):
            ax[0, col].set_title(column_titles[col])

        for validation in range(1, num_validations + 1):
            dataset = run.datasplit.validate[0].name
            validation_it = validation * run.validation_interval
            # break
            raw = zarr.open(f"{run_path}/validation.zarr/inputs/{dataset}/raw")
            gt = zarr.open(f"{run_path}/validation.zarr/inputs/{dataset}/gt")
            pred_path = (
                f"{run_path}/validation.zarr/{validation_it}/{dataset}/prediction"
            )
            out_path = f"{run_path}/validation.zarr/{validation_it}/{dataset}/output/WatershedPostProcessorParameters(id=2, bias=0.5, context=(32, 32, 32))"
            output = zarr.open(out_path)[:]
            prediction = zarr.open(pred_path)[0]
            c = (raw.shape[2] - gt.shape[1]) // 2
            if c != 0:
                raw = raw[:, :, c:-c, c:-c]
            ax[validation - 1, 0].imshow(raw[1, raw.shape[1] // 2])
            ax[validation - 1, 1].imshow(
                gt[gt.shape[0] // 2], cmap=label_cmap, interpolation="none"
            )
            ax[validation - 1, 2].imshow(prediction[prediction.shape[0] // 2])
            ax[validation - 1, 3].imshow(
                output[output.shape[0] // 2], cmap=label_cmap, interpolation="none"
            )
            ax[validation - 1, 0].set_ylabel(f"Validation {validation_it}")
        plt.show()
