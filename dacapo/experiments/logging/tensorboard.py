import tensorboard
from torch.utils.tensorboard import SummaryWriter

from shutil import rmtree

import attr

from .logging_backend import LoggingBackend

from dacapo.options import Options

from dacapo.experiments import RunConfig, ValidationIterationScores


class TensorboardLogger(LoggingBackend):
    def __init__(self):
        self.writers = {}

    def log_dir(self, run: RunConfig):
        dacapo_config = Options.instance()
        return f"{dacapo_config.runs_base_dir}/tb_logs/{run.name}"

    def get_writer(self, run: RunConfig) -> SummaryWriter:
        if run.name not in self.writers:
            self.writers[run.name] = SummaryWriter(log_dir=self.log_dir(run))
        return self.writers[run.name]

    def log_training_iteration(self, run: RunConfig, training_stats: dict[str, float]):
        writer = self.get_writer(run)
        writer.add_scalar(
            "Loss/train",
            scalar_value=training_stats["loss"],
            global_step=training_stats["iteration"],
        )

    def retrieve_training_stats(self, run: RunConfig):
        raise NotImplementedError("tensorboard retrieval isn't implemented yet.")

    def log_validation_iteration_scores(
        self, run: RunConfig, iteration: int, validation_scores: ValidationIterationScores
    ):
        writer = self.get_writer(run)
        for k, v in validation_scores.to_dict().items():
            writer.add_scalar(k, v, global_step=iteration)

    def retrieve_validation_iteration_scores(self, run: RunConfig):
        raise NotImplementedError("tensorboard retrieval isn't implemented yet.")

    def delete_training_stats(self, run: RunConfig):
        log_dir = self.log_dir(run)
        rmtree(log_dir)

    def visualize(self, run: RunConfig):
        tracking_address = self.log_dir(run)
        tb = tensorboard.program.TensorBoard()
        tb.configure(argv=[None, "--logdir", tracking_address])
        url = tb.launch()
        print(f"Tensorflow listening on {url}")
        input("Hit enter to exit!")
