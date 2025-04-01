import wandb

import attr

from .logging_backend import LoggingBackend

from dacapo.experiments import RunConfig, ValidationIterationScores


class WandBLogger(LoggingBackend):
    def __init__(self):
        wandb.login()
        self.runs = {}

    def run(self, run: RunConfig) -> wandb.run:
        if run.name not in self.runs:
            self.runs[run.name] = wandb.init(
                project="dacapo",
                id=run.name,
                resume="allow",
            )
        return self.runs[run.name]

    def log_training_iteration(self, run: RunConfig, training_stats: dict[str, float]):
        run = self.run(run)
        run.log(training_stats)

    def retrieve_training_stats(self, run: RunConfig):
        raise NotImplementedError("wandb retrieval isn't implemented yet.")

    def log_validation_iteration_scores(
        self, run: RunConfig, iteration: int, validation_scores: ValidationIterationScores
    ):
        run = self.run(run)
        run.log({"iteration": iteration, **validation_scores.to_dict()})

    def retrieve_validation_iteration_scores(self, run: RunConfig):
        raise NotImplementedError("wandb retrieval isn't implemented yet.")

    def delete_training_stats(self, run: RunConfig):
        api = wandb.Api()
        run = api.run(f"pattonw/dacapo/{run.name}")
        run.delete()

    def visualize(self, run: RunConfig):
        input("go to wandb.com")
