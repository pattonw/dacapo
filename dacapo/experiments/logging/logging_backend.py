from abc import ABC, abstractmethod
from dacapo.experiments import RunConfig, ValidationIterationScores


class LoggingBackend(ABC):
    @abstractmethod
    def log_training_iteration(self, run: RunConfig, training_stats: dict):
        """
        Log an iteration of training stats.
        training_stats should include: {'loss': float, 'iteration': int}
        """
        pass

    @abstractmethod
    def retrieve_training_stats(self, run: RunConfig):
        """
        Retrieve all iterations of training stats for a given run.
        Returns a list of training stats dictionaries.
        """
        pass

    @abstractmethod
    def log_validation_iteration_scores(
        self,
        run: RunConfig,
        iteration: int,
        validation_scores: ValidationIterationScores,
    ):
        """
        Log an iteration of validation scores.
        validation_scores should include a grid like:
        {'param_combo': {'dataset': {'metric': float}}}
        """
        pass

    @abstractmethod
    def retrieve_validation_iteration_scores(self, run: RunConfig):
        """
        Retrieve all iterations of validation scores for a given run.
        Returns a list of validation scores dictionaries.
        """
        pass

    @abstractmethod
    def delete_training_stats(self, run: RunConfig):
        """
        Delete the training stats associated with a specific run.
        """
        pass

    @abstractmethod
    def visualize(self, run: RunConfig):
        """
        Open the preferred GUI or link for visualization of the given tool.
        """
        pass
