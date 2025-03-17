from abc import ABC, abstractmethod
import attr

from .training_stats import TrainingStats, TrainingIterationStats

@attr.s()
class StatsStorage(ABC):
    

    @abstractmethod
    def add_iteration_stats(self, iteration_stats: TrainingIterationStats) -> None:
        pass

    @abstractmethod
    def delete_after(self, iteration: int) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> TrainingStats:
        pass

    @abstractmethod
    def set_stats(self, stats: TrainingStats) -> None:
        pass

    @abstractmethod
    def visualize(self):
        pass

