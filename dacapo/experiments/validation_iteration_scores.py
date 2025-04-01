from typing import List
import attr

import numpy as np
from itertools import product

@attr.s
class ValidationIterationScores:
    """
    A class used to represent the validation iteration scores in an organized structure.

    Attributes:
        iteration (int): The iteration associated with these validation scores.
        scores (List[List[List[float]]]): A list of scores per dataset, post processor
        parameters, and evaluation criterion.
    Note:
        The scores list is structured as follows:
        - The outer list contains the scores for each dataset.
        - The middle list contains the scores for each post processor parameter.
        - The inner list contains the scores for each evaluation criterion.

    """

    iteration: int = attr.ib(
        metadata={"help_text": "The iteration associated with these validation scores."}
    )
    scores: List[List[List[float]]] = attr.ib(
        metadata={
            "help_text": "A list of scores per dataset, post processor "
            "parameters, and evaluation criterion."
        }
    )
    datasets: list[str] = attr.ib()
    parameters: list[str] = attr.ib()
    criteria: list[str] = attr.ib()

    def to_dict(self) -> dict[str, float]:
        ds, ps, cs = (
            self.datasets,
            self.parameters,
            self.criteria,
        )
        scores_array = np.array(self.scores).reshape(
            (len(ds), len(ps), len(cs))
        )
        scores_dict = {}
        for i, j, k in product(range(len(ds)), range(len(ps)), range(len(cs))):
            scores_dict[f"{ds[i]}-{ps[j]}-{cs[k]}"] = scores_array[i, j, k]
        return scores_dict