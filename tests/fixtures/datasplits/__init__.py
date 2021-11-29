from .dummy_datasplit import mk_dummy_datasplit
from .twelve_class_datasplit import (
    mk_twelve_class_datasplit,
    mk_six_class_distance_datasplit,
)

DATASPLIT_MK_FUNCTIONS = [
    mk_dummy_datasplit,
    mk_twelve_class_datasplit,
    mk_six_class_distance_datasplit,
]
