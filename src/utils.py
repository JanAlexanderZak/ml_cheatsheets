"""
Convert imperical to numerical units.
"""
import os
import sys
import traceback

from typing import Dict
from dotenv import load_dotenv

import numpy as np

from src.constants import PLOTLY_DEFAULT_COLORS

load_dotenv()
MY_ENV_VAR = os.getenv('API_KEY')
#print(MY_ENV_VAR)


def greyscale(_num: int, _a: float = 1.0) -> np.ndarray:
    """[summary]

    Args:
        _num ([type]): [description]
        _a (float, optional): [description]. Defaults to 1.0.

    Returns:
        np.array: Array of rgba strings
    """
    _rgba_list: np.ndarray = np.array([])
    for i in np.linspace(0, 255, _num):
        _rgba_list = np.append(_rgba_list, f"rgba({int(i)}, {int(i)}, {int(i)}, {_a})")
    return _rgba_list


def p_color(selector: int, _a: float = 1.0) -> str:
    """ Plotly default color list
    Cuts the bracket
    """
    try:
        assert isinstance(_a, float)
        assert 0.0 <= _a <= 1.0

    except AssertionError:
        _, _, trace = sys.exc_info()
        traceback.print_tb(trace)
        print("Alpha value is out of bounds.")
        sys.exit(1)

    _rgb = PLOTLY_DEFAULT_COLORS[selector]
    _rgba = f"{_rgb[:-1]}, {_a})"
    return _rgba


def convert_si_unit(val: float, unit_in: str, unit_out: str) -> float:
    """ Converts SI units.

    Args:
        val (float): value to convert
        unit_in (str): input si unit
        unit_out (str): output si unit

    Returns:
        float: output value
    """
    si_dict: Dict[str, float] = {
        'mm': 0.001,
        'cm': 0.01,
        'm': 1.0,
        'km': 1000.0,
    }
    return val * si_dict[unit_in] / si_dict[unit_out]
