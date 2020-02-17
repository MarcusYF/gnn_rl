"""Load correlation matrix
"""
from typing import List
import scipy.io
import numpy as np
import pathlib
from copy import deepcopy
from typing import List

import dataclasses
from terminaltables import AsciiTable

import json


@dataclasses.dataclass
class ColocationConfig:
    """Configuration for running a colocation optimizer
    """
    task: str = 'strict_ga'
    job_name: str = 'untitled_job'
    seed: int = 1
    corr_matrix_path: str = ''
    output_path: str = './__output__/'
    total_room_count: int = 51
    total_type_count: int = 4
    selected_rooms: List[int] = dataclasses.field(default_factory=list)
    room_count: int = 51
    type_count: int = 4
    population_count: int = 300
    replaced_count: int = 150
    survivor_count: int = 20
    max_iteration: int = 5000
    crossing_over_rate: float = 0.0
    mutation_rate: float = 0.001
    mutation_weighted: bool = False
    searching_count: int = 0
    verbose: bool = True
    print_final_solution: bool = True
    visualize: bool = True
    show_figure: bool = True
    save_figure: bool = False
    save_results: bool = True
    plot_fitness_density: bool = False
    plot_fitness_accuracy: bool = True
    profile: bool = False

    def __str__(self):
        data = [['key', 'value']] + [[k, v] for k, v in vars(self).items()]
        return AsciiTable(data, title='Config').table

    def to_json(self):
        """Generate a json string

        Returns:
            str: a string of json
        """
        return json.dumps(dataclasses.asdict(self), indent=4)

    def copy(self) -> 'ColocationConfig':
        """Return a copy of self
        """
        return deepcopy(self)

    @property
    def base_file_name(self):
        """Get the base file path
        """
        return self.output_path + self.job_name + '/'

    def join_name(self, name: str):
        """Join the name based on a base name of the config
        """
        path = pathlib.Path(self.output_path).joinpath(
            pathlib.Path(self.job_name))
        return path.joinpath(name)


def load_config(config_file_path: str) -> ColocationConfig:
    """Load a configuration json file

    Args:
        config_file_path (str): Config file's path
    """
    with open(config_file_path, 'rb') as file:
        config = json.load(file)

    # mypy cannot recognize dataclass correctly
    config = ColocationConfig(**config)  # type: ignore

    return config


def load_matrix(matrix_file: str):
    """Load correlational coefficient matrix, and apply absolute value
    """
    # pylint: disable=E1101
    mat = scipy.io.loadmat(matrix_file)['corr']
    #mat = np.fabs(mat)
    return mat


def select_rooms(matrix: np.ndarray, selected_rooms: List[int],
                 type_count: int):
    """Select rows and columns from matrix based on a list of rooms

    Args:
        matrix (np.ndarray): correlational coefficient matrix
        selected_rooms (List[int]): IDs of selected rooms
        type_count (int): Number of types in current matrix

    Returns:
        np.ndarray: a matrix of correlational coefficient
        Let ``sensor_count`` be ``type_count * len(selected_rooms)``,
        The dimension of the returned matrix is (sensor_count, sensor_count)
    """

    sensor_ids = np.array([[r_id * type_count + j for j in range(type_count)]
                           for r_id in selected_rooms]).reshape(-1)
    return matrix[sensor_ids][:, sensor_ids]


def select_types(matrix: np.ndarray, selected_types: List[int],
                 original_type_count: int, room_count: int):
    """Select rows and columns from matrix according to selected types

    Args:
        matrix (np.ndarray): correlational coefficient matrix
        selected_types (List[int]): types selected
        original_type_count (int): original number of types
        room_count (int): the number of rooms

    Returns:
        np.ndarray: a new coefficient matrix,
        with side length = ``len(selected_types) * room_count``
    """

    sensor_ids = np.array(
        [[r_id * original_type_count + t_id for t_id in selected_types]
         for r_id in range(room_count)]).reshape(-1)
    return matrix[sensor_ids][:sensor_ids]
