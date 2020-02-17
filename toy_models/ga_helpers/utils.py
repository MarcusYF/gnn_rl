"""Printing Helper Functions
"""
from terminaltables import AsciiTable
import cProfile
import pstats
import time
import subprocess
import pathlib
from typing import Dict
import argparse
import sys
from toy_models.ga_helpers import data_loader
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
from toy_models.ga_helpers.data_loader import ColocationConfig


def as_table(data) -> str:
    """Format data as a table string

    Args:
        data : data

    Returns:
        str: table
    """
    if isinstance(data, dict):
        data = list(data.items())
    return AsciiTable(data).table


def compile_vprint_function(verbose):
    """Compile a verbose print function

    Args:
        verbose (bool): is verbose or not

    Returns:
        [msg, *args]->None: a vprint function
    """

    if verbose:
        return lambda msg, *args: print(msg.format(*args))

    return lambda *_: None

def get_cache(config) -> Dict[str, list]:
    """Prepare a cache dictionary

    Args:
        config (ColocationConfig): configuration file

    Returns:
        Dict[str, list]: cache dict
    """

    cache: Dict[str, list] = {}
    if config.plot_fitness_density:
        cache['fitnesses'] = []
    if config.plot_fitness_accuracy:
        cache['best_fitness'] = []
        cache['accuracies'] = []
    return cache


def my_parse_args(myargs) -> data_loader.ColocationConfig:

    config = data_loader.load_config(myargs[3])

    config.corr_matrix_path = myargs[1]

    return config


def parse_args() -> data_loader.ColocationConfig:
    """parse argument
    """
    parser = argparse.ArgumentParser(prog='colocation')
    parser.add_argument(
        '-g', '--gen-config', dest='gen_config', action='store_true')
    parser.add_argument('-c', '--config', dest='config_path', required=True)
    parser.add_argument('-m', '--corr-matrix-path', dest='corr_matrix_path')
    parser.add_argument('-o', '--output-path', dest='output_path')
    parser.add_argument('-j', '--job_name', dest='job_name')
    commands = parser.parse_args(sys.argv[1:])
    if commands.gen_config:
        config = data_loader.ColocationConfig()
        print(config)
        with open(commands.config_path, 'w') as file:
            file.write(config.to_json())
        exit(0)

    config = data_loader.load_config(commands.config_path)

    if commands.corr_matrix_path is not None:
        config.corr_matrix_path = commands.corr_matrix_path

    return config


def create_dir(dir_name):
    """Create a dir if it does not already exist

    Args:
        dir_name (str): the directory name

    Raises:
        FileExistsError: If the path specified already exists and is a file
    """

    path = pathlib.Path(dir_name)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    elif path.is_file():
        raise FileExistsError('Log Path already exists and is not a dir')


class Profiler:
    """A wrapper around cProfile
    """

    def __init__(self, config):
        self.profiler = cProfile.Profile()
        self.has_run = False
        self.config = config
        self.tik = 0
        self.tok = 0

    def start(self):
        """start
        """
        self.tik = time.time()
        if self.config.profile:
            self.has_run = True
            self.profiler.enable()

    def stop(self):
        """stop
        """
        self.tok = time.time()
        if not self.has_run:
            return
        self.profiler.disable()
        self.profiler.dump_stats(self.config.base_file_name + 'profile.dump')

    def print_results(self):
        """print results
        """
        print("Total time used: {}".format(self.tok - self.tik))
        print("In average: {:0.2f} iterations / second".format(
            self.config.max_iteration / (self.tok - self.tik)))
        if not self.config.profile:
            return
        stats: pstats.Stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumtime')
        stats.print_stats()
        self.save_call_graph()

    def save_call_graph(self):
        """save the call graph
        """
        subprocess.run(
            [
                'gprof2dot -f pstats {} | dot -Tpng -o {}'.format(
                    self.config.base_file_name + 'profile.dump',
                    self.config.base_file_name + 'call-graph.png')
            ],
            shell=True,
            check=True)


FONT = {'family': 'monospace', 'size': '8'}


def plot_text(axes: plt.Axes, text: str):
    """Plot text on an axes

    Args:
        axes (plt.Axes): an axes object
        text (str): the text
    """

    axes.axis('off')
    axes.grid('off')
    axes.text(x=0, y=0, s=text, horizontalalignment='left', fontdict=FONT)


def plot_histo(axes: plt.Axes, data, bins=None, num_bins=50):
    """plot a histogram with probability density curve

    Args:
        axes (plt.Axes): axes object
        data (np.ndarray): an np array to be plotted
    """
    if bins is None:
        _, bins, _ = axes.hist(data, bins=num_bins)
    else:
        axes.hist(data, bins=bins)
    axes.set_ylabel('Frequency')

    return bins


def plot_box(axes: plt.Axes, data):
    """plot a box diagram
    """
    axes.boxplot(data)


def plot_1d(axes: plt.Axes, data):
    """plot 1D array
    """
    axes.plot(data)


def plot_fitness_accuracy(config: ColocationConfig, cache):
    """Plot the fitness and accuracy
    """
    fig: plt.Figure = plt.figure(figsize=(6, 4))
    plt.title('Correlational Score for {}'.format(config.job_name))
    axes = plt.subplot(121)
    axes.set_title("Correlational Score")
    axes.plot(cache['best_fitness'])
    axes = plt.subplot(122)
    axes.set_title("Accuracy")
    axes.plot(cache['accuracies'])
    plt.tight_layout()
    if config.save_figure:
        fig.savefig(config.base_file_name + 'fitness_accuracy.png', dpi=300)



def plot_cache(cache: dict, config: ColocationConfig):
    """Plot a cache dictionary
    """
    if config.plot_fitness_accuracy:
        plot_fitness_accuracy(config, cache)


@contextmanager
def visualizing(*,
                nrows=1,
                ncols=1,
                figsize=(6, 4),
                dpi=300,
                filename='',
                **kwargs):
    """Visualization figure
    """

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    yield axes
    plt.tight_layout()
    fig.savefig(filename, dpi=dpi)
