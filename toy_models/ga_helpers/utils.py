"""Printing Helper Functions
"""
from terminaltables import AsciiTable
from typing import Dict

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