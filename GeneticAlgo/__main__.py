"""Main Module to run colocation optimizer
"""
import numpy as np
# from GeneticAlgo import tasks
from GeneticAlgo import visualization
from GeneticAlgo import paths
from GeneticAlgo import arg_parser
from GeneticAlgo import profiling


def main():
    """Running colocation solver
    """
    config = arg_parser.parse_args()
    print(config)

    # Process the configuration
    paths.create_dir(config.base_file_name)
    profiler = profiling.Profiler(config)
    np.random.seed(config.seed)

    if config.task in tasks.TASKS:
        task = tasks.TASKS[config.task]
        profiler.start()
        _, _, cache = task.run(config)
        profiler.stop()

        if config.visualize:
            visualization.plot_cache(cache, config)

        profiler.print_results()
    else:
        print("unknown task name.")


if __name__ == '__main__':
    main()
