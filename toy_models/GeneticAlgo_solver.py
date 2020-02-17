"""Strict GA Task
"""
import numpy as np
from toy_models.ga_helpers import strict_genetic_algorithm as ga
from toy_models.ga_helpers import calcs
from toy_models.ga_helpers import data_loader
from toy_models.ga_helpers import utils
from toy_models.ga_helpers import corr_score


def run(config: data_loader.ColocationConfig):
    """run a strict Genetic optimizer

    Args:
        config (dict): configurations

    Returns:
        best fitness (float), accuracy of best solution, cache
    """
    # Prepare cache
    cache = utils.get_cache(config)

    # Prepare verbose print function
    vprint = utils.printing.compile_vprint_function(config.verbose)

    # Load matrix
    corr_matrix = data_loader.load_matrix(config.corr_matrix_path)

    # Compile functions
    corr_func = corr_score.compile_solution_func(corr_matrix, config.type_count)

    weight_func = corr_score.compile_room_func(
        corr_matrix, config.type_count) if config.mutation_weighted else None

    population = ga.initialize_population(config.population_count, config.room_count,
                                          config.type_count)
    best_fitness = 0
    best_solution = None

    assert not np.isnan(population).any(), 'population has nan'
    for iteration in range(config.max_iteration):
        fitnesses: np.ndarray = ga.fitness(population, corr_func)

        best, winners, losers = ga.suvival_of_fittest(fitnesses, config.survivor_count,
                                                      config.replaced_count)

        best_fitness = fitnesses[best]
        best_solution = np.copy(population[best])

        if iteration % 100 == 0 and config.verbose:
            vprint('Iteration [{}]: {}', iteration, fitnesses[best])
            '''
            recalls = []
            for i in range(len(population)):
                recalls.append(cal_acc(population[i]))
            recalls = np.array(recalls)

            ids = np.argsort(fitnesses)
            for id in ids:
                print(id, " Fitness %f; Recall %f" % (fitnesses[id], recalls[id]))
            '''
        if config.plot_fitness_density:
            cache['fitnesses'].append(fitnesses)
        if config.plot_fitness_accuracy:
            cache['best_fitness'].append(fitnesses[best])
            cache['accuracies'].append(calcs.calculate_accuracy(population[best]))

        population = ga.next_gen(
            population,
            winners,
            losers,
            config.crossing_over_rate,
            config.mutation_rate,
            weight_func=weight_func)

    g_t = ga.ground_truth(population[0])
    fit: np.ndarray = ga.fitness(np.array([g_t], dtype=np.int32), corr_func)

    recalls = []
    for i in range(len(population)):
        recalls.append(ga.cal_acc(population[i]))
    recalls = np.array(recalls)

    ids = np.argsort(fitnesses)
    for id in ids:
        print("Fitness %f; Recall %f" % (fitnesses[id], recalls[id]))

    print("Ground Truth fitness:", fit[0])
    if config.print_final_solution:
        print('Final Solution:')
        print(best_solution)

    return best_solution, calcs.calculate_accuracy(best_solution), cache, fit[0], best_fitness