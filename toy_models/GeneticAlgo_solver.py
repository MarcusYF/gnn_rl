"""Strict GA Task
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import numpy as np
from toy_models.ga_helpers import strict_genetic_algorithm as ga
from imp import reload
reload(ga)
from toy_models.ga_helpers import calcs
from toy_models.ga_helpers import data_loader
from toy_models.ga_helpers import utils
import corr_score.c_score_func
from . import corr_score
from toy_models.ga_helpers.data_loader import dump_matrix
from k_cut import *
import dgl
from DQN import to_cuda
from toy_models.Qiter import vis_g
from tqdm import tqdm


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
    vprint = utils.compile_vprint_function(config.verbose)

    # Load matrix
    corr_matrix = data_loader.load_matrix(config.corr_matrix_path)

    # Compile functions
    # corr_func = corr_score.compile_solution_func(corr_matrix, config.type_count)
    def corr_func(solution):
        """Wrapper for a mean correlational score function.
        """
        score = corr_score.c_score_func.solution_mean_generic(corr_matrix, solution.astype(np.int32))
        return score

    weight_func = corr_score.compile_room_func(
        corr_matrix, config.type_count) if config.mutation_weighted else None

    population = ga.initialize_population(config.population_count, config.room_count,
                                          config.type_count)
    best_fitness = 0
    best_solution = None

    # print('pop', population.shape)
    # print('mat', corr_matrix.shape)

    assert not np.isnan(population).any(), 'population has nan'
    for iteration in range(config.max_iteration):
        fitnesses: np.ndarray = ga.fitness(population, corr_func)

        best, winners, losers = ga.suvival_of_fittest(fitnesses, config.survivor_count,
                                                      config.replaced_count)

        best_fitness = fitnesses[best]
        best_solution = np.copy(population[best])

        # if iteration % 100 == 0 and config.verbose:
        #     vprint('Iteration [{}]: {}', iteration, fitnesses[best])
        if config.plot_fitness_density:
            cache['fitnesses'].append(fitnesses)
        if config.plot_fitness_accuracy:
            cache['best_fitness'].append(fitnesses[best])
            cache['accuracies'].append(calcs.calculate_accuracy(population[best]))
        # print('polulation 0', population)


        population = ga.next_gen(
            population,
            winners,
            losers,
            config.crossing_over_rate,
            config.mutation_rate,
            weight_func=weight_func,
            manner=config.manner)
        # print('polulation 1', population)

    ids = np.argsort(fitnesses)
    # for id in ids:
    #     print("Fitness %f;" % (fitnesses[id]))

    if config.print_final_solution:
        print('Final Solution:')
        print(best_solution)

    return best_solution, calcs.calculate_accuracy(best_solution), cache, best_fitness


def run_ga(path_m, path_c):
    """Running colocation solver
    """
    #config = arg_parser.parse_args()
    config = utils.my_parse_args(['-m', path_m, '-c', path_c])
    # print(config)

    # Process the configuration
    utils.create_dir(config.base_file_name)
    profiler = utils.Profiler(config)
    np.random.seed(config.seed)

    profiler.start()
    best_solution, acc, cache, best_fitness = run(config)
    profiler.stop()

    if config.visualize:
        utils.plot_cache(cache, config)

    profiler.print_results()

    return best_solution, acc, best_fitness


if __name__ == '__main__':

    path_m = os.path.abspath(os.path.join(os.getcwd())) + '/toy_models/ga_helpers/corr_mat/dqn_5by6.mat'
    path_c = os.path.abspath(os.path.join(os.getcwd())) + '/toy_models/ga_helpers/configs/default.json'
    num_trial = 100
    config = utils.my_parse_args(['-m', path_m, '-c', path_c])
    k, m = config.room_count, config.type_count

    init_S = []
    best_S = []
    problem = KCut_DGL(k=k, m=m, adjacent_reserve=1, hidden_dim=1, mode='complete', sample_episode=1)

    for t in tqdm(range(num_trial)):
        problem.reset()
        init_S.append(problem.calc_S())

        _, _, sq_dist_matrix = dgl.transform.knn_graph(problem.g.ndata['x'], 1, extend_info=True)
        mat_5by6 = (2 - torch.sqrt(F.relu(sq_dist_matrix, inplace=True))[0]).numpy().astype('float64')
        m_path = os.path.abspath(os.path.join(os.getcwd())) + '/toy_models/ga_helpers/corr_mat/dqn_5by6.mat'
        dump_matrix(mat_5by6, m_path)

        # path_m = os.path.abspath(os.path.join(os.getcwd())) + '/ga_helpers/corr_mat/dqn_5by6.mat'
        # path_c = os.path.abspath(os.path.join(os.getcwd())) + '/ga_helpers/configs/default.json'
        best_solution, acc, best_fitness = run_ga(path_m, path_c)

        best_label = np.zeros(k*m).astype('int')
        for i in range(k):
            best_label[best_solution[i, :]] += i
        problem.g = to_cuda(problem.g)
        problem.reset_label(label=best_label)
        # print('Final S1:', problem.calc_S())
        # print('Final S2:', k * m * (m - 1) - k * m * (m - 1) / 2 * best_fitness)
        best_S.append(problem.calc_S())
        # path = os.path.abspath(os.path.join(os.getcwd())) + '/toy_models/figs/test1'
        # vis_g(problem, name=path, topo='cut')

    print('init S:', sum(init_S) / num_trial)
    print('best S:', sum(best_S) / num_trial)
    print('gain ratio:', (sum(init_S) - sum(best_S)) / sum(init_S))
