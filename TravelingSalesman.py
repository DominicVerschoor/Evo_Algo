import math
import random
import numpy
from matplotlib import pyplot as plt


def generate_cities(cities_size):
    return list(range(cities_size))

def generate_population(cities_list, pop_size):
    population = []
    for i in range(pop_size):
        cand_path = cities_list.copy()
        random.shuffle(cand_path)
        population.append(cand_path)
    return population


def ranking_population(population):
    scores_list = []

    for path in population:
        scores_list.append(get_fitness_score(path))

    scores_list, sorted_pop = zip(*sorted(zip(scores_list, population), reverse=True))
    return sorted_pop, scores_list


def tournament_selection(population, scores, prob=0.8, k=5):
    # Use tournament selection to get the fittest individuals of the population
    mating_pool = []
    while len(mating_pool) != len(population):
        group_idx = random.sample(range(len(population)), k)
        group = [population[i] for i in group_idx]
        group_score = [scores[i] for i in group_idx]
        # selects champion from each pool
        champion = select_champion(group, group_score, prob)
        mating_pool.append(champion)

    return mating_pool


def select_champion(group, group_score, prob):
    sorted_score, sorted_group = zip(*sorted(zip(group_score, group), reverse=True))
    # the champion is not always the highest scored, but sometimes second or third depending on the probability
    for i in range(len(sorted_group)):
        # decreasing probability of being chosen the lower ranked item
        prob = prob * ((1 - prob) ** i)
        if random.random() < prob:
            return sorted_group[i]

    return sorted_group[0]


def get_unit_distance(city1, city2, cities_size):
    raw_angle = 360 * abs(city1 - city2) / cities_size
    angle = min(raw_angle, 360 - raw_angle)
    distance = math.sqrt(2 - 2 * math.cos(angle))
    return distance


def get_fitness_score(cand_list):
    total_distance = 0
    i = 0
    while i + 1 < len(cand_list):
        total_distance += get_unit_distance(cand_list[i], cand_list[i + 1], len(cand_list))
        i += 1

    return total_distance


def get_population_fitness_scores(population):
    scores_list = []
    for path in population:
        scores_list.append(get_fitness_score(path))

    return scores_list



def pmx_crossover(parent1, parent2, cross_prob):
    if random.random() <= cross_prob:
        midpoint = random.randint(1, len(parent1) - 2)
        cutoff_1 = random.randint(0, midpoint)
        cutoff_2 = random.randint(midpoint, len(parent1) - 1)

        offspring1 = [-1] * len(parent1)
        offspring2 = [-1] * len(parent2)

        offspring1[cutoff_1:cutoff_2] = parent2[cutoff_1:cutoff_2]
        offspring2[cutoff_1:cutoff_2] = parent1[cutoff_1:cutoff_2]
        # print(offspring1)
        # print(offspring2)

        for i, (o1, p1) in enumerate(zip(offspring1, parent1)):
            # print(i, o1, p1)
            if o1 == -1:
                if p1 in offspring1:
                    idx = offspring1.index(p1)
                    # print(f'outer {i}: idx: {idx} val: {offspring2[idx]}')
                    while offspring2[idx] in offspring1:
                        idx = offspring1.index(offspring2[idx])
                        # print(f'inner {i}: idx: {idx} val: {offspring2[idx]}')
                    offspring1[i] = offspring2[idx]
                else:
                    offspring1[i] = p1
            # print(offspring1)

        # print(offspring1)
        # print(offspring2)
        for i, (o2, p2) in enumerate(zip(offspring2, parent2)):
            # print(i, o1, p1)
            if o2 == -1:
                if p2 in offspring2:
                    idx = offspring2.index(p2)
                    while offspring1[idx] in offspring2:
                        idx = offspring2.index(offspring1[idx])
                    offspring2[i] = offspring1[idx]
                else:
                    offspring2[i] = p2
            # print(offspring2)
    else:
        offspring1 = parent1
        offspring2 = parent2

    return offspring1, offspring2


def ox_crossover(parent1, parent2, cross_prob):
    if random.random() <= cross_prob:
        midpoint = random.randint(1, len(parent1) - 2)
        cutoff_1 = random.randint(0, midpoint)
        cutoff_2 = random.randint(midpoint, len(parent1) - 1)

        offspring1 = [-1] * len(parent1)
        offspring2 = [-1] * len(parent2)

        offspring1[cutoff_1:cutoff_2] = parent2[cutoff_1:cutoff_2]
        offspring2[cutoff_1:cutoff_2] = parent1[cutoff_1:cutoff_2]
        # print(offspring1)
        # print(offspring2)

        for i in range(cutoff_2, len(parent1) + cutoff_1):
            i = i % len(parent1)
            idx = i
            # print(f'outer {idx} val {parent1[idx]}')
            while parent1[idx] in offspring1:
                idx = (idx + 1) % len(parent1)
            offspring1[i] = parent1[idx]

            idx = i
            while parent2[idx] in offspring2:
                idx = (idx + 1) % len(parent1)
            offspring2[i] = parent2[idx]
            # print(offspring1)
    else:
        offspring1 = parent1
        offspring2 = parent2

    return offspring1, offspring2


def mutation(child):
    from_idx = random.randint(0, len(child)-1)
    to_idx = random.randint(0, len(child)-1)

    temp = child[from_idx]
    child[from_idx] = child[to_idx]
    child[to_idx] = temp

    return child


def next_generation(population, cross_version="pmx", cross_prob=0.8, mutate_prob=0.2):
    # generates the next generation
    pop = population.copy()
    next_gen = []

    while len(pop) != 0:
        # choose 2 random parents
        parents = random.sample(pop, k=2)

        # crossover the parents
        if cross_version.lower() == "pmx":
            child1, child2 = pmx_crossover(parents[0], parents[1], cross_prob)
        elif cross_version.lower() == "ox":
            child1, child2 = pmx_crossover(parents[0], parents[1], cross_prob)
        else:
            print("Please enter a valid form of crossover: 'pmx', 'ox'")
            break

        # randomly mutate
        if random.random() < mutate_prob:
            child1 = mutation(child1)

        if random.random() < mutate_prob:
            child2 = mutation(child2)

        # add child to next generation
        next_gen.append(child1)
        next_gen.append(child2)

        # remove parents from gene pool
        for parent in parents:
            pop.remove(parent)

    return next_gen


def genetic_algorithm(num_generations=100, cities_size=10, population_size=100, cross_version=["ox", "pmx"],
                      num_best_lists=2):
    print("Starting genetic Algorithm: Number of cities = {} Population size = {}...".format(cities_size,
                                                                                             population_size))
    print("---------------------------------------------------------------------------\n")
    best_lists = [[] for _ in range(num_best_lists)]

    # initialize all possible items and generate a population from them
    cities_list = generate_cities(cities_size)
    population = generate_population(cities_list, population_size)
    population_scores = get_population_fitness_scores(population)
    # first iteration of selection
    pop_pool = tournament_selection(population, population_scores)

    # append best score
    for i in range(num_best_lists):
        best_lists[i].append(ranking_population(pop_pool)[1][0])

    # generate N generations
    prev = pop_pool.copy()
    for i in range(num_generations):
        # generate different results for different parameters
        for j in range(num_best_lists):
            tmp = next_generation(prev, cross_version=cross_version[j])
            best_lists[j].append(ranking_population(tmp)[1][0])
            tmp_scores = get_population_fitness_scores(tmp)

            prev = tournament_selection(tmp, tmp_scores)

        print("Completed Generation", i)

    print("Algorithm Complete :)\n------------------------------------------------------------------\n")
    return best_lists


def plot(all_scores):
    # Plot each list in the same plot
    for i, sublist in enumerate(all_scores):
        plt.plot(sublist, label=f"Algorithm {i + 1}")

    # Add legend and labels
    plt.legend()
    plt.title("Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fittest Score")
    plt.show()


if __name__ == '__main__':
    random.seed(100)
    score_list = genetic_algorithm(num_generations=100,
                                   cities_size=10,
                                   population_size=100,
                                   cross_version=['pmx', 'ox'],
                                   num_best_lists=2)
    plot(score_list)

