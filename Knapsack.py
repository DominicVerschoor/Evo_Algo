import random
import matplotlib.pyplot as plt

def generate_item_list(weight_limit):
    pop = {}
    for n in range(1, weight_limit+1):
        pop[n] = random.randint(1,weight_limit)
    return pop

def generate_population(item_list, weight_limit, pop_size):
    pops = []
    for i in range(pop_size):
        group = {}
        keys = random.sample(list(item_list), k=len(item_list))
        for key in keys:
            weight = item_list[key]
            if (sum(group.values()) + weight <= weight_limit):
                group[key] = weight
        pops.append(group)
    return pops

def ranking_population(population):
    sorted_pop = sorted(population, key=lambda d: sum(d.keys()), reverse=True)
    scores_list = []
    for group in sorted_pop:
        scores_list.append(sum(group.keys()))
    return sorted_pop, scores_list

def tournament_selection(population, prob=0.8, k=5):
    mating_pool = []
    while (len(mating_pool) != len(population)):
        group = random.sample(population, k)
        champion = select_chapion(group, prob)
        mating_pool.append(champion)

    return mating_pool

def select_chapion(group, prob):
    sorted_group = sorted(group, key=lambda d: sum(d.keys()), reverse=True)
    for i in range(len(sorted_group)):
        prob = prob * ((1 - prob) ** i)
        if random.random() < prob:
            return sorted_group[i]

    return sorted_group[0]

def single_point_crossover(parent1, parent2, prob=0.8):
    child1 = parent1.copy()
    child2 = parent2.copy()
    if (random.random() < prob):
        list1 = list(parent1.items())
        list2 = list(parent2.items())
        cross_pt = random.randint(0, min(len(list1), len(list2)))

        # Create children by slicing and concatenating the lists
        child1_list = list1[:cross_pt] + list2[cross_pt:]
        child2_list = list2[:cross_pt] + list1[cross_pt:]

        # Convert lists back to dictionaries
        child1 = dict(child1_list)
        child2 = dict(child2_list)

    return child1, child2

import random

def double_point_crossover(parent1, parent2, prob=0.8):
    child1 = parent1.copy()
    child2 = parent2.copy()
    if random.random() < prob:
        list1 = list(parent1.items())
        list2 = list(parent2.items())

        # Ensure the crossover points are different
        if len(list1) >= 2 and len(list2) >= 2:
            cross_pt1, cross_pt2 = sorted(random.sample(range(min(len(list1), len(list2))), 2))

            # Create children by slicing and concatenating the lists
            child1_list = list1[:cross_pt1] + list2[cross_pt1:cross_pt2] + list1[cross_pt2:]
            child2_list = list2[:cross_pt1] + list1[cross_pt1:cross_pt2] + list2[cross_pt2:]

            # Convert lists back to dictionaries
            child1 = dict(child1_list)
            child2 = dict(child2_list)
        else:
            # Handle the case when the input dictionaries are too small
            child1, child2 = parent1.copy(), parent2.copy()

    return child1, child2   

def mutation(child, weight_limit):
    mutation_pt = random.choice(list(child))   
    del child[mutation_pt]

    new_value = random.randint(1, weight_limit) 
    while new_value in child:
        new_value = random.randint(1, weight_limit)
    
    child[new_value] = random.randint(1, weight_limit)

    return child

def next_generation(population, weight_limit, cross_version='single', cross_prob=0.8, mutate_prob=0.2):
    pop = population.copy()
    next_gen = []
    while len(pop) != 0:
        parents = random.sample(pop, k=2)

        if cross_version.lower() == 'single':
            child1, child2 = single_point_crossover(parents[0], parents[1], cross_prob)
        elif cross_version.lower() == 'double':
            child1, child2 = double_point_crossover(parents[0], parents[1], cross_prob)
        else:
            print('Please entera valid form of crossover: \'single\', \'double\'')
            break

        if random.random() < mutate_prob:
                    if random.random() < 0.5:
                        child1 = mutation(child1, weight_limit)
                    else:
                        child2 = mutation(child2, weight_limit)


        # discard children who go over weight limit
        while sum(child1.values()) > weight_limit or sum(child2.values()) > weight_limit:
            parents = random.sample(pop, k=2)

            if cross_version.lower() == 'single':
              child1, child2 = single_point_crossover(parents[0], parents[1], cross_prob)
            elif cross_version.lower() == 'double':
                child1, child2 = double_point_crossover(parents[0], parents[1], cross_prob)

            if random.random() < mutate_prob:
                    if random.random() < 0.5:
                        child1 = mutation(child1, weight_limit)
                    else:
                        child2 = mutation(child2, weight_limit)

        next_gen.append(child1)
        next_gen.append(child2)
        
        for parent in parents:
            pop.remove(parent)

    return next_gen

def genetic_algorithm(num_generations=100, weight_limit=10, population_size=100, cross_version='single'):
    print('Starting genetic Algorithm: Weight limit = {} Population size = {}...'.format(weight_limit, population_size))
    print('---------------------------------------------------------------------------\n')
    best_list = []
    best_list2 = []

    item_list = generate_item_list(weight_limit)
    population = generate_population(item_list, weight_limit, population_size)

    pop_pool = tournament_selection(population)
    best_list.append(ranking_population(pop_pool)[1][0])
    best_list2.append(ranking_population(pop_pool)[1][0])

    prev = pop_pool.copy()
    prev2 = pop_pool.copy()
    for i in range(num_generations):
        tmp = next_generation(prev, weight_limit, cross_version='single')
        tmp2 = next_generation(prev2, weight_limit, cross_version='double')
        
        best_list.append(ranking_population(tmp)[1][0])
        best_list2.append(ranking_population(tmp2)[1][0])

        prev = tournament_selection(tmp)
        prev2 = tournament_selection(tmp2)
        print('Completed Generation', i)

    print('Algorithm Complete :)\n------------------------------------------------------------------\n')
    return best_list, best_list2

def plot(all_scores):
    # Plot each list in the same plot
    for i, sublist in enumerate(all_scores):
        plt.plot(sublist, label=f'Algorithm {i + 1}')

    # Add legend and labels
    plt.legend()
    plt.title('Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fittest Score')    
    plt.show()

random.seed(100)
score_list1, score_list2 = genetic_algorithm(num_generations=1000, weight_limit=2000, population_size=2000)

plot([score_list1, score_list2])