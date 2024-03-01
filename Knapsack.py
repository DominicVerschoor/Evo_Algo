import random
import matplotlib.pyplot as plt


def generate_item_list(weight_limit):
    # generate all possible scores assigned with random weights
    pop = {}
    for n in range(1, weight_limit + 1):
        pop[n] = random.randint(1, weight_limit)
    return pop


def generate_population(item_list, weight_limit, pop_size):
    # generate a population of randomly chosen items
    pops = []
    for i in range(pop_size):
        group = {}
        keys = random.sample(list(item_list), k=len(item_list))
        for key in keys:
            weight = item_list[key]
            # make sure that the item is valid
            if sum(group.values()) + weight <= weight_limit:
                group[key] = weight
        pops.append(group)
    return pops


def ranking_population(population):
    # reorders population from best-->worst item
    # also returns score of each item
    scores_list = []
    sorted_pop = sorted(population, key=lambda d: sum(d.keys()), reverse=True)
    for group in sorted_pop:
        scores_list.append(sum(group.keys()))
    return sorted_pop, scores_list


def tournament_selection(population, prob=0.8, k=5):
    # Use tournament selection to get the fittest individuals of the population
    mating_pool = []
    while len(mating_pool) != len(population):
        group = random.sample(population, k)
        # selects champion from each pool
        champion = select_champion(group, prob)
        mating_pool.append(champion)

    return mating_pool

def select_champion(group, prob):
    sorted_group = sorted(group, key=lambda d: sum(d.keys()), reverse=True)
    # the champion is not always the highest scored, but sometimes second or third depending on the probability
    for i in range(len(sorted_group)):
        # decreasing probability of being chosen the lower ranked item
        prob = prob * ((1 - prob) ** i)
        if random.random() < prob:
            return sorted_group[i]

    return sorted_group[0]

def adjust_weight(child, weight_limit):
    # adjust the weights of the child if it is over the threshold
    total_weight = sum(child.values())
    if total_weight > weight_limit:
        # Calculate the negative difference
        diff = total_weight - weight_limit
        for key in child:
            child[key] -= min(child[key], diff)
        return child
    else:
        return child

def single_point_crossover(parent1, parent2, weight_limit, prob=0.8, fitness="cutoff"):
    child1, child2 = parent1.copy(), parent2.copy()

    if random.random() < prob:
        list1 = list(parent1.items())
        list2 = list(parent2.items())
        cross_pt = random.randint(0, min(len(list1), len(list2)))

        # Create children by slicing and concatenating the lists
        child1_list = list1[:cross_pt] + list2[cross_pt:]
        child2_list = list2[:cross_pt] + list1[cross_pt:]

        child1 = dict(child1_list)
        child2 = dict(child2_list)

        # Check if child values exceed the weight limit
        if sum(child1.values()) > weight_limit or sum(child2.values()) > weight_limit:
            if (fitness.lower() == 'cutoff'):
                # If cutoff, copy parents
                child1, child2 = parent1.copy(), parent2.copy()
            elif (fitness.lower() == 'adjusted'):
                # if adjusted, assign negative weight
                child1 = adjust_weight(child1, weight_limit)
                child2 = adjust_weight(child2, weight_limit)
            else:
                print("Please enter a valid form of crossover: 'cutoff', 'adjusted'")

    return child1, child2


def double_point_crossover(parent1, parent2, weight_limit, prob=0.8, fitness="cutoff"):
    child1, child2 = parent1.copy(), parent2.copy()

    if random.random() < prob:
        list1 = list(parent1.items())
        list2 = list(parent2.items())

        # Ensure the crossover points are different
        if len(list1) >= 2 and len(list2) >= 2:
            cross_pt1, cross_pt2 = sorted(random.sample(range(min(len(list1), len(list2))), 2))

            # Create children by slicing and concatenating the lists
            child1_list = (list1[:cross_pt1] + list2[cross_pt1:cross_pt2] + list1[cross_pt2:])
            child2_list = (list2[:cross_pt1] + list1[cross_pt1:cross_pt2] + list2[cross_pt2:])

            child1 = dict(child1_list)
            child2 = dict(child2_list)

            # Check if child values exceed the weight limit
            if (sum(child1.values()) > weight_limit or sum(child2.values()) > weight_limit):
                if (fitness.lower() == 'cutoff'):
                    # If cutoff, copy parents
                    child1, child2 = parent1.copy(), parent2.copy()
                elif (fitness.lower() == 'adjusted'):
                    # if adjusted, assign negative weight
                    child1 = adjust_weight(child1, weight_limit)
                    child2 = adjust_weight(child2, weight_limit)
                else:
                    print("Please enter a valid form of crossover: 'cutoff', 'adjusted'")
        else:
            # Handle the case when the input dictionaries are too small
            child1, child2 = parent1.copy(), parent2.copy()

    return child1, child2


def mutation(child, weight_limit):
    # random mutation to a random child
    mutation_pt = random.choice(list(child))
    del child[mutation_pt]

    # give the mutation a valid score and weight
    new_value = random.randint(1, weight_limit)
    while new_value in child:
        new_value = random.randint(1, weight_limit)

    child[new_value] = random.randint(1, weight_limit)

    return child


def next_generation(population, weight_limit, cross_version="single", cross_prob=0.8, mutate_prob=0.2, fitness="cutoff"):
    # generates the next generation
    pop = population.copy()
    next_gen = []

    while len(pop) != 0:
        # choose 2 random parents
        parents = random.sample(pop, k=2)

        # crossover the parents
        if cross_version.lower() == "single":
            child1, child2 = single_point_crossover(parents[0], parents[1], weight_limit=weight_limit ,prob=cross_prob, fitness=fitness)
        elif cross_version.lower() == "double":
            child1, child2 = double_point_crossover(parents[0], parents[1], weight_limit=weight_limit, prob=cross_prob, fitness=fitness)
        else:
            print("Please enter a valid form of crossover: 'single', 'double'")
            break

        # randomly mutate
        if random.random() < mutate_prob:
            if random.random() < 0.5:
                child1 = mutation(child1, weight_limit)
            else:
                child2 = mutation(child2, weight_limit)

        # add child to next generation
        next_gen.append(child1)
        next_gen.append(child2)

        # remove parents from gene pool
        for parent in parents:
            pop.remove(parent)

    return next_gen


def genetic_algorithm(num_generations=100, weight_limit=10, population_size=100, cross_version=["single", "double"], fitness=["cutoff", "adjusted"], num_best_lists=2):
    
    print("Starting genetic Algorithm: Weight limit = {} Population size = {}...".format(weight_limit, population_size))
    print("---------------------------------------------------------------------------\n")
    best_lists = [[] for _ in range(num_best_lists)]  

    # initialize all possible items and generate a population from them
    item_list = generate_item_list(weight_limit)
    population = generate_population(item_list, weight_limit, population_size)

    # first iteration of selection
    pop_pool = tournament_selection(population)

    # append best score
    for i in range(num_best_lists):
        best_lists[i].append(ranking_population(pop_pool)[1][0])

    # generate N generations
    prev = pop_pool.copy()
    for i in range(num_generations):
        # generate different results for different parameters
        for j in range(num_best_lists):
            tmp = next_generation(prev, weight_limit, cross_version=cross_version[j], fitness=fitness[j])
            best_lists[j].append(ranking_population(tmp)[1][0])
            prev = tournament_selection(tmp)

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


random.seed(100)
score_list = genetic_algorithm(num_generations=1000, 
                               weight_limit=2000, 
                               population_size=2000, 
                               cross_version=["single", "single", 'double', 'double'], 
                               fitness=['cutoff', 'adjusted', 'cutoff', 'adjusted'], 
                               num_best_lists=4)

plot(score_list)
