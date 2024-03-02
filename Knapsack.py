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

def ranking_population(population, weight_limit, fitness="cutoff"):
    # Reorders population from best-->worst item and returns the best score
    best_score = 0  # Initialize the best score
    for child in population:
        if (fitness.lower() == 'cutoff'):
            if not (sum(child.values()) > weight_limit):
                current_score = sum(child.keys())
                best_score = max(best_score, current_score)
        elif (fitness.lower() == 'adjusted'):
            current_score = adjust_score(child, weight_limit)
            best_score = max(best_score, current_score)

    return best_score

def adjust_score(child, weight_limit):
    # adjust the weights of the child if it is over the threshold
    total_weight = sum(child.values())
    if total_weight > weight_limit:
        # Calculate the negative difference
        diff = total_weight - weight_limit
        return (sum(child.keys()) - diff)
    else:
        return sum(child.keys())

def tournament_selection(population, weight_limit, fitness='cutoff', prob=0.8, k=5):
    # Use tournament selection to get the fittest individuals of the population
    mating_pool = []
    while len(mating_pool) != len(population):
        group = random.sample(population, k)
        # selects champion from each pool
        champion = select_champion(group, prob, weight_limit, fitness)
        # makes sure champion is not none
        while champion is None:
            group = random.sample(population, k)
            # selects champion from each pool
            champion = select_champion(group, prob, weight_limit, fitness)

        mating_pool.append(champion)

    return mating_pool

def select_champion(group, prob, weight_limit, fitness='cutoff'):

    def fitness_score(child):
        if fitness.lower() == 'cutoff':
            return sum(child.keys())
        elif fitness.lower() == 'adjusted':
            weight_difference = sum(child.values()) - weight_limit
            return sum(child.keys()) - weight_difference if weight_difference > 0 else sum(child.keys())
        
    cut_group = group.copy()
    if fitness.lower() == 'cutoff':
        # Filter out individuals with total weight exceeding the weight limit
        cut_group = [child for child in group if sum(child.values()) <= weight_limit]

    sorted_group = sorted(cut_group, key=fitness_score, reverse=True)

    total_prob = sum((1 - prob) ** i for i in range(len(sorted_group)))
    selected_prob = random.uniform(0, total_prob)
    cumulative_prob = 0
    for i, child in enumerate(sorted_group):
        cumulative_prob += (1 - prob) ** i
        if cumulative_prob >= selected_prob:
            return child

    return None

def single_point_crossover(parent1, parent2, prob=0.8):
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

    return child1, child2


def double_point_crossover(parent1, parent2, prob=0.8):
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
    child[new_value] = random.randint(1, weight_limit)

    while sum(child.values()) > weight_limit:
        child[new_value] = random.randint(1, weight_limit)

    return child


def next_generation(population, weight_limit, cross_version="single", cross_prob=0.8, mutate_prob=0.2):
    # generates the next generation
    pop = population.copy()
    next_gen = []

    while len(pop) != 0:
        # choose 2 random parents
        parents = random.sample(pop, k=2)

        # crossover the parents
        if cross_version.lower() == "single":
            child1, child2 = single_point_crossover(parents[0], parents[1],prob=cross_prob)
        elif cross_version.lower() == "double":
            child1, child2 = double_point_crossover(parents[0], parents[1], prob=cross_prob)
        else:
            print("Please enter a valid form of crossover: 'single', 'double'")
            break

        # mutate child1
        if random.random() < mutate_prob and sum(child1.values()) <= weight_limit:
            child1 = mutation(child1, weight_limit)

        # mutate child2
        if random.random() < mutate_prob and sum(child2.values()) <= weight_limit:
            child2 = mutation(child2, weight_limit)

        # add child to next generation
        next_gen.append(child1)
        next_gen.append(child2)

        # remove parents from gene pool
        for parent in parents:
            pop.remove(parent)

    return next_gen


def genetic_algorithm(num_generations=100, weight_limit=10, population_size=100, cross_version=["single", "double"], fitness=["cutoff", "adjusted"]):
    if len(cross_version) != len(fitness):
        print("Please enter have the same number of cross_version and fitness")
        return

    num_best_lists = len(cross_version)
    print("Starting genetic Algorithm: Weight limit = {} Population size = {}...".format(weight_limit, population_size))
    print("---------------------------------------------------------------------------\n")
    best_lists = [[] for _ in range(num_best_lists)]  

    # initialize all possible items and generate a population from them
    item_list = generate_item_list(weight_limit)
    population = generate_population(item_list, weight_limit, population_size)

    prev = [random.sample(population, population_size) for _ in range(num_best_lists)]

    # generate N generations
    for i in range(num_generations):
        # generate different results for different parameters
        for j in range(num_best_lists):
            tmp = next_generation(prev[j], weight_limit, cross_version=cross_version[j])
            prev[j] = tournament_selection(tmp, weight_limit, fitness=fitness[j], prob=0.8, k=5)

            best_lists[j].append(ranking_population(tmp, weight_limit=weight_limit, fitness=fitness[j]))

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
                               weight_limit=5000, 
                               population_size=1000, 
                               cross_version=["single", "single", "double", "double"], 
                               fitness=['cutoff', 'adjusted', 'cutoff', 'adjusted'])

plot(score_list)