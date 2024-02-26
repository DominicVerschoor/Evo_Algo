import random


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

def tournament_selection(population, k=5):
    mating_pool = []
    while (len(mating_pool) != len(population)):
        group = random.sample(population, k)
        champion = sorted(group, key=lambda d: sum(d.keys()), reverse=True)[0]
        mating_pool.append(champion)

    return mating_pool

def crossover(parent1, parent2, prob=0.8):
    child1 = parent1.copy()
    child2 = parent2.copy()
    if (random.random() < prob):
        cross_pt = random.randint(0, min(len(parent1), len(parent2))-2)
        child1 = parent1[:cross_pt] + parent2[cross_pt:]
        child2 = parent2[:cross_pt] + parent1[cross_pt:]


weight_limit = 10
item_list = generate_item_list(weight_limit)

pop_size = 100
population = generate_population(item_list, weight_limit, pop_size)
reordered_population = ranking_population(population)[0]

pop_pool = tournament_selection(reordered_population)





