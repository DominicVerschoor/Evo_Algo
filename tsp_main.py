import math
import random
import numpy

cities_size = 10
cities_list = range(cities_size)


def generate_population(pop_size, cities_list):
    population = []
    for i in pop_size:
        population.append(random.shuffle(cities_list.copy()))
    return population


def get_distance(city1, city2):
    raw_angle = 360 * abs(city1 - city2) / cities_size
    angle = min(raw_angle, 360 - raw_angle)

    distance = math.sqrt(2 - 2 * math.cos(angle))
    return distance


def get_fitness_score(cand_list):
    total_distance = 0
    i = 0
    while i+1 < len(cand_list):
        total_distance += get_distance(cand_list[i], cand_list[i+1])
        i += 1
    return total_distance


def tournament_selection(pop):
    pass


def pmx_crossover(parent1, parent2):
    midpoint = random.randint(1, len(parent1) - 1)
    cutoff_1 = random.randint(0, midpoint)
    cutoff_2 = random.randint(midpoint, len(parent1))

    offspring1 = [-1] * len(parent1)
    offspring2 = [-1] * len(parent2)

    offspring1[cutoff_1:cutoff_2] = parent2[cutoff_1:cutoff_2]
    offspring2[cutoff_1:cutoff_2] = parent1[cutoff_1:cutoff_2]
    print(offspring1)
    print(offspring2)

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

    return offspring1, offspring2


def ox_crossover(parent1, parent2):
    midpoint = random.randint(1, len(parent1) - 1)
    cutoff_1 = random.randint(0, midpoint)
    cutoff_2 = random.randint(midpoint, len(parent1))

    offspring1 = [-1] * len(parent1)
    offspring2 = [-1] * len(parent2)

    offspring1[cutoff_1:cutoff_2] = parent2[cutoff_1:cutoff_2]
    offspring2[cutoff_1:cutoff_2] = parent1[cutoff_1:cutoff_2]
    print(offspring1)
    print(offspring2)

    for i in range(cutoff_2, len(parent1) + cutoff_1):
        i = i % len(parent1)
        idx = i
        # print(f'outer {idx} val {parent1[idx]}')
        while parent1[idx] in offspring1:
            idx = (idx+1) % len(parent1)
        offspring1[i] = parent1[idx]

        idx = i
        while parent2[idx] in offspring2:
            idx = (idx+1) % len(parent1)
        offspring2[i] = parent2[idx]
        # print(offspring1)

    return offspring1, offspring2



if __name__ == '__main__':
    off1, off2 = ox_crossover([3, 4, 8, 2, 7, 1, 6, 5], [4, 2, 5, 1, 6, 8, 3, 7])
    print(f'final: {off1}')
    print(f'final: {off2}')