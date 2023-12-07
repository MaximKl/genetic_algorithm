import random
import numpy as np

class Genetic_Algorithm:
    def __init__(self, desired_answer, accuracy, max_generations):
        self.__desired_answer = desired_answer
        self.__accuracy = accuracy
        self.__generations = max_generations

    def __fitness_function(self, x, y, z):
        return 6*x**3 +9*y**2+90*z - self.__desired_answer
    
    def test_fitness(self, x, y, z):
        return 6*x**3 +9*y**2+90*z
    
    def __selection(self, x, y, z):
        ans = self.__fitness_function(x, y, z)
        if ans == 0:
            return 9_999_999
        else:
            return abs(1/ans)

    def __get_best(self, ranked_population):
        elements = []
        parents = ranked_population[:int(len(ranked_population)/10)]
        for p in parents:
            elements.append(p[1][0])
            elements.append(p[1][1])
            elements.append(p[1][2])
        return elements

    def __mutation(self, gen_size, best_values):
        new_generation = []
        for _ in range(gen_size):
            e1 = random.choice(best_values) * random.uniform(0.99,1.01)
            e2 = random.choice(best_values) * random.uniform(0.99,1.01)
            e3 = random.choice(best_values) * random.uniform(0.99,1.01)
            new_generation.append([e1, e2,e3])
        return new_generation

    def get_random_generation(self, gen_size, random_boundary):
        population = []
        for _ in range(gen_size):
            population.append([random.uniform(0, random_boundary),
                            random.uniform(0, random_boundary),
                            random.uniform(0, random_boundary)])
        return population

    def genetic_algorithm(self, population):
        best_ancestors = []
        best_results = []
        for i in range(self.__generations):
            ranked_population = []
            for p in population:
                ranked_population.append( [self.__selection(p[0],p[1],p[2]), p] )
            ranked_population.sort(reverse=True)
            best_results.insert(0, [i, ranked_population[0][0]])
            best_ancestors.insert(0, [i, ranked_population[0][1]])

            if ranked_population[0][0] > self.__accuracy:
                return best_results[0], best_ancestors, np.array(best_results)
            
            population = self.__mutation(len(population), self.__get_best(ranked_population))
        best_results = np.array(best_results)
        return best_results[best_results[:,1].argmax()], best_ancestors, best_results

    def get_best_ancestor(self, best_generation, ancestors):
        ancestors = ancestors = np.array(ancestors, dtype="object")
        return ancestors[np.where(ancestors[:,0]==best_generation)][0][1]

    def get_accuracy(self, best_ancestor):
        acc = 1 - abs(self.__desired_answer - self.test_fitness(best_ancestor[0],best_ancestor[1],best_ancestor[2]))
        if acc>1 or acc<0:
            return 1/abs(acc)
        else:
            return acc