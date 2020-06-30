


import numpy
import ga
from scipy.spatial import distance

"""
The y=target is to maximize this equation ASAP:
    y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)
    What are the best values for the 6 weights w1 to w6?
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""

# Inputs of the equation.
# store the best solution for each variables for each iteration

bestSolutionsLits = {}
bestSolutionsResp = {}
bestSolutionsMasq = {}
bestSolutionsMed = {}
bestSolutionsTest = {}
bestSolutionsVeh = {}



# Inputs of the equation.
# Inputs ofthe equation for each variable
equation_inputs         = [4,2,3,5,11,4,3,4,2,6,6,8]
# equation_inputs y1 = beds inputs : y1 = [4,-2,3.5,5,-11,-4.7]
equation_inputsLits     = [4,2,3,5,11,4,3,4,2,6,6,8]
# equation_inputs y2 = respirators inputs : y2 = [4,-2,3.5,5,-11,-4.7]
equation_inputsResp     = [7,4,8,23,56,13,72,26,34,27,26,18]
# equation_inputs y3 = Masques inputs : y3 = [4,-2,3.5,5,-11,-4.7]
equation_inputsMasq     = [14,22,23,15,51,48,32,46,22,61,68,81]
# equation_inputs y4 = Medecin inputs : y4 = [4,-2,3.5,5,-11,-4.7]
equation_inputsMed      = [43,21,34,57,11,49,33,49,52,60,62,18]
# equation_inputs y5 = Test inputs : y5 = [4,-2,3.5,5,-11,-4.7]
equation_inputsTest     = [43,21,35,57,11,49,13,44,72,86,46,28]
# equation_inputs y6 = Vehicles inputs : y6 = [4,-2,3.5,5,-11,-4.7]
equation_inputsVeh      = [84,42,93,25,81,74,63,94,22,46,26,78]



distance_matrice = [
[84,42,93,25,81,74,63,94,22,46,26,78],
[84,42,93,25,81,74,63,94,22,46,26,78],
[84,42,93,25,81,74,63,94,22,46,26,78],
[84,42,93,25,81,74,63,94,22,46,26,78],
[84,42,93,25,81,74,63,94,22,46,26,78],
[84,42,93,25,81,74,63,94,22,46,26,78]
                    ]

resource_priority_matrice = [84,42,93,25,81,74,63,94,22,46,26,78] # weights are priority or cost for respective resources


# Number of the weights we are looking to optimize.

# num_weights = 6

num_weights = 12

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 8
num_parents_mating = 4

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
# new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)

#Creating the initial population for each axes variables yi
new_population = numpy.random.uniform(low=4.0, high=100.0, size=pop_size)
print(new_population)

new_populationLits = numpy.random.uniform(low=4.0, high=100.0, size=pop_size)
print(new_population)
new_populationResp = numpy.random.uniform(low=4.0, high=100.0, size=pop_size)
print(new_population)
new_populationMasq = numpy.random.uniform(low=4.0, high=100.0, size=pop_size)
print(new_population)
new_populationMed = numpy.random.uniform(low=4.0, high=100.0, size=pop_size)
print(new_population)
new_populationTest = numpy.random.uniform(low=4.0, high=100.0, size=pop_size)
print(new_population)
new_populationVeh = numpy.random.uniform(low=4.0, high=100.0, size=pop_size)
print(new_population)



num_generations = 5
for generation in range(num_generations):
    print("Generation : ", generation)

    # Measing the fitness of each chromosome in the population. for each variables or optim axes
    fitness = ga.cal_pop_fitness(equation_inputs, new_population)
    #fitnessY1 = ga.cal_pop_fitness(equation_inputs, new_population)
    fitnessY1 = ga.cal_pop_fitness(equation_inputsLits, new_populationLits)
    fitnessY2 = ga.cal_pop_fitness(equation_inputsResp, new_populationResp)
    fitnessY3 = ga.cal_pop_fitness(equation_inputsMasq, new_populationMasq)
    fitnessY4 = ga.cal_pop_fitness(equation_inputsMed, new_populationMed)
    fitnessY5 = ga.cal_pop_fitness(equation_inputsTest, new_populationTest)
    fitnessY6 = ga.cal_pop_fitness(equation_inputsVeh, new_populationVeh)

    # Selecting the best parents in the population for mating.
    parents   = ga.select_mating_pool(new_population, fitness,num_parents_mating)
    parentsY1 = ga.select_mating_pool(new_populationLits, fitnessY1,num_parents_mating)
    parentsY2 = ga.select_mating_pool(new_populationResp, fitnessY2,num_parents_mating)
    parentsY3 = ga.select_mating_pool(new_populationMasq, fitnessY3,num_parents_mating)
    parentsY4 = ga.select_mating_pool(new_populationMed, fitnessY4,num_parents_mating)
    parentsY5 = ga.select_mating_pool(new_populationTest, fitnessY5,num_parents_mating)
    parentsY6 = ga.select_mating_pool(new_populationVeh, fitnessY6,num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover   = ga.crossover(parents,offspring_size=(pop_size[0]-parents.shape[0], num_weights))
    offspring_crossoverY1 = ga.crossover(parentsY1,offspring_size=(pop_size[0]-parentsY1.shape[0], num_weights))
    offspring_crossoverY2 = ga.crossover(parentsY2,offspring_size=(pop_size[0]-parentsY2.shape[0], num_weights))
    offspring_crossoverY3 = ga.crossover(parentsY3,offspring_size=(pop_size[0]-parentsY3.shape[0], num_weights))
    offspring_crossoverY4 = ga.crossover(parentsY4,offspring_size=(pop_size[0]-parentsY4.shape[0], num_weights))
    offspring_crossoverY5 = ga.crossover(parentsY5,offspring_size=(pop_size[0]-parentsY5.shape[0], num_weights))
    offspring_crossoverY6 = ga.crossover(parentsY6,offspring_size=(pop_size[0]-parentsY6.shape[0], num_weights))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation   = ga.mutation(offspring_crossover)
    offspring_mutationY1 = ga.mutation(offspring_crossoverY1)
    offspring_mutationY2 = ga.mutation(offspring_crossoverY2)
    offspring_mutationY3 = ga.mutation(offspring_crossoverY3)
    offspring_mutationY4 = ga.mutation(offspring_crossoverY4)
    offspring_mutationY5 = ga.mutation(offspring_crossoverY5)
    offspring_mutationY6 = ga.mutation(offspring_crossoverY6)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :]       = parents
    new_population[parents.shape[0]:, :]        = offspring_mutation
    #Y1
    new_populationLits[0:parentsY1.shape[0], :] = parentsY1
    new_populationLits[parentsY1.shape[0]:, :]  = offspring_mutationY1
    #Y2
    new_populationResp[0:parentsY2.shape[0], :] = parentsY2
    new_populationResp[parentsY2.shape[0]:, :]  = offspring_mutationY2
    #Y3
    new_populationMasq[0:parentsY3.shape[0], :] = parentsY3
    new_populationMasq[parentsY3.shape[0]:, :]  = offspring_mutationY3
    #Y4
    new_populationMed[0:parentsY4.shape[0], :]  = parentsY4
    new_populationMed[parentsY4.shape[0]:, :]   = offspring_mutationY4
    #Y5
    new_populationTest[0:parentsY5.shape[0], :] = parentsY5
    new_populationTest[parentsY5.shape[0]:, :]  = offspring_mutationY5
    #Y6
    new_populationVeh[0:parentsY6.shape[0], :]  = parentsY6
    new_populationVeh[parentsY6.shape[0]:, :]   = offspring_mutationY6

    # The best result in the current iteration.
    print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))

# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = ga.cal_pop_fitness(equation_inputs, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))
print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])

# Y1: Getting the best solution after iterating finishing all generations.
fitnessY1 = ga.cal_pop_fitness(equation_inputsLits, new_populationLits)
# Then return the index of that solution corresponding to the best fitness.
best_match_idxY1 = numpy.where(fitnessY1 == numpy.max(fitnessY1))
print("Best solution Y1: ", new_populationLits[best_match_idx, :])
print("Best solution fitnessY1 : ", fitnessY1[best_match_idxY1])

# Y2: Getting the best solution after iterating finishing all generations.
fitnessY2 = ga.cal_pop_fitness(equation_inputsResp, new_populationResp)
# Then return the index of that solution corresponding to the best fitness.
best_match_idxY2 = numpy.where(fitnessY2 == numpy.max(fitnessY2))
print("Best solution Y2: ", new_populationResp[best_match_idxY2, :])
print("Best solution fitnessY2 : ", fitnessY2[best_match_idxY2])
