import numpy as np


def generate_population(bounds_min, bounds_max, population_size):
  population = np.zeros(shape=(population_size, len(bounds_min)), dtype=np.int32)
  for i in range(len(bounds_min)):
    population[:, i] = np.random.randint(bounds_min[i], bounds_max[i], size=population_size)
  return population

def check_bounds(array, bounds_min, bounds_max):
  return np.maximum(bounds_min, np.minimum(array, bounds_max))

def diff_evaluation(score, bounds_min, bounds_max, max_iters = 100, population_size = 100, crossover_p = 0.5):
  
  count = 0
  f = lambda x: x
  all_possible_indices = np.arange(population_size)
  genes_size = len(bounds_min)
  population = generate_population(bounds_min, bounds_max, population_size) #np.random.randint(-5, 5, size=genes_size*population_size).reshape(population_size, genes_size)

  scores = np.apply_along_axis(score, 1, population)

  while count < max_iters: 
    for individual_ind in range(population_size):
      # get random indices to mutate
      individual = population[individual_ind]
      random_indices = np.random.choice(all_possible_indices, 4, replace=False)
      if individual in random_indices:
        ind1, ind2, ind3 = random_indices[random_indices!=individual_ind]
      else:
        ind1, ind2, ind3 = random_indices[:3]
      # mutation
      donor = population[ind1] + f(population[ind2] - population[ind3])
      donor = check_bounds(donor, bounds_min, bounds_max)

      # crossover
      binomial = np.random.binomial(1, crossover_p, genes_size)
      trial = binomial*donor + (1 - binomial) * individual

      # selection
      score_trial = score(trial)
      if score_trial > scores[individual]:
        population[individual_ind] = trial
        scores[individual_ind] = score_trial
    count+=1
  return population[scores.argmax()], scores.argmax()
  # population