import numpy as np

def _is_unique_row(arr):
  sorted_idices = np.lexsort(arr.T, )
  sorted_data =  arr[sorted_idices,:]
  is_unique = np.concatenate( ([True], np.any(np.diff(sorted_data, axis=0), axis=1) ))
  is_unique_original = np.zeros_like(is_unique)
  is_unique_original[sorted_idices] = is_unique
  return is_unique_original
  
  
def _generate_population(bounds_min, bounds_max, population_size):
  population = np.zeros(shape=(population_size, len(bounds_min)), dtype=np.int32)
  for i in range(len(bounds_min)):
#     if (i%5 <3):
      population[:, i] = np.random.randint(bounds_min[i], bounds_max[i], size=population_size)
#     else:
#       population[: ,i] = np.random.normal(128, 127, size=population_size)
  population = _check_bounds(population, bounds_min, bounds_max)
  return population

def _check_bounds(array, bounds_min, bounds_max):
  return np.maximum(bounds_min, np.minimum(array, bounds_max))

def diff_evaluation(batch_score, bounds_min, bounds_max, max_iters = 100, population_size = 100, crossover_p = 0.5, f = None):
  count = 0
  if f is None:
    f = lambda x: x
  all_possible_indices = np.arange(population_size)
  genes_size = len(bounds_min)
  population = _generate_population(bounds_min, bounds_max, population_size) 
  # delete duplicates
  unique_rows = _is_unique_row(population)
  population = population[unique_rows]
  # scoring
  scores = batch_score(population)

  while count < max_iters: 
    new_semi_generation = np.zeros(shape=(population_size, len(bounds_min)), dtype=np.int32)
    for individual_ind in range(len(population)):
      # get random indices to mutate
      individual = population[individual_ind]
      random_indices = np.random.choice(all_possible_indices, 4, replace=False)
      
      # pick 3 indices which != individual_ind
      ind1, ind2, ind3 = random_indices[random_indices!=individual_ind][:3]
      
      # mutation
      donor = population[ind1] + f(population[ind2] - population[ind3])
      donor = _check_bounds(donor, bounds_min, bounds_max)

      # crossover
      binomial = np.random.binomial(1, crossover_p, genes_size)
      trial = binomial*individual + (1 - binomial) * donor

      new_semi_generation[individual_ind] = trial
      
    # delete duplicates 
    new_semi_generation = new_semi_generation[_is_unique_row(new_semi_generation)]
    
    # selection
    new_semi_generation_scores = batch_score(new_semi_generation)
    selection_generation = np.vstack((population, new_semi_generation))
    selection_generation_scores = np.hstack((scores, new_semi_generation_scores))
    # delete duplicates
    unique_rows = _is_unique_row(selection_generation)
    selection_generation = selection_generation[unique_rows]
    selection_generation_scores = selection_generation_scores[unique_rows]
    
    # take individuals with best scores
    selected_indices = selection_generation_scores.argsort()[:population_size]
    population = selection_generation[selected_indices]
    scores = selection_generation_scores[selected_indices]
      
    count+=1
  
  return population[scores.argsort()], scores[scores.argsort()]
  