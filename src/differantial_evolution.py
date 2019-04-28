import numpy as np

def _is_unique_row(arr):
  """for each row in arr return 'false' if arr have duplicate of this row above,
  in other words, if we delete rows with false, we get array without duplicates, 
  but amount of unique rows will be saved
  
  Arguments:
      arr {[type]} -- array of rows
  
  Returns:
      array of bool 
  """
  sorted_idices = np.lexsort(arr.T, )
  sorted_data =  arr[sorted_idices,:]
  is_unique = np.concatenate( ([True], np.any(np.diff(sorted_data, axis=0), axis=1) ))
  is_unique_original = np.zeros_like(is_unique)
  is_unique_original[sorted_idices] = is_unique
  return is_unique_original
  
  
def _generate_population(bounds_min, bounds_max, population_size):
  """generate population with bounds
  
  Arguments:
      bounds_min {[type]} -- array of minimum values for current gene
      bounds_max {[type]} -- array of maximum values for current gene
      population_size {[type]} -- population size
  """
  population = np.zeros(shape=(population_size, len(bounds_min)), dtype=np.int32)
  for i in range(len(bounds_min)):
#     if (i%5 <3):
      population[:, i] = np.random.randint(bounds_min[i], bounds_max[i], size=population_size)
#     else:
#       population[: ,i] = np.random.normal(128, 127, size=population_size)
  population = _check_bounds(population, bounds_min, bounds_max)
  return population

def _check_bounds(array, bounds_min, bounds_max):
  """truncate array with bounds
  (if array[i]>bounds_max[i]) then array[i] = bounds_max[i])
  (if array[i]<bounds_min[i]) then array[i] = bounds_min[i])
  Arguments:
      array {[type]} -- array to truncate
      bounds_min {[array]} -- array of minimal value for given gene
      bounds_max {[array]} -- array of maximal value for given gene
  
  Returns:
      [array] -- [truncated array]
  """
  return np.maximum(bounds_min, np.minimum(array, bounds_max))

def diff_evaluation(batch_score, bounds_min, bounds_max, max_iters = 100, population_size = 100, crossover_p = 0.5, f = None, no_changes_max_iters = 7):
  """DE algo
  
  Arguments:
      batch_score {[type]} -- scoring function which works with batches
      bounds_min {[type]} -- array of minimal value for given gene
      bounds_max {[type]} -- array of maximal value for given gene
  
  Keyword Arguments:
      max_iters {int} -- maximum iterations (default: {100})
      population_size {int} -- population size (default: {100})
      crossover_p {float} -- proba for crossover with donor (default: {0.5})
      f {[type]} -- function f (default: {None})
      no_changes_max_iters {int} -- stopping criteria, if for N iteration we best score not changed then stop  (default: {7})
  
  Returns:
      return last population (generation) sorted by score with corresponded scores
  """
  iter = 0
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

  # for stopping criteria
  no_changes = False
  no_changes_iters = 0
  no_changes_max_iters = 10
  min_score = np.inf

  while iter < max_iters and not no_changes: 
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
    if min_score > scores.min():
      min_score = scores.min()
      no_changes_iters = 0
    else:
      no_changes_iters += 1
      if no_changes_iters >= no_changes_max_iters:
        no_changes = True
    iter+=1
  
  return population[scores.argsort()], scores[scores.argsort()]
  