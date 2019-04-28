from black_box_model import ModelPyTorch
from differantial_evolution import diff_evaluation
from utils import *
try:
    from IPython.display import clear_output
except:
    clear_output = lambda:()

def loss_batch(x_batch, numpy_image, model, label):
  """Computing score for differantial evalution
  
  Arguments:
      x_batch {[type]} -- array of petrubations
      numpy_image {[type]} -- image which attacked
      model {[type]} -- model
      label {[type]} -- label of original image
  
  Returns:
      return proba of correct label for petrubated images
  """
  images = []
  for petrubation in x_batch:
    image = numpy_image.copy()
    # petrubate image (it might be few pixels)
    for k in range(0, len(petrubation), 5):
      x_ = petrubation[k :k+5]
      x, y, r, g, b =  x_
      image[x][y] = (r, g, b)
    images.append(image)
  return model.get_probas_batch(images)[:, label]

def pixel_attack(image, img_label_idx, model_, pixels_per_iter = 1, max_pixels = 50176, max_iters=100, 
  population_size=None, no_changes_max_iters=10,  show_info=False):
    """implementation of attack
    
    Arguments:
        image {[type]} -- original image
        img_label_idx {[type]} -- original label of image
        model_ {[ModelPyTorch]} -- model
    
    Keyword Arguments:
        pixels_per_iter {int} -- the number of immediately attacked pixel  (default: {1})
        max_pixels {int} -- maximal amount of pixel which can be modified (default: {50176})
        max_iters {int} -- maximum iters for DE algo (default: {100})
        population_size {[type]} --  population_size for DE algo(default: {None})
        no_changes_max_iters {int} -- stoping criteria: if score not have changes for N iters, then stop (default: {10})
        show_info {bool} -- show images and probas (default: {False})
    
    Returns:
        [numpy_array] -- return numpy representation of adversarial image
    """
    adversarial = image.copy()
    i = 0 # number of iteration (if pixels_per_iter=1, then it equal to amount of attacked pixels)
    label = img_label_idx
    model_.requests = 0
    num_pixels = pixels_per_iter
    if population_size is None:
      population_size = pixels_per_iter**2*400
    if show_info:
        show_stats(model_, image, adversarial, i*num_pixels, label)
    
    while model_.get_probas(adversarial)[label] > 0.05 and max_pixels > i:
    
        res = diff_evaluation(lambda x: loss_batch(x, adversarial, model_, label), [0, 0, 0, 0, 0]*num_pixels, [223, 223, 255, 255, 255]*num_pixels,
                            max_iters=100, population_size=population_size, f= lambda x:x//2, crossover_p = 0.5, no_changes_max_iters=no_changes_max_iters)
        adversarial = petrubate_img(adversarial, res[0][0])
        
        i+=1
        if show_info:
            clear_output()
            show_stats(model_, image, adversarial, i*num_pixels, label)
            
    return adversarial

