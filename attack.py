from black_box_model import ModelPyTorch
from differantial_evolution import diff_evaluation
from utils import *
try:
    from IPython.display import clear_output
except:
    clear_output = lambda:()

def loss_batch(x_batch, numpy_image, model, label):
  images = []
  for petrubation in x_batch:
    image = numpy_image.copy()
    for k in range(0, len(petrubation), 5):
      x_ = petrubation[k :k+5]
      x, y, r, g, b =  x_

      
      image[x][y] = (r, g, b)
    images.append(image)
  return model.get_probas_batch(images)[:, label]

def pixel_attack(image, img_label_idx, model_, pixels_per_iter = 1, max_pixels = 50176, show_info=False):
    adversarial = image.copy()
    i = 0
    num_pixels = pixels_per_iterlabel = img_label_idx
    model_.requests = 0
    show_stats(model_, image, adversarial, i*num_pixels, label)
    
    while model_.get_probas(adversarial)>0.05 and max_pixels > i:
    
        res = diff_evaluation(lambda x: loss_batch(x, adversarial, model_, label), [0, 0, 0, 0, 0]*num_pixels, [223, 223, 255, 255, 255]*num_pixels,
                            max_iters=100, population_size=(pixels_per_iter**2*100), f= lambda x:x//2, crossover_p = 0.5)
        adversarial = petrubate_img(adversarial, res[0][0])
        i+=1
        if show_info:
            clear_output()
            show_stats(model_, image, adversarial, i*num_pixels, label)
    return adversarial

