import numpy as np
import matplotlib.pyplot as plt
import json
import os

def petrubate_img(numpy_image, petrubation):
  adversarial = numpy_image.copy()
  for k in range(0, len(petrubation), 5):
    x, y, r, g, b = petrubation[k: k+5]
    adversarial[x][y] = (r, g, b)
  return adversarial

def load_idx2label():
    with open('imagenet_class_index.json', 'r') as f:
        class_idx = json.load(f)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        label2idx = {label : index for index, label in enumerate(idx2label)}
    return idx2label, label2idx

def show_stats(model_, original, adversarial, pixels=0, orig_label_ind=None):
  idx2label, _ = load_idx2label()
  probas_orig = model_.get_probas(original)
  if orig_label_ind is None:
    orig_label_ind = np.argmax(probas_orig)
 
  orig_label = idx2label[orig_label_ind]
  orig_proba = probas_orig[orig_label_ind]

  probas_adver = model_.get_probas(adversarial)
  adver_label_ind = np.argmax(probas_adver)
  adver_label = idx2label[adver_label_ind]
  adver_proba = probas_adver[adver_label_ind]
  orig_proba_on_adver = probas_adver[orig_label_ind]

  plt.figure(figsize=(21, 21))

  plt.subplot(1, 3, 1)
  plt.title('Original - {}:{:.2f}\n'.format(orig_label, orig_proba))
  plt.imshow(np.uint8(original))
  plt.axis('off')

  plt.subplot(1, 3, 2)
  plt.title('Adverserial - {}:{:.2f}\nOriginal - {}:{:.2f}'.format(adver_label, adver_proba, orig_label, orig_proba_on_adver))
  plt.imshow(np.uint8(adversarial))
  plt.axis('off')

  plt.subplot(1, 3, 3)
  plt.title('Diff: pixels %s'%pixels)
  difference = (original - adversarial)[:,:, 0]
  petrub_pixels = (difference!=0).nonzero()
  adversarial_highlight = adversarial.copy()
  for pixel in zip(*petrub_pixels):
    highlight_pixel(adversarial_highlight, *pixel)
  plt.imshow(np.uint8(adversarial_highlight))
  plt.axis('off')
  plt.show()


def highlight_pixel(img, x, y, radius = 5):
  max_x, max_y = img.shape[:2]
  min_x, min_y = (0, 0)
  coors = np.array([0, radius])
  for tetta in np.linspace(0, np.pi*2, 40):
    rot_matrix = np.array([
        [np.cos(tetta), -np.sin(tetta)],
        [np.sin(tetta), np.cos(tetta)]
    ])
    x_delta, y_delta = coors.dot(rot_matrix).astype(np.int8)
    if ((x + x_delta >= min_x) & (x + x_delta < max_x) & (y + y_delta >= min_y) & (y + y_delta < max_y)):
      img[x + x_delta, y + y_delta] = (255, 0, 0)


def load_images_google():
  images_labels = []
  _, label2idx = load_idx2label()
  for root, dirs, files in os.walk("./"):
    for file in files:
      if file.endswith(".jpg"):
        path = (os.path.join(root, file))
        label = path.split('/')[1]
        if label in label2idx:     
          images_labels.append((path, label2idx[label]))
  return images_labels

def load_name2label():
  imagenet_name2label = {}
  with open('ILSVRC2012_val.txt', 'r') as f:
    for line in f:
      key, value = line.split()
      imagenet_name2label[key] = int(value)
  return imagenet_name2label

def load_images_imagenet():
  imagenet_name2label = load_name2label()
  images_labels = []
  for root, dirs, files in os.walk("./"):
    for file in files:
      if file.endswith(".JPEG"):
        path = (os.path.join(root, file))

        if file in imagenet_name2label:     
          images_labels.append((path, imagenet_name2label[file]))
  return images_labels