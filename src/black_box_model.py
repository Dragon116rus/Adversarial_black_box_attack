import numpy as np
import cv2
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

class ModelPyTorch:
  def __init__(self, model):
    """initilize some parameters
    
    Arguments:
        model  -- Interface for working with model
    """
    model.eval() # for deactivate training of dropout and batch norm
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
      model.cuda()
    self.model = model
    # for preprocessing image
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    self.loader = transforms.Compose([
      transforms.ToTensor(),
      self.normalize])
    # for returning proba
    self.softmax = torch.nn.Softmax(dim=1)
    
  def preprocess(self, numpy_image):
    """Convert numpy_image(array) to tensor
    
    Arguments:
        numpy_image -- numpy array which represent image
    
    Returns:
        tensor
    """
    tensor = self.loader(numpy_image)
    # adding dimension
    tensor = tensor.unsqueeze(0)
    return tensor.to(self.device)
  
  def get_probas(self, numpy_image):
    """Return predicted probas
    
    Arguments:
        numpy_image -- numpy array which represent image
    
    Returns:
        probas for each class
    """
    tensor_image = self.preprocess(numpy_image)
    predictions = self.model(tensor_image)

    probas = self.softmax(predictions)
    
    return probas[0].cpu().detach().numpy()
  
  def preprocess_batch(self, numpy_images):
    """Same as preprocess, but for few images
    
    Arguments:
        numpy_images -- array of images
    
    Returns:
        tensor
    """
    tensor_imgs = torch.zeros(size = (len(numpy_images), 3, 224, 224), device=self.device)
    for k, numpy_image in enumerate(numpy_images):
      tensor = self.loader(numpy_image)
      tensor_imgs[k] = tensor
    return tensor_imgs.to(self.device)
  
  def get_probas_batch(self, numpy_images):
    """Same as get_probas, but for few images
    
    Arguments:
        numpy_images -- array of images
    
    Returns:
        array of probas for each class
    """
    tensor_images = self.preprocess_batch(numpy_images)
    predictions = self.model(tensor_images)
    probas = self.softmax(predictions)
    return probas.cpu().detach().numpy()
  
  def get_label(self, numpy_image):
    """Return number of class with maximum proba
    
    Arguments:
        numpy_image -- numpy array which represent image
    
    Returns:
        [int] -- which corresponds to maximum probabal class
    """
    predictions = (self.get_probas(numpy_image))
    return np.argmax(predictions)