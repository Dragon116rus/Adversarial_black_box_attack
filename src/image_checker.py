import cv2
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

class ImageChecker:
    def __init__(self):
        """Class to check if image correctly classified by each model
        """
        resnet50 = models.resnet50(pretrained=True)
        resnet50.eval()
        alexnet = models.alexnet(pretrained=True)
        vgg16 = models.vgg16(pretrained=True)
        densenet = models.densenet161(pretrained=True)
        densenet.eval()
        self.models = [densenet, resnet50, vgg16, alexnet]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
    def check(self, img_binary, label_ind):
        img_tensor = self.preprocess(img_binary)
        for model in self.models:
            predict = model(img_tensor)[0].argmax()
            if predict != label_ind:
                return False
        return True

    def preprocess(self, img_binary):
        bgr_img = cv2.imdecode(np.frombuffer(img_binary, dtype=np.uint8), flags=cv2.IMREAD_UNCHANGED)
        bgr_img = cv2.resize(bgr_img, (224, 224)) 
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        img = rgb_img
        img = img.swapaxes(0, 1).swapaxes(0, 2) / 255
        tensor = torch.from_numpy(img)
        tensor = self.normalize(tensor.float())
        tensor = tensor.unsqueeze(0)
        return tensor