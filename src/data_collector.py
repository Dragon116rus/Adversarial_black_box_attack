from bs4 import BeautifulSoup
import requests
import os
import json
from image_checker import ImageChecker
try:
    from tqdm import tqdm as tqdm
except:
    tqdm = lambda x : x

def get_imagenet_labels():
    """Return list of imagnet labels
    
    Returns:
        [list(str)] -- list of imagnet labels
    """
    with open('imagenet_class_index.json', 'r') as f:
        class_idx = json.load(f)
    imagenet_labels = [class_idx[str(k)][1] for k in range(len(class_idx))]
    return imagenet_labels

def get_binary_images(label):
    """Download img from google by label
    
    Arguments:
        label -- query to download
    """
    query = label.replace("_", "%20")
    content = requests.get("https://www.google.ru/search?q={}&tbm=isch".format(query)).content
    soup = BeautifulSoup(content)
    table_row = soup.find('table', {'class':'images_table'}).tr
    for number, td in enumerate(table_row):
        img_src = (td.img.attrs['src'])
        img_binary = requests.get(img_src).content
        yield img_binary

def create_dir(name):
    if not os.path.isdir(name):
        os.mkdir(name)

def save_jpg(img_binary, dir_name, file_number):
    with open(os.path.join(dir_name, "img{}.jpg".format(file_number)), "wb") as f:
        f.write(img_binary)

if __name__ == "__main__":
    checker = ImageChecker()
    imagenet_labels = get_imagenet_labels()
    for label_num, label in tqdm(enumerate(imagenet_labels)):
        create_dir(label)
        for i, img_binary in enumerate(get_binary_images(label)):
            if checker.check(img_binary, label_num): # check if all models correctly classified this image
                save_jpg(img_binary, dir_name = label, file_number = i)
                print(label, i)