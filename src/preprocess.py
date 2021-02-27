import os
import uuid
import requests
import numpy as np
from os import path
from PIL import Image
from tempfile import NamedTemporaryFile as TmpFile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import logging


data_path = "data"
raw_path = "raw"
url_path = "urls"
processed_path = "processed"
url_file = "test_urls.txt"
progress_file = "progress.txt"
data_group = "test/{}"
data_file = "chunk_{}.npy"


def download_url(url, file):
    request = requests.get(url, timeout=10) 
    file.write(request.content)

def file_save_image(url, file, path):
    download_url(url, file)
    return Image.open(path)

def get_image(url, save):
    if (save):
        name = str(uuid.uuid1()) + ".jpg"
        full_path = path.join(path.join(data_path, raw_path), name)
        with open(full_path, 'wb') as file:
            img = file_save_image(url, file, full_path)
    else:
        with TmpFile(suffix=".jpg") as file:
            img = file_save_image(url, file, file.name)
    
    return img

def url_to_np_arr(url, save):
    img = get_image(url, save)
    return np.asarray(img)

# converts range from 0 - 255 to 0 to 1
def rerange(img):
    return img / 255

def resize(img, new_shape):
    return cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

def blur(img, kernel_size):
    return cv2.GaussianBlur(img, kernel_size, 0)

# takes in np arr
def preprocess_image(img, img_size):
    new_img = rerange(img)
    new_img = resize(new_img, img_size)
    return blur(new_img, (5, 5))

def get_preprocessed_img(url, save, img_size):
    img = url_to_np_arr(url, save)
    if (len(img.shape) == 2):
        # it is black and white so reshape into 3 channel image
        img = np.stack((img,)*3, axis=-1)

    if (img.shape[2] == 4):
        # includes alpha channel
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return preprocess_image(img, img_size)

def gauss_noise(img):
    row,col,ch= img.shape
    mean = 0.01
    var = 0.005
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    return img + gauss


def salt_pepper_noise(img):
    #change to random amounts of amount and svp
    row,col,ch = img.shape
    s_vs_p = 0.5
    amount = 0.01
    out = np.copy(img)
    
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[tuple(coords)] = 1


    num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    out[tuple(coords)] = 0
    return out

def speckle_noise(img):
    row,col,ch = img.shape
    weight = 0.15
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)        
    return img + img * gauss * weight

noises = [
    gauss_noise,
    salt_pepper_noise,
    speckle_noise,
]

def add_noise(img):
    noise_activation = np.random.randint(0, 2, len(noises))
    new_img = img
    for index in range(len(noises)):
        if noise_activation[index] == 1:
            new_img = noises[index](new_img)
    
    return new_img

def load_urls(text_file_name):
    full_path = path.join(path.join(data_path, url_path), text_file_name)
    with open(full_path, "r") as file:
        urls = file.readlines()

    for i in range(len(urls)):
        urls[i] = urls[i].strip()

    return urls

def add_x_data(data):
    ret = []
    for dp in data:
        ret.append(add_noise(dp))
    
    return np.array(ret)

def save_data(urls, chunk_size=64, img_size=(256,256), save=False):
    path_name = path.join(path.join(path.join(data_path, processed_path), data_group), data_file)
    curr_chunk = []
    chunk_index = 0
    for index in range(len(urls)):
        logging.info("index: {}".format(index))

        try:
            curr_chunk.append(get_preprocessed_img(urls[index], save, img_size))
        except Exception as e:
            logging.exception(e)
            logging.info(urls[index])

        if len(curr_chunk) > chunk_size or index == len(urls) - 1:            
            y_data = np.array(curr_chunk)

            with open(path_name.format("y", chunk_index), "wb") as file:
                np.save(file, y_data)

            x_data = add_x_data(y_data)

            with open(path_name.format("x", chunk_index), "wb") as file:
                np.save(file, x_data)

            curr_chunk = []
            chunk_index += 1

        
def main():
    urls = load_urls(url_file)
    save_data(urls)

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    main()