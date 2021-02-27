import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Display one image
def display_one(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()

# Display two images
def display(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()

def display_three(a, b, c, title1="Original", title2="Pred", title3="GroundTruth"):
    plt.subplot(1, 3, 1), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(c), plt.title(title3)
    plt.xticks([]), plt.yticks([])
    plt.show()

def display_data_point(x):
    display(x[0], x[1])

class config:
    data_path = "./data/processed/train"
    test_data_path = "./data/processed/test"
    should_load = True
    model_path = "./models"
    load_model = "denoiser_50_256.pb"
    model_name = "denoiser_{}_{}.pb" # epochs, img_shape

def sort_by_num(x):
    return int(x[x.find("_")+1:x.find(".")])

class DataSet:
    def __init__(self, path):
        self.get_file_list(path)
        self.data_path = path
        self.full_shape = None

    def get_full_path(self, data_file, x_y="x"):
        if x_y == "x":
            return os.path.join(os.path.join(self.data_path, "x"), data_file)
        else: 
            return os.path.join(os.path.join(self.data_path, "y"), data_file)

    def get_file_list(self, path):
        self.files_x = os.listdir(os.path.join(path, "x"))
        self.files_y = os.listdir(os.path.join(path, "y"))
        self.files_x.sort(key=sort_by_num)
        self.files_y.sort(key=sort_by_num)

    def num_chunks(self):
        return len(self.files_x)

    def get_full_shape(self):
        data = np.load(self.get_full_path(self.files_x[0]))
        self.full_shape = list(data.shape)
        self.full_shape[0] = None
        self.full_shape = tuple(self.full_shape)


    def get_img_shape(self):
        if not self.full_shape:
            self.get_full_shape()
        return (self.full_shape[1], self.full_shape[2], self.full_shape[3])

    def get_data_gen(self, epochs):
        for e in range(epochs):
            for file in self.files_x:
                x = np.load(self.get_full_path(file, "x"))
                y = np.load(self.get_full_path(file, "y"))
                yield (x, y)

    def get_random_sample(self):
        random1 = np.random.randint(self.num_chunks())
        x = np.load(self.get_full_path(self.files_x[random1], "x"))
        y = np.load(self.get_full_path(self.files_x[random1], "y"))
        random2 = np.random.randint(x.shape[0])
        print(x[random2].shape)
        return x[random2], y[random2]


def add_conv_layer(layer_params):
    return Conv2D(filters=layer_params["filters"], kernel_size=layer_params["kernel_size"], activation=layer_params["activation"], padding='same')
    

def get_model(img_shape):
    model_params = {
        "layers": [
            {"filters": 5, "kernel_size": 3, "activation": "relu"},
            {"filters": 10, "kernel_size": 4, "activation": "relu"},
            {"filters": 20, "kernel_size": 5, "activation": "relu"},
            {"filters": 10, "kernel_size": 4, "activation": "relu"},
            {"filters": 5, "kernel_size": 3, "activation": "relu"},
            {"filters": img_shape[2], "kernel_size": 3, "activation": "relu"},
        ]
    }

    model = Sequential()
    model.add(Input(shape=img_shape))
    for layer_params in model_params["layers"]:
        model.add(add_conv_layer(layer_params))

    return model

def train(model, dataset, prev_epochs=0):
    params = {
        "epochs": 50,
        "loss": 'mae',
        "optimizer": 'adam',
    }

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=config.model_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq= int(4 * dataset.num_chunks()))

    model.compile(loss=params["loss"], optimizer=params["optimizer"])
    history = model.fit(dataset.get_data_gen(params["epochs"]), epochs=params["epochs"], callbacks=[cp_callback], steps_per_epoch=dataset.num_chunks())
    model.save(os.path.join(config.model_path, config.model_name.format(params["epochs"] + prev_epochs, dataset.get_img_shape()[0])))


def demo(model, dataset):
    x, y_gt = dataset.get_random_sample()
    y_pred = model.predict(np.array([x]))[0]
    display_three(x, y_pred, y_gt)

def test(model, test_dataset, compile=True):
    params = {
        "loss": 'mae',
        "optimizer": 'adam',
    }

    if compile:
        model.compile(loss=params["loss"], optimizer=params["optimizer"])

    model.evaluate(test_dataset.get_data_gen(1), steps=test_dataset.num_chunks())
    

def main():
    train_dataset = DataSet(config.data_path)
    if (config.should_load):
        model = load_model(os.path.join(config.model_path, config.load_model))
    else:
        model = get_model(train_dataset.get_img_shape())
    model.summary()
    train(model, train_dataset, 50)

    test_dataset = DataSet(config.test_data_path)
    test(model, test_dataset, True)
    demo(model, test_dataset)

if __name__ == '__main__':
    main()