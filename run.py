import argparse

import numpy as np
from keras import layers, models, utils
from keras.datasets import mnist, fashion_mnist
import tensorflow as tf
from src.animate_scatter import AnimateScatter
from src.whale_optimization import WhaleOptimization
from sklearn.model_selection import KFold
from keras.datasets import reuters
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


num_folds = 3

cat2batch = {
    1: 8,
    2: 16,
    3: 32,
    4: 64,
    5: 128,
    6: 256,
    7: 512
}

cat2opt = {
    1: "SGD",
    2: "rmsprop",
    3: "adam",
    4: "adadelta",
    5: "adagrad",
    6: "adamax"
}

class NeuralNetwork:
    def __init__(self, train_samples, test_samples, fashion=False, optimizer="rmsprop"):
        self.opt = optimizer
        self.train_samples_no = train_samples
        self.test_samples_no = test_samples
        self._fashion = fashion
        self.train_images = (
            self.train_labels
        ) = self.test_images = self.test_labels = None
        self.categorical_train_labels = self.categorical_test_labels = None
        self.network = None
        self.prepareDataset()

    def prepareDataset(self):
        if self._fashion:
            (self.train_images, self.train_labels), (
                self.test_images,
                self.test_labels,
            ) = fashion_mnist.load_data()
        else:
            (self.train_images, self.train_labels), (
                self.test_images,
                self.test_labels,
            ) = mnist.load_data()
        self.train_images.astype(float)
        self.train_images = self.train_images / 255
        self.test_images.astype(float)
        self.test_images = self.test_images / 255
        self.train_images.reshape(self.train_samples_no, 28 * 28)
        self.test_images.reshape(self.test_samples_no, 28 * 28)
        self.categorical_train_labels = utils.to_categorical(self.train_labels)
        self.categorical_test_labels = utils.to_categorical(self.test_labels)

    def createArchitecture(self):
        self.network = None
        self.network = models.Sequential()
        self.network.add(layers.Flatten())
        self.network.add(layers.Dense(512, activation="relu"))
        self.network.add(layers.Dense(10, activation="softmax"))


class MultiNN:
    def __init__(self, optimizer="rmsprop"):
        self.opt = optimizer
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) \
            = reuters.load_data(num_words=10000)
        self.x_train = self.vectorize_sequences(self.train_data)
        # Our vectorized test data
        self.x_test = self.vectorize_sequences(self.test_data)
        # Vectorize labels
        self.y_train_raw = np.asarray(self.train_labels).astype('float32')
        self.y_test_raw = np.asarray(self.test_labels).astype('float32')

        # ONE-HOT ENCODING (convert categorical data into numbers)
        self.y_train = to_categorical(self.y_train_raw)
        self.y_test = to_categorical(self.y_test_raw)

        # Divide TEST dataset into TEST & VALIDATION sets
        #self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.2,
        #                                                                      shuffle=True)
        self.network = None

    def vectorize_sequences(self, sequences, dimension=10000):
        # Create an all-zero matrix of shape (len(sequences), dimension)
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.  # set specific indices of results[i] to 1s
        return results

    def prepareNetwork(self):
        self.network = None
        self.network = models.Sequential()
        self.network.add(layers.Dense(units=64, activation='relu', input_shape=(self.x_train[0].size,)))
        self.network.add(layers.Dense(units=64, activation='relu'))
        self.network.add(layers.Dense(units=46, activation='softmax'))


def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nsols",
        type=int,
        default=50,
        dest="nsols",
        help="number of solutions per generation, default: 50",
    )
    parser.add_argument(
        "-ngens",
        type=int,
        default=30,
        dest="ngens",
        help="number of generations, default: 20",
    )
    parser.add_argument(
        "-a",
        type=float,
        default=2.0,
        dest="a",
        help="woa algorithm specific parameter, controls search spread default: 2.0",
    )
    parser.add_argument(
        "-b",
        type=float,
        default=0.5,
        dest="b",
        help="woa algorithm specific parameter, controls spiral, default: 0.5",
    )
    parser.add_argument(
        "-c",
        type=float,
        default=None,
        dest="c",
        help="absolute solution constraint value, default: None, will use default constraints",
    )
    parser.add_argument(
        "-func",
        type=str,
        default="booth",
        dest="func",
        help="function to be optimized, default: booth; options: matyas, cross, eggholder, schaffer, booth",
    )
    parser.add_argument(
        "-r",
        type=float,
        default=0.25,
        dest="r",
        help="resolution of function meshgrid, default: 0.25",
    )
    parser.add_argument(
        "-t",
        type=float,
        default=0.1,
        dest="t",
        help="animate sleep time, lower values increase animation speed, default: 0.1",
    )
    parser.add_argument(
        "-max",
        default=False,
        dest="max",
        action="store_true",
        help="enable for maximization, default: False (minimization)",
    )

    args = parser.parse_args()
    return args


# optimization functions from https://en.wikipedia.org/wiki/Test_functions_for_optimization


def DNN(X, Y):
    out = []
    for i in range(0, len(X)):
        test_n = NeuralNetwork(train_samples=60000, test_samples=10000, fashion=True)
        kfold = KFold(n_splits=num_folds, shuffle=True)
        acc = []
        for train, test in kfold.split(test_n.train_images, test_n.train_images):
            trainX, valX = test_n.train_images[train], test_n.train_images[test]
            trainY, valY = test_n.categorical_train_labels[train], test_n.categorical_train_labels[test]
            test_n.createArchitecture()
            test_n.network.compile(optimizer=cat2opt[X[i]], loss='categorical_crossentropy',
                                   metrics=['accuracy'])
            score = test_n.network.fit(
                trainX,
                trainY,
                batch_size=128,
                epochs=round(Y[i]),
                verbose=0,
                validation_data=(valX, valY)
            )
            try:
                acc.append(score.history['val_accuracy'][-1])
            except:
                print(f"Error in val_accuracy readout: {score.history}")
        print(f"Optimiser {cat2opt[X[i]]}, epoch {Y[i]}, val accuracy kFold: {round(np.mean(acc), 4)}")
        out.append(np.mean(acc))

    return out

def mNN(X, Y):
    out = []
    for i in range(0, len(X)):
        test_n = MultiNN()
        kfold = KFold(n_splits=num_folds, shuffle=True)
        acc = []
        for train, test in kfold.split(test_n.x_train, test_n.x_train):
            trainX, valX = test_n.x_train[train], test_n.x_train[test]
            trainY, valY = test_n.y_train[train], test_n.y_train[test]
            test_n.prepareNetwork()
            test_n.network.compile(optimizer=cat2opt[X[i]], loss='categorical_crossentropy', metrics=['accuracy'])
            score = test_n.network.fit(
                trainX,
                trainY,
                batch_size=128,
                epochs=round(Y[i]),
                verbose=0,
                validation_data=(valX, valY)
            )
            try:
                acc.append(score.history['val_accuracy'][-1])
            except:
                print(f"Error in val_accuracy readout: {score.history}")
        print(f"Optimizer {cat2opt[X[i]]}, epoch {Y[i]}, val accuracy kFold: {round(np.mean(acc), 4)}")
        out.append(np.mean(acc))

    return out


def schaffer(X, Y):
    """constraints=100, minimum f(0,0)=0"""
    numer = np.square(np.sin(X ** 2 - Y ** 2)) - 0.5
    denom = np.square(1.0 + (0.001 * (X ** 2 + Y ** 2)))

    return 0.5 + (numer * (1.0 / denom))


def eggholder(X, Y):
    """constraints=512, minimum f(512, 414.2319)=-959.6407"""
    y = Y + 47.0
    a = (-1.0) * (y) * np.sin(np.sqrt(np.absolute((X / 2.0) + y)))
    b = (-1.0) * X * np.sin(np.sqrt(np.absolute(X - y)))
    return a + b


def booth(X, Y):
    """constraints=10, minimum f(1, 3)=0"""
    return ((X) + (2.0 * Y) - 7.0) ** 2 + ((2.0 * X) + (Y) - 5.0) ** 2


def matyas(X, Y):
    """constraints=10, minimum f(0, 0)=0"""
    return (0.26 * (X ** 2 + Y ** 2)) - (0.48 * X * Y)


def cross_in_tray(X, Y):
    """constraints=10,
    minimum f(1.34941, -1.34941)=-2.06261
    minimum f(1.34941, 1.34941)=-2.06261
    minimum f(-1.34941, 1.34941)=-2.06261
    minimum f(-1.34941, -1.34941)=-2.06261
    """
    B = np.exp(np.absolute(100.0 - (np.sqrt(X ** 2 + Y ** 2) / np.pi)))
    A = np.absolute(np.sin(X) * np.sin(Y) * B) + 1
    return -0.0001 * (A ** 0.1)


def levi(X, Y):
    """constraints=10,
    minimum f(1,1)=0.0
    """
    A = np.sin(3.0 * np.pi * X) ** 2
    B = ((X - 1) ** 2) * (1 + np.sin(3.0 * np.pi * Y) ** 2)
    C = ((Y - 1) ** 2) * (1 + np.sin(2.0 * np.pi * Y) ** 2)
    return A + B + C


def main():

    args = parse_cl_args()

    nsols = args.nsols
    ngens = args.ngens

    funcs = {
        "DNN": DNN,
        "mNN": mNN,
        "schaffer": schaffer,
        "eggholder": eggholder,
        "booth": booth,
        "matyas": matyas,
        "cross": cross_in_tray,
        "levi": levi,
    }
    func_constraints = {
        "DNN": 100.0,
        "mNN": 100.0,
        "schaffer": 100.0,
        "eggholder": 512.0,
        "booth": 10.0,
        "matyas": 10.0,
        "cross": 10.0,
        "levi": 10.0,
    }

    if args.func in funcs:
        func = funcs[args.func]
    else:
        print(
            "Missing supplied function "
            + args.func
            + " definition. Ensure function defintion exists or use command line options."
        )
        return

    if args.c is None:
        if args.func in func_constraints:
            args.c = func_constraints[args.func]
        else:
            print(
                "Missing constraints for supplied function "
                + args.func
                + ". Define constraints before use or supply via command line."
            )
            return

    C = args.c
    # first is batch size, second epoch
    constraints = [[1, 6], [1, 50]]

    opt_func = func

    b = args.b
    a = args.a
    a_step = a / ngens

    maximize = args.max
    solutions = []
    opt_alg = WhaleOptimization(opt_func, constraints, nsols, b, a, a_step, maximize)
    solutions.append(opt_alg.get_solutions2())
    colors = [[1.0, 1.0, 1.0] for _ in range(nsols)]

    # a_scatter = AnimateScatter(constraints[0][0],
    # constraints[0][1],
    # constraints[1][0],
    # constraints[1][1],
    # solutions, colors, opt_func, args.r, args.t)
    import time
    _time = []
    t2 = time.time()
    for i in range(ngens):
        #print("Gen {}".format(i))
        t1 = time.time()
        opt_alg.optimize()
        _time.append(time.time() - t1)
        solutions.append(opt_alg.get_solutions2())
        # a_scatter.update(solutions)
    gen_time = time.time() - t2
    _sols = opt_alg.print_best_solutions()
    _best = sorted(_sols, key=lambda x: x[0], reverse=maximize)[0]
    print("Results:")
    print(gen_time)
    print(_sols)
    print(_best)


if __name__ == "__main__":
    print("MNIST, epoch (0-50), discrete opt, batch 128")
    main()
