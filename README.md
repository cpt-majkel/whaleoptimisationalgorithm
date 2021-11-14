# Whale Optimization Algorithm
Python implementation of the Whale Optimization Algorithm.

Whale Optimisation Algorithm (WOA) based on the swarm foraging behavior of humpback whales used to optimise 
neural network hyperparameters. Additionally, we have implemented a third dimension feature analysis to the 
original WOA algorithm to utilize 3D search space (3D-WOA).

### Description
Python implementation of the [Whale Optimization Algorithm](https://www.sciencedirect.com/science/article/pii/S0965997816300163). 
Additional information can be found at the [algorithm's webpage](http://www.alimirjalili.com/WOA.html).
Forked from [GitHub](https://github.com/docwza/woa)

## Installation

### Dependencies

- [Python](https://www.python.org/) 3.6
- [SciPy](https://keras.io/) 2.2.4
- TensorFlow2
- Keras

## Running the code

`python run.py` will run the WOA with default arguments. Run `python run.py --help` to learn more about command 
line arguments. 

To run 3D DNN example (which is default for now) run `python run.py -func DNN -nsols 5 -ngens 4 -max`. Specify 
constraints accordingly.
