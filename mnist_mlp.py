#!/usr/bin/env python

from blocks.bricks import Linear, Rectifier, Softmax, Sigmoid, Tanh
from theano import tensor
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.algorithms import GradientDescent, CompositeRule, Scale, StepRule
from blocks.utils import shared_floatx
from blocks.theano_expressions import l2_norm
from collections import OrderedDict
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.plot import Plot
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing

class StepSwitcher(StepRule):
    """Rescales an entire step if its L2 norm exceeds a threshold.
    When the previous steps are the gradients, this step rule performs
    gradient clipping.
    Parameters
    ----------
    threshold : float, optional
        The maximum permitted L2 norm for the step. The step
        will be rescaled to be not higher than this quanity.
        If ``None``, no rescaling will be applied.
    Attributes
    ----------
    threshold : :class:`.tensor.TensorSharedVariable`
        The shared variable storing the clipping threshold used.
    """
    def __init__(self, threshold=None, scale_higher_than_threshold = 0.2):
        if threshold:
            self.threshold = shared_floatx(threshold)
        self.scale_higher_than_threshold = scale_higher_than_threshold

    def compute_steps(self, previous_steps):
        if not hasattr(self, 'threshold'):
            return previous_steps
        norm = l2_norm(previous_steps.values())
        multiplier = tensor.switch(norm <= self.threshold,
                                   1, self.scale_higher_than_threshold)
        steps = OrderedDict(
            (param, step * multiplier)
            for param, step in previous_steps.items())
        return steps, []

def main():
    print("Build the network")
    input_of_image = tensor.matrix('features')
    input_to_hidden = Linear(name = 'input_to_hidden', input_dim = 784, output_dim = 100)
    h = Tanh().apply(input_to_hidden.apply(input_of_image))
    hidden_to_output = Linear(name = 'hidden_to_output', input_dim = 100, output_dim = 10)
    output_hat = Softmax().apply(hidden_to_output.apply(h))

    output = tensor.lmatrix('targets')
    cost = CategoricalCrossEntropy().apply(output.flatten(), output_hat)
    correct_rate =  1 - MisclassificationRate().apply(output.flatten(), output_hat)
    correct_rate.name = 'correct_rate'
    print(type(correct_rate))
    cost.name = 'cost'

    cg = ComputationGraph(cost)

    # Initialize the parameters
    input_to_hidden.weights_init = hidden_to_output.weights_init = IsotropicGaussian(0.01)
    input_to_hidden.biases_init = hidden_to_output.biases_init = Constant(0)
    input_to_hidden.initialize()
    hidden_to_output.initialize()

    # Train
    print("Prepare the data.")
    mnist_train = MNIST("train")
    mnist_test = MNIST("test")

    ## Carve the data into lots of batches.
    data_stream_train = DataStream(mnist_train, iteration_scheme=SequentialScheme(
        mnist_train.num_examples, batch_size = 256))

    ## Set the algorithm for the training.
    algorithm = GradientDescent(cost = cost, params = cg.parameters,
                                step_rule = CompositeRule([Scale(0.9), StepSwitcher(0.05,0.1)]) )

    ## Add a monitor extension for the training.
    data_stream_test = DataStream(mnist_test, iteration_scheme = SequentialScheme(
        mnist_test.num_examples, batch_size = 1024))

    test_monitor = DataStreamMonitoring(variables = [cost, correct_rate], data_stream = data_stream_test,
                                   prefix = "test" , after_every_epoch = True)

    train_monitor = TrainingDataMonitoring(variables = [cost, correct_rate, algorithm.total_step_norm],
                                           prefix = 'train',
                                           after_every_batch = True)

    ## Add a plot monitor.
    plot = Plot(document = 'new',
                channels=[['train_correct_rate','test_correct_rate']],
                start_server = True,
                after_every_batch = True)

    print("Start training")
    main_loop = MainLoop(algorithm=algorithm, data_stream=data_stream_train,
                         extensions=[plot,
                                     test_monitor,
                                     train_monitor,
                                     FinishAfter(after_n_epochs = 20),
                                     Printing()
                                     ])
    main_loop.run()

if __name__ == "__main__":
    main()

