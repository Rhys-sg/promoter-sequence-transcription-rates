import random
import numpy as np
from deap import tools # type: ignore

"""
Evaluation methods for genetic and search algorithms.

Class structure:
    Each class is a "callable class," initialized with the necessary parameters
    and then called with the population to evaluate. The __call__ methods takes
    a population of individuals (either infills or full sequences) and returns
    a list of fitness values for each individual in the population.

The classes are currently only used in the MOGA, but may be used in other
algorithms in the future.

"""

class evalPredict():
    """
    Model-based evaluation method. Using a pre-trained model, it predicts the
    values of the reconstructed population and compares each to the target value.

    Attributes:
        model: A predictive model with a `predict()` method.
        target: Target value used to compute fitness scores.
    """
    def __init__(self, model, target):
        self.model = model
        self.target = target
    
    def __call__(self, reconstructed_population, **kwargs):
        predictions = self.model.predict(reconstructed_population)
        return 1 - abs(self.target - predictions)
    
class evalMaxDiversity():
    """
    Evaluation method that computes the MAXIMUM diversity of the infill
    population using pairwise Hamming distance.
    """
    def __call__(self, infill_population, **kwargs):
        return _evalDiversity(infill_population, max)
    
class evalMinDiversity():
    """
    Evaluation method that computes the MINIMUM diversity of the infill
    population using pairwise Hamming distance.
    """
    def __call__(self, infill_population, **kwargs):
        return _evalDiversity(infill_population, min)

def _evalDiversity(infill_population, method):
    fitnesses = []
    for i, current_ind in enumerate(infill_population, start=1):
        fitnesses.append(1-method([_hamming_distance(current_ind, other_ind) for other_ind in infill_population[:i:]]))
    return fitnesses

def _hamming_distance(ind1, ind2):
    return sum([1 for s, t in zip(ind1, ind2) if s != t]) / len(ind1)

class evalMaxDivergence(): 
    """
    Evaluation method that computes the MAXIMUM divergence between an infill
    and the previous best infill using Hamming distance.
    """
    def __call__(self, population, **kwargs):
        return _evalDivergence(population, max)

class evalMinDivergence():
    """
    Evaluation method that computes the MAXIMUM divergence between an infill
    and the previous best infill using Hamming distance.
    """
    def __call__(self, population, **kwargs):
        return _evalDivergence(population, min)

def _evalDivergence(self, population, method):
    if len(self.lineage_objects) == 0:
        return np.zeros(len(population))
    fitnesses = []
    for current_ind in population:
        fitnesses.append(1-method([self.hamming_distance(current_ind, previous_ind.best_sequence[0]) for previous_ind in self.lineage_objects]))
    return fitnesses

# TODO
# def evalNaturalHamming(population, reconstruct_sequence):
#     real_sequences = self.real_sequences
#     population = [self.reconstruct_sequence(ind) for ind in population]
#     fitnesses = []
#     for current_ind in population:
#         fitnesses.append(1 - min([hamming_distance(current_ind, real_sequence) for real_sequence in real_sequences]))
#     return fitnesses