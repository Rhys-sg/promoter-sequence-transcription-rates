import random
import math
from collections import Counter
from deap import tools  # type: ignore

class MogaSelectionMethod():
    '''
    This class implements various Multi-objective DEAP selection methods for genetic algorithms and stores selection parameters.
    It contains method references to the DEAP library.

    For single-objective selection methods, refer to SelectionMethod.py.
    
    '''
    def __init__(self, boltzmann_temperature, tournsize):
        self.boltzmann_temperature = boltzmann_temperature
        self.tournsize = tournsize
        
    def selLexicase(self, *args, **kwargs):
        return self.selLexicase(*args, **kwargs)
    
    def selBest(self, *args, **kwargs):
        return self.selBest(*args, **kwargs)
    
    def selEpsilonLexicase(self, *args, **kwargs):
        return self.selEpsilonLexicase(*args, **kwargs)
    
    def selAutomaticEpsilonLexicase(self, *args, **kwargs):
        return self.selAutomaticEpsilonLexicase(*args, **kwargs)
    
    def get_all_methods():
        return [method for method in dir(MogaSelectionMethod) if method.startswith('sel')]