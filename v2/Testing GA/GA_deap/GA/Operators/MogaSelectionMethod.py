import random
import math
from collections import Counter
from deap import tools  # type: ignore

class MogaSelectionMethod():
    '''
    This class implements various Multi-objective DEAP selection methods for genetic algorithms and stores selection parameters.
    It contains method references to the DEAP library.

    It does not include selEpsilonLexicase, as selAutomaticEpsilonLexicase is a more advanced version of this method.

    For single-objective selection methods, refer to SelectionMethod.py.
    
    '''
    def __init__(self):
        pass
        
    def selLexicase(self, *args, **kwargs):
        return self.selLexicase(*args, **kwargs)
    
    def selAutomaticEpsilonLexicase(self, *args, **kwargs):
        return self.selAutomaticEpsilonLexicase(*args, **kwargs)
    
    '''
    Non dominant selection methods:
    '''

    def selNSGA2(self, *args, **kwargs):
        return self.selNSGA2(*args, **kwargs)
    
    def selNSGA3(self, *args, **kwargs):
        return self.selNSGA3(*args, **kwargs)
    
    def selNSGA3WithMemory(self, *args, **kwargs):
        return self.selNSGA3WithMemory(*args, **kwargs)
    
    def selSPEA2(self, *args, **kwargs):
        return self.selSPEA2(*args, **kwargs)
    
    def sortNondominated(self, *args, **kwargs):
        return self.sortNondominated(*args, **kwargs)
    
    def sortLogNondominated(self, *args, **kwargs):
        return self.sortLogNondominated(*args, **kwargs)
    
    def selTournamentDCD(self, *args, **kwargs):
        return self.selTournamentDCD(*args, **kwargs)
    
    def uniform_reference_points(self, *args, **kwargs):
        return self.uniform_reference_points(*args, **kwargs)
    
    def get_all_methods():
        return [method for method in dir(MogaSelectionMethod) if method.startswith('sel')]