import random
import math
from deap import base, creator, tools  # type: ignore

from .Lineage import Lineage
from .CNN import CNN

from .Operators.CrossoverMethod import CrossoverMethod
from .Operators.MutationMethod import MutationMethod
from .Operators.SelectionMethod import SelectionMethod

class GeneticAlgorithm:
    def __init__(
            self,
            cnn_model_path,
            masked_sequence,
            target_expression,
            use_cache=True,
            population_size=100,
            generations=100,
            crossover_rate=1,

            # Mutation parameters
            mutation_method='constant',
            mutation_rate=0.05,
            mutation_rate_start=0,
            mutation_rate_end=0.5,
            mutation_rate_degree=2
    ):
        # Genetic Algorithm attributes
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        # CNN model and attributes
        self.cnn = CNN(cnn_model_path)
        self.use_cache = use_cache

        # Evaluation attributes
        self.masked_sequence = self.cnn.one_hot_sequence(masked_sequence)
        self.mask_indices = self._get_mask_indices(self.masked_sequence)
        self.target_expression = target_expression

        # Mutation method
        self.mutation_method = getattr(MutationMethod(mutation_rate, mutation_rate_start, mutation_rate_end, mutation_rate_degree, generations), mutation_method)

        # Setup DEAP
        self.toolbox = base.Toolbox()
        self._setup_deap()

        # Lineage objects
        self.lineage_objects = []

    def _get_mask_indices(self, masked_sequence):
        return [i for i, element in enumerate(masked_sequence) if all(math.isclose(e, 0.25, rel_tol=1e-9) for e in element)]

    def _setup_deap(self):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        def generate_one_hot():
            nucleotide = [0, 0, 0, 0]
            nucleotide[random.randint(0, 3)] = 1
            return tuple(nucleotide)

        def generate_individual():
            return [generate_one_hot() for _ in range(len(self.mask_indices))]

        # Batch evaluation
        def eval_fitness_batch(population):
            population = [self._reconstruct_sequence(ind) for ind in population]
            predictions = self.cnn.predict(population, use_cache=self.use_cache)
            fitnesses = [1 - abs(pred - self.target_expression) for pred in predictions]
            return [(fit,) for fit in fitnesses]
        
        # Override map to process individuals in batches
        def batch_map(evaluate, individuals):
            return evaluate(individuals)

        self.toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", eval_fitness_batch)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", self._cx_one_hot)
        self.toolbox.register("mutate", self.mutation_method)

        self.toolbox.register("map", batch_map)
        
    def _reconstruct_sequence(self, infill):
        sequence = list(self.masked_sequence)
        for idx, char in zip(self.mask_indices, infill):
            sequence[idx] = char
        return sequence

    def _cx_one_hot(self, ind1, ind2):
        """Crossover: Swap nucleotides between individuals."""
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2

    def run(self, lineages=1):
        """Run multiple lineages of the Genetic Algorithm."""
        for lineage_id in range(lineages):
            lineage = Lineage(
                toolbox=self.toolbox,
                population_size=self.population_size,
                generations=self.generations,
                crossover_rate=self.crossover_rate,
                reconstruct_sequence=self._reconstruct_sequence,
                reverse_one_hot_sequence=self.cnn.reverse_one_hot_sequence,
                cnn=self.cnn
            )
            
            lineage.run()
            self.lineage_objects.append(lineage)
    
    @property
    def best_sequences(self):
        return [lineage.best_sequence for lineage in self.lineage_objects]
    
    @property
    def best_fitnesses(self):
        return [lineage.best_fitness for lineage in self.lineage_objects]
    
    @property
    def best_predictions(self):
        return [lineage.best_prediction for lineage in self.lineage_objects]
