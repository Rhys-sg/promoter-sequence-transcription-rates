import pandas as pd
import time
from tqdm import tqdm
import itertools

from GA.ParallelGeneticAlgorithm import ParallelGeneticAlgorithm
from .function_module import format_time

def test_params(param_ranges, target_expressions, lineages, kwargs, to_csv=None, iteration=1):
    results = []
    initial_time = time.time()
    
    # Generate all combinations of parameters
    param_keys = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    param_combinations = list(itertools.product(*param_values))

    total_combinations = len(param_combinations) * len(target_expressions) * lineages
    current_combination = 0
    progress_bar = tqdm(total=total_combinations, desc='Processing combinations', position=0)
    
    for param_combination in param_combinations:
        params = dict(zip(param_keys, param_combination))
        
        for target_expression in target_expressions:
            for lineage in range(lineages):

                if 'seed' in kwargs:
                    kwargs['seed'] += 1
                
                kwargs = {**kwargs, **params}
                
                # Create genetic algorithm with the current parameter combination
                ga = ParallelGeneticAlgorithm(**kwargs, target_expression=target_expression)

                start = time.time()
                ga.run()
                end = time.time()
                
                # Store results
                result = {
                    'target_expression': target_expression,
                    'lineage': lineage,
                    'sequence': ga.best_sequences[0],
                    'error': abs(target_expression - ga.best_predictions[0]),
                    'prediction': ga.best_predictions[0],
                    'run_time': end - start
                }
                results.append({**params, **result})

                # Update progress bar
                current_combination += 1
                progress_bar.update(1)
                elapsed_time = time.time() - initial_time
                progress_bar.set_postfix({
                    "Elapsed": format_time(elapsed_time),
                    "ETA": format_time(((elapsed_time / current_combination) * (total_combinations - current_combination)))
                })
    
    # Close progress bar
    progress_bar.close()

    results_df = pd.DataFrame(results)
    if to_csv != None:
        results_df.to_csv(to_csv, index=False)

    return results_df