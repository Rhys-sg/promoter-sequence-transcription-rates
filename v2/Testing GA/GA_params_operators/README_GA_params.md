# GA Parameter Optimization

### Description

These jupyter notebooks find the optimal values the GA hyperparameters based on (1) minimized error, (2) algorithm runtime, and (3) lineage divergence. The algorithm can accurately explore the sequence landscape and find pLac inserts that meet the relative expressions of 0.1 and higher. However, it struggles to generate pLac inserts (-10, spacer, and -35) that change the sequence's relative expressions to be close to 0. For robustness, we focus on target expression=0, but still test for 0.5 and 1. Because there is stochasticity in the results, we run each combination of parameters multiple times, with seeds for reproducibility.

Parameters:
* pop_size
* generations
* base_mutation_rate
* chromosomes
* elitist_rate
* lineage_divergence_alpha
* islands
    * gene_flow_rate
* surval_rate
* num_parents
* selection
    * num_competitors
    * boltzmann_temperature

### Test by file
We split the testing into XXX jupyter notebooks for readability and to avoid storing unnessesary data.

* **1_sub_params.ipynb** Some parameters are dependant on others. islands is conditional to gene_flow_rate while num_competitors and boltzmann_temperature only apply to the tournament and boltzmann selection methods, respectively. So, we find the optimal values for these parameters first.

* **2_param_grid_search.ipynb** We do a broad grid search for each parameter (or combination if interdependant) seperately. Then, if necessary, we repeat the grid searh, narrowing in on the minimum error and run time combinations. 

* **3_bayesian_search.ipynb** We use a probabilistic model (Gaussian process) to predict promising parameter combinations and iteratively improve them. Then, we compare these parameter combinations to see if they result in different minima compared to grid search. If they do, this means there are more interdependant parameters.

* **4_param_evaluations.ipynb** We evaluate the algorithm with optimized hyperparameters to see what limitations it still has.

### Notes:
Should we add code to test all combinations of high and low levels of parameters (full factorial design) to explicitly test for interactions, interdependance, or influence on each other. We would define a ranges for each hyperparameter (each with 5 values except for selection), compare influence?, then analyze results using ANOVA to identify significant interaction effects.