# GA Parameter Optimization

### Description

These jupyter notebooks find the optimal values the GA hyperparameters to minimize error and algorithm runtime. The algorithm can accurately explore the sequence landscape and find pLac inserts that meet the relative expressions of 0.1 and higher. However, it struggles to generate pLac inserts (-10, spacer, and -35) that change the sequence's relative expressions to be close to 0.

Parameters:
* pop_size
* generations
* base_mutation_rate
* chromosomes
* elitist_rate
* islands
    * gene_flow_rate
* surval_rate
* num_parents
* num_competitors
* selection
    * boltzmann_temperature

### Test by file
We split the testing into XXX jupyter notebooks for readability and to avoid storing unnessesary data.

* **1_sub_params.ipynb** We know gene_flow_rate is directly related to islands and boltzmann_temperature is only needed for boltzmann selection. So, we find the optimal values for these two parameters first, and set them in the interaction analysis.

* **2_param_interdependance.ipynb** We test all combinations of high and low levels of parameters (full factorial design) to explicitly test for interactions, interdependance, or influence on each other.

* **3_param_grid_search.ipynb** We do a broad grid search for each parameter (or combination if interdependant) seperately. Then, we repeat the grid searh, narrowing in on the minimum error and run time combinations. 

* **4_param_evaluations.ipynb** We evaluate the algorithm with optimized hyperparameters to see what limitations it still has.

* **5_GA_analysis.ipynb** We see how varying the masked section of pLac can change the upper and lower bounds of it's expression.

* **6_GA_insert_analysis.ipynb** The size of the masked region affects how well the GA can explore the expression landscape and reach the desired expression level. Here, we vary the mask length from 0-150 and analyze how well the GA can reach the desired expression level.

### Notes:
* Because there is stochasticity in the results, we run each combination of parameters multiple times, with seeds for reproducibility.
* More robust testing may optimize for a grid or random relative expression levels.
* Other optimization approaches include Bayesian Optimization, Random Search, or Grid Search without identifying and seperating the parameter space.