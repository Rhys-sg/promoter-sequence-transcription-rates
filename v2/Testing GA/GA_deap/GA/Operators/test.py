import math

def rate_over_generations(start_rate, end_rate, generation_idx, generations, mode='linear'):
    """
    Calculate the rate for a given generation, supporting linear and non-linear progression.

    Parameters:
        start_rate (float): The initial rate at generation 0.
        end_rate (float): The final rate at the last generation.
        generation_idx (int): The current generation index (0-based).
        generations (int): The total number of generations.
        mode (str): 'linear' or 'non-linear' (exponential easing). Default is 'linear'.

    Returns:
        float: The calculated rate for the current generation.
    """
    if generations <= 0:
        raise ValueError("Number of generations must be greater than zero.")
    if generation_idx < 0 or generation_idx > generations:
        raise ValueError("Generation index must be within the range [0, generations].")
    if mode not in ['linear', 'non-linear']:
        raise ValueError("Mode must be either 'linear' or 'non-linear'.")

    if mode == 'linear':
        # Linear interpolation
        rate = start_rate + (end_rate - start_rate) * (generation_idx / generations)
    elif mode == 'non-linear':
        # Non-linear (exponential progression)
        t = generation_idx / generations
        rate = start_rate + (end_rate - start_rate) * (math.pow(t, 1/3))

    return rate


# Example Usage
if __name__ == "__main__":
    start_rate = 1
    end_rate = 0.1
    generations = 100
    
    print("Linear Mode:")
    for i in range(generations + 1):
        print(f"Generation {i}: {rate_over_generations(start_rate, end_rate, i, generations, mode='linear')}")
    
    print("\nNon-Linear Mode:")
    for i in range(generations + 1):
        print(f"Generation {i}: {rate_over_generations(start_rate, end_rate, i, generations, mode='non-linear')}")

    # plot the rates
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(generations + 1)
    y_linear = [rate_over_generations(start_rate, end_rate, i, generations, mode='linear') for i in x]
    y_non_linear = [rate_over_generations(start_rate, end_rate, i, generations, mode='non-linear') for i in x]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_linear, label='Linear Mode')
    plt.plot(x, y_non_linear, label='Non-Linear Mode')
    plt.xlabel('Generation')
    plt.ylabel('Mutation Rate')
    plt.title('Mutation Rate Over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()
