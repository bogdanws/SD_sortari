import numpy as np
import random
import string

def generate_random_uniform(size, min_val=0, max_val=1000000):
    return np.random.randint(min_val, max_val + 1, size=size).tolist()

def generate_sorted(size, min_val=0, max_val=1000000):
    return sorted(generate_random_uniform(size, min_val, max_val))

def generate_reverse_sorted(size, min_val=0, max_val=1000000):
    return sorted(generate_random_uniform(size, min_val, max_val), reverse=True)

def generate_partially_sorted(size, min_val=0, max_val=1000000, sorted_fraction=0.9):
    data = generate_sorted(size, min_val, max_val)
    num_unsorted = int(size * (1 - sorted_fraction))
    indices_to_randomize = random.sample(range(size), num_unsorted)

    for i in indices_to_randomize:
        data[i] = random.randint(min_val, max_val)
    
    return data

def generate_few_unique(size, min_val=0, max_val=1000000, unique_fraction=0.1):
    num_unique = max(1, int(size * unique_fraction))
    unique_values = np.random.randint(min_val, max_val + 1, size=num_unique)
    return np.random.choice(unique_values, size=size).tolist()

def generate_floats(size, min_val=0.0, max_val=1.0):
    return (np.random.rand(size) * (max_val - min_val) + min_val).tolist()

def generate_strings(size, length=10, chars=string.ascii_letters + string.digits):
    return [''.join(random.choice(chars) for _ in range(length)) for _ in range(size)]

GENERATORS = {
    'random_uniform_int': generate_random_uniform,
    'sorted_int': generate_sorted,
    'reverse_sorted_int': generate_reverse_sorted,
    'partially_sorted_int': generate_partially_sorted,
    'few_unique_int': generate_few_unique,
    'random_float': generate_floats,
    'random_ascii_string': generate_strings,
}

DATA_SIZES = [100, 1000, 10000, 100000, 1000000]

def get_datasets():
    datasets = {}
    for name, generator in GENERATORS.items():
        datasets[name] = {}
        for size in DATA_SIZES:
            max_val = size * 10 if 'int' in name else 1000000

            if 'float' in name:
                datasets[name][size] = generator(size)
            
            elif 'string' in name:
                datasets[name][size] = generator(size)
            
            else: # int types
                datasets[name][size] = generator(size, max_val=max_val)
    return datasets
