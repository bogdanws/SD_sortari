from typing import List, Literal, Callable, Tuple
import math
import numpy as np

def shell_sort(arr: list[int], gap_sequence: Literal['shell', 'knuth', 'ciura', 'tokuda'] = 'shell') -> Tuple[list[int], int, int, int]:
    comparisons = 0
    swaps = 0
    n = len(arr)

    if n <= 1:
        return arr[:], 0, 0, 0

    arr_copy = arr[:]

    try:
        if gap_sequence == 'shell':
            gaps = _shell_gaps(n)
        elif gap_sequence == 'knuth':
            gaps = _knuth_gaps(n)
        elif gap_sequence == 'ciura':
            gaps = _ciura_gaps(n)
        elif gap_sequence == 'tokuda':
            gaps = _tokuda_gaps(n)
        else:
            raise ValueError("Gap sequence must be 'shell', 'knuth', 'ciura', or 'tokuda'.")

        for gap in gaps:
            for i in range(gap, n):
                temp = arr_copy[i] # store the element to be inserted
                swaps += 1 # count reading into temp as one movement
                j = i
                # shift earlier gap-sorted elements up until the correct location for arr_copy[i] is found
                while j >= gap:
                    comparisons += 1 # compare arr_copy[j - gap] with temp
                    if arr_copy[j - gap] > temp:
                        arr_copy[j] = arr_copy[j - gap] # shift element up
                        swaps += 1 # count shift as one movement
                        j -= gap
                    else:
                        # found the correct position (or element is smaller), stop shifting for this temp
                        break
                # place temp (the original arr_copy[i]) in its correct location
                if j != i: # only count as swap if position changed
                    arr_copy[j] = temp
                    swaps += 1 # count placing temp back as one movement

        return arr_copy, comparisons, swaps, 0
    except Exception as e:
        print(f"Warning: Shell sort ({gap_sequence}) on size {n} failed: {e}. Returning original.")
        return arr, np.nan, np.nan, np.nan

def _shell_gaps(n: int) -> List[int]:
    # n/2, n/4, ..., 1
    gaps = []
    gap = n // 2
    while gap > 0:
        gaps.append(gap)
        gap //= 2
    return gaps

def _knuth_gaps(n: int) -> List[int]:
    # (3^k - 1) / 2, for k = 1, 2, 3, ...
    gaps = []
    k = 1
    while True:
        gap = (3**k - 1) // 2
        if gap >= n:
            break
        gaps.append(gap)
        k += 1
    
    return gaps[::-1]

def _ciura_gaps(n: int) -> List[int]:
    # 1, 4, 10, 23, 57, 132, 301, 701, 1750, ...
    ciura = [1, 4, 10, 23, 57, 132, 301, 701, 1750]
    
    # generate larger gaps if needed
    k = len(ciura)
    while ciura[-1] < n // 2.25:
        ciura.append(int(ciura[-1] * 2.25))
        k += 1
    
    # return only the gaps that are less than n
    return [gap for gap in ciura[::-1] if gap < n]

def _tokuda_gaps(n: int) -> List[int]:
    # ceil(n/(2^(n+1))) and ends with 1
    gaps = []
    k = 1
    while True:
        gap = math.ceil((n / (9.0 * (9.0/4)**k - 4.0/9)))
        if gap <= 1:
            break
        gaps.append(gap)
        k += 1
    
    gaps.append(1)
    return gaps

# wrapper functions
def shell_sort_original(arr: list[int]) -> Tuple[list[int], int, int, int]:
    try:
        return shell_sort(arr, 'shell')
    except Exception as e:
        print(f"Warning: shell_sort_original failed: {e}")
        return arr, np.nan, np.nan, np.nan

def shell_sort_knuth(arr: list[int]) -> Tuple[list[int], int, int, int]:
    try:
        return shell_sort(arr, 'knuth')
    except Exception as e:
        print(f"Warning: shell_sort_knuth failed: {e}")
        return arr, np.nan, np.nan, np.nan

def shell_sort_ciura(arr: list[int]) -> Tuple[list[int], int, int, int]:
    try:
        return shell_sort(arr, 'ciura')
    except Exception as e:
        print(f"Warning: shell_sort_ciura failed: {e}")
        return arr, np.nan, np.nan, np.nan

def shell_sort_tokuda(arr: list[int]) -> Tuple[list[int], int, int, int]:
    try:
        return shell_sort(arr, 'tokuda')
    except Exception as e:
        print(f"Warning: shell_sort_tokuda failed: {e}")
        return arr, np.nan, np.nan, np.nan

if __name__ == "__main__":
    import random
    
    # Test original Shell sequence
    test1_orig = list(range(10))
    random.shuffle(test1_orig)
    print("Test 1 (Shuffled):", test1_orig)
    res1, comps1, swaps1, allocs1 = shell_sort_original(test1_orig)
    print(f"Sorted (Shell): {res1} (Comps: {comps1}, Swaps: {swaps1}, Aux Elements: {allocs1})")
    assert res1 == list(range(10)), "Test 1 Failed"
    
    # Test Knuth sequence
    test2_orig = list(range(10, 0, -1))
    print("\nTest 2 (Reversed):", test2_orig)
    res2, comps2, swaps2, allocs2 = shell_sort_knuth(test2_orig)
    print(f"Sorted (Knuth): {res2} (Comps: {comps2}, Swaps: {swaps2}, Aux Elements: {allocs2})")
    assert res2 == list(range(1, 11)), "Test 2 Failed"
    
    # Test Ciura sequence
    test3_orig = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    print("\nTest 3 (Duplicates):", test3_orig)
    res3, comps3, swaps3, allocs3 = shell_sort_ciura(test3_orig)
    print(f"Sorted (Ciura): {res3} (Comps: {comps3}, Swaps: {swaps3}, Aux Elements: {allocs3})")
    assert res3 == sorted(test3_orig), "Test 3 Failed"
    
    # Test Tokuda sequence
    test4_orig = [5, -2, 0, -3, 8, 1]
    print("\nTest 4 (Negatives):", test4_orig)
    res4, comps4, swaps4, allocs4 = shell_sort_tokuda(test4_orig)
    print(f"Sorted (Tokuda): {res4} (Comps: {comps4}, Swaps: {swaps4}, Aux Elements: {allocs4})")
    assert res4 == sorted(test4_orig), "Test 4 Failed"
    
    print("All tests passed.")