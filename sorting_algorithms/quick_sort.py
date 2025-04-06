from typing import Literal
import copy
import random
import sys
import numpy as np

# increase the recursion limit
sys.setrecursionlimit(10**6)

from typing import Literal, Tuple

def quick_sort(arr: list[int], pivot_method: Literal['first', 'last', 'median', 'random'] = 'median') -> Tuple[list[int], int, int, int]:
    arr_copy = copy.deepcopy(arr)
    comparisons = 0
    swaps = 0
    try:
        # use iterative implementation for very large arrays to avoid recursion limit
        threshold = 2000 # sys.getrecursionlimit() is typically 1000, iterative is safer
        if len(arr_copy) > threshold:
            comparisons, swaps = _iterative_quicksort(arr_copy, 0, len(arr_copy) - 1, pivot_method)
        else:
            comparisons, swaps = _quicksort(arr_copy, 0, len(arr_copy) - 1, pivot_method)
        return arr_copy, comparisons, swaps, 0
    except RecursionError:
        print(f"Warning: Quick sort ({pivot_method}) on size {len(arr_copy)} hit recursion depth limit. Returning original.")
        # return original array and 0 counts as it failed
        return arr, np.nan, np.nan, np.nan
    except Exception as e:
        print(f"Warning: Quick sort ({pivot_method}) on size {len(arr_copy)} failed: {e}. Returning original.")
        return arr, np.nan, np.nan, np.nan

def _quicksort(arr: list[int], low: int, high: int, pivot_method: str) -> Tuple[int, int]:
    total_comparisons = 0
    total_swaps = 0
    if low < high:
        pivot_index, p_comparisons, p_swaps = _partition(arr, low, high, pivot_method)
        total_comparisons += p_comparisons
        total_swaps += p_swaps

        left_comparisons, left_swaps = _quicksort(arr, low, pivot_index - 1, pivot_method)
        total_comparisons += left_comparisons
        total_swaps += left_swaps

        right_comparisons, right_swaps = _quicksort(arr, pivot_index + 1, high, pivot_method)
        total_comparisons += right_comparisons
        total_swaps += right_swaps
    return total_comparisons, total_swaps

def _iterative_quicksort(arr: list[int], low: int, high: int, pivot_method: str) -> Tuple[int, int]:
    total_comparisons = 0
    total_swaps = 0
    # create an auxiliary stack
    stack = []

    # push initial values of low and high to stack
    stack.append((low, high))

    # keep popping from stack while it's not empty
    while stack:
        # pop low and high
        low, high = stack.pop()

        # if there are elements to sort
        if low < high:
            # partition the array and get pivot index, comparisons, and swaps
            pivot_index, p_comparisons, p_swaps = _partition(arr, low, high, pivot_method)
            total_comparisons += p_comparisons
            total_swaps += p_swaps

            # if there are elements on the left side of pivot,
            # push left side boundaries to stack
            if pivot_index - 1 > low:
                stack.append((low, pivot_index - 1))

            # if there are elements on the right side of pivot,
            # push right side boundaries to stack
            if pivot_index + 1 < high:
                stack.append((pivot_index + 1, high))
    return total_comparisons, total_swaps

def _partition(arr: list[int], low: int, high: int, pivot_method: str) -> Tuple[int, int, int]:
    comparisons = 0
    swaps = 0
    pivot_index = -1 # initialize pivot index

    if pivot_method == 'first':
        pivot_index = low
    elif pivot_method == 'last':
        pivot_index = high
    elif pivot_method == 'median':
        pivot_index, m_comparisons = _median_of_three(arr, low, high)
        comparisons += m_comparisons
    elif pivot_method == 'random':
        pivot_index = random.randint(low, high)
    else:
        raise ValueError("Invalid pivot method. Choose 'first', 'last', 'median', or 'random'.")

    # move pivot to high index for Lomuto partition scheme
    # check if pivot is already at high to avoid unnecessary swap
    if pivot_index != high:
        arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
        swaps += 1
    pivot = arr[high]

    i = low - 1 # index of smaller element

    for j in range(low, high):
        comparisons += 1 # comparison: arr[j] vs pivot
        if arr[j] <= pivot:
            i += 1
            # swap arr[i] and arr[j] only if they are different elements
            if i != j:
                arr[i], arr[j] = arr[j], arr[i]
                swaps += 1

    # swap pivot (arr[high]) with the element at arr[i + 1]
    # check if swap is necessary
    if (i + 1) != high:
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        swaps += 1

    return i + 1, comparisons, swaps # return pivot's final position, comparisons, swaps

def _median_of_three(arr: list[int], low: int, high: int) -> Tuple[int, int]:
    comparisons = 0
    mid = (low + high) // 2
    # ensure indices are distinct if possible
    if low == high: return low, 0
    if mid == low: # adjust mid if low+high is odd and low=0
        if low + 1 <= high:
             mid = low + 1
        else: # only two elements, low and high
             comparisons += 1
             return low if arr[low] <= arr[high] else high, comparisons

    a, b, c = arr[low], arr[mid], arr[high]
    comparisons += 3 # approximately 3 comparisons needed in the worst/average case

    # determine median
    if (a <= b <= c) or (c <= b <= a):
        return mid, comparisons
    elif (b <= a <= c) or (c <= a <= b):
        return low, comparisons
    else: # (b <= c <= a) or (a <= c <= b)
        return high, comparisons

def quicksort_first_pivot(arr: list[int]) -> Tuple[list[int], int, int, int]:
    try:
        return quick_sort(arr, 'first')
    except Exception as e:
        print(f"Warning: quicksort_first_pivot failed: {e}")
        return arr, np.nan, np.nan, np.nan

def quicksort_last_pivot(arr: list[int]) -> Tuple[list[int], int, int, int]:
    try:
        return quick_sort(arr, 'last')
    except Exception as e:
        print(f"Warning: quicksort_last_pivot failed: {e}")
        return arr, np.nan, np.nan, np.nan

def quicksort_median_pivot(arr: list[int]) -> Tuple[list[int], int, int, int]:
    try:
        return quick_sort(arr, 'median')
    except Exception as e:
        print(f"Warning: quicksort_median_pivot failed: {e}")
        return arr, np.nan, np.nan, np.nan

def quicksort_random_pivot(arr: list[int]) -> Tuple[list[int], int, int, int]:
    try:
        return quick_sort(arr, 'random')
    except Exception as e:
        print(f"Warning: quicksort_random_pivot failed: {e}")
        return arr, np.nan, np.nan, np.nan

# Example usage
if __name__ == "__main__":
    # Test different pivot methods
    test_data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    print("Original:", test_data)
    res_first, comps_first, swaps_first, allocs_first = quicksort_first_pivot(test_data[:])
    print(f"First pivot:  {res_first} (Comps: {comps_first}, Swaps: {swaps_first}, Aux Elements: {allocs_first})")
    assert res_first == sorted(test_data), "Test 1 Failed"
    res_last, comps_last, swaps_last, allocs_last = quicksort_last_pivot(test_data[:])
    print(f"Last pivot:   {res_last} (Comps: {comps_last}, Swaps: {swaps_last}, Aux Elements: {allocs_last})")
    assert res_last == sorted(test_data), "Test 2 Failed"
    res_median, comps_median, swaps_median, allocs_median = quicksort_median_pivot(test_data[:])
    print(f"Median pivot: {res_median} (Comps: {comps_median}, Swaps: {swaps_median}, Aux Elements: {allocs_median})")
    assert res_median == sorted(test_data), "Test 3 Failed"
    res_random, comps_random, swaps_random, allocs_random = quicksort_random_pivot(test_data[:])
    print(f"Random pivot: {res_random} (Comps: {comps_random}, Swaps: {swaps_random}, Aux Elements: {allocs_random})")
    assert res_random == sorted(test_data), "Test 4 Failed"
    
    # Test with negative numbers
    test_neg = [5, -2, 0, -3, 8, 1]
    print("\nWith negatives:", test_neg)
    res_neg, comps_neg, swaps_neg, allocs_neg = quick_sort(test_neg[:]) # Use copy
    print(f"Sorted (median): {res_neg} (Comps: {comps_neg}, Swaps: {swaps_neg}, Aux Elements: {allocs_neg})")
    assert res_neg == sorted(test_neg), "Test 5 Failed"

    print("All tests passed.")