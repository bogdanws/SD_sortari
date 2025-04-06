from typing import List, Literal, Tuple
import numpy as np

def heap_sort(arr: List[int], heap_type: Literal['max', 'min'] = 'max') -> Tuple[List[int], int, int, int]:
    comparisons = 0
    swaps = 0
    try:
        arr_copy = arr[:]
        n = len(arr_copy)
        if n <= 1:
            return arr_copy, 0, 0, 0

        # build heap (max or min)
        build_comps, build_swaps = _build_heap(arr_copy, n, heap_type)
        comparisons += build_comps
        swaps += build_swaps

        # extract elements one by one
        for i in range(n - 1, 0, -1):
            # swap root (max/min element) with the last element of the heap
            arr_copy[0], arr_copy[i] = arr_copy[i], arr_copy[0]
            swaps += 1
            # heapify the reduced heap
            heapify_comps, heapify_swaps = _heapify(arr_copy, i, 0, heap_type)
            comparisons += heapify_comps
            swaps += heapify_swaps

        # if min heap was used, the result is in descending order, reverse it
        if heap_type == 'min':
            arr_copy.reverse()

        return arr_copy, comparisons, swaps, 0 # 0 auxiliary elements for in-place sort
    except RecursionError:
        print(f"Warning: Heap sort ({heap_type}) on size {len(arr)} hit recursion depth limit. Returning original.")
        return arr, np.nan, np.nan, np.nan
    except Exception as e:
        print(f"Warning: Heap sort ({heap_type}) on size {len(arr)} failed: {e}. Returning original.")
        return arr, np.nan, np.nan, np.nan

def _build_heap(arr: List[int], n: int, heap_type: str) -> Tuple[int, int]:
    total_comparisons = 0
    total_swaps = 0
    # start from the last non-leaf node and heapify down
    for i in range(n // 2 - 1, -1, -1):
        comps, swaps = _heapify(arr, n, i, heap_type)
        total_comparisons += comps
        total_swaps += swaps
    return total_comparisons, total_swaps

def _heapify(arr: List[int], n: int, i: int, heap_type: str) -> Tuple[int, int]:
    comparisons = 0
    swaps = 0
    current_root = i
    left = 2 * i + 1
    right = 2 * i + 2

    if heap_type == 'max':
        # find largest among root, left child, and right child
        if left < n:
            comparisons += 1
            if arr[left] > arr[current_root]:
                current_root = left

        if right < n:
            comparisons += 1
            if arr[right] > arr[current_root]:
                current_root = right

        # if largest is not root
        if current_root != i:
            arr[i], arr[current_root] = arr[current_root], arr[i]
            swaps += 1
            # recursively heapify the affected sub-tree
            rec_comps, rec_swaps = _heapify(arr, n, current_root, heap_type)
            comparisons += rec_comps
            swaps += rec_swaps
    else: # min heap
        # find smallest among root, left child, and right child
        if left < n:
            comparisons += 1
            if arr[left] < arr[current_root]:
                current_root = left

        if right < n:
            comparisons += 1
            if arr[right] < arr[current_root]:
                current_root = right

        # if smallest is not root
        if current_root != i:
            arr[i], arr[current_root] = arr[current_root], arr[i]
            swaps += 1
            # recursively heapify the affected sub-tree
            rec_comps, rec_swaps = _heapify(arr, n, current_root, heap_type)
            comparisons += rec_comps
            swaps += rec_swaps

    return comparisons, swaps

# wrapper functions
def heapsort_max_heap(arr: List[int]) -> Tuple[List[int], int, int, int]:
    try:
        return heap_sort(arr, 'max')
    except Exception as e:
        print(f"Warning: heapsort_max_heap failed: {e}")
        return arr, np.nan, np.nan, np.nan

def heapsort_min_heap(arr: List[int]) -> Tuple[List[int], int, int, int]:
    try:
        return heap_sort(arr, 'min')
    except Exception as e:
        print(f"Warning: heapsort_min_heap failed: {e}")
        return arr, np.nan, np.nan, np.nan

if __name__ == "__main__":
    # Test basic functionality
    test_data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    print("Original:", test_data)
    res_max, comps_max, swaps_max, allocs_max = heapsort_max_heap(test_data[:])
    print(f"Sorted (max heap): {res_max} (Comps: {comps_max}, Swaps: {swaps_max}, Aux Elements: {allocs_max})")
    assert res_max == sorted(test_data), "Test 1 Failed"
    res_min, comps_min, swaps_min, allocs_min = heapsort_min_heap(test_data[:])
    print(f"Sorted (min heap): {res_min} (Comps: {comps_min}, Swaps: {swaps_min}, Aux Elements: {allocs_min})")
    assert res_min == sorted(test_data), "Test 2 Failed"
    # Test with negative numbers
    test_neg = [5, -2, 0, -3, 8, 1]
    print("\nWith negatives:", test_neg)
    res_neg_max, comps_neg_max, swaps_neg_max, allocs_neg_max = heapsort_max_heap(test_neg[:])
    print(f"Sorted (max heap): {res_neg_max} (Comps: {comps_neg_max}, Swaps: {swaps_neg_max}, Aux Elements: {allocs_neg_max})")
    assert res_neg_max == sorted(test_neg), "Test 3 Failed"
    res_neg_min, comps_neg_min, swaps_neg_min, allocs_neg_min = heapsort_min_heap(test_neg[:])
    print(f"Sorted (min heap): {res_neg_min} (Comps: {comps_neg_min}, Swaps: {swaps_neg_min}, Aux Elements: {allocs_neg_min})")
    assert res_neg_min == sorted(test_neg), "Test 4 Failed"
    # Test empty list
    res_empty, comps_empty, swaps_empty, allocs_empty = heap_sort([])
    print(f"\nEmpty list: {res_empty} (Comps: {comps_empty}, Swaps: {swaps_empty}, Aux Elements: {allocs_empty})")
    assert res_empty == [], "Test 5 Failed"

    print("All tests passed.")