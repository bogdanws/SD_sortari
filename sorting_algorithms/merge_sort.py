from typing import Tuple
import numpy as np

def merge_sort(arr: list[int], in_place: bool = False) -> Tuple[list[int], int, int, int]:
    arr_copy = arr[:]
    comparisons = 0
    movements = 0
    method = "in-place" if in_place else "out-of-place"
    try:
        if in_place:
            comparisons, movements = _merge_sort_in_place(arr_copy, 0, len(arr_copy) - 1)
            return arr_copy, comparisons, movements, 0
        else:
            sorted_list, comparisons, movements, allocated_elements = _merge_sort_out_of_place(arr_copy)
            return sorted_list, comparisons, movements, allocated_elements
    except RecursionError:
        print(f"Warning: Merge sort ({method}) on size {len(arr)} hit recursion depth limit. Returning original.")
        return arr, np.nan, np.nan, np.nan
    except Exception as e:
        print(f"Warning: Merge sort ({method}) on size {len(arr)} failed: {e}. Returning original.")
        return arr, np.nan, np.nan, np.nan

def _merge_sort_out_of_place(arr: list[int]) -> Tuple[list[int], int, int, int]:
    n = len(arr)
    if n <= 1:
        # copying the list of size 0 or 1 counts as n movements
        # base case: copy uses n space, max allocation is n
        return arr[:], 0, n, n

    comparisons = 0
    movements = 0
    mid = n // 2

    left_sorted, left_comps, left_moves, left_allocs = _merge_sort_out_of_place(arr[:mid])
    comparisons += left_comps
    movements += left_moves

    right_sorted, right_comps, right_moves, right_allocs = _merge_sort_out_of_place(arr[mid:])
    comparisons += right_comps
    movements += right_moves

    merged_list, merge_comps, merge_moves, merge_allocs = _merge(left_sorted, right_sorted)
    comparisons += merge_comps
    movements += merge_moves

    # the max auxiliary space is the max of children calls and the current merge operation
    max_allocated_elements = max(left_allocs, right_allocs, merge_allocs)
    return merged_list, comparisons, movements, max_allocated_elements

def _merge_sort_in_place(arr: list[int], start: int, end: int) -> Tuple[int, int]:
    comparisons = 0
    movements = 0
    if start < end:
        mid = (start + end) // 2

        # recursive calls
        left_comps, left_moves = _merge_sort_in_place(arr, start, mid)
        comparisons += left_comps
        movements += left_moves

        right_comps, right_moves = _merge_sort_in_place(arr, mid + 1, end)
        comparisons += right_comps
        movements += right_moves

        # merge step
        merge_comps, merge_moves = _merge_in_place(arr, start, mid, end)
        comparisons += merge_comps
        movements += merge_moves

    # base case (start >= end) returns 0, 0
    return comparisons, movements

def _merge(left: list[int], right: list[int]) -> Tuple[list[int], int, int, int]:
    merged = []
    comparisons = 0
    movements = 0 # counts appends/extends to the new 'merged' list
    i = j = 0
    len_left, len_right = len(left), len(right)

    while i < len_left and j < len_right:
        comparisons += 1
        if left[i] <= right[j]:
            merged.append(left[i])
            movements += 1
            i += 1
        else:
            merged.append(right[j])
            movements += 1
            j += 1

    # add remaining elements - these don't require comparisons
    remaining_left = len_left - i
    if remaining_left > 0:
        merged.extend(left[i:])
        movements += remaining_left

    remaining_right = len_right - j
    if remaining_right > 0:
        merged.extend(right[j:])
        movements += remaining_right

    return merged, comparisons, movements, len(merged)

def _merge_in_place(arr: list[int], start: int, mid: int, end: int) -> Tuple[int, int]:
    comparisons = 0
    movements = 0 # counts reads into temp arrays + writes back to original array

    # create temporary arrays - counts as read movements
    left = arr[start:mid+1]
    right = arr[mid+1:end+1]
    movements += len(left) + len(right) # cost of creating temp arrays

    # merge the temp arrays back into arr[start:end+1]
    i = j = 0 # pointers for left and right temp arrays
    k = start # pointer for the main array section being merged

    len_left, len_right = len(left), len(right)

    while i < len_left and j < len_right:
        comparisons += 1
        if left[i] <= right[j]:
            arr[k] = left[i]
            movements += 1 # write movement
            i += 1
        else:
            arr[k] = right[j]
            movements += 1 # write movement
            j += 1
        k += 1

    # copy remaining elements of left[] if any
    while i < len_left:
        arr[k] = left[i]
        movements += 1 # write movement
        i += 1
        k += 1

    # copy remaining elements of right[] if any
    while j < len_right:
        arr[k] = right[j]
        movements += 1 # write movement
        j += 1
        k += 1

    return comparisons, movements

# wrapper functions
def merge_sort_out_of_place(arr: list[int]) -> Tuple[list[int], int, int, int]:
    try:
        return merge_sort(arr, in_place=False)
    except Exception as e:
        print(f"Warning: merge_sort_out_of_place failed: {e}")
        return arr, np.nan, np.nan, np.nan

def merge_sort_in_place(arr: list[int]) -> Tuple[list[int], int, int, int]:
    try:
        return merge_sort(arr, in_place=True)
    except Exception as e:
        print(f"Warning: merge_sort_in_place failed: {e}")
        return arr, np.nan, np.nan, np.nan

if __name__ == "__main__":
    # test with positive and negative numbers
    test_data = [38, 27, 43, 3, 9, 82, 10, -5, 0, -12, 15]
    print("Original array:", test_data)
    res_out, comps_out, moves_out, allocs_out = merge_sort_out_of_place(test_data[:]) # Use copy
    print(f"Sorted array (out-of-place): {res_out} (Comps: {comps_out}, Moves: {moves_out}, Aux Elements: {allocs_out})")
    assert res_out == sorted(test_data), "Test 1 Failed"
    # in-place modifies the copy made inside merge_sort, so pass original test_data
    res_in, comps_in, moves_in, allocs_in = merge_sort_in_place(test_data)
    print(f"Sorted array (in-place): {res_in} (Comps: {comps_in}, Moves: {moves_in}, Aux Elements: {allocs_in})")
    assert res_in == sorted(test_data), "Test 2 Failed"

    print("All tests passed.")