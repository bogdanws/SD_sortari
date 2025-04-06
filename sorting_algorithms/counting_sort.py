from typing import List, Tuple
import copy
import numpy as np

def counting_sort(arr: List[int], stable: bool = True) -> Tuple[List[int], int, int, int]:
    comparisons = 0 # counting sort is non-comparative
    movements = 0   # count writes to count array and output array
    n = len(arr)

    if n == 0:
        return [], 0, 0, 0 # 0 auxiliary elements for empty array

    arr_copy = arr[:]

    try:
        # finding min/max involves comparisons, but not between elements being sorted in the main phase. we don't count these towards the comparisons
        min_val = min(arr_copy)
        max_val = max(arr_copy)
        k = max_val - min_val + 1

        # check if the range is too large
        if k > 5 * 10**7:
            print(f"Warning: Range too large for counting sort (k={k}). Aborting.")
            return arr, np.nan, np.nan, np.nan

        # initialize count array
        count = [0] * k
        movements += k # initialization counts as k writes

        # populate count array
        for num in arr_copy:
            index = num - min_val
            count[index] += 1
            movements += 1 # write to count array

        output = [0] * n # initialize output array
        allocated_elements = k + n # peak auxiliary space = count array + output array

        if stable:
            # calculate prefix sum for stable sorting
            for i in range(1, k):
                count[i] += count[i-1]
                movements += 1 # write to count array (prefix sum update)

            # build the output array (stable)
            # iterate backwards through the input array
            for i in range(n - 1, -1, -1):
                num = arr_copy[i]
                index = num - min_val
                output[count[index] - 1] = num
                movements += 1 # write to output array
                count[index] -= 1
                movements += 1 # write back to count array
        else:
            # build the output array (unstable)
            output_idx = 0
            for i in range(k):
                val = i + min_val
                freq = count[i]
                for _ in range(freq):
                    output[output_idx] = val
                    movements += 1 # write to output array
                    output_idx += 1

        return output, comparisons, movements, allocated_elements
    except MemoryError:
        print(f"Warning: Counting sort (stable={stable}) on size {n} failed due to MemoryError (range k={k}). Returning original.")
        return arr, np.nan, np.nan, np.nan
    except IndexError as e:
         print(f"Warning: Counting sort (stable={stable}) on size {n} failed due to IndexError (likely large k={k}): {e}. Returning original.")
         return arr, np.nan, np.nan, np.nan
    except Exception as e:
        print(f"Warning: Counting sort (stable={stable}) on size {n} failed: {e}. Returning original.")
        return arr, np.nan, np.nan, np.nan

# wrapper functions for stable/unstable versions
def counting_sort_stable(arr: List[int]) -> Tuple[List[int], int, int, int]:
    try:
        return counting_sort(arr, True)
    except Exception as e:
        print(f"Warning: counting_sort_stable failed: {e}")
        return arr, np.nan, np.nan, np.nan

def counting_sort_unstable(arr: List[int]) -> Tuple[List[int], int, int, int]:
    try:
        return counting_sort(arr, False)
    except Exception as e:
        print(f"Warning: counting_sort_unstable failed: {e}")
        return arr, np.nan, np.nan, np.nan

if __name__ == "__main__":
    # Test basic functionality
    test1 = [4, 2, 2, 8, 3, 3, 1]
    print("Original:", test1)
    res_s, comps_s, moves_s, allocs_s = counting_sort_stable(test1[:])
    print(f"Stable sorted:   {res_s} (Comps: {comps_s}, Moves: {moves_s}, Aux Elements: {allocs_s})")
    assert res_s == sorted(test1), "Test 1 Failed"
    res_u, comps_u, moves_u, allocs_u = counting_sort_unstable(test1[:])
    print(f"Unstable sorted: {res_u} (Comps: {comps_u}, Moves: {moves_u}, Aux Elements: {allocs_u})")
    assert res_u == sorted(test1), "Test 2 Failed"
    
    # Test with negative numbers
    test2 = [-5, 2, -3, 4, 1, 0, -1]
    print("\nWith negatives:", test2)
    res_neg_s, comps_neg_s, moves_neg_s, allocs_neg_s = counting_sort_stable(test2[:])
    print(f"Stable sorted: {res_neg_s} (Comps: {comps_neg_s}, Moves: {moves_neg_s}, Aux Elements: {allocs_neg_s})")
    assert res_neg_s == sorted(test2), "Test 3 Failed"
    res_neg_u, comps_neg_u, moves_neg_u, allocs_neg_u = counting_sort_unstable(test2[:])
    print(f"Unstable sorted: {res_neg_u} (Comps: {comps_neg_u}, Moves: {moves_neg_u}, Aux Elements: {allocs_neg_u})")
    assert res_neg_u == sorted(test2), "Test 4 Failed"
    # Test empty array
    test3 = []
    print("\nEmpty array:", test3)
    res_empty_s, comps_empty_s, moves_empty_s, allocs_empty_s = counting_sort_stable(test3[:])
    print(f"Stable sorted: {res_empty_s} (Comps: {comps_empty_s}, Moves: {moves_empty_s}, Aux Elements: {allocs_empty_s})")
    assert res_empty_s == [], "Test 5 Failed"
    res_empty_u, comps_empty_u, moves_empty_u, allocs_empty_u = counting_sort_unstable(test3[:])
    print(f"Unstable sorted: {res_empty_u} (Comps: {comps_empty_u}, Moves: {moves_empty_u}, Aux Elements: {allocs_empty_u})")
    assert res_empty_u == [], "Test 6 Failed"
    # Test single element
    test4 = [5]
    print("\nSingle element:", test4)
    res_single_s, comps_single_s, moves_single_s, allocs_single_s = counting_sort_stable(test4[:])
    print(f"Stable sorted: {res_single_s} (Comps: {comps_single_s}, Moves: {moves_single_s}, Aux Elements: {allocs_single_s})")
    assert res_single_s == [5], "Test 7 Failed"
    res_single_u, comps_single_u, moves_single_u, allocs_single_u = counting_sort_unstable(test4[:])
    print(f"Unstable sorted: {res_single_u} (Comps: {comps_single_u}, Moves: {moves_single_u}, Aux Elements: {allocs_single_u})")
    assert res_single_u == [5], "Test 8 Failed"
    
    print("All tests passed.")
    