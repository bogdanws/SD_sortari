from typing import Literal, Tuple
import copy
import numpy as np

def radix_sort(arr: list[int], base: Literal[10, 65536]) -> Tuple[list[int], int, int, int]:
    comparisons = 0 # radix sort is non-comparative
    movements = 0
    n = len(arr)

    if n == 0:
        return [], 0, 0, 0

    arr_copy = arr[:]

    try:
        if base not in (10, 65536):
            raise ValueError("Base must be either 10 or 65536.")

        # determine min/max for offset calculation
        min_val = min(arr_copy) if arr_copy else 0
        max_val = max(arr_copy) if arr_copy else 0

        # use offset to handle negative numbers
        offset = 0
        if min_val < 0:
            offset = -min_val
            max_val += offset
            # applying offset counts as n movements (read + write)
            arr_copy = [num + offset for num in arr_copy]
            movements += n

        # calculate maximum number of passes needed
        max_num = max_val
        total_passes = 0
        if max_num == 0:
             total_passes = 1 # need at least one pass even for all zeros
        elif base == 10:
            power = 0
            temp_max = max_num
            while temp_max > 0:
                temp_max //= base
                power += 1
            total_passes = power if power > 0 else 1
        else:  # base is 2^16
            bits = max_num.bit_length()
            total_passes = (bits + 15) // 16 if bits > 0 else 1 # ceil(bits/16)

        # calculate peak auxiliary space (count array + output array used in _counting_sort)
        allocated_elements = base + n

        # perform counting sort for each digit/chunk
        for current_pass in range(total_passes):
            # _counting_sort returns the new array state and movements for that pass
            arr_copy, pass_movements = _counting_sort(arr_copy, base, current_pass)
            movements += pass_movements

        # remove offset if applied
        if offset > 0:
            arr_copy = [num - offset for num in arr_copy]
            movements += n # removing offset counts as n movements

        return arr_copy, comparisons, movements, allocated_elements
    except MemoryError:
        print(f"Warning: Radix sort (base {base}) on size {n} failed due to MemoryError. Returning original.")
        return arr, np.nan, np.nan, np.nan # Return nan for allocs on error
    except Exception as e:
        print(f"Warning: Radix sort (base {base}) on size {n} failed: {e}. Returning original.")
        return arr, np.nan, np.nan, np.nan # Return nan for allocs on error

def _counting_sort(arr: list[int], base: int, current_pass: int) -> Tuple[list[int], int]:
    n = len(arr)
    movements = 0

    # initialize count array and output array
    count = [0] * base
    output = [0] * n
    movements += base # initialization of count array
    movements += n   # initialization of output array

    # populate count array based on the current digit/chunk
    for num in arr:
        digit = _get_digit(num, base, current_pass)
        if 0 <= digit < base:
            count[digit] += 1
            movements += 1 # write to count array
        else:
             # should not happen if _get_digit is correct
             raise ValueError(f"Digit {digit} out of range for base {base}")


    # calculate cumulative count (prefix sum)
    for i in range(1, base):
        count[i] += count[i-1]
        movements += 1 # write to count array (prefix sum update)

    # build the output array (stable) by placing elements in sorted order
    # iterate backwards through the input array to maintain stability
    for i in range(n - 1, -1, -1):
        num = arr[i]
        digit = _get_digit(num, base, current_pass)
        output_index = count[digit] - 1
        output[output_index] = num
        movements += 1 # write to output array
        count[digit] -= 1
        movements += 1 # write back to count array

    return output, movements

def _get_digit(num: int, base: int, current_pass: int) -> int:
    if base == 10:
        divisor = 10 ** current_pass
        return (num // divisor) % 10
    else:  # base 2^16
        shift = 16 * current_pass
        return (num >> shift) & 0xFFFF

def radix_sort_base10(arr: list[int]) -> Tuple[list[int], int, int, int]:
    try:
        return radix_sort(arr, 10)
    except Exception as e:
        print(f"Warning: radix_sort_base10 failed: {e}")
        return arr, np.nan, np.nan, np.nan

def radix_sort_base2_16(arr: list[int]) -> Tuple[list[int], int, int, int]:
    try:
        return radix_sort(arr, 65536)
    except Exception as e:
        print(f"Warning: radix_sort_base2_16 failed: {e}")
        return arr, np.nan, np.nan, np.nan

if __name__ == "__main__":
    # Test base 10 sorting
    test_data = [170, 45, 75, -90, 802, -24, 2, 66]
    res10, comps10, moves10, allocs10 = radix_sort_base10(test_data[:])
    print(f"Base 10 sorted: {res10} (Comps: {comps10}, Moves: {moves10}, Aux Elements: {allocs10})")
    assert res10 == sorted(test_data), "Test 1 Failed"
    
    # Test base 2^16 sorting
    large_numbers = [0xABCD1234, 0xFFFF0000, 0x1A2B3C4D, -0x76543210]
    res16, comps16, moves16, allocs16 = radix_sort_base2_16(large_numbers[:])
    print(f"Base 2^16 sorted: {res16} (Comps: {comps16}, Moves: {moves16}, Aux Elements: {allocs16})")
    assert res16 == sorted(large_numbers), "Test 2 Failed"

    print("All tests passed.")