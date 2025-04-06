import time
import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Any
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
import traceback

# Import sorting algorithms
from sorting_algorithms.counting_sort import counting_sort_stable, counting_sort_unstable
from sorting_algorithms.radix_sort import radix_sort_base10, radix_sort_base2_16
from sorting_algorithms.heap_sort import heapsort_min_heap, heapsort_max_heap
from sorting_algorithms.merge_sort import merge_sort_in_place, merge_sort_out_of_place
from sorting_algorithms.shell_sort import (
    shell_sort_original, shell_sort_knuth,
    shell_sort_ciura, shell_sort_tokuda
)
from sorting_algorithms.quick_sort import (
    quicksort_first_pivot, quicksort_last_pivot,
    quicksort_median_pivot, quicksort_random_pivot
)

from benchmarking.dataset_generator import get_datasets, DATA_SIZES


ALGORITHMS = {
    'Counting Sort': [counting_sort_stable, counting_sort_unstable],
    'Radix Sort': [radix_sort_base10, radix_sort_base2_16],
    'Heap Sort': [heapsort_min_heap, heapsort_max_heap],
    'Merge Sort': [merge_sort_in_place, merge_sort_out_of_place],
    'Shell Sort': [shell_sort_original, shell_sort_knuth,
                   shell_sort_ciura, shell_sort_tokuda],
    'Quick Sort': [quicksort_first_pivot, quicksort_last_pivot,
                   quicksort_median_pivot, quicksort_random_pivot],
    'Python sorted()': [sorted]
}


def execute_single_run(args):
    """
    Executes a single run of a sorting algorithm, measures performance,
    and returns results.
    """
    # Unpack arguments
    category, algo_func, dataset_name, size, data_copy, _, run_index = args
    algo_name = algo_func.__name__ if hasattr(algo_func, '__name__') else str(algo_func)
    # Log start of execution in worker
    print(f"  [Worker] Starting run {run_index+1} for {algo_name} on {dataset_name} size {size}...", flush=True)

    start_time = 0.0
    end_time = 0.0
    error = None
    return_value = None
    comparisons = np.nan
    swaps_moves = np.nan
    sorted_data = None
    status = "OK" # Default status
    allocated_elements = np.nan

    try:
        start_time = time.perf_counter()
        # --- Execute the sorting function ---
        return_value = algo_func(data_copy)
        end_time = time.perf_counter()

        # --- Unpack results ---
        if isinstance(return_value, tuple):
            if len(return_value) == 4: # Instrumented: (data, comps, swaps, allocs)
                sorted_data, comparisons, swaps_moves, allocated_elements = return_value
            elif len(return_value) == 3: # Fallback for older/uninstrumented: (data, comps, swaps)
                 sorted_data, comparisons, swaps_moves = return_value
                 allocated_elements = 0 # Assume 0 if not reported
            else:
                 error = f"Unexpected tuple return length: {len(return_value)}"
                 status = "Error"
        elif isinstance(return_value, list): # Handle non-instrumented (e.g., sorted())
            sorted_data = return_value
            # comparisons, swaps_moves, allocated_elements remain np.nan
        else:
            error = f"Unexpected return type: {type(return_value).__name__}"
            status = "Error"

        # --- Basic Validation ---
        if error is None:
            if not isinstance(sorted_data, list):
                error = f"Invalid data type after sort: {type(sorted_data).__name__}"
                status = "Error"
            elif len(sorted_data) != len(data_copy):
                error = f"Output size mismatch: expected {len(data_copy)}, got {len(sorted_data)}"
                status = "Error"

    except Exception as e:
        end_time = time.perf_counter() # Record time until error
        error = f"Exception in run: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        status = "Error"

    elapsed_time = end_time - start_time if start_time > 0 else 0.0

    # --- Prepare result dictionary for this single run ---
    is_warmup = (run_index == -1)
    run_result = {
        # Identification fields for grouping later
        'Algorithm Category': category,
        'Algorithm Name': algo_name,
        'Dataset Name': dataset_name,
        'Dataset Size': size,
        'Run Index': run_index,
        'Warmup': is_warmup,
        # Measured metrics for this run
        'Time (s)': elapsed_time if status == "OK" else np.nan,
        'Comparisons': comparisons if status == "OK" else np.nan,
        'Swaps/Moves': swaps_moves if status == "OK" else np.nan,
        'Auxiliary Elements': allocated_elements if status == "OK" else np.nan, # New metric
        # Status and potentially sorted data for correctness check
        'Status': status,
        'Error Detail': error,
        'Sorted Data': sorted_data if status == "OK" else None # Only include if successful
    }
    return run_result


# --- Aggregation Function ---
def aggregate_results(all_run_results: List[Dict]) -> List[Dict]:
    """Aggregates results from individual runs into per-combination summaries."""
    aggregated = {}
    # Group results by combination key
    for run_res in all_run_results:
        key = (run_res['Algorithm Category'], run_res['Algorithm Name'],
               run_res['Dataset Name'], run_res['Dataset Size'])
        if key not in aggregated:
            aggregated[key] = {'runs': [], 'original_data_size': run_res['Dataset Size']} # Store original size if needed
        aggregated[key]['runs'].append(run_res)

    final_results = []
    for key, combo_data in aggregated.items():
        category, algo_name, dataset_name, size = key
        runs = combo_data['runs']
        # Filter out warmup runs *before* calculating stats
        valid_runs = [r for r in runs if not r.get('Warmup', False)]
        actual_runs_for_stats = len(valid_runs)

        # Extract metrics from successful, non-warmup runs
        ok_runs = [r for r in valid_runs if r['Status'] == 'OK']
        times = [r['Time (s)'] for r in ok_runs]
        comps = [r['Comparisons'] for r in ok_runs]
        swaps = [r['Swaps/Moves'] for r in ok_runs]
        allocs = [r['Auxiliary Elements'] for r in ok_runs] # New metric

        # Calculate averages, handling cases with no successful runs
        avg_time = np.nanmean(times) if times else np.nan
        avg_comps = np.nanmean(comps) if comps and not all(np.isnan(comps)) else np.nan
        avg_swaps = np.nanmean(swaps) if swaps and not all(np.isnan(swaps)) else np.nan
        avg_allocs = np.nanmean(allocs) if allocs and not all(np.isnan(allocs)) else np.nan

        # Determine overall status and correctness
        final_status = "OK"
        correct = True
        # Check status based on valid (non-warmup) runs
        first_ok_run = next((r for r in valid_runs if r['Status'] == 'OK'), None)

        if not first_ok_run:
            # No successful runs, determine status from failures
            if any(r['Status'] == 'Timeout' for r in valid_runs):
                final_status = "Timeout"
                avg_time = float('inf') # Represent timeout with infinity
            elif any(r['Status'] == 'Error' for r in valid_runs):
                 # Find first error message
                 first_error = next((r['Error Detail'] for r in valid_runs if r['Status'] == 'Error'), "Unknown Error")
                 final_status = f"Error ({first_error[:50]}...)" # Truncate long errors
            else:
                 final_status = "Failed (Unknown)" # Should not happen if status is always set for valid runs
            correct = False
        else:
            pass # Assume correct if Status is OK for now

            # Check if any later runs failed
            # Check partial failures among non-warmup runs
            if len(ok_runs) < actual_runs_for_stats:
                 if any(r['Status'] == 'Timeout' for r in valid_runs): final_status = "Partial Timeout"
                 elif any(r['Status'] == 'Error' for r in valid_runs): final_status = "Partial Error"
                 else: final_status = "Partial Failure" # Should have specific status


        final_results.append({
            'Algorithm Category': category,
            'Algorithm Name': algo_name,
            'Dataset Name': dataset_name,
            'Dataset Size': size,
            f'Avg Time ({actual_runs_for_stats} runs) (s)': avg_time,
            'Avg Comparisons': avg_comps,
            'Avg Swaps/Moves': avg_swaps,
            'Avg Auxiliary Elements': avg_allocs,
            'Correct': correct,
            'Status': final_status
        })

    return final_results

# --- Helper Function for Task Preparation ---

def _prepare_benchmark_tasks(algorithms_dict, datasets_dict, runs_per_combination):
    """Generates the list of arguments for each individual benchmark run."""
    run_args_list = []
    combination_keys = set()
    total_combinations = 0

    print("Algorithm Registry:")
    for category, funcs in algorithms_dict.items():
        print(f"  {category}: {[f.__name__ for f in funcs]}")
    print("-" * 30)

    for category, funcs in algorithms_dict.items():
        for algo_func in funcs:
            algo_name = algo_func.__name__ if hasattr(algo_func, '__name__') else str(algo_func)
            for dataset_name, size_data in datasets_dict.items():
                # Skip incompatible combinations
                is_integer_algo = category in ['Counting Sort', 'Radix Sort']
                is_non_integer_data = 'int' not in dataset_name
                if is_integer_algo and is_non_integer_data:
                    continue

                for size, data in size_data.items():
                    # Skip specific slow combinations if needed
                    if (algo_name == 'shell_sort_knuth' and size > 10000) or ((algo_name == 'quicksort_last_pivot' or algo_name == 'quicksort_first_pivot') and size > 100000):
                        print(f"Skipping {algo_name} for size {size} (too slow).")
                        continue # Skip this specific size for this algorithm

                    combo_key = (category, algo_name, dataset_name, size)
                    if combo_key not in combination_keys:
                        total_combinations += 1
                        combination_keys.add(combo_key)

                    # Add the warmup run first (run_index = -1)
                    run_args_list.append(
                        (category, algo_func, dataset_name, size, data[:], None, -1) # Warmup run
                    )
                    # Add arguments for the actual measurement runs
                    for run_index in range(runs_per_combination):
                        # Pass a copy of the data for each run
                        run_args_list.append(
                            (category, algo_func, dataset_name, size, data[:], None, run_index) # Measurement runs (0 to N-1)
                        )

    total_runs = len(run_args_list)
    print(f"Total unique combinations: {total_combinations}, Total runs to execute (incl. warmups): {total_runs}")
    if total_runs == 0:
        print("No valid runs found to benchmark.")
        return None, 0 # Indicate no runs

    return run_args_list, total_runs


# --- Main Benchmark Runner ---
def run_benchmark():
    """Runs the full benchmark suite in parallel."""
    print("Starting benchmark...")
    datasets = get_datasets()
    all_run_results = [] # Store results from *each individual run*
    runs_per_combination = 4

    # Prepare arguments using the helper function
    run_args_list, total_runs = _prepare_benchmark_tasks(ALGORITHMS, datasets, runs_per_combination)

    if not run_args_list: # Check if preparation yielded any tasks
        return

    # Determine number of workers
    try:
        num_workers = mp.cpu_count()
        if not num_workers or num_workers < 1: num_workers = 2
    except NotImplementedError:
        num_workers = 2
    num_workers = max(1, num_workers - 1) if num_workers > 1 else 1 # Leave one core free
    print(f"\nRunning benchmarks in parallel using {num_workers} workers...")

    # Use a Pool for parallel execution
    with Pool(processes=num_workers) as pool:
        try:
            # Map execute_single_run to all the run arguments
            # imap_unordered yields results as they complete
            results_iterator = pool.imap_unordered(execute_single_run, run_args_list)

            completed_runs = 0
            start_time_total = time.perf_counter()

            for run_result in results_iterator:
                completed_runs += 1
                if run_result: # Check if result is valid
                    all_run_results.append(run_result)
                    # Progress indicator based on runs completed
                    run_idx_display = run_result.get('Run Index', '?')
                    run_label = "Warmup" if run_idx_display == -1 else f"Run {run_idx_display + 1}"
                    print(f"Progress: {completed_runs}/{total_runs} runs ({((completed_runs)/total_runs)*100:.1f}%) - "
                          f"Completed: {run_label} for {run_result.get('Algorithm Name', '')} "
                          f"on {run_result.get('Dataset Name', '')} size {run_result.get('Dataset Size', '')} "
                          f"-> Status: {run_result.get('Status', 'Unknown')}")
                    # Optionally print detailed errors immediately
                    if run_result.get('Status') == 'Error':
                        # Limit printing potentially long tracebacks
                        error_detail = run_result.get('Error Detail', 'N/A')
                        print(f"  ERROR Detail: {error_detail[:500]}...") # Print first 500 chars
                else:
                     # Should not happen if execute_single_run always returns a dict
                     print(f"Progress: {completed_runs}/{total_runs} runs ({((completed_runs)/total_runs)*100:.1f}%) - Worker returned None (unexpected).")

            end_time_total = time.perf_counter()
            print(f"\nParallel execution finished in {end_time_total - start_time_total:.2f} seconds.")

        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user. Terminating pool...")
            pool.terminate()
            pool.join()
            print("Pool terminated.")
        except Exception as e:
            print(f"\nAn error occurred during parallel execution: {e}\n{traceback.format_exc()}")
            pool.terminate()
            pool.join()

    print("\nBenchmark processing complete. Aggregating results...")

    # Aggregate the results from individual runs
    if all_run_results:
        # Add a check for correctness during aggregation if possible
        # This requires the original data or a way to verify sorted order
        # For now, we rely on the 'Correct' flag set during aggregation based on status
        aggregated_results = aggregate_results(all_run_results) # Pass the list of run dictionaries

        if aggregated_results:
            aggregated_results_df = pd.DataFrame(aggregated_results)
            # Sort results for consistency and better readability
            aggregated_results_df = aggregated_results_df.sort_values(by=['Algorithm Category', 'Algorithm Name', 'Dataset Name', 'Dataset Size'])

            print("\nAggregated Results Summary (first 5 rows):")
            print(aggregated_results_df.head().to_string())

            aggregated_results_filename = "benchmark_results.csv"
            try:
                aggregated_results_df.to_csv(aggregated_results_filename, index=False)
                print(f"\nResults saved to {aggregated_results_filename}")
            except Exception as e:
                print(f"\nError saving results to {aggregated_results_filename}: {e}")
        else:
             print("\nAggregation produced no results (all runs might have failed).")
    else:
        print("\nNo results were collected from runs (benchmark might have been interrupted early or failed).")
        aggregated_results_df = None # Ensure df is None if no results

    # Save the individual run results as CSV, excluding the sorted data
    individual_results_filename = "individual_run_results.csv"
    if all_run_results:
        try:
            individual_df = pd.DataFrame(all_run_results)
            # Drop the 'Sorted Data' column before saving
            if 'Sorted Data' in individual_df.columns:
                individual_df = individual_df.drop(columns=['Sorted Data'])
            individual_df.to_csv(individual_results_filename, index=False)
            print(f"\nIndividual run results saved to {individual_results_filename}")
        except Exception as e:
            print(f"\nError saving individual run results to {individual_results_filename}: {e}")
    else:
        # Handle case where there are no results (e.g., early interruption)
        # Optionally create an empty file or just skip saving
        print("\nNo individual run results to save.")

if __name__ == "__main__":
    # Ensure multiprocessing works correctly when script is run directly
    mp.freeze_support() # Needed for multiprocessing on some platforms (e.g., Windows)
    run_benchmark()