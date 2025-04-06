import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from adjustText import adjust_text

# Ignore RuntimeWarnings from nanmean of empty slice if a combination failed entirely
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# --- Configuration ---
RESULTS_FILE = 'benchmark_results.csv'
PLOTS_DIR = 'plots'
# Define metrics to plot
TIME_METRIC = 'Avg Time (4 runs) (s)'
MEMORY_METRIC = 'Avg Auxiliary Elements'
COMPS_METRIC = 'Avg Comparisons'
SWAPS_METRIC = 'Avg Swaps/Moves'

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Helper Functions ---

def save_plot(fig, filename, subfolder=None):
    """Saves the given figure to the plots directory, optionally in a subfolder."""
    target_dir = PLOTS_DIR
    if subfolder:
        target_dir = os.path.join(PLOTS_DIR, subfolder)
        os.makedirs(target_dir, exist_ok=True) # Ensure subfolder exists

    filepath = os.path.join(target_dir, filename)
    try:
        fig.savefig(filepath, bbox_inches='tight', dpi=150)
        print(f"Saved plot: {filepath}")
    except Exception as e:
        print(f"Error saving plot {filepath}: {e}")
    plt.close(fig) # Close the figure to free memory

def plot_metric_vs_size(df, metric, dataset_name_filter, title_suffix, filename_suffix, log_scale_y=False, log_scale_x=False, y_limit_quantile=None):
    """
    Generates a line plot of a metric vs. dataset size for a specific dataset type.
    Optionally limits the y-axis based on a quantile to exclude extreme outliers.
    """
    df_filtered = df[df['Dataset Name'] == dataset_name_filter].copy()
    # Convert inf time (timeout) to NaN for plotting, or handle appropriately
    df_filtered[metric] = df_filtered[metric].replace([np.inf, -np.inf], np.nan)
    # Drop rows where the metric is NaN to avoid plotting issues
    df_filtered.dropna(subset=[metric], inplace=True)

    if df_filtered.empty:
        print(f"No data to plot for {metric} with filter {dataset_name_filter}")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Use seaborn lineplot
    sns.lineplot(
        data=df_filtered,
        x='Dataset Size',
        y=metric,
        hue='Algorithm Name',
        style='Algorithm Category',
        markers=True,
        dashes=False,
        ax=ax
    )

    ax.set_title(f'{metric} vs. Dataset Size ({title_suffix})')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel(metric)
    if log_scale_y:
        ax.set_yscale('log')
        ax.set_ylabel(f'{metric} (log scale)')
    if log_scale_x:
        ax.set_xscale('log')
        ax.set_xlabel('Dataset Size (log scale)')

    # Apply y-axis limit based on quantile if specified
    if y_limit_quantile is not None and not df_filtered.empty:
        try:
            # Calculate the quantile, ignoring NaN values
            upper_limit = df_filtered[metric].quantile(y_limit_quantile)
            current_ylim = ax.get_ylim()
            # Set ylim, ensuring lower bound is reasonable (e.g., 0 or current min)
            # Avoid setting limit if quantile is NaN or non-positive for log scale
            if pd.notna(upper_limit) and (not log_scale_y or upper_limit > 0):
                 ax.set_ylim(bottom=current_ylim[0] if not log_scale_y else max(current_ylim[0], 1e-9), top=upper_limit * 1.05) # Add 5% margin
                 print(f"Applied y-limit based on {y_limit_quantile*100:.0f}th percentile: {upper_limit:.4g}")
            else:
                 print(f"Could not apply y-limit based on quantile {y_limit_quantile} (limit={upper_limit}). Keeping default limits.")

        except Exception as e:
            print(f"Warning: Could not calculate or apply y-limit quantile: {e}")


    # Place legend outside the plot
    ax.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    # Adjust layout to prevent legend overlapping title/plot
    plt.subplots_adjust(right=0.75) # Adjust right margin

    # Sanitize metric name for filename (replace spaces and slashes)
    safe_metric_name = metric.replace(' ', '_').replace('/', '_').lower()
    filename = f"{safe_metric_name}_vs_size_{filename_suffix}.png"
    save_plot(fig, filename, subfolder='vs_size')


# --- Visualization Stages ---

def stage1_raw_metrics(df):
    """Plots raw performance metrics (time, memory) vs. dataset size."""
    print("\n--- Stage 1: Raw Performance Metrics ---")
    # Plot for a representative dataset type, e.g., random uniform integers
    dataset_filter = 'random_uniform_int'
    plot_metric_vs_size(df, TIME_METRIC, dataset_filter, 'Random Uniform Integers', 'random_int', log_scale_y=True, log_scale_x=True)
    plot_metric_vs_size(df, MEMORY_METRIC, dataset_filter, 'Random Uniform Integers', 'random_int', log_scale_y=False, log_scale_x=True)

    # Plot for another type, e.g., reverse sorted
    dataset_filter_rev = 'reverse_sorted_int'
    plot_metric_vs_size(df, TIME_METRIC, dataset_filter_rev, 'Reverse Sorted Integers', 'reverse_sorted_int', log_scale_y=True, log_scale_x=True)

    # Plot comparisons and swaps
    plot_metric_vs_size(df, COMPS_METRIC, dataset_filter, 'Random Uniform Integers', 'random_int', log_scale_y=True, log_scale_x=True)
    plot_metric_vs_size(df, SWAPS_METRIC, dataset_filter, 'Random Uniform Integers', 'random_int', log_scale_y=True, log_scale_x=True)


def stage2_normalized_scores(df):
    """Calculates and plots normalized scores relative to a baseline."""
    print("\n--- Stage 2: Normalized Scores ---")
    # Use Python's sorted() as the baseline
    baseline_algo = 'sorted'
    df_norm = df.copy()
    # Ensure numeric types for calculation
    for metric in [TIME_METRIC, MEMORY_METRIC, COMPS_METRIC, SWAPS_METRIC]:
         df_norm[metric] = pd.to_numeric(df_norm[metric], errors='coerce')

    # Calculate normalized scores (Algo Time / Baseline Time) for each group
    metrics_to_normalize = [TIME_METRIC, MEMORY_METRIC, COMPS_METRIC, SWAPS_METRIC]
    normalized_cols = {}

    for metric in metrics_to_normalize:
        norm_col_name = f'Normalized {metric}'
        normalized_cols[metric] = norm_col_name
        # Group by dataset type and size, then normalize within each group
        df_norm[norm_col_name] = df_norm.groupby(['Dataset Name', 'Dataset Size'])[metric].transform(
            lambda x: x / x[df_norm['Algorithm Name'] == baseline_algo].iloc[0] if not x[df_norm['Algorithm Name'] == baseline_algo].empty else np.nan
        )
        # Handle division by zero or NaN baseline -> result NaN
        df_norm[norm_col_name].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Plot normalized time for random data
    dataset_filter = 'random_uniform_int'
    norm_time_col = normalized_cols[TIME_METRIC]
    # Apply quantile limit to prevent outliers from dominating the scale
    plot_metric_vs_size(df_norm, norm_time_col, dataset_filter, 'Random Uniform Integers (Normalized to sorted())', 'random_int_norm_time', log_scale_y=False, log_scale_x=True, y_limit_quantile=0.95) # Limit y-axis to 95th percentile


def stage4_efficiency_metrics(df):
    """Plots derived efficiency metrics."""
    print("\n--- Stage 4: Efficiency Metrics ---")
    df_eff = df.copy()
    # Avoid division by zero or NaN time
    df_eff[TIME_METRIC] = df_eff[TIME_METRIC].replace(0, np.nan)

    # Calculate Comparisons per second (higher is better)
    df_eff['Comps per Second'] = df_eff[COMPS_METRIC] / df_eff[TIME_METRIC]
    df_eff['Comps per Second'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Calculate Swaps/Moves per second (higher is better, but interpretation varies)
    df_eff['Swaps per Second'] = df_eff[SWAPS_METRIC] / df_eff[TIME_METRIC]
    df_eff['Swaps per Second'].replace([np.inf, -np.inf], np.nan, inplace=True)

    dataset_filter = 'random_uniform_int'
    plot_metric_vs_size(df_eff, 'Comps per Second', dataset_filter, 'Random Uniform Integers', 'random_int_comps_per_sec', log_scale_y=True, log_scale_x=True)
    plot_metric_vs_size(df_eff, 'Swaps per Second', dataset_filter, 'Random Uniform Integers', 'random_int_swaps_per_sec', log_scale_y=True, log_scale_x=True)


def plot_barchart_comparison(df, dataset_name_filter, data_size_filter, metric=TIME_METRIC):
    """Plots a bar chart comparing algorithms for a specific dataset and size."""
    print(f"\n--- Plotting Bar Chart for {dataset_name_filter} Size {data_size_filter} ---")
    df_filtered = df[(df['Dataset Name'] == dataset_name_filter) & (df['Dataset Size'] == data_size_filter)].copy()
    df_filtered[metric] = pd.to_numeric(df_filtered[metric], errors='coerce')
    df_filtered.dropna(subset=[metric], inplace=True)
    # Replace potential infinity from timeouts with a large value or handle separately
    large_value_for_inf = df_filtered[metric][np.isfinite(df_filtered[metric])].max() * 1.2 if np.isfinite(df_filtered[metric]).any() else 1
    df_filtered[metric] = df_filtered[metric].replace(np.inf, large_value_for_inf)


    if df_filtered.empty:
        print(f"No data for bar chart: {dataset_name_filter} size {data_size_filter}")
        return

    df_sorted = df_filtered.sort_values(by=metric)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df_sorted, x='Algorithm Name', y=metric, hue='Algorithm Category', dodge=False, ax=ax) # dodge=False if x is unique

    ax.set_title(f'{metric} Comparison for {dataset_name_filter} (Size {data_size_filter})')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    filename = f"barchart_{metric.replace(' ', '_').lower()}_{dataset_name_filter}_size_{data_size_filter}.png"
    save_plot(fig, filename, subfolder='fixed_size/comparison')


def plot_algorithm_across_datasets(df, algo_name_filter, data_size_filter, metric=TIME_METRIC):
    """Plots a bar chart comparing dataset types for a specific algorithm and size."""
    print(f"\n--- Plotting Bar Chart for {algo_name_filter} Size {data_size_filter} Across Datasets ---")
    df_filtered = df[(df['Algorithm Name'] == algo_name_filter) & (df['Dataset Size'] == data_size_filter)].copy()
    df_filtered[metric] = pd.to_numeric(df_filtered[metric], errors='coerce')
    df_filtered.dropna(subset=[metric], inplace=True)
    # Replace potential infinity from timeouts
    large_value_for_inf = df_filtered[metric][np.isfinite(df_filtered[metric])].max() * 1.2 if np.isfinite(df_filtered[metric]).any() else 1
    df_filtered[metric] = df_filtered[metric].replace(np.inf, large_value_for_inf)

    if df_filtered.empty:
        print(f"No data for bar chart: {algo_name_filter} size {data_size_filter}")
        return

    df_sorted = df_filtered.sort_values(by=metric)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_sorted, x='Dataset Name', y=metric, ax=ax)

    ax.set_title(f'{metric} for {algo_name_filter} (Size {data_size_filter}) Across Dataset Types')
    ax.set_xlabel('Dataset Type')
    ax.set_ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()

    filename = f"barchart_{metric.replace(' ', '_').lower()}_{algo_name_filter}_size_{data_size_filter}_datasets.png"
    save_plot(fig, filename, subfolder='fixed_size/comparison')



def plot_bubble_chart(df, data_size_filter, x_metric=TIME_METRIC, y_metric=MEMORY_METRIC, size_metric=COMPS_METRIC):
    """
    Creates a bubble chart where:
    - X-axis is a performance metric (e.g., time)
    - Y-axis is another metric (e.g., memory usage)
    - Bubble size represents a third metric (e.g., comparisons)
    - Color represents algorithm category
    """
    print(f"\n--- Plotting Bubble Chart for Size {data_size_filter} ---")
    df_filtered = df[df['Dataset Size'] == data_size_filter].copy()
    # Select only random uniform dataset for this visualization
    df_filtered = df_filtered[df_filtered['Dataset Name'] == 'random_uniform_int'].copy()
    
    # Ensure numeric types and handle infinity
    metrics = [x_metric, y_metric, size_metric]
    for metric in metrics:
        df_filtered[metric] = pd.to_numeric(df_filtered[metric], errors='coerce')
        df_filtered[metric] = df_filtered[metric].replace([np.inf, -np.inf], np.nan)
    
    # Drop NaN values to avoid plotting issues
    df_plot = df_filtered.dropna(subset=metrics).copy()
    
    if df_plot.empty:
        print(f"No data for bubble chart at size {data_size_filter}")
        return
    
    # Use a color map based on algorithm category with more distinct colors
    categories = df_plot['Algorithm Category'].unique()
    # Use tab10 for more distinct colors
    category_colors = dict(zip(categories, plt.cm.tab10.colors[:len(categories)]))
    
    # Create figure with fixed size - wider for better readability
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Determine if we need log scales and set axis limits upfront
    use_x_log = False
    use_y_log = False
    
    try:
        x_ratio = df_plot[x_metric].max() / (df_plot[x_metric].min() + 1e-10)
        if x_ratio > 10:
            use_x_log = True
    except:
        use_x_log = True  # Default to log scale if error
        
    try:
        y_ratio = df_plot[y_metric].max() / (df_plot[y_metric].min() + 1e-10)
        use_y_log = False
    except:
        use_y_log = True  # Default to log scale if error
    
    # Apply log scales if needed
    if use_x_log:
        ax.set_xscale('log')
    if use_y_log:
        ax.set_yscale('log')
    
    # Define limits explicitly early in the function
    x_min = df_plot[x_metric].min() * 0.8
    x_max = df_plot[x_metric].max() * 1.2
    y_min = df_plot[y_metric].min() * 0.8
    y_max = df_plot[y_metric].max() * 1.2
    
    # Ensure positive values for log scales
    if use_x_log:
        x_min = max(x_min, 1e-10)
    if use_y_log:
        y_min = max(y_min, 1e-10)
    
    # Create bubble chart with fixed maximum bubble size
    max_bubble_size = 1000  # Larger maximum for better visibility
    min_bubble_size = 100   # Minimum size to ensure visibility
    
    # Store bubble positions and sizes for later legend
    bubble_info = {}
    
    # Create dictionary to store scatter plots by category
    category_scatters = {}
    
    for cat, group in df_plot.groupby('Algorithm Category'):
        # Normalize the size metric for better visualization
        size_values = group[size_metric]
        
        # Handle case where all values are the same
        if size_values.max() == size_values.min():
            size_normalized = [max_bubble_size/3] * len(size_values)
        else:
            # More conservative sizing with absolute max cap
            size_normalized = min_bubble_size + (size_values - size_values.min()) / (size_values.max() - size_values.min() + 1e-10) * (max_bubble_size - min_bubble_size)
        
        # Store for legend
        for i, algo in enumerate(group['Algorithm Name']):
            bubble_info[algo] = {
                'x': group[x_metric].iloc[i],
                'y': group[y_metric].iloc[i],
                'size': size_normalized.iloc[i] if hasattr(size_normalized, 'iloc') else size_normalized[i],
                'raw_size': size_values.iloc[i],
                'category': cat
            }
        
        # Create scatter plot
        scatter = ax.scatter(
            group[x_metric], 
            group[y_metric], 
            s=size_normalized,
            c=[category_colors[cat]] * len(group),
            alpha=0.7,  # Slightly more opaque
            edgecolors='white',
            linewidths=1.5,
            label=cat
        )
        
        category_scatters[cat] = scatter
    
    # Set explicit limits to prevent excessive scaling
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add text labels using adjustText
    texts = []
    for algo, info in bubble_info.items():
        label = algo
        # Shorten label if needed, adjustText might handle this better though
        # if len(label) > 15:
        #     label = label[:12] + '...'
        texts.append(ax.text(info['x'], info['y'], label, fontsize=8))
    
    # Adjust text positions to avoid overlaps
    adjust_text(texts,
                ax=ax,
                # Remove arrowprops as per warning and potential conflict
                # arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5),
                # Add some padding around points
                # expand_points=(1.2, 1.2),
                # Add some padding around text
                # expand_text=(1.2, 1.2),
                # Force text away from points
                # force_points=0.2,
                # Force text away from other text
                # force_text=0.5,
                # Limit iterations to prevent excessive adjustments
                lim=200
                )
    
    # Set axis labels and title with better formatting
    x_label = x_metric.replace('Avg ', '')
    y_label = y_metric.replace('Avg ', '')
    size_label = size_metric.replace('Avg ', '')
    
    # Define custom formatter for values in scientific notation
    def sci_notation(x, pos):
        if x == 0:
            return '0'
        exponent = int(np.log10(x))
        coeff = x / 10**exponent
        if coeff == 1:
            return r'$10^{%d}$' % exponent
        else:
            return r'$%.1f \times 10^{%d}$' % (coeff, exponent)
    
    # Apply custom formatter if using log scale
    if use_y_log:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(sci_notation))
    
    ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
    
    # Create a more informative title
    plt.title(f'Algorithm Comparison: {x_label} vs {y_label}\nBubble Size = {size_label} (Size {data_size_filter:,})', 
              fontsize=12, fontweight='bold', pad=15)
    
    # Create a better category legend
    legend1 = ax.legend(
        handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                           label=cat, markersize=10) for cat, color in category_colors.items()],
        title='Algorithm Category',
        fontsize=9,
        title_fontsize=10,
        bbox_to_anchor=(1.02, 1),
        loc='upper left'
    )
    ax.add_artist(legend1)
    
    # Create a size legend with examples
    # Get min, median and max values
    size_values = np.array([info['raw_size'] for info in bubble_info.values()])
    if len(size_values) >= 3:
        size_legend_values = [
            np.min(size_values),
            np.percentile(size_values, 50),
            np.max(size_values)
        ]
        
        # Calculate corresponding bubble sizes
        size_normalized_legend = min_bubble_size + (np.array(size_legend_values) - np.min(size_values)) / (np.max(size_values) - np.min(size_values) + 1e-10) * (max_bubble_size - min_bubble_size)
        
        # Create legend handles
        legend_handles = []
        for i, (value, size) in enumerate(zip(size_legend_values, size_normalized_legend)):
            val_formatted = f'{value:,.0f}' if value >= 1000 else f'{value:.2f}'
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color='gray', alpha=0.7,
                          label=f'{val_formatted}', markersize=np.sqrt(size)/3)
            )
        
        # Add size legend
        legend2 = ax.legend(
            handles=legend_handles,
            title=f'Bubble Size: {size_label}',
            fontsize=9,
            title_fontsize=10,
            bbox_to_anchor=(1.02, 0.5),
            loc='center left'
        )
        ax.add_artist(legend2)
    
    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Apply tight layout *before* saving
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Sanitize metric names for filename
    x_metric_name = x_metric.replace(' ', '_').replace('/', '_').lower()
    y_metric_name = y_metric.replace(' ', '_').replace('/', '_').lower()
    
    filename = f"bubble_chart_{x_metric_name}_vs_{y_metric_name.replace('avg_auxiliary_elements', 'aux_elements')}_size_{data_size_filter}.png"
    
    # Save the plot (moved tight_layout before this)
    save_plot(fig, filename, subfolder='fixed_size/comparison')


# --- Complexity Definitions ---

# Default complexities (assume average case, e.g., for random data)
DEFAULT_COMPLEXITIES = {
    'quicksort_first_pivot': ['O(n log n)', 'O(n²)'], # Can degrade
    'quicksort_last_pivot': ['O(n log n)', 'O(n²)'],  # Can degrade
    'quicksort_median_pivot': ['O(n log n)'],        # More robust
    'quicksort_random_pivot': ['O(n log n)'],        # More robust
    'merge_sort_in_place': ['O(n log n)'],
    'merge_sort_out_of_place': ['O(n log n)'],
    'heapsort_max_heap': ['O(n log n)'],
    'heapsort_min_heap': ['O(n log n)'],
    'sorted': ['O(n log n)'], # Python's Timsort
    'shell_sort_ciura': ['O(n log n)', 'O(n²)'], # Empirical fit varies
    'shell_sort_knuth': ['O(n log n)', 'O(n²)'], # O(n^1.5) theoretical
    'shell_sort_original': ['O(n log n)', 'O(n²)'], # O(n^2) worst
    'shell_sort_tokuda': ['O(n log n)', 'O(n²)'],
    'counting_sort_stable': ['O(n)'],
    'counting_sort_unstable': ['O(n)'],
    'radix_sort_base10': ['O(n)'], # O(d*(n+k))
    'radix_sort_base2_16': ['O(n)'], # O(d*(n+k))
    # Add others if needed
}

# Complexities for already sorted data
SORTED_COMPLEXITIES = {
    **DEFAULT_COMPLEXITIES,
    'quicksort_first_pivot': ['O(n²)'], # Worst case
    'quicksort_last_pivot': ['O(n²)'], # Worst case
    'sorted': ['O(n)'], # Timsort best case
}

# Complexities for reverse sorted data
REVERSE_SORTED_COMPLEXITIES = {
    **DEFAULT_COMPLEXITIES,
    'quicksort_first_pivot': ['O(n²)'], # Worst case
    'quicksort_last_pivot': ['O(n²)'], # Worst case
    'sorted': ['O(n)'], # Timsort best case (detects reverse order)
}

# Complexities for data with few unique values (relevant for counting/radix)
FEW_UNIQUE_COMPLEXITIES = {
    **DEFAULT_COMPLEXITIES,
    # Counting sort is particularly good here, O(n+k) where k is small
    'counting_sort_stable': ['O(n)'],
    'counting_sort_unstable': ['O(n)'],
}

# Map dataset names (or patterns) to their complexity dictionaries
DATASET_SPECIFIC_COMPLEXITIES = {
    'random': DEFAULT_COMPLEXITIES, # Use default for anything containing 'random'
    'sorted': SORTED_COMPLEXITIES, # Use for anything containing 'sorted' (catches reverse_sorted too initially)
    'reverse_sorted': REVERSE_SORTED_COMPLEXITIES, # More specific override for reverse
    'few_unique': FEW_UNIQUE_COMPLEXITIES, # Use for anything containing 'few_unique'
}

def get_expected_complexities(dataset_name):
    """Selects the appropriate complexity dictionary based on the dataset name."""
    # Check for specific overrides first
    if 'reverse_sorted' in dataset_name:
        return REVERSE_SORTED_COMPLEXITIES
    if 'sorted' in dataset_name: # Check after reverse_sorted
        return SORTED_COMPLEXITIES
    if 'few_unique' in dataset_name:
        return FEW_UNIQUE_COMPLEXITIES
    if 'random' in dataset_name:
        return DEFAULT_COMPLEXITIES
    # Fallback to default if no specific match
    return DEFAULT_COMPLEXITIES


def plot_complexity_fit(df, dataset_name_filter,
                          metric=TIME_METRIC,
                          metric_label='Time',
                          algorithms_to_include=None,
                          plot_title_suffix='',
                          filename_prefix='time'):
    """
    Fits complexity models (O(n), O(n log n), O(n²)) to empirical data (time or operations)
    for a specific dataset type and plots the results. Can filter algorithms.
    """
    print(f"\n--- Plotting {metric_label} Complexity Fit for {dataset_name_filter}{plot_title_suffix} ---")
    
    df_filtered = df[df['Dataset Name'] == dataset_name_filter].copy()
    # Ensure both potential metrics and size are numeric
    df_filtered[TIME_METRIC] = pd.to_numeric(df_filtered.get(TIME_METRIC), errors='coerce')
    df_filtered[COMPS_METRIC] = pd.to_numeric(df_filtered.get(COMPS_METRIC), errors='coerce')
    df_filtered['Dataset Size'] = pd.to_numeric(df_filtered['Dataset Size'], errors='coerce')
    
    # Remove inf/nan values for the specific metric being plotted
    df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric, 'Dataset Size'])
    
    if df_filtered.empty:
        print(f"No data for {metric_label} complexity fit with filter {dataset_name_filter}")
        return
    
    # Group by algorithm and analyze time complexity
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Theoretical complexity models for fitting
    def linear(n, a):
        return a * n

    def n_log_n(n, a):
        # Add epsilon to avoid log(0) if n ever happens to be 0 (unlikely for size)
        return a * n * np.log(n + 1e-9)

    def quadratic(n, a):
        return a * n**2

    model_functions = {
        'O(n)': linear,
        'O(n log n)': n_log_n,
        'O(n²)': quadratic
    }

    # Get the expected complexities for this specific dataset type
    expected_complexities = get_expected_complexities(dataset_name_filter)
    
    # Set up colors for different algorithms
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Track best fit for each algorithm
    results = []
    
    from scipy.optimize import curve_fit
    
    algo_counter = 0 # Use a separate counter for colors/markers after filtering
    for algo_name, group in df_filtered.groupby('Algorithm Name'):
        # Filter algorithms if a list is provided
        if algorithms_to_include is not None and algo_name not in algorithms_to_include:
            continue
            
        # Need at least 3 points for meaningful fit
        if len(group) < 3:
            print(f"  - Skipping {algo_name}: Not enough data points ({len(group)}) for fit.")
            continue
        
        # Extract x and y data for fitting
        x_data = group['Dataset Size'].values
        y_data = group[metric].values
        
        # Sort data by size
        sort_idx = np.argsort(x_data)
        x_data = x_data[sort_idx]
        y_data = y_data[sort_idx]
        
        # Skip if all y values are the same (can't fit)
        if np.all(y_data == y_data[0]):
            continue
        
        # Dictionary to store R-squared values for each model
        r_squared = {}
        fitted_params = {}
        
        # Determine which models to try based on expected complexity
        # Default to trying all models if algorithm not in the specific list, or if list is empty
        models_to_try = expected_complexities.get(algo_name) or list(model_functions.keys())

        r_squared = {}
        fitted_params = {}

        try:
            # Fit only the expected models for this algorithm
            for model_name in models_to_try:
                model_func = model_functions.get(model_name)
                if not model_func:
                    print(f"  - Warning: Model function for '{model_name}' not found.")
                    continue

                try:
                    # Add bounds=(0, np.inf) to ensure coefficient 'a' is non-negative
                    # Use a small initial guess p0 to help convergence
                    params, _ = curve_fit(model_func, x_data, y_data, p0=[1e-6], bounds=(0, np.inf), maxfev=5000)
                    fitted_params[model_name] = params

                    # Calculate R-squared robustly
                    y_pred = model_func(x_data, *params)
                    mean_y = np.mean(y_data)
                    ss_total = np.sum((y_data - mean_y)**2)

                    # Handle cases where all y_data are (almost) identical
                    if ss_total < 1e-10:
                         # If prediction is also identical, R^2 is 1, otherwise 0
                         r_squared[model_name] = 1.0 if np.allclose(y_data, y_pred) else 0.0
                    else:
                        ss_residual = np.sum((y_data - y_pred)**2)
                        # Ensure R-squared doesn't go significantly below 0 due to numerical issues
                        r_squared[model_name] = max(0.0, 1 - (ss_residual / ss_total))

                except RuntimeError as rt_e: # curve_fit often raises RuntimeError on failure
                    print(f"  - Failed to fit {model_name} to {algo_name}: {rt_e}")
                    r_squared[model_name] = -np.inf # Mark as failed fit
                except Exception as model_e: # Catch other potential errors
                    print(f"  - Error fitting {model_name} to {algo_name}: {type(model_e).__name__} - {model_e}")
                    r_squared[model_name] = -np.inf

            # Find the best fitting model *among the ones tried*
            if fitted_params: # Check if any model was successfully fitted
                # Filter out failed fits before finding max R-squared
                valid_fits = {k: v for k, v in r_squared.items() if v > -np.inf}

                if valid_fits:
                    best_model_name = max(valid_fits, key=valid_fits.get)
                    best_r2 = valid_fits[best_model_name]

                    # Plotting logic: Only plot if the best fit is reasonably good (e.g., R² > 0.7)
                    if best_r2 > 0.7:
                        color = colors[algo_counter % len(colors)]
                        marker = markers[algo_counter % len(markers)]
                        ax.scatter(x_data, y_data, color=color, marker=marker, label=f"{algo_name} ({metric_label} data)")

                        # Plot the best fit curve
                        x_fit = np.linspace(min(x_data), max(x_data), 100)
                        best_model_func = model_functions[best_model_name]
                        best_params = fitted_params[best_model_name]
                        y_fit = best_model_func(x_fit, *best_params)

                        ax.plot(x_fit, y_fit, '--', color=color,
                                label=f"{algo_name} fit: {best_model_name} ({metric_label}, R² = {best_r2:.2f})")

                        # Store result
                        results.append({
                            'Algorithm': algo_name,
                            'Best Fit': best_model_name,
                            'R-squared': best_r2,
                            'Metric': metric_label
                        })
                    else:
                        print(f"  - Best {metric_label} fit for {algo_name} ({best_model_name}) has low R² ({best_r2:.2f}). Not plotting fit line.")
                        # Optionally plot just the data points even if fit is poor
                        # color = colors[i % len(colors)]
                        # marker = markers[i % len(markers)]
                        # ax.scatter(x_data, y_data, color=color, marker=marker, label=f"{algo_name} (data, no good fit)")

                else:
                     print(f"  - No valid {metric_label} model fits found for {algo_name} among expected types.")
            else:
                print(f"  - No {metric_label} models were successfully fitted for {algo_name}.")
        
        except Exception as e: # Catch errors processing this specific algorithm
            print(f"Error processing {algo_name} for {metric_label}: {type(e).__name__} - {str(e)}")
            
        algo_counter += 1
    
    # Set up plot
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel(f'{metric_label} ({metric})')
    ax.set_title(f'{metric_label} Complexity Analysis ({dataset_name_filter}{plot_title_suffix})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Handle legend - if too many entries, move outside plot
    if len(results) > 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.7)
    else:
        ax.legend()
    
    fig.tight_layout()
    
    safe_dataset_name = dataset_name_filter.replace(' ', '_').lower()
    filename = f"{filename_prefix}_complexity_fit_{safe_dataset_name}.png"
    save_plot(fig, filename, subfolder='complexity')
    
    # Print summary of results
    if results:
        print(f"\n{metric_label} Complexity Analysis Results ({dataset_name_filter}{plot_title_suffix}):")
        for result in sorted(results, key=lambda x: x['Algorithm']):
            print(f"  {result['Algorithm']:<25}: Best fit = {result['Best Fit']:<10} (R² = {result['R-squared']:.3f})")
    else:
        print(f"\nNo valid {metric_label} complexity fits found for {dataset_name_filter}{plot_title_suffix}.")


def plot_algorithm_efficiency_index(df, data_size_filter, dataset_name_filter='random_uniform_int'):
    """
    Creates a custom 'efficiency index' that balances time, memory, and operation counts,
    then ranks algorithms by this composite score.
    """
    print(f"\n--- Plotting Algorithm Efficiency Index for Size {data_size_filter} ---")
    
    df_filtered = df[(df['Dataset Size'] == data_size_filter) & 
                    (df['Dataset Name'] == dataset_name_filter)].copy()
    
    # Ensure numeric types
    metrics = [TIME_METRIC, MEMORY_METRIC, COMPS_METRIC, SWAPS_METRIC]
    for metric in metrics:
        df_filtered[metric] = pd.to_numeric(df_filtered[metric], errors='coerce')
        df_filtered[metric] = df_filtered[metric].replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN values in any metric
    df_filtered = df_filtered.dropna(subset=metrics)
    
    if df_filtered.empty:
        print(f"No data for efficiency index at size {data_size_filter}")
        return
    
    # Create normalized versions of each metric (0-1 scale, where 0 is worst and 1 is best)
    for metric in metrics:
        min_val = df_filtered[metric].min()
        max_val = df_filtered[metric].max()
        if max_val > min_val:  # Prevent division by zero
            # For all metrics, lower is better, so invert normalization
            df_filtered[f'Norm {metric}'] = 1 - ((df_filtered[metric] - min_val) / (max_val - min_val))
        else:
            df_filtered[f'Norm {metric}'] = 1  # All equal
    
    # Create weighted efficiency index
    # Weights for Time, Memory, Comparisons, Swaps (adjust as needed)
    weights = {
        f'Norm {TIME_METRIC}': 0.4,          # Time is most important
        f'Norm {MEMORY_METRIC}': 0.2,        # Memory is next
        f'Norm {COMPS_METRIC}': 0.25,        # Comparisons 
        f'Norm {SWAPS_METRIC}': 0.15         # Swaps/Moves least weight
    }
    
    # Calculate weighted efficiency index
    df_filtered['Efficiency Index'] = sum(df_filtered[metric] * weight 
                                        for metric, weight in weights.items())
    
    # Sort by efficiency index
    df_sorted = df_filtered.sort_values(by='Efficiency Index', ascending=False)
    
    # Create horizontal bar chart
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color bars by algorithm category
    categories = df_sorted['Algorithm Category'].unique()
    category_colors = dict(zip(categories, plt.cm.Set2.colors[:len(categories)]))
    bar_colors = [category_colors[cat] for cat in df_sorted['Algorithm Category']]
    
    # Plot the bars
    bars = ax.barh(df_sorted['Algorithm Name'], df_sorted['Efficiency Index'], color=bar_colors)
    
    # Add value labels to the right of each bar
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{df_sorted["Efficiency Index"].iloc[i]:.2f}', 
                va='center')
    
    # Highlight component scores
    # Create a stacked bar chart to visualize contribution of each metric
    left = np.zeros(len(df_sorted))
    norm_metrics = [f'Norm {metric}' for metric in metrics]
    legend_elements = []
    
    for i, (metric, weight) in enumerate(weights.items()):
        # Calculate weighted contribution
        contribution = df_sorted[metric] * weight
        
        # Display only if space allows (adjust threshold as needed)
        if len(df_sorted) <= 15:
            ax.barh(df_sorted['Algorithm Name'], contribution, left=left, 
                    alpha=0.5, color=plt.cm.tab10(i), height=0.5)
            
        left += contribution
        
        # Add to legend
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=plt.cm.tab10(i), alpha=0.5,
                                           label=f'{metric.replace("Norm ", "")} ({weight:.2f})'))
    
    # Add category legend
    for cat, color in category_colors.items():
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=f'Category: {cat}'))
    
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Set labels and title
    ax.set_xlabel('Efficiency Index (higher is better)')
    ax.set_title(f'Algorithm Efficiency Index for {dataset_name_filter} (Size {data_size_filter})')
    ax.set_xlim(0, 1.1)  # Give space for labels
    
    # Grid for readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    
    filename = f"efficiency_index_{dataset_name_filter}_size_{data_size_filter}.png"
    save_plot(fig, filename, subfolder='fixed_size/comparison')


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Loading results from {RESULTS_FILE}...")
    try:
        results_df = pd.read_csv(RESULTS_FILE)
        print("Results loaded successfully.")

        # Data Cleaning / Preparation
        # Convert relevant columns to numeric, coercing errors
        numeric_cols = [TIME_METRIC, MEMORY_METRIC, COMPS_METRIC, SWAPS_METRIC, 'Dataset Size']
        for col in numeric_cols:
             if col in results_df.columns:
                  results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

        # Run visualization stages
        try:
            stage1_raw_metrics(results_df)
        except Exception as e:
            print(f"Error in stage1_raw_metrics: {e}")
        
        try:
            stage2_normalized_scores(results_df)
        except Exception as e:
            print(f"Error in stage2_normalized_scores: {e}")
        
        try:
            stage4_efficiency_metrics(results_df)
        except Exception as e:
            print(f"Error in stage4_efficiency_metrics: {e}")

        try:
            plot_barchart_comparison(results_df, dataset_name_filter='reverse_sorted_int', data_size_filter=10000)
        except Exception as e:
            print(f"Error in plot_barchart_comparison: {e}")
        
        # Plot all algorithms across datasets with better error handling
        for algo_name in results_df['Algorithm Name'].unique():
            try:
                plot_algorithm_across_datasets(results_df, algo_name_filter=algo_name, data_size_filter=10000)
            except Exception as e:
                print(f"Error plotting algorithm {algo_name} across datasets: {e}")
             
        # New creative visualizations
        print("\n--- Creating advanced visualizations... ---")
        
        # Bubble chart - compare three metrics at once
        try:
            # Use a slightly different setup for the bubble chart to avoid issues
            plot_bubble_chart(results_df, data_size_filter=100000, 
                            x_metric=TIME_METRIC, 
                            y_metric=MEMORY_METRIC,  # Use memory instead of comparisons
                            size_metric=COMPS_METRIC)  # Use comparisons as size
        except Exception as e:
            print(f"Error in plot_bubble_chart: {e}")
        
        # Complexity fit analysis for different dataset types
        print("\n--- Plotting Complexity Fits for Various Datasets ---")
        dataset_names = results_df['Dataset Name'].unique()
        
        # Define comparison-based sorts (adjust if new ones are added)
        comparison_sorts = [
            'quicksort_first_pivot', 'quicksort_last_pivot', 'quicksort_median_pivot', 'quicksort_random_pivot',
            'merge_sort_in_place', 'merge_sort_out_of_place',
            'heapsort_max_heap', 'heapsort_min_heap',
            'shell_sort_ciura', 'shell_sort_knuth', 'shell_sort_original', 'shell_sort_tokuda',
            'sorted' # Python's Timsort is comparison-based
        ]
        
        # Datasets to generate complexity plots for
        complexity_datasets_to_plot = [
            'random_uniform_int',
            'sorted_int',
            'reverse_sorted_int',
            'few_unique_int' # Add others if they exist and are interesting
        ]
        
        for dataset_name in complexity_datasets_to_plot:
            if dataset_name in dataset_names:
                # Plot Time Complexity Fit (All Algos)
                try:
                    plot_complexity_fit(results_df, dataset_name_filter=dataset_name,
                                        metric=TIME_METRIC, metric_label='Time',
                                        filename_prefix='time')
                except Exception as e:
                    print(f"Error in plot_complexity_fit (Time) for {dataset_name}: {e}")

                # Plot Comparison Complexity Fit (Comparison Sorts Only)
                try:
                    plot_complexity_fit(results_df, dataset_name_filter=dataset_name,
                                        metric=COMPS_METRIC, metric_label='Comparisons',
                                        algorithms_to_include=comparison_sorts,
                                        plot_title_suffix=' (Comparison Sorts)',
                                        filename_prefix='comparisons')
                except Exception as e:
                    print(f"Error in plot_complexity_fit (Comparisons) for {dataset_name}: {e}")
            else:
                print(f"Skipping complexity plots for {dataset_name}: Not found in results.")
        
        # Algorithm efficiency index - composite score visualization
        try:
            plot_algorithm_efficiency_index(results_df, data_size_filter=10000)
        except Exception as e:
            print(f"Error in plot_algorithm_efficiency_index: {e}")
        
        # Try another efficiency index with a different dataset type
        try:
            plot_algorithm_efficiency_index(results_df, data_size_filter=10000, 
                                        dataset_name_filter='reverse_sorted_int')
        except Exception as e:
            print(f"Error in plot_algorithm_efficiency_index (reverse): {e}")

        print("\nVisualization complete. Plots saved in:", PLOTS_DIR)

    except FileNotFoundError:
        print(f"Error: Results file not found at {RESULTS_FILE}")
        print("Please run the benchmark script (main.py) first.")
    except Exception as e:
        print(f"An error occurred during visualization: {e}")
        import traceback
        print(traceback.format_exc())