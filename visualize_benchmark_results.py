#!/usr/bin/env python3
"""
Benchmark Results Visualization Script

Parses and visualizes results from index-microbench benchmark runs.
Results are organized by thread count, key type, workload, and index type.

File naming convention: result_{threads}_{key_type}_{workload}_{index}_{run}
- threads: number of threads (1, 2, 4, 6, 8, etc.)
- key_type: mono (monotonic) or rand (random)
- workload: a, c, e (YCSB workloads)
- index: artolc, btreeolc, bwtree, masstree
- run: run number

Filters out results containing "PinToCore() returns non-0" as these indicate failed runs.
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class BenchmarkResult:
    """Represents a single benchmark result."""
    threads: int
    key_type: str  # mono or rand
    workload: str  # a, c, e
    index: str     # artolc, btreeolc, bwtree, masstree
    run: int
    insert_throughput: Optional[float] = None
    operation_throughput: Optional[float] = None  # read/update or insert/scan
    operation_type: Optional[str] = None  # "read/update" or "insert/scan"
    is_valid: bool = True
    error_message: Optional[str] = None


def parse_filename(filename: str) -> Optional[Tuple[int, str, str, str, int]]:
    """
    Parse the filename to extract benchmark parameters.
    
    Expected format: result_{threads}_{key_type}_{workload}_{index}_{run}
    Returns: (threads, key_type, workload, index, run) or None if parsing fails
    """
    # Handle .tmp files
    filename = filename.replace('.tmp', '')
    
    pattern = r'result_(\d+)_(\w+)_(\w)_(\w+)_(\d+)'
    match = re.match(pattern, filename)
    
    if match:
        threads = int(match.group(1))
        key_type = match.group(2)
        workload = match.group(3)
        index = match.group(4)
        run = int(match.group(5))
        return threads, key_type, workload, index, run
    return None


def parse_result_file(filepath: Path) -> BenchmarkResult:
    """
    Parse a benchmark result file and extract metrics.
    
    Returns a BenchmarkResult object with extracted data.
    """
    filename = filepath.name
    parsed = parse_filename(filename)
    
    if not parsed:
        return BenchmarkResult(
            threads=0, key_type="", workload="", index="", run=0,
            is_valid=False, error_message=f"Could not parse filename: {filename}"
        )
    
    threads, key_type, workload, index, run = parsed
    result = BenchmarkResult(
        threads=threads,
        key_type=key_type,
        workload=workload,
        index=index,
        run=run
    )
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        result.is_valid = False
        result.error_message = f"Could not read file: {e}"
        return result
    
    # Check for PinToCore error - these are invalid results
    if "PinToCore() returns non-0" in content:
        result.is_valid = False
        result.error_message = "PinToCore() returns non-0"
        return result
    
    # Extract insert throughput (ANSI color codes: \x1b[1;32m ... \x1b[0m)
    # Pattern: insert <throughput>
    insert_pattern = r'insert\s+([\d.]+)'
    insert_matches = re.findall(insert_pattern, content)
    if insert_matches:
        # Take the last insert throughput if multiple exist
        result.insert_throughput = float(insert_matches[-1])
    
    # Extract read/update throughput
    read_update_pattern = r'read/update\s+([\d.]+)'
    read_update_matches = re.findall(read_update_pattern, content)
    if read_update_matches:
        result.operation_throughput = float(read_update_matches[-1])
        result.operation_type = "read/update"
    
    # Extract insert/scan throughput (for workload e)
    insert_scan_pattern = r'insert/scan\s+([\d.]+)'
    insert_scan_matches = re.findall(insert_scan_pattern, content)
    if insert_scan_matches:
        result.operation_throughput = float(insert_scan_matches[-1])
        result.operation_type = "insert/scan"
    
    # Mark as invalid if no throughput data found
    if result.insert_throughput is None and result.operation_throughput is None:
        result.is_valid = False
        result.error_message = "No throughput data found"
    
    return result


def load_all_results(results_dir: Path) -> List[BenchmarkResult]:
    """Load and parse all result files from the directory."""
    results = []
    
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist.")
        return results
    
    for filepath in results_dir.iterdir():
        if filepath.is_file() and filepath.name.startswith('result_'):
            result = parse_result_file(filepath)
            results.append(result)
    
    return results


def filter_valid_results(results: List[BenchmarkResult]) -> List[BenchmarkResult]:
    """Filter out invalid results."""
    return [r for r in results if r.is_valid]


def print_summary(results: List[BenchmarkResult]) -> None:
    """Print a summary of the loaded results."""
    valid_results = filter_valid_results(results)
    invalid_results = [r for r in results if not r.is_valid]
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"\nTotal files found: {len(results)}")
    print(f"Valid results: {len(valid_results)}")
    print(f"Invalid/skipped results: {len(invalid_results)}")
    
    if invalid_results:
        print("\n--- Skipped Results (Invalid) ---")
        error_counts = defaultdict(int)
        for r in invalid_results:
            error_counts[r.error_message] += 1
        for error, count in sorted(error_counts.items()):
            print(f"  {error}: {count} files")
    
    # Summary by parameter
    if valid_results:
        print("\n--- Data Summary ---")
        threads = sorted(set(r.threads for r in valid_results))
        key_types = sorted(set(r.key_type for r in valid_results))
        workloads = sorted(set(r.workload for r in valid_results))
        indexes = sorted(set(r.index for r in valid_results))
        
        print(f"Thread counts: {threads}")
        print(f"Key types: {key_types}")
        print(f"Workloads: {workloads}")
        print(f"Index types: {indexes}")


def create_dataframe(results: List[BenchmarkResult]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame."""
    data = []
    for r in results:
        if r.is_valid:
            data.append({
                'threads': r.threads,
                'key_type': r.key_type,
                'workload': r.workload,
                'index': r.index,
                'run': r.run,
                'insert_throughput': r.insert_throughput,
                'operation_throughput': r.operation_throughput,
                'operation_type': r.operation_type
            })
    return pd.DataFrame(data)


def identify_outstanding_records(df: pd.DataFrame) -> pd.DataFrame:
    """Identify outstanding records (best and worst performers)."""
    records = []
    
    # Best insert throughput per workload
    for workload in df['workload'].unique():
        wl_df = df[df['workload'] == workload].dropna(subset=['insert_throughput'])
        if not wl_df.empty:
            best_idx = wl_df['insert_throughput'].idxmax()
            worst_idx = wl_df['insert_throughput'].idxmin()
            records.append({
                'category': f'Best Insert Throughput (Workload {workload.upper()})',
                **wl_df.loc[best_idx].to_dict()
            })
            records.append({
                'category': f'Worst Insert Throughput (Workload {workload.upper()})',
                **wl_df.loc[worst_idx].to_dict()
            })
    
    # Best operation throughput per workload
    for workload in df['workload'].unique():
        wl_df = df[df['workload'] == workload].dropna(subset=['operation_throughput'])
        if not wl_df.empty:
            best_idx = wl_df['operation_throughput'].idxmax()
            worst_idx = wl_df['operation_throughput'].idxmin()
            op_type = wl_df.loc[best_idx, 'operation_type']
            records.append({
                'category': f'Best {op_type} (Workload {workload.upper()})',
                **wl_df.loc[best_idx].to_dict()
            })
            records.append({
                'category': f'Worst {op_type} (Workload {workload.upper()})',
                **wl_df.loc[worst_idx].to_dict()
            })
    
    return pd.DataFrame(records)


def plot_throughput_by_threads(df: pd.DataFrame, output_dir: Path) -> None:
    """Create throughput plots comparing indexes across thread counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    workloads = sorted(df['workload'].unique())
    key_types = sorted(df['key_type'].unique())
    indexes = sorted(df['index'].unique())
    
    # Color palette for indexes
    colors = {
        'artolc': '#1f77b4',
        'btreeolc': '#ff7f0e',
        'bwtree': '#2ca02c',
        'masstree': '#d62728'
    }
    
    # Markers for key types
    markers = {'mono': 'o', 'rand': 's'}
    
    for workload in workloads:
        wl_df = df[df['workload'] == workload]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'YCSB Workload {workload.upper()} Performance', fontsize=14, fontweight='bold')
        
        # --- Insert Throughput ---
        ax1 = axes[0]
        for key_type in key_types:
            kt_df = wl_df[wl_df['key_type'] == key_type]
            for index in indexes:
                idx_df = kt_df[kt_df['index'] == index]
                if idx_df.empty or idx_df['insert_throughput'].isna().all():
                    continue
                
                # Group by threads and compute mean
                grouped = idx_df.groupby('threads')['insert_throughput'].mean().reset_index()
                
                linestyle = '-' if key_type == 'mono' else '--'
                label = f"{index} ({key_type})"
                ax1.plot(grouped['threads'], grouped['insert_throughput'],
                        marker=markers.get(key_type, 'o'),
                        linestyle=linestyle,
                        color=colors.get(index, 'gray'),
                        label=label,
                        linewidth=2,
                        markersize=8)
        
        ax1.set_xlabel('Number of Threads', fontsize=12)
        ax1.set_ylabel('Insert Throughput (Mops/sec)', fontsize=12)
        ax1.set_title('Insert Throughput', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9)
        
        # Set x-axis ticks to thread counts
        thread_counts = sorted(wl_df['threads'].unique())
        ax1.set_xticks(thread_counts)
        
        # --- Operation Throughput ---
        ax2 = axes[1]
        op_type = wl_df['operation_type'].dropna().iloc[0] if not wl_df['operation_type'].dropna().empty else 'read/update'
        
        for key_type in key_types:
            kt_df = wl_df[wl_df['key_type'] == key_type]
            for index in indexes:
                idx_df = kt_df[kt_df['index'] == index]
                if idx_df.empty or idx_df['operation_throughput'].isna().all():
                    continue
                
                # Group by threads and compute mean
                grouped = idx_df.groupby('threads')['operation_throughput'].mean().reset_index()
                
                linestyle = '-' if key_type == 'mono' else '--'
                label = f"{index} ({key_type})"
                ax2.plot(grouped['threads'], grouped['operation_throughput'],
                        marker=markers.get(key_type, 'o'),
                        linestyle=linestyle,
                        color=colors.get(index, 'gray'),
                        label=label,
                        linewidth=2,
                        markersize=8)
        
        ax2.set_xlabel('Number of Threads', fontsize=12)
        ax2.set_ylabel(f'{op_type.capitalize()} Throughput (Mops/sec)', fontsize=12)
        ax2.set_title(f'{op_type.capitalize()} Throughput', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=9)
        ax2.set_xticks(thread_counts)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'workload_{workload}_throughput.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: workload_{workload}_throughput.png")


def plot_scalability_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Create scalability comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    indexes = sorted(df['index'].unique())
    workloads = sorted(df['workload'].unique())
    
    colors = {
        'artolc': '#1f77b4',
        'btreeolc': '#ff7f0e',
        'bwtree': '#2ca02c',
        'masstree': '#d62728'
    }
    
    # Scalability: Compare throughput normalized to single-thread performance
    fig, axes = plt.subplots(len(workloads), 2, figsize=(14, 5*len(workloads)))
    if len(workloads) == 1:
        axes = [axes]
    
    fig.suptitle('Index Scalability Analysis', fontsize=14, fontweight='bold', y=1.02)
    
    for i, workload in enumerate(workloads):
        wl_df = df[df['workload'] == workload]
        
        for j, key_type in enumerate(['mono', 'rand']):
            kt_df = wl_df[wl_df['key_type'] == key_type]
            ax = axes[i][j] if len(workloads) > 1 else axes[0][j]
            
            for index in indexes:
                idx_df = kt_df[kt_df['index'] == index]
                if idx_df.empty or idx_df['insert_throughput'].isna().all():
                    continue
                
                grouped = idx_df.groupby('threads')['insert_throughput'].mean().reset_index()
                
                if len(grouped) > 0 and 1 in grouped['threads'].values:
                    base_perf = grouped[grouped['threads'] == 1]['insert_throughput'].values[0]
                    if base_perf > 0:
                        grouped['speedup'] = grouped['insert_throughput'] / base_perf
                        ax.plot(grouped['threads'], grouped['speedup'],
                               marker='o', color=colors.get(index, 'gray'),
                               label=index, linewidth=2, markersize=8)
            
            # Ideal linear speedup
            thread_counts = sorted(kt_df['threads'].unique())
            if thread_counts:
                ax.plot(thread_counts, thread_counts, 'k--', alpha=0.5, label='Ideal')
            
            ax.set_xlabel('Number of Threads', fontsize=11)
            ax.set_ylabel('Speedup (vs 1 thread)', fontsize=11)
            ax.set_title(f'Workload {workload.upper()} - {key_type.capitalize()} Keys', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
            if thread_counts:
                ax.set_xticks(thread_counts)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scalability_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: scalability_comparison.png")


def plot_index_comparison_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Create heatmaps comparing index performance across configurations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate by index, workload, and threads
    agg_df = df.groupby(['index', 'workload', 'threads', 'key_type']).agg({
        'insert_throughput': 'mean',
        'operation_throughput': 'mean'
    }).reset_index()
    
    indexes = sorted(df['index'].unique())
    workloads = sorted(df['workload'].unique())
    key_types = sorted(df['key_type'].unique())
    
    for key_type in key_types:
        kt_df = agg_df[agg_df['key_type'] == key_type]
        
        fig, axes = plt.subplots(1, len(workloads), figsize=(6*len(workloads), 5))
        if len(workloads) == 1:
            axes = [axes]
        
        fig.suptitle(f'Insert Throughput Heatmap ({key_type.capitalize()} Keys)', 
                     fontsize=14, fontweight='bold')
        
        for i, workload in enumerate(workloads):
            wl_df = kt_df[kt_df['workload'] == workload]
            
            # Create pivot table
            pivot = wl_df.pivot(index='index', columns='threads', values='insert_throughput')
            
            # Reindex to ensure consistent ordering
            pivot = pivot.reindex(indexes)
            
            im = axes[i].imshow(pivot.values, cmap='YlOrRd', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.set_label('Mops/sec', fontsize=10)
            
            # Set labels
            axes[i].set_xticks(range(len(pivot.columns)))
            axes[i].set_xticklabels(pivot.columns)
            axes[i].set_yticks(range(len(pivot.index)))
            axes[i].set_yticklabels(pivot.index)
            axes[i].set_xlabel('Threads', fontsize=11)
            axes[i].set_ylabel('Index', fontsize=11)
            axes[i].set_title(f'Workload {workload.upper()}', fontsize=12)
            
            # Add text annotations
            for y in range(len(pivot.index)):
                for x in range(len(pivot.columns)):
                    value = pivot.values[y, x]
                    if not np.isnan(value):
                        text_color = 'white' if value > (pivot.values.max() + pivot.values.min()) / 2 else 'black'
                        axes[i].text(x, y, f'{value:.1f}', ha='center', va='center', 
                                    color=text_color, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'heatmap_{key_type}_keys.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: heatmap_{key_type}_keys.png")


def plot_key_type_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar charts comparing mono vs rand key performance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    indexes = sorted(df['index'].unique())
    workloads = sorted(df['workload'].unique())
    thread_counts = sorted(df['threads'].unique())
    
    colors = {'mono': '#3498db', 'rand': '#e74c3c'}
    
    for workload in workloads:
        wl_df = df[df['workload'] == workload]
        
        fig, axes = plt.subplots(1, len(indexes), figsize=(4*len(indexes), 5))
        if len(indexes) == 1:
            axes = [axes]
        
        fig.suptitle(f'Workload {workload.upper()}: Mono vs Random Keys', 
                     fontsize=14, fontweight='bold')
        
        for i, index in enumerate(indexes):
            idx_df = wl_df[wl_df['index'] == index]
            
            mono_df = idx_df[idx_df['key_type'] == 'mono'].groupby('threads')['insert_throughput'].mean()
            rand_df = idx_df[idx_df['key_type'] == 'rand'].groupby('threads')['insert_throughput'].mean()
            
            x = np.arange(len(thread_counts))
            width = 0.35
            
            mono_vals = [mono_df.get(t, 0) for t in thread_counts]
            rand_vals = [rand_df.get(t, 0) for t in thread_counts]
            
            axes[i].bar(x - width/2, mono_vals, width, label='Mono', color=colors['mono'], alpha=0.8)
            axes[i].bar(x + width/2, rand_vals, width, label='Rand', color=colors['rand'], alpha=0.8)
            
            axes[i].set_xlabel('Threads', fontsize=11)
            axes[i].set_ylabel('Insert Throughput (Mops/sec)', fontsize=11)
            axes[i].set_title(index.upper(), fontsize=12)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(thread_counts)
            axes[i].legend(fontsize=9)
            axes[i].grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'key_comparison_workload_{workload}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: key_comparison_workload_{workload}.png")


def print_trend_analysis(df: pd.DataFrame) -> None:
    """Print analysis of trends in the data."""
    print("\n" + "="*80)
    print("TREND ANALYSIS")
    print("="*80)
    
    indexes = sorted(df['index'].unique())
    workloads = sorted(df['workload'].unique())
    
    # Analyze scalability trends
    print("\n--- Scalability Trends ---")
    for index in indexes:
        idx_df = df[df['index'] == index]
        grouped = idx_df.groupby('threads')['insert_throughput'].mean()
        if len(grouped) >= 2:
            first_threads = grouped.index.min()
            last_threads = grouped.index.max()
            first_perf = grouped.iloc[0]
            last_perf = grouped.iloc[-1]
            if first_perf > 0:
                speedup = last_perf / first_perf
                ideal_speedup = last_threads / first_threads
                efficiency = (speedup / ideal_speedup) * 100
                print(f"  {index.upper()}: {speedup:.2f}x speedup from {first_threads} to {last_threads} threads "
                      f"({efficiency:.1f}% parallel efficiency)")
    
    # Best performer per workload
    print("\n--- Best Performers ---")
    for workload in workloads:
        wl_df = df[df['workload'] == workload]
        if not wl_df['insert_throughput'].isna().all():
            best_idx = wl_df.loc[wl_df['insert_throughput'].idxmax()]
            print(f"  Workload {workload.upper()} Insert: {best_idx['index'].upper()} "
                  f"with {best_idx['insert_throughput']:.2f} Mops/sec "
                  f"({best_idx['threads']} threads, {best_idx['key_type']} keys)")
        
        if not wl_df['operation_throughput'].isna().all():
            best_idx = wl_df.loc[wl_df['operation_throughput'].idxmax()]
            op_type = best_idx['operation_type']
            print(f"  Workload {workload.upper()} {op_type}: {best_idx['index'].upper()} "
                  f"with {best_idx['operation_throughput']:.2f} Mops/sec "
                  f"({best_idx['threads']} threads, {best_idx['key_type']} keys)")
    
    # Mono vs Random comparison
    print("\n--- Key Type Comparison (Insert Throughput) ---")
    for index in indexes:
        idx_df = df[df['index'] == index]
        mono_mean = idx_df[idx_df['key_type'] == 'mono']['insert_throughput'].mean()
        rand_mean = idx_df[idx_df['key_type'] == 'rand']['insert_throughput'].mean()
        if mono_mean > 0 and rand_mean > 0:
            ratio = mono_mean / rand_mean
            better = "mono" if ratio > 1 else "rand"
            diff = abs(ratio - 1) * 100
            print(f"  {index.upper()}: {better} keys are {diff:.1f}% better on average")


def main():
    """Main function to run the visualization."""
    # Determine results directory
    script_dir = Path(__file__).parent
    results_dir = script_dir / "_results"
    output_dir = script_dir / "_plots"
    
    print(f"Looking for results in: {results_dir}")
    print(f"Output plots will be saved to: {output_dir}")
    
    # Load all results
    results = load_all_results(results_dir)
    
    if not results:
        print("No results found.")
        sys.exit(1)
    
    # Print summary
    print_summary(results)
    
    # Filter valid results
    valid_results = filter_valid_results(results)
    
    if not valid_results:
        print("\nNo valid results to visualize.")
        sys.exit(1)
    
    # Create DataFrame
    df = create_dataframe(valid_results)
    
    # Print trend analysis
    print_trend_analysis(df)
    
    # Print outstanding records
    print("\n" + "="*80)
    print("OUTSTANDING RECORDS")
    print("="*80)
    outstanding = identify_outstanding_records(df)
    if not outstanding.empty:
        for _, row in outstanding.iterrows():
            print(f"\n{row['category']}:")
            print(f"  Index: {row['index'].upper()}, Threads: {row['threads']}, Key Type: {row['key_type']}")
            if pd.notna(row.get('insert_throughput')):
                print(f"  Insert Throughput: {row['insert_throughput']:.2f} Mops/sec")
            if pd.notna(row.get('operation_throughput')):
                print(f"  Operation Throughput: {row['operation_throughput']:.2f} Mops/sec")
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    try:
        plot_throughput_by_threads(df, output_dir)
        plot_scalability_comparison(df, output_dir)
        plot_index_comparison_heatmap(df, output_dir)
        plot_key_type_comparison(df, output_dir)
        
        print(f"\nAll plots saved to: {output_dir}")
    except Exception as e:
        print(f"\nError generating plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Save DataFrame to CSV for further analysis
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results data saved to: {csv_path}")


if __name__ == "__main__":
    main()
