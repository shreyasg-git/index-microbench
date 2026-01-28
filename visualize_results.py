#!/usr/bin/env python3
"""
Visualize benchmark results from _results/ directory.
Generates comparison charts for different index structures.
"""

import os
import re
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Result file pattern: result_{threads}_{keytype}_{workload}_{index}_1
RESULT_PATTERN = re.compile(
    r'result_(\d+)_(mono|rand)_([a-z])_([a-z]+)_1'
)

# ANSI color code pattern for extracting metrics
INSERT_PATTERN = re.compile(r'insert\s+([\d.]+)')
READ_UPDATE_PATTERN = re.compile(r'read/update\s+([\d.]+)')

# Colors for each index type
INDEX_COLORS = {
    'artolc': '#FF6B6B',
    'btreeolc': '#4ECDC4',
    'bwtree': '#45B7D1',
    'masstree': '#96CEB4'
}

INDEX_LABELS = {
    'artolc': 'ART-OLC',
    'btreeolc': 'BTree-OLC',
    'bwtree': 'BwTree',
    'masstree': 'Masstree'
}

WORKLOAD_LABELS = {
    'a': 'Workload A (50% read, 50% update)',
    'c': 'Workload C (100% read)',
    'e': 'Workload E (95% scan, 5% insert)'
}

KEYTYPE_LABELS = {
    'mono': 'Monotonic Keys',
    'rand': 'Random Keys'
}


def parse_result_file(filepath):
    """Parse a result file and extract insert/read-update throughput."""
    insert_throughput = None
    read_update_throughput = None
    
    with open(filepath, 'r', errors='ignore') as f:
        content = f.read()
        
        # Remove ANSI color codes
        content = re.sub(r'\x1b\[[0-9;]*m', '', content)
        
        insert_match = INSERT_PATTERN.search(content)
        read_update_match = READ_UPDATE_PATTERN.search(content)
        
        if insert_match:
            insert_throughput = float(insert_match.group(1))
        if read_update_match:
            read_update_throughput = float(read_update_match.group(1))
    
    return insert_throughput, read_update_throughput


def load_all_results(results_dir):
    """Load all results from the directory into a structured dict."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    for filepath in glob.glob(os.path.join(results_dir, 'result_*')):
        filename = os.path.basename(filepath)
        
        # Skip tmp files
        if filename.endswith('.tmp'):
            continue
            
        match = RESULT_PATTERN.match(filename)
        if not match:
            continue
        
        threads, keytype, workload, index = match.groups()
        threads = int(threads)
        
        insert_tp, read_update_tp = parse_result_file(filepath)
        
        results[threads][keytype][workload][index] = {
            'insert': insert_tp,
            'read_update': read_update_tp
        }
    
    return results


def plot_by_threads(results, output_dir='.'):
    """Create bar charts comparing index performance by thread count."""
    thread_counts = sorted(results.keys())
    indexes = sorted(set(
        idx for tc in results.values() 
        for kt in tc.values() 
        for wl in kt.values() 
        for idx in wl.keys()
    ))
    
    for keytype in ['mono', 'rand']:
        for workload in ['a', 'c', 'e']:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(
                f'{KEYTYPE_LABELS.get(keytype, keytype)} - {WORKLOAD_LABELS.get(workload, workload)}',
                fontsize=14, fontweight='bold'
            )
            
            # Insert throughput
            ax1 = axes[0]
            x = np.arange(len(thread_counts))
            width = 0.2
            
            for i, index in enumerate(indexes):
                insert_vals = []
                for tc in thread_counts:
                    val = results.get(tc, {}).get(keytype, {}).get(workload, {}).get(index, {}).get('insert', 0) or 0
                    insert_vals.append(val)
                
                offset = (i - len(indexes)/2 + 0.5) * width
                bars = ax1.bar(x + offset, insert_vals, width, 
                              label=INDEX_LABELS.get(index, index),
                              color=INDEX_COLORS.get(index, '#888888'))
            
            ax1.set_xlabel('Thread Count')
            ax1.set_ylabel('Throughput (Mops/s)')
            ax1.set_title('Insert Throughput')
            ax1.set_xticks(x)
            ax1.set_xticklabels(thread_counts)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # Read/Update throughput
            ax2 = axes[1]
            
            for i, index in enumerate(indexes):
                read_vals = []
                for tc in thread_counts:
                    val = results.get(tc, {}).get(keytype, {}).get(workload, {}).get(index, {}).get('read_update', 0) or 0
                    read_vals.append(val)
                
                offset = (i - len(indexes)/2 + 0.5) * width
                bars = ax2.bar(x + offset, read_vals, width,
                              label=INDEX_LABELS.get(index, index),
                              color=INDEX_COLORS.get(index, '#888888'))
            
            ax2.set_xlabel('Thread Count')
            ax2.set_ylabel('Throughput (Mops/s)')
            ax2.set_title('Read/Update Throughput')
            ax2.set_xticks(x)
            ax2.set_xticklabels(thread_counts)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            output_file = os.path.join(output_dir, f'benchmark_{keytype}_{workload}.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f'Saved: {output_file}')
            plt.close()


def plot_scalability(results, output_dir='.'):
    """Create line charts showing scalability across thread counts."""
    thread_counts = sorted(results.keys())
    indexes = sorted(set(
        idx for tc in results.values() 
        for kt in tc.values() 
        for wl in kt.values() 
        for idx in wl.keys()
    ))
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Index Scalability Comparison', fontsize=16, fontweight='bold')
    
    for col, workload in enumerate(['a', 'c', 'e']):
        for row, keytype in enumerate(['mono', 'rand']):
            ax = axes[row, col]
            
            for index in indexes:
                throughputs = []
                for tc in thread_counts:
                    val = results.get(tc, {}).get(keytype, {}).get(workload, {}).get(index, {}).get('read_update', 0) or 0
                    throughputs.append(val)
                
                ax.plot(thread_counts, throughputs, 'o-',
                       label=INDEX_LABELS.get(index, index),
                       color=INDEX_COLORS.get(index, '#888888'),
                       linewidth=2, markersize=6)
            
            ax.set_xlabel('Thread Count')
            ax.set_ylabel('Throughput (Mops/s)')
            ax.set_title(f'{KEYTYPE_LABELS.get(keytype, keytype)[:4]} - WL-{workload.upper()}')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_xticks(thread_counts)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'scalability_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_file}')
    plt.close()


def print_summary(results):
    """Print a text summary of results."""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    
    for threads in sorted(results.keys()):
        print(f"\n{'='*40}")
        print(f"THREADS: {threads}")
        print(f"{'='*40}")
        
        for keytype in ['mono', 'rand']:
            if keytype not in results[threads]:
                continue
            print(f"\n  {KEYTYPE_LABELS.get(keytype, keytype)}")
            print(f"  {'-'*35}")
            
            for workload in ['a', 'c', 'e']:
                if workload not in results[threads][keytype]:
                    continue
                print(f"\n    Workload {workload.upper()}:")
                
                for index, metrics in sorted(results[threads][keytype][workload].items()):
                    insert = metrics.get('insert', 'N/A')
                    read_update = metrics.get('read_update', 'N/A')
                    insert_str = f"{insert:.2f}" if isinstance(insert, float) else str(insert)
                    read_str = f"{read_update:.2f}" if isinstance(read_update, float) else str(read_update)
                    print(f"      {INDEX_LABELS.get(index, index):12s}: Insert={insert_str:>8} Mops/s, Read/Update={read_str:>8} Mops/s")


def main():
    # Find results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '_results')
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    print(f"Loading results from: {results_dir}")
    results = load_all_results(results_dir)
    
    if not results:
        print("No results found!")
        return 1
    
    # Print summary
    print_summary(results)
    
    # Generate plots
    output_dir = os.path.join(script_dir, '_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating plots in: {output_dir}")
    plot_by_threads(results, output_dir)
    plot_scalability(results, output_dir)
    
    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())
