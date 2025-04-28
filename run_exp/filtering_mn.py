import sys
import os
import json
import time
import multiprocessing as mp
from functools import partial
from pathlib import Path
from tqdm import tqdm

# Set root path and add to system path
ROOT_PATH = ".."
sys.path.append(ROOT_PATH)

# Import required modules
import fasttext
from utils.preprocessing import get_program_lang, get_doc_type
from pipeline.compute_quality_signals import ComputeCodeQualitySignal
from pipeline.compute_filtering import CodeFilter

# Import all functions from filtering.py
from filtering import (
    process_file,
    compute_quality_signal,
    apply_filtering,
    find_all_files,
    chunk_list,
    process_files_in_parallel,
    save_filtered_files
)

# Get command line arguments
if len(sys.argv) != 5:
    print("Usage: python filtering_mn.py <input_dir> <output_dir> <num_nodes> <num_cores>")
    sys.exit(1)

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
NUM_NODES = int(sys.argv[3])
NUM_CORES = int(sys.argv[4]) if sys.argv[4] != "None" else None

# Constants
CHUNK_SIZE = 100  # Number of files to process in each chunk (same as filtering.py)
NUM_PROCESSES = NUM_CORES if NUM_CORES else mp.cpu_count()

# Get node ID from hostname (assuming hostname ends with node number)
node_id = int(os.popen('hostname').read().strip()[-2:])
print(f'Node ID: {node_id}')

def main():
    """Main function to process and filter files in a multi-node setup"""
    start_time = time.time()

    # Initialize paths
    input_dir = os.path.join(ROOT_PATH, INPUT_DIR) if INPUT_DIR else os.path.join(ROOT_PATH, 'test_data/raw_code')
    output_dir = os.path.join(ROOT_PATH, OUTPUT_DIR) if OUTPUT_DIR else os.path.join(ROOT_PATH, 'test_data/clean_code')

    # Print configuration
    print("\nConfiguration:")
    print(f"- Node ID: {node_id}")
    print(f"- Total Nodes: {NUM_NODES}")
    print(f"- Chunk size: {CHUNK_SIZE} files")
    print(f"- Number of processes per node: {NUM_PROCESSES}")
    print(f"- Input directory: {input_dir}")
    print(f"- Output directory: {output_dir}\n")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Find all files
    all_files = find_all_files(input_dir)
    print(f"Found {len(all_files)} total files")

    # Distribute files among nodes
    node_files = [f for idx, f in enumerate(all_files) if idx % NUM_NODES == node_id % NUM_NODES]
    print(f"This node ({node_id}) will process {len(node_files)} files")

    if not node_files:
        print("No files assigned to this node")
        return

    # Process files in chunks to limit memory usage
    total_chunks = (len(node_files) + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_processed = 0
    total_with_signals = 0
    total_filtered = 0
    total_saved = 0

    print(f"\nProcessing files in {total_chunks} chunks of {CHUNK_SIZE} files each")

    for chunk_idx, chunk_start in enumerate(range(0, len(node_files), CHUNK_SIZE)):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(node_files))
        current_chunk = node_files[chunk_start:chunk_end]

        print(f"\nProcessing chunk {chunk_idx+1}/{total_chunks} ({len(current_chunk)} files)...")

        # Step 1: Process files to extract attributes for this chunk
        print(f"  Extracting file attributes...")
        processed_chunk = process_files_in_parallel(
            current_chunk,
            process_file,
            NUM_PROCESSES,
            f"Chunk {chunk_idx+1}: Extracting"
        )
        total_processed += len(processed_chunk)

        # Step 2: Compute quality signals for this chunk
        print(f"  Computing quality signals...")
        signals_chunk = process_files_in_parallel(
            processed_chunk,
            compute_quality_signal,
            NUM_PROCESSES,
            f"Chunk {chunk_idx+1}: Signals"
        )
        total_with_signals += len(signals_chunk)

        # Step 3: Apply filtering for this chunk
        print(f"  Applying filtering...")
        filtered_chunk = process_files_in_parallel(
            signals_chunk,
            apply_filtering,
            NUM_PROCESSES,
            f"Chunk {chunk_idx+1}: Filtering"
        )

        # Count effective files in this chunk
        effective_chunk = [f for f in filtered_chunk if f.get('effective') == '1']
        total_filtered += len(effective_chunk)
        print(f"  Found {len(effective_chunk)} effective files in chunk {chunk_idx+1}")

        # Step 4: Save filtered files from this chunk
        print(f"  Saving filtered files...")
        saved_count = save_filtered_files(filtered_chunk, output_dir)
        total_saved += saved_count
        print(f"  Saved {saved_count} files from chunk {chunk_idx+1}")

        # Explicitly clear references to free memory
        processed_chunk = None
        signals_chunk = None
        filtered_chunk = None
        effective_chunk = None

    # Print node summary
    print(f"\nNode {node_id} processing complete:")
    print(f"- Total files processed: {total_processed}")
    print(f"- Total quality signals computed: {total_with_signals}")
    print(f"- Total effective after filtering: {total_filtered}")
    print(f"- Total files saved: {total_saved}")
    print(f"- Output directory: {output_dir}")

    # Report total execution time for this node
    total_time = time.time() - start_time
    print(f"\nTotal execution time for node {node_id}: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 