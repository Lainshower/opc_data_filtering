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

# Get command line arguments
if len(sys.argv) != 5:
    print("Usage: python filtering_repo_mn.py <input_dir> <output_dir> <num_nodes> <num_cores>")
    sys.exit(1)

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
NUM_NODES = int(sys.argv[3])
NUM_CORES = int(sys.argv[4]) if sys.argv[4] != "None" else None

# Constants
CHUNK_SIZE = 3000  # Number of files to process in each chunk
NUM_PROCESSES = NUM_CORES if NUM_CORES else mp.cpu_count()  # Use specified cores or all available

# Get node ID from hostname (assuming hostname ends with node number)
node_id = int(os.popen('hostname').read().strip()[-2:])
print(f'Node ID: {node_id}')

# Load language prediction model once at module level
_lang_predictor = fasttext.load_model(f'{ROOT_PATH}/artifacts/lang_predictor.bin')

# Import all functions from filtering_repo.py
from filtering_repo import (
    process_file,
    compute_quality_signal,
    apply_filtering,
    find_repositories,
    find_files_in_repository,
    chunk_list,
    process_files_in_parallel,
    generate_tree_structure,
    save_repository_data,
    process_file_with_repo,
    process_repository
)

def main():
    """Main function to process and filter repositories in a multi-node setup"""
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
    
    # Find all repositories
    repositories = find_repositories(input_dir)
    print(f"Found {len(repositories)} total repositories")
    
    # Distribute repositories among nodes
    node_repositories = [repo for idx, repo in enumerate(repositories) if idx % NUM_NODES == node_id % NUM_NODES]
    print(f"This node ({node_id}) will process {len(node_repositories)} repositories")
    
    if not node_repositories:
        print("No repositories assigned to this node")
        return
    
    # Process repositories assigned to this node
    total_stats = {
        'processed': 0,
        'signals': 0,
        'filtered': 0,
        'saved': 0
    }
    
    for repo_path in node_repositories:
        processed, signals, filtered, saved = process_repository(
            repo_path,
            output_dir,
            chunk_size=CHUNK_SIZE,
            num_processes=NUM_PROCESSES
        )
        
        total_stats['processed'] += processed
        total_stats['signals'] += signals
        total_stats['filtered'] += filtered
        total_stats['saved'] += saved
    
    # Print node summary
    print(f"\nNode {node_id} processing complete:")
    print(f"- Repositories processed: {len(node_repositories)}")
    print(f"- Total files processed: {total_stats['processed']}")
    print(f"- Total quality signals computed: {total_stats['signals']}")
    print(f"- Total effective after filtering: {total_stats['filtered']}")
    print(f"- Total files saved: {total_stats['saved']}")
    print(f"- Output directory: {output_dir}")
    
    # Report total execution time for this node
    total_time = time.time() - start_time
    print(f"\nTotal execution time for node {node_id}: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 