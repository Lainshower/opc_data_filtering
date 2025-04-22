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

# Constants
CHUNK_SIZE = 3000  # Number of files to process in each chunk
NUM_PROCESSES = mp.cpu_count()  # Default: Use all available CPU cores

# Load language prediction model once at module level
_lang_predictor = fasttext.load_model(f'{ROOT_PATH}/artifacts/lang_predictor.bin')

def process_file(file_path, repo_path):
    """Process a single file and extract its attributes
    
    Args:
        file_path (str): Path to the file to process
        repo_path (str): Path to the repository root
        
    Returns:
        dict: File metadata dictionary or None if processing failed
    """
    try:
        # Access the global language predictor model
        global _lang_predictor
        
        # Convert to Path objects for easier handling
        file_path = Path(file_path)
        repo_path = Path(repo_path)
        
        # Get relative path from repository root
        rel_path = file_path.relative_to(repo_path)
        name = file_path.stem
        ext = file_path.suffix[1:] if file_path.suffix else ''
        
        # Read file content
        with open(file_path, 'r', errors='ignore') as f:
            text = f.read()
            
        # Get file size
        file_size_in_byte = os.path.getsize(file_path)
        
        # Predict language
        predictions = _lang_predictor.predict(text.lower().replace("\n", " "))
        lang = predictions[0][0].replace('__label__', '')
        
        # Get programming language and doc type
        program_lang = get_program_lang(name, ext)
        doc_type = get_doc_type(program_lang)
        
        # Create file metadata dictionary
        file_data = {
            'file_path': str(file_path),
            'rel_path': str(rel_path),
            'filename': name,
            'ext': ext,
            'text': text,
            'lang': lang,
            'file_size_in_byte': file_size_in_byte,
            'program_lang': program_lang,
            'doc_type': doc_type
        }
        
        return file_data
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def compute_quality_signal(file_data):
    """Compute quality signals for a single file
    
    Args:
        file_data (dict): File metadata dictionary
        
    Returns:
        dict: Updated file metadata with quality signals or None if processing failed
    """
    if file_data is None:
        return None
        
    try:
        ccqs = ComputeCodeQualitySignal()
        result = ccqs.evaluate(
            text=file_data['text'],
            filename=file_data['filename'],
            lang=file_data['lang'],
            ext=file_data['ext'],
            file_size_in_byte=file_data['file_size_in_byte'],
            program_lang=file_data['program_lang'],
            doc_type=file_data['doc_type']
        )
        
        # Parse result and add quality signal to file data
        result_json = json.loads(result)
        if 'quality_signal' in result_json:
            file_data['quality_signal'] = result_json['quality_signal']
        else:
            file_data['quality_signal'] = None
            
        return file_data
        
    except Exception as e:
        print(f"Error computing quality signal for {file_data['file_path']}: {str(e)}")
        return None

def apply_filtering(file_data):
    """Apply filtering to a single file based on quality signals
    
    Args:
        file_data (dict): File metadata dictionary with quality signals
        
    Returns:
        dict: Updated file metadata with filtering results or None if processing failed
    """
    if file_data is None or file_data.get('quality_signal') is None:
        return None
        
    try:
        # Special case: __init__.py files are always kept
        if (file_data.get('ext') == 'py' and 
            'init' in file_data.get('filename', '').lower()):
            file_data['effective'] = '1'
            file_data['hit_map'] = '{}'
            return file_data
            
        code_filter = CodeFilter()
        result = code_filter.evaluate(
            doc_type=file_data['doc_type'],
            lang=file_data['lang'],
            program_lang=file_data['program_lang'],
            quality_signal=file_data['quality_signal']
        )
        
        # Parse result and add filtering results to file data
        filter_result = json.loads(result)
        file_data.update(filter_result)
        
        return file_data
        
    except Exception as e:
        print(f"Error filtering file {file_data['file_path']}: {str(e)}")
        return None

def find_repositories(directory):
    """Find all repositories in a directory
    
    Args:
        directory (str): Directory to search for repositories
        
    Returns:
        list: List of repository paths
    """
    return [os.path.join(directory, d) for d in os.listdir(directory) 
            if os.path.isdir(os.path.join(directory, d))]

def find_files_in_repository(repo_path):
    """Find all files in a repository
    
    Args:
        repo_path (str): Repository path
        
    Returns:
        list: List of file paths
    """
    all_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def chunk_list(lst, chunk_size):
    """Split a list into chunks of specified size
    
    Args:
        lst (list): List to split
        chunk_size (int): Size of each chunk
        
    Yields:
        list: Chunks of the original list
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def process_files_in_parallel(file_paths, processor_func, num_processes, desc, *args):
    """Process files in parallel using multiprocessing
    
    Args:
        file_paths (list): List of file paths to process
        processor_func (function): Function to apply to each file
        num_processes (int): Number of processes to use
        desc (str): Description for the progress bar
        *args: Additional arguments to pass to processor_func
        
    Returns:
        list: List of processed results (None results are filtered out)
    """
    if not file_paths:
        return []
        
    # Create a partial function with additional arguments if provided
    if args:
        func = partial(processor_func, *args)
    else:
        func = processor_func
        
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(func, file_paths),
            total=len(file_paths),
            desc=desc
        ))
    return [r for r in results if r is not None]

def generate_tree_structure(repo_path, filtered_files):
    """Generate tree structure of the repository for filtered files
    
    Args:
        repo_path (str): Path to the repository
        filtered_files (list): List of filtered file data dictionaries
        
    Returns:
        str: Tree structure string
    """
    # Get relative paths of filtered files and sort them
    rel_paths = [Path(f['rel_path']) for f in filtered_files if f.get('effective') == '1']
    rel_paths.sort()
    
    if not rel_paths:
        return ""
    
    # Convert paths to directory structure
    dirs = {}
    for path in rel_paths:
        parts = path.parts
        current = dirs
        
        # Process directories
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Add file to its directory
        if '__files__' not in current:
            current['__files__'] = []
        current['__files__'].append(parts[-1])
    
    # Generate tree lines recursively
    lines = []
    
    def add_to_tree(node, prefix=''):
        # Sort directories and files
        dirs = sorted([k for k in node.keys() if k != '__files__'])
        files = sorted(node.get('__files__', []))
        
        # Process directories
        for i, d in enumerate(dirs):
            is_last_dir = (i == len(dirs) - 1) and not files
            lines.append(f"{prefix}{'└── ' if is_last_dir else '├── '}{d}/")
            add_to_tree(node[d], prefix + ('    ' if is_last_dir else '│   '))
        
        # Process files
        for i, f in enumerate(files):
            is_last = i == len(files) - 1
            lines.append(f"{prefix}{'└── ' if is_last else '├── '}{f}")
    
    add_to_tree(dirs)
    return '\n'.join(lines)

def save_repository_data(filtered_data, repo_path, output_dir):
    """Save filtered files from a repository to a single JSONL file
    
    Args:
        filtered_data (list): List of filtered file data dictionaries
        repo_path (str): Path to the repository
        output_dir (str): Directory to save output files
        
    Returns:
        int: Number of files successfully saved
    """
    try:
        # Create output directory structure
        repo_name = os.path.basename(repo_path)
        repo_output_dir = os.path.join(output_dir, repo_name)
        os.makedirs(repo_output_dir, exist_ok=True)
        
        # Output file path
        output_file = os.path.join(repo_output_dir, f"{repo_name}.jsonl")
        
        # Sort filtered data by relative path to maintain directory structure
        filtered_data.sort(key=lambda x: x.get('rel_path', ''))
        
        # Get effective files only
        effective_files = [f for f in filtered_data if f.get('effective') == '1']
        
        # Generate tree structure as string
        tree_str = generate_tree_structure(repo_path, effective_files)
        
        saved_count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            # First, write the tree structure
            f.write(json.dumps({
                "type": "tree_structure",
                "tree": tree_str
            }, ensure_ascii=False) + '\n')
            
            # Then write the filtered files
            for file_data in effective_files:
                # Get relative path from repository root
                rel_path = file_data['rel_path']
                
                # Prepare data for jsonl
                json_data = {
                    "text": file_data['text'],
                    "file_name": file_data['filename'],
                    "ext": file_data['ext'],
                    "path": rel_path,  # Full relative path including filename
                    "file_path": str(Path(repo_name) / rel_path)  # repo/path/to/file format
                }
                
                # Write to file
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')
                saved_count += 1
                
        return saved_count
        
    except Exception as e:
        print(f"Error saving repository data for {repo_path}: {str(e)}")
        return 0

def process_file_with_repo(args):
    """Wrapper function to process a file with repository path
    
    Args:
        args (tuple): Tuple containing (file_path, repo_path)
        
    Returns:
        dict: File metadata dictionary or None if processing failed
    """
    file_path, repo_path = args
    return process_file(file_path, repo_path)

def process_repository(repo_path, output_dir, chunk_size=CHUNK_SIZE, num_processes=NUM_PROCESSES):
    """Process a single repository"""
    repo_name = os.path.basename(repo_path)
    print(f"\nProcessing repository: {repo_name}")
    
    try:
        # Find all files in repository
        all_files = find_files_in_repository(repo_path)
        if not all_files:
            print(f"No files found in repository: {repo_name}")
            return 0, 0, 0, 0

        total_processed = 0
        total_with_signals = 0
        total_filtered = 0
        processed_files = []

        for chunk in chunk_list(all_files, chunk_size):
            try:
                # Create list of (file_path, repo_path) tuples for processing
                chunk_with_repo = [(f, repo_path) for f in chunk]
                
                # Process files
                processed_chunk = process_files_in_parallel(
                    chunk_with_repo,
                    process_file_with_repo,
                    num_processes,
                    f"{repo_name}: Processing"
                )
                if not processed_chunk:  # 빈 리스트이거나 None인 경우
                    raise Exception("File processing failed")
                total_processed += len(processed_chunk)
                
                # Compute quality signals
                signals_chunk = process_files_in_parallel(
                    processed_chunk,
                    compute_quality_signal,
                    num_processes,
                    f"{repo_name}: Signals"
                )
                if not signals_chunk:
                    raise Exception("Signal computation failed")
                total_with_signals += len(signals_chunk)
                
                # Apply filtering
                filtered_chunk = process_files_in_parallel(
                    signals_chunk,
                    apply_filtering,
                    num_processes,
                    f"{repo_name}: Filtering"
                )
                if not filtered_chunk:
                    raise Exception("Filtering failed")
                
                # Add filtered files to the list
                processed_files.extend(filtered_chunk)

            except Exception as chunk_error:
                print(f"Error processing chunk in repository {repo_name}: {str(chunk_error)}")
                return 0, 0, 0, 0

        # Mark __init__.py files as effective
        for file_data in processed_files:
            if (file_data.get('ext') == 'py' and 
                'init' in file_data.get('filename', '').lower()):
                file_data['effective'] = '1'
                
        # Count effective files
        effective_files = [f for f in processed_files if f.get('effective') == '1']
        total_filtered = len(effective_files)
        
        # Save repository data
        total_saved = save_repository_data(processed_files, repo_path, output_dir)
        
        print(f"Repository {repo_name} summary:")
        print(f"- Files processed: {total_processed}")
        print(f"- Quality signals computed: {total_with_signals}")
        print(f"- Effective after filtering: {total_filtered}")
        print(f"- Successfully saved: {total_saved}")
        
        return total_processed, total_with_signals, total_filtered, total_saved
        
    except Exception as e:
        print(f"Error processing repository {repo_name}: {str(e)}")
        return 0, 0, 0, 0

def main(chunk_size=CHUNK_SIZE, num_processes=NUM_PROCESSES, input_dir=None, output_dir=None):
    """Main function to process and filter repositories
    
    Args:
        chunk_size (int): Number of files to process in each chunk
        num_processes (int): Number of CPU cores to use
        input_dir (str): Directory containing repositories
        output_dir (str): Directory to save filtered data
    """
    start_time = time.time()
    
    # Initialize paths
    input_dir = os.path.join(ROOT_PATH, input_dir) if input_dir else os.path.join(ROOT_PATH, 'test_data/raw_code')
    output_dir = os.path.join(ROOT_PATH, output_dir) if output_dir else os.path.join(ROOT_PATH, 'test_data/clean_code')
    
    # Print configuration
    print("\nConfiguration:")
    print(f"- Chunk size: {chunk_size} files")
    print(f"- Number of processes: {num_processes}")
    print(f"- Input directory: {input_dir}")
    print(f"- Output directory: {output_dir}\n")
    
    # Find all repositories
    repositories = find_repositories(input_dir)
    print(f"Found {len(repositories)} repositories")
    
    # Process each repository
    total_stats = {
        'processed': 0,
        'signals': 0,
        'filtered': 0,
        'saved': 0
    }
    
    for repo_path in repositories:
        processed, signals, filtered, saved = process_repository(
            repo_path,
            output_dir,
            chunk_size,
            num_processes
        )
        
        total_stats['processed'] += processed
        total_stats['signals'] += signals
        total_stats['filtered'] += filtered
        total_stats['saved'] += saved
    
    # Print final summary
    print(f"\nProcessing complete:")
    print(f"- Total repositories: {len(repositories)}")
    print(f"- Total files processed: {total_stats['processed']}")
    print(f"- Total quality signals computed: {total_stats['signals']}")
    print(f"- Total effective after filtering: {total_stats['filtered']}")
    print(f"- Total files saved: {total_stats['saved']}")
    print(f"- Output directory: {output_dir}")
    
    # Report total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process and filter code repositories')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE,
                      help=f'Number of files to process in each chunk (default: {CHUNK_SIZE})')
    parser.add_argument('--num-processes', type=int, default=NUM_PROCESSES,
                      help=f'Number of CPU cores to use (default: all available cores)')
    parser.add_argument('--input-dir', type=str,
                      help='Directory containing repositories')
    parser.add_argument('--output-dir', type=str,
                      help='Directory to save filtered data')
    
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(
        chunk_size=args.chunk_size,
        num_processes=args.num_processes,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    ) 