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
from pipeline.compute_filtering_script import CodeFilter

# Constants - These can be adjusted based on your system resources and dataset size
CHUNK_SIZE = 100  # Number of files to process in each chunk (increase for better I/O efficiency with large datasets)
NUM_PROCESSES = mp.cpu_count()  # Default: Use all available CPU cores
MAX_FILE_SIZE_GB = 20  # Maximum size of each output JSONL file in GB

# Load language prediction model once at module level
_lang_predictor = fasttext.load_model(f'{ROOT_PATH}/artifacts/lang_predictor.bin')

def process_file(file_path):
    """Process a single file and extract its attributes
    
    Args:
        file_path (str): Path to the file to process
        
    Returns:
        dict: File metadata dictionary or None if processing failed
    """
    try:
        # Access the global language predictor model
        global _lang_predictor
        
        # Convert to Path object for easier handling
        file_path = Path(file_path)
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

# Main processing functions
def find_all_files(directory):
    """Find all files in a directory recursively
    
    Args:
        directory (str): Directory to search for files
        
    Returns:
        list: List of file paths
    """
    all_files = []
    for root, _, files in os.walk(directory):
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

def save_filtered_files(filtered_data, target_dir, max_file_size_gb=MAX_FILE_SIZE_GB):
    """Save filtered files to jsonl files by language and filename
    
    Args:
        filtered_data (list): List of filtered file data dictionaries
        target_dir (str): Directory to save output files
        max_file_size_gb (int): (Unused) Kept for compatibility
        
    Returns:
        int: Number of files successfully saved
    """
    saved_count = 0
    for file_data in filtered_data:
        if file_data.get('effective') == '1':
            lang = file_data['program_lang']
            lang_dir = os.path.join(target_dir, lang)
            os.makedirs(lang_dir, exist_ok=True)
            file_name = f"{file_data['filename']}.jsonl"
            file_path = os.path.join(lang_dir, file_name)
            json_data = {
                "problem": file_data['text'],
                "file_name": file_data['filename'],
                "ext": file_data['ext']
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json_str = json.dumps(json_data, ensure_ascii=False) + '\n'
                f.write(json_str)
                saved_count += 1
    return saved_count

# Main execution
def main(chunk_size=CHUNK_SIZE, num_processes=NUM_PROCESSES, max_file_size_gb=MAX_FILE_SIZE_GB, 
         input_dir=None, output_dir=None):
    """Main function to process and filter code files
    
    Args:
        chunk_size (int): Number of files to process in each chunk. Larger values improve I/O efficiency
                         but require more memory.
        num_processes (int): Number of CPU cores to use for parallel processing.
        max_file_size_gb (int): Maximum size of each output JSONL file in GB.
        input_dir (str, optional): Directory containing raw code files. Default is test_data/raw_code.
        output_dir (str, optional): Directory to save filtered code files. Default is test_data/clean_code.
        
    Returns:
        None
    """
    start_time = time.time()
    
    # Initialize paths
    input_dir = f'{ROOT_PATH}/{input_dir}'
    output_dir = f'{ROOT_PATH}/{output_dir}'
    
    # Print configuration
    print("\nConfiguration:")
    print(f"- Chunk size: {chunk_size} files")
    print(f"- Number of processes: {num_processes}")
    print(f"- Maximum output file size: {max_file_size_gb} GB")
    print(f"- Input directory: {input_dir}")
    print(f"- Output directory: {output_dir}\n")
    
    # Step 1: Find all files
    print("Finding all files...")
    all_files = find_all_files(input_dir)
    print(f"Found {len(all_files)} files")
    
    # Process files in chunks to limit memory usage
    total_chunks = (len(all_files) + chunk_size - 1) // chunk_size  # Ceiling division
    total_processed = 0
    total_with_signals = 0
    total_filtered = 0
    total_saved = 0
    
    print(f"\nProcessing files in {total_chunks} chunks of {chunk_size} files each")
    
    # Process each chunk separately to limit memory usage
    for chunk_idx, chunk_start in enumerate(range(0, len(all_files), chunk_size)):
        chunk_end = min(chunk_start + chunk_size, len(all_files))
        current_chunk = all_files[chunk_start:chunk_end]
        
        print(f"\nProcessing chunk {chunk_idx+1}/{total_chunks} ({len(current_chunk)} files)...")
        
        # Step 2: Process files to extract attributes for this chunk
        print(f"  Extracting file attributes...")
        processed_chunk = process_files_in_parallel(
            current_chunk,
            process_file,
            num_processes,
            f"Chunk {chunk_idx+1}: Extracting"
        )
        total_processed += len(processed_chunk)
        
        # Step 3: Compute quality signals for this chunk
        print(f"  Computing quality signals...")
        signals_chunk = process_files_in_parallel(
            processed_chunk,
            compute_quality_signal,
            num_processes,
            f"Chunk {chunk_idx+1}: Signals"
        )
        total_with_signals += len(signals_chunk)
        
        # Step 4: Apply filtering for this chunk
        print(f"  Applying filtering...")
        filtered_chunk = process_files_in_parallel(
            signals_chunk,
            apply_filtering,
            num_processes,
            f"Chunk {chunk_idx+1}: Filtering"
        )
        
        # Count effective files in this chunk
        effective_chunk = [f for f in filtered_chunk if f.get('effective') == '1']
        total_filtered += len(effective_chunk)
        print(f"  Found {len(effective_chunk)} effective files in chunk {chunk_idx+1}")
        
        # Step 5: Save filtered files from this chunk
        print(f"  Saving filtered files...")
        saved_count = save_filtered_files(filtered_chunk, output_dir, max_file_size_gb)
        total_saved += saved_count
        print(f"  Saved {saved_count} files from chunk {chunk_idx+1}")
        
        # Explicitly clear references to free memory
        processed_chunk = None
        signals_chunk = None
        filtered_chunk = None
        effective_chunk = None
    
    # Print summary
    print(f"\nProcessing complete:")
    print(f"- Total files found: {len(all_files)}")
    print(f"- Successfully processed: {total_processed} files")
    print(f"- Quality signals computed: {total_with_signals} files")
    print(f"- Effective after filtering: {total_filtered} files")
    print(f"- Successfully saved: {total_saved} files to path: {output_dir}")
    
    # Report total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process and filter code files using multiprocessing')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help=f'Number of files to process in each chunk (default: {CHUNK_SIZE})')
    parser.add_argument('--num-processes', type=int, default=NUM_PROCESSES, help=f'Number of CPU cores to use (default: all available cores)')
    parser.add_argument('--max-file-size', type=int, default=MAX_FILE_SIZE_GB, help=f'Maximum size of each output JSONL file in GB (default: {MAX_FILE_SIZE_GB})')
    parser.add_argument('--input-dir', type=str, help='Directory containing raw code files')
    parser.add_argument('--output-dir', type=str, help='Directory to save filtered code files')
    
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(
        chunk_size=args.chunk_size,
        num_processes=args.num_processes,
        max_file_size_gb=args.max_file_size,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )