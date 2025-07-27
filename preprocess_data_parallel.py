#!/usr/bin/env python3
"""
Multiprocessing version of preprocess_data.sh
Handles the 4-step preprocessing pipeline with proper dependency management.
"""

import os
import sys
import argparse
import subprocess
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_mesh_files(data_path, folder_to_parse):
    """Discover all mesh files and organize them into (category, filename) tuples."""
    all_ids = []
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist")
    
    # Find all category folders (those starting with '0')
    folders = sorted([item for item in os.listdir(data_path) if item.startswith('0')])
    
    for folder in folders:
        folder_path = data_path / folder / folder_to_parse
        if folder_path.exists():
            cur_ids = sorted(os.listdir(folder_path))
            cur_ids = [(folder, item) for item in cur_ids if not item.startswith('.')]
            all_ids.extend(cur_ids)
    
    logger.info(f"Found {len(all_ids)} files in {len(folders)} categories")
    return all_ids

def create_chunks(all_ids, chunk_size):
    """Split all_ids into chunks for parallel processing."""
    chunks = [all_ids[i:i + chunk_size] for i in range(0, len(all_ids), chunk_size)]
    logger.info(f"Created {len(chunks)} chunks with chunk_size={chunk_size}")
    return chunks

def run_preprocessing_step(step_name, script_path, chunk_id, args_dict, show_output=True):
    """Run a single preprocessing step for a specific chunk."""
    try:
        # Build command arguments
        cmd_args = [
            'python', '-u', script_path,
            '--chunk_id', str(chunk_id)
        ]
        
        # Add all other arguments
        for key, value in args_dict.items():
            if isinstance(value, bool) and value:
                cmd_args.append(f'--{key}')
            elif not isinstance(value, bool):
                cmd_args.extend([f'--{key}', str(value)])
        
        logger.info(f"Starting {step_name} chunk {chunk_id}")
        logger.debug(f"Command: {' '.join(cmd_args)}")
        
        if show_output:
            # Run with real-time output
            with subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            ) as process:
                
                output_lines = []
                for line in process.stdout:
                    line = line.rstrip()
                    if line:  # Only log non-empty lines
                        # Prefix each line with step and chunk info
                        prefixed_line = f"[{step_name}-{chunk_id}] {line}"
                        logger.info(prefixed_line)
                        output_lines.append(line)
                
                # Wait for process to complete
                process.wait(timeout=3600)  # 1 hour timeout
                
                if process.returncode != 0:
                    error_msg = f"{step_name} chunk {chunk_id} failed with return code {process.returncode}"
                    logger.error(error_msg)
                    # Show last few lines of output for context
                    if output_lines:
                        logger.error("Last 10 lines of output:")
                        for line in output_lines[-10:]:
                            logger.error(f"  {line}")
                    return False, chunk_id, error_msg
        else:
            # Run without showing output (fallback mode)
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode != 0:
                error_msg = f"{step_name} chunk {chunk_id} failed with return code {result.returncode}"
                logger.error(error_msg)
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False, chunk_id, error_msg
        
        logger.info(f"✓ {step_name} chunk {chunk_id} completed successfully")
        return True, chunk_id, None
        
    except subprocess.TimeoutExpired:
        error_msg = f"{step_name} chunk {chunk_id} timed out after 1 hour"
        logger.error(error_msg)
        return False, chunk_id, error_msg
    except Exception as e:
        error_msg = f"{step_name} chunk {chunk_id} failed with exception: {e}"
        logger.error(error_msg)
        return False, chunk_id, error_msg

def run_step_parallel(step_name, script_name, num_chunks, max_workers, base_args, show_output=True):
    """Run a preprocessing step in parallel across all chunks."""
    logger.info(f"Starting {step_name} with {num_chunks} chunks using {max_workers} workers")
    
    script_path = Path(__file__).parent / 'preprocessing_scripts' / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script {script_path} not found")
    
    successful = 0
    failed = 0
    failed_chunks = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(run_preprocessing_step, step_name, str(script_path), chunk_id, base_args, show_output): chunk_id
            for chunk_id in range(num_chunks)
        }
        
        # Process results with progress bar
        with tqdm(total=num_chunks, desc=f"{step_name} chunks", position=0, leave=True) as pbar:
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    success, returned_chunk_id, error = future.result()
                    if success:
                        successful += 1
                        pbar.set_description(f"{step_name} chunks (✓ {successful})")
                    else:
                        failed += 1
                        failed_chunks.append(returned_chunk_id)
                        logger.warning(f"{step_name} chunk {returned_chunk_id} failed: {error}")
                        pbar.set_description(f"{step_name} chunks (✗ {failed})")
                except Exception as e:
                    failed += 1
                    failed_chunks.append(chunk_id)
                    logger.error(f"Exception in {step_name} chunk {chunk_id}: {e}")
                
                pbar.update(1)
                pbar.set_postfix(success=successful, failed=failed)
    
    if failed > 0:
        logger.warning(f"{step_name} completed with {failed} failed chunks: {failed_chunks}")
    else:
        logger.info(f"✓ {step_name} completed successfully (all {successful} chunks)")
    
    return successful, failed, failed_chunks

def main():
    parser = argparse.ArgumentParser(description='Parallel preprocessing of mesh data')
    
    # Data paths
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to input data directory')
    parser.add_argument('--script_path', type=str, 
                       default=str(Path(__file__).parent / 'preprocessing_scripts'),
                       help='Path to preprocessing scripts directory')
    
    # Parallelization settings
    parser.add_argument('--max_workers', type=int, default=mp.cpu_count(),
                       help='Maximum number of parallel workers')
    parser.add_argument('--chunk_size', type=int, default=10,
                       help='Number of files per chunk')
    
    # Output control
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress subprocess output (only show errors)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging (debug level)')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Save logs to file in addition to console')
    
    # Step selection
    parser.add_argument('--skip_simplification', action='store_true',
                       help='Skip mesh simplification step')
    parser.add_argument('--skip_skeletons', action='store_true',
                       help='Skip skeleton computation step')
    parser.add_argument('--skip_skelrays', action='store_true',
                       help='Skip skelrays computation step')
    parser.add_argument('--skip_downsampling', action='store_true',
                       help='Skip skeleton downsampling step')
    
    # Processing parameters
    parser.add_argument('--simple_mesh_folder', type=str, default='watertight_simple',
                       help='Output folder for simplified meshes')
    parser.add_argument('--num_faces', type=int, default=200000,
                       help='Number of faces for mesh simplification')
    parser.add_argument('--skel_folder', type=str, default='skeletons_min_sdf_iter_50',
                       help='Skeleton folder name')
    parser.add_argument('--skel_num_iter', type=int, default=50,
                       help='Number of iterations for skeleton computation')
    parser.add_argument('--skel_nn', type=int, default=8,
                       help='Number of nearest neighbors for envelope size estimation')
    parser.add_argument('--skel_downsample_k', type=int, default=1024,
                       help='Downsampling size for skeletons')
    parser.add_argument('--reg_sample', type=int, default=40000,
                       help='Size of regularization sample')
    parser.add_argument('--num_sphere_points', type=int, default=1000,
                       help='Number of directions for implicit function estimation')
    parser.add_argument('--sampling_mode', type=str, default='random',
                       help='Mode for sampling directions')
    parser.add_argument('--skelray_output_folder', type=str, 
                       default='skelrays_min_sdf_iter_50_1000rays_random',
                       help='Output folder for skelrays')
    parser.add_argument('--skel_downsample_k_final', type=int, default=10000,
                       help='Final downsampling size for skeletons')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Set up handlers
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True  # Override the basic config set at module level
    )
    
    # Show subprocess output unless --quiet is specified
    show_subprocess_output = not args.quiet
    
    # Validate paths
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path {data_path} does not exist")
        return
    
    script_path = Path(args.script_path)
    if not script_path.exists():
        logger.error(f"Script path {script_path} does not exist")
        return
    
    # Discover all mesh files and create chunks
    try:
        all_ids = get_all_mesh_files(data_path, '4_watertight_scaled')
        if not all_ids:
            logger.error("No mesh files found to process")
            return
        
        chunks = create_chunks(all_ids, args.chunk_size)
        num_chunks = len(chunks)
        
    except Exception as e:
        logger.error(f"Error discovering mesh files: {e}")
        return
    
    logger.info(f"=== GEM3D Parallel Preprocessing Pipeline ===")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Workers: {args.max_workers}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Subprocess output: {'Hidden' if args.quiet else 'Shown'}")
    if args.log_file:
        logger.info(f"Log file: {args.log_file}")
    logger.info(f"===============================================")
    
    # Step 1: Mesh Simplification
    if not args.skip_simplification:
        simplify_args = {
            'in_path': str(data_path),
            'out_path': str(data_path),
            'folder_to_parse': '4_watertight_scaled',
            'chunk_size': args.chunk_size,
            'num_faces': args.num_faces,
            'output_format': 'off',
            'output_folder': args.simple_mesh_folder
        }
        
        success, failed, failed_chunks = run_step_parallel(
            "Mesh Simplification", 
            "simplify_mesh_shapenet.py",
            num_chunks, 
            args.max_workers,
            simplify_args,
            show_subprocess_output
        )
        
        if failed > 0:
            logger.warning(f"Mesh simplification had {failed} failed chunks, continuing...")
    
    # Step 2: Skeleton Computation (depends on step 1)
    if not args.skip_skeletons:
        skeleton_args = {
            'data_path': str(data_path),
            'out_path': str(data_path),
            'folder_to_parse': args.simple_mesh_folder,
            'chunk_size': args.chunk_size,
            'output_folder': args.skel_folder,
            'ignore_starting_zero': True,
            'num_iter': args.skel_num_iter,
            'use_min_sdf_skel': True
        }
        
        success, failed, failed_chunks = run_step_parallel(
            "Skeleton Computation",
            "generate_skeletons_shapenet.py",
            num_chunks,
            args.max_workers,
            skeleton_args,
            show_subprocess_output
        )
        
        if failed > 0:
            logger.warning(f"Skeleton computation had {failed} failed chunks, continuing...")
    
    # Steps 3 & 4: Run skelrays and downsampling in parallel (both depend on step 2)
    remaining_steps = []
    
    # Step 3: Skelrays computation
    if not args.skip_skelrays:
        skelrays_args = {
            'data_path': str(data_path),
            'out_path': str(data_path),
            'mesh_folder': args.simple_mesh_folder,
            'chunk_size': args.chunk_size,
            'output_folder': args.skelray_output_folder,
            'ignore_starting_zero': True,
            'skel_folder': args.skel_folder,
            'skel_nn': args.skel_nn,
            'skel_downsample_k': args.skel_downsample_k,
            'return_reg_points': True,
            'reg_sample': args.reg_sample,
            'num_sphere_points': args.num_sphere_points,
            'sampling_mode': args.sampling_mode,
            'load_min_sdf': True,
            'store_directions': True
        }
        remaining_steps.append(("Skelrays Computation", "generate_skelrays.py", skelrays_args))
    
    # Step 4: Skeleton downsampling
    if not args.skip_downsampling:
        downsample_args = {
            'data_path': str(data_path),
            'out_path': str(data_path),
            'mesh_folder': args.simple_mesh_folder,
            'chunk_size': args.chunk_size,
            'output_folder': args.skel_folder,
            'ignore_starting_zero': True,
            'skel_folder': args.skel_folder,
            'load_min_sdf': True,
            'skel_downsample_k': args.skel_downsample_k_final
        }
        remaining_steps.append(("Skeleton Downsampling", "clean_and_downsample_ply_skeletons.py", downsample_args))
    
    # Run remaining steps in parallel
    if remaining_steps:
        logger.info(f"Running {len(remaining_steps)} remaining steps in parallel")
        
        with ProcessPoolExecutor(max_workers=len(remaining_steps)) as executor:
            future_to_step = {}
            
            for step_name, script_name, step_args in remaining_steps:
                future = executor.submit(
                    run_step_parallel,
                    step_name,
                    script_name,
                    num_chunks,
                    args.max_workers // len(remaining_steps) + 1,  # Distribute workers
                    step_args,
                    show_subprocess_output
                )
                future_to_step[future] = step_name
            
            # Wait for all steps to complete
            for future in as_completed(future_to_step):
                step_name = future_to_step[future]
                try:
                    success, failed, failed_chunks = future.result()
                    if failed > 0:
                        logger.warning(f"{step_name} had {failed} failed chunks")
                    else:
                        logger.info(f"{step_name} completed successfully")
                except Exception as e:
                    logger.error(f"Exception in {step_name}: {e}")
    
    logger.info("Preprocessing pipeline completed!")

if __name__ == "__main__":
    main() 