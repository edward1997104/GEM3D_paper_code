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

def run_preprocessing_step(step_name, script_path, chunk_id, args_dict):
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
        
        logger.debug(f"Running {step_name} chunk {chunk_id}: {' '.join(cmd_args)}")
        
        # Run the command
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per chunk
        )
        
        if result.returncode != 0:
            error_msg = f"{step_name} chunk {chunk_id} failed with return code {result.returncode}"
            logger.error(error_msg)
            logger.error(f"STDERR: {result.stderr}")
            return False, chunk_id, error_msg
        
        logger.debug(f"{step_name} chunk {chunk_id} completed successfully")
        return True, chunk_id, None
        
    except subprocess.TimeoutExpired:
        error_msg = f"{step_name} chunk {chunk_id} timed out"
        logger.error(error_msg)
        return False, chunk_id, error_msg
    except Exception as e:
        error_msg = f"{step_name} chunk {chunk_id} failed with exception: {e}"
        logger.error(error_msg)
        return False, chunk_id, error_msg

def run_step_parallel(step_name, script_name, num_chunks, max_workers, base_args):
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
            executor.submit(run_preprocessing_step, step_name, str(script_path), chunk_id, base_args): chunk_id
            for chunk_id in range(num_chunks)
        }
        
        # Process results with progress bar
        with tqdm(total=num_chunks, desc=f"{step_name} chunks") as pbar:
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    success, returned_chunk_id, error = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        failed_chunks.append(returned_chunk_id)
                        logger.warning(f"{step_name} chunk {returned_chunk_id} failed: {error}")
                except Exception as e:
                    failed += 1
                    failed_chunks.append(chunk_id)
                    logger.error(f"Exception in {step_name} chunk {chunk_id}: {e}")
                
                pbar.update(1)
                pbar.set_postfix(success=successful, failed=failed)
    
    if failed > 0:
        logger.warning(f"{step_name} completed with {failed} failed chunks: {failed_chunks}")
    else:
        logger.info(f"{step_name} completed successfully (all {successful} chunks)")
    
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
    
    logger.info(f"Processing pipeline with {args.max_workers} workers")
    
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
            simplify_args
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
            skeleton_args
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
                    step_args
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