"""
Created 2024-02-08 by Seth Woodbury (woodbuse@uw.edu)
This script parses all the AF2 JSON files in an input directory and outputs af2_out_parsed.csv.
Efficient logic and handling of extra data are implemented.
"""

import os
import json
import pandas as pd
import glob
from multiprocessing import Pool, cpu_count
import argparse
import time

def safe_loads(json_str):
    """
    Attempt to parse a string as JSON. Handle cases where extra data follows a valid JSON object.
    """
    try:
        return json.loads(json_str), None
    except json.JSONDecodeError as e:
        if e.pos < len(json_str):
            # Try to parse up to the position where extra data begins.
            return json.loads(json_str[:e.pos]), json_str[e.pos:]
        raise

def parse_json(json_file):
    try:
        with open(json_file, 'r') as f:
            data, extra = safe_loads(f.read())
            if extra:
                print(f"Warning: Extra data ignored in {json_file}")

            pdb_path = json_file.replace('prediction_results.json', 'unrelaxed.pdb')

            # If 'mean_plddt' appears more than once and is a list, take the lower value.
            mean_plddt = data.get('mean_plddt', 'NaN')
            rmsd_to_input = data.get('rmsd_to_input', 'NaN')
            mean_pae_intra_chain = data.get('mean_pae_intra_chain', 'NaN')
            mean_pae = data.get('mean_pae', 'NaN')
            pTMscore = data.get('pTMscore', 'NaN')
            af2_convergence_tol = data.get('tol', 'NaN')
            af2_elapsed_folding_time = data.get('elapsed_time', 'NaN')
            
            if isinstance(mean_plddt, list):
                mean_plddt = max(mean_plddt)
                
            if isinstance(rmsd_to_input, list):
                rmsd_to_input = min(rmsd_to_input)
    
            if isinstance(mean_pae_intra_chain, list):
                mean_pae_intra_chain = min(mean_pae_intra_chain)

            if isinstance(mean_pae, list):
                mean_pae = min(mean_pae)
                
            if isinstance(pTMscore, list):
                pTMscore = max(pTMscore)
                
            if isinstance(af2_convergence_tol, list):
                af2_convergence_tol = min(af2_convergence_tol)

            if isinstance(af2_elapsed_folding_time, list):
                af2_elapsed_folding_time = min(af2_elapsed_folding_time)

            return {
                'af2_json_path': json_file,
                'pdb_path': pdb_path,
                'mean_plddt': mean_plddt,
                'rmsd_to_input': data.get('rmsd_to_input', 'NaN'),
                'mean_pae_intra_chain': data.get('mean_pae_intra_chain', 'NaN'),
                'mean_pae': data.get('mean_pae', 'NaN'),
                'pTMscore': data.get('pTMscore', 'NaN'),
                'af2_convergence_tol': data.get('tol', 'NaN'),
                'af2_elapsed_folding_time': data.get('elapsed_time', 'NaN')
            }
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return {}

def process_directory(directory, cpus=None, output_basename=None):
    start_time = time.time()
    json_files = glob.glob(os.path.join(directory, '*.json'))

    if not json_files:
        print("No JSON files found in the directory.")
        return

    cpus = max(1, cpu_count() - 5) if cpus is None else cpus
    total_files = len(json_files)
    print(f"Found {total_files} JSON files. Available CPUs: {cpu_count()}, Utilizing: {cpus} CPUs.")

    results = []
    with Pool(processes=cpus) as pool:
        for i, result in enumerate(pool.imap_unordered(parse_json, json_files), 1):
            results.append(result)
            if i in [100, 1000, 10000] or i % 10000 == 0:
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / i * total_files
                print(f"Processed {i} files. Elapsed Time: {elapsed_time:.2f} seconds. Estimated Time Remaining: {(estimated_total_time - elapsed_time):.2f} seconds.")

    results_filtered = [result for result in results if result]
    df = pd.DataFrame.from_records(results_filtered)

    if output_basename:
        if not any(output_basename.endswith(ext) for ext in ['.csv', '.sc', '.json']):
            output_basename += '.csv'  # Default to .csv if no valid extension is present
        output_csv = os.path.join(directory, output_basename)
    else:
        output_csv = os.path.join(directory, 'af2_out_parsed.csv')
    
    df.to_csv(output_csv, index=False)
    print(f"Total Time Taken: {time.time() - start_time:.2f} seconds. OUTPUT SAVED HERE: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process AF2 JSON files and generate a summary CSV.')
    parser.add_argument('directory', help='Directory containing AF2 JSON files.')
    parser.add_argument('--cpus', type=int, default=None, help='Specify the number of CPUs to use. Defaults to CPU count - 5.')
    parser.add_argument('--optional_basename_for_summary_stats', type=str, default=None, help='Optional base name for the output file with a valid file extension.')
    
    args = parser.parse_args()
    
    # Pass the optional base name for output file to the processing function
    process_directory(args.directory, cpus=args.cpus, output_basename=args.optional_basename_for_summary_stats)
