import pandas as pd
import glob
import os
import multiprocessing
import argparse
import time

def parse_metrics_from_pdb(file_path):
    """
    Parse metrics from the end of a pdb file, specified after the '#END_POSE_ENERGIES_TABLE' marker.
    """
    metrics = {
        'pdb_path': file_path,
        'secondary_structure': 'NaN',
        'secondary_structure_dssp_reduced_alphabet': 'NaN',
        'sap_score': 'NaN',
        'aliphatic_residues_in_design': 'NaN',
        'bad_torsion_preproline': 'NaN',
        'holes_in_design_lower_is_better': 'NaN',
        'hydrophobic_exposure_sasa_in_design': 'NaN',
        'hydrophobic_residues_in_design': 'NaN',
        'longest_cont_apolar_seg': 'NaN',
        'longest_cont_polar_seg': 'NaN',
        'net_charge_in_design_not_w_his': 'NaN',
        'number_dssp_helices_in_design': 'NaN',
        'number_dssp_loops_in_design': 'NaN',
        'number_dssp_sheets_in_design': 'NaN',
        'total_pose_sasa': 'NaN',
        'total_residues_in_design_plus_ligand': 'NaN',
        'total_rosetta_energy_metric': 'NaN'
    }

    with open(file_path, 'r') as file:
        content = file.read()

    start = content.find("#END_POSE_ENERGIES_TABLE")
    if start != -1:
        content = content[start:].split("\n")[1:]  # Skip the line with the marker
        for line in content:
            line = line.strip()
            if line:
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts[0].strip().replace(' ', '_').lower(), parts[1].strip()
                    if key in metrics:
                        metrics[key] = value

    return metrics

def parse_energy_tables(directory_path, n_processes):
    file_paths = glob.glob(os.path.join(directory_path, '*.pdb'))
    total_files = len(file_paths)
    print(f"Total files to process: {total_files}")

    start_time = time.time()
    results = []

    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.map(parse_metrics_from_pdb, file_paths)

    actual_time_taken = time.time() - start_time
    print(f"Actual time taken: {actual_time_taken:.2f} seconds.")

    # Create DataFrame from results and export
    df = pd.DataFrame(results)
    output_path = os.path.join(directory_path, 'parsed_energy_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Output CSV: {os.path.abspath(output_path)}")
    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse PDB files to extract energy tables.')
    parser.add_argument('directory', help='Directory containing PDB files')
    parser.add_argument('--nproc', type=int, default=multiprocessing.cpu_count() - 1, help='Number of processes to use')
    args = parser.parse_args()

    N_PROCESSES = int(os.getenv("OMP_NUM_THREADS", args.nproc))
    print(f"Using {N_PROCESSES} processes")

    parse_energy_tables(args.directory, N_PROCESSES)
