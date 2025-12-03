import argparse
import os

def read_pdb_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

def update_atom_nums(pdb_lines):
    num_update_dic, iter_num = {}, 1
    for line in pdb_lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom_number = line[6:11].strip()
            num_update_dic[atom_number] = iter_num
            iter_num += 1
    
    updated_pdb_lines = []
    for line in pdb_lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom_number = line[6:11].strip()
            updated_pdb_line = line[:6]+str(num_update_dic[atom_number]).rjust(5)+line[11:]
            updated_pdb_lines.append(updated_pdb_line)
        elif line.startswith("CONECT"):
            line_numbers = line.split()
            atom_number = line_numbers[0]
            connected_numbers = [str(num_update_dic[num]) for num in line_numbers[1:]]
            updated_pdb_lines.append(atom_number + ' ' + ' '.join(connected_numbers) + '\n')
        else:
            updated_pdb_lines.append(line)
    return updated_pdb_lines
            
def write_pdb_file(file_path, pdb_lines):
    updated_pdb_lines = update_atom_nums(pdb_lines)
    with open(file_path, 'w') as f:     
        f.writelines(updated_pdb_lines)

def filter_atoms(input_pdb, output_pdb, ori_lig_name, new_lig_name, atoms_to_remove=None, atoms_to_keep=None):
    pdb_name = os.path.splitext(os.path.basename(input_pdb))[0]

    # Read input PDB file
    pdb_lines = read_pdb_file(input_pdb)

    # Filter atoms
    filtered_lines = []
    deleted_atom_numbers = set()
    
    for line in pdb_lines:
        if line.startswith('HETATM'):
            atom_number = line[6:11].strip()
            atom_name = line[12:16].strip()
            if (atoms_to_remove and atom_name not in atoms_to_remove) or (atoms_to_keep and atom_name in atoms_to_keep):
                if (ori_lig_name and new_lig_name):
                    line = line.replace(ori_lig_name, new_lig_name)
                filtered_lines.append(line)
            else: deleted_atom_numbers.add(atom_number)
        else:
            filtered_lines.append(line)

    # Update CONECT lines removing the atom numbers corresponding to the deleted HETATMs
    updated_filtered_lines = []
    for line in filtered_lines:
        if line.startswith('CONECT'):
            line_numbers = line.split()[1:]
            atom_number = line_numbers[0]
            connected_numbers = [num for num in line_numbers[1:] if num not in deleted_atom_numbers]
            if connected_numbers:  # only keep CONECT lines that still have at least one atom connection
                updated_filtered_lines.append(atom_number + ' ' + ' '.join(connected_numbers) + '\n')
        else:
            updated_filtered_lines.append(line)
                
                
    # Write filtered lines to output PDB file
    write_pdb_file(output_pdb, updated_filtered_lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter atoms from a PDB file and save the result.')
    parser.add_argument('--input', type=str, help='Path to the input PDB file.')
    parser.add_argument('--output', type=str, help='Path to the output PDB file.')
    parser.add_argument('--ori_lig_name', type=str, help='Original ligand name.')
    parser.add_argument('--new_lig_name', type=str, help='New ligand name.')
    parser.add_argument('--atoms_to_remove', nargs='+', help='List of atom names to remove.')
    parser.add_argument('--atoms_to_keep', nargs='+', help='List of atom names to keep.')
    args = parser.parse_args()
    
    filter_atoms(args.input, args.output, args.ori_lig_name, args.new_lig_name, args.atoms_to_remove, args.atoms_to_keep)
