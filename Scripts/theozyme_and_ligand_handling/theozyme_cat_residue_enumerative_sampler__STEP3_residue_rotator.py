#!/usr/bin/env python3
"""
SCRIPT NAME
    theozyme_cat_residue_enumerative_sampler__STEP3_residue_rotator.py

PURPOSE
    Define and apply arbitrary residue‐level rotations around user‐specified axes.
    For each rotation spec you can rotate a single residue by a fixed angle
    around an axis defined by two atoms, repeated `periodicity` times, and
    dump each permuted PDB with a `_rotP_XX` suffix.

USAGE
    python theozyme_cat_residue_enumerative_sampler__STEP3_residue_rotator.py \
      --input_pdb PATH/to/input.pdb \
      --output_dir PATH/to/outdir \
      --rotation_config '[  
          {"residue":"C3","pivot":"NZ","axis":["Z9:C1","C3:NZ"],"degrees":180,"periodicity":2}
       ]'

ARGUMENTS
    --input_pdb PATH             Path to the PDB to rotate
    --output_dir PATH            Directory to write rotated PDBs (defaults to input folder)
    --rotation_config JSON|FILE  JSON list of rotation specs (or path to JSON file). Each spec:
        {
          "residue": "<chain><resnum>",
          "pivot":   "<atomName>",
          "axis":    ["<chain2>:<res2>:<atom2>", "<chain1>:<res1>:<atom1>"],
          "degrees": <float degrees per step>,
          "periodicity": <int number of steps>
        }

    * You can list multiple specs to enumerate all combinations.

EXAMPLE
    python Scripts/theozyme_and_ligand_handling/STEP3_residue_rotator.py \
      --input_pdb /path/to/project/.../test_E.pdb \
      --output_dir /path/to/project/.../rotations \
      --rotation_config '[{"residue":"C3","pivot":"NZ","axis":["Z9:C1","C3:NZ"],"degrees":180,"periodicity":2}]'
"""
import os, sys, argparse, json, copy, itertools
import numpy as np

##############################################
### PDB I/O ##################################
##############################################
class Atom:
    def __init__(self, line):
        self._orig = line.rstrip("\n")
        self.name = self._orig[12:16].strip()
        self.resname = self._orig[17:20].strip()
        self.chain = self._orig[21]
        self.resnum = int(self._orig[22:26])
        x = float(self._orig[30:38]); y = float(self._orig[38:46]); z = float(self._orig[46:54])
        self.coord = np.array([x,y,z],dtype=float)
    def format_line(self):
        line = self._orig
        name_f = f"{self.name:>4}"      # cols 13-16
        coords = f"{self.coord[0]:8.3f}{self.coord[1]:8.3f}{self.coord[2]:8.3f}"
        return line[:12] + name_f + line[16:30] + coords + line[54:] + "\n"

def parse_pdb(path):
    recs = []
    with open(path) as f:
        for L in f:
            if L.startswith(("ATOM  ","HETATM")):
                recs.append(Atom(L))
            else:
                recs.append(L)
    return recs

def write_pdb(recs, outp):
    with open(outp,'w') as w:
        for r in recs:
            if isinstance(r,Atom): w.write(r.format_line())
            else:                 w.write(r)

##############################################
### Geometry ###############################
##############################################
def rotate_around_axis(pt, p1, p2, angle_deg):
    # Rodrigues rotation of pt about line p1->p2 by angle_deg
    v = pt - p1
    axis = p2 - p1
    axis = axis / np.linalg.norm(axis)
    θ = np.radians(angle_deg)
    c, s = np.cos(θ), np.sin(θ)
    # Rodrigues
    v_rot = v*c + np.cross(axis,v)*s + axis*(np.dot(axis,v)*(1-c))
    return p1 + v_rot

##############################################
### Main ####################################
##############################################

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_pdb',      required=True)
    p.add_argument('--output_dir',     default=None)
    p.add_argument('--rotation_config',required=True,
                   help='JSON list or path of rotation specs')
    args = p.parse_args()

    # load rotation specs
    raw = args.rotation_config
    if os.path.isfile(raw):
        specs = json.load(open(raw))
    else:
        specs = json.loads(raw)
    if isinstance(specs,dict): specs=[specs]

    # parse each spec
    parsed = []
    for S in specs:
        # 1) residue & pivot
        ch,   res   = S['residue'][0], int(S['residue'][1:])
        pivot      = S['pivot']

        # 2) parse axis strings "Z9:C1"
        chain_res2, atom2 = S['axis'][0].split(':')
        chain2 = chain_res2[0]
        res2   = int(chain_res2[1:])
        chain_res1, atom1 = S['axis'][1].split(':')
        chain1 = chain_res1[0]
        res1   = int(chain_res1[1:])

        # 3) degrees & periodicity
        deg = float(S['degrees'])
        per = int(S['periodicity'])

        # 4) collect
        parsed.append({
            'chain':       ch,
            'resnum':      res,
            'pivot':       pivot,
            'axis0':       (chain2, res2, atom2),
            'axis1':       (chain1, res1, atom1),
            'degrees':     deg,
            'periodicity': per,
        })

    # read PDB
    records = parse_pdb(args.input_pdb)
    # collect mapping: (chain,resnum)-> list of record indices
    idx_map = {}
    for i,r in enumerate(records):
        if isinstance(r,Atom):
            key=(r.chain,r.resnum)
            idx_map.setdefault(key,[]).append(i)

    # build all index‐tuples
    all_idx = [range(s['periodicity']) for s in parsed]
    for combo in itertools.product(*all_idx):
        recs2 = copy.deepcopy(records)
        # apply each rotation
        for spec,i_step in zip(parsed,combo):
            angle = spec['degrees']*(i_step+1)
            chain,resnum = spec['chain'],spec['resnum']
            # find pivot coords
            pivot_atom = next(r for r in recs2 if isinstance(r,Atom)
                              and r.chain==chain and r.resnum==resnum
                              and r.name==spec['pivot'])
            p1 = pivot_atom.coord
            # anchor
            a0ch,a0rn,a0nm = spec['axis0']
            anchor = next(r for r in recs2 if isinstance(r,Atom)
                          and r.chain==a0ch and r.resnum==a0rn and r.name==a0nm)
            p2 = anchor.coord
            # rotate all atoms in that residue
            for idx in idx_map[(chain,resnum)]:
                atom = recs2[idx]
                atom.coord = rotate_around_axis(atom.coord,p1,p2,angle)
            # re‐fix pivot & anchor exactly
            pivot_atom.coord = p1
            anchor.coord    = p2

        # write out
        base = os.path.splitext(os.path.basename(args.input_pdb))[0]
        # build suffix parts
        parts=[]
        for spec,i_step in zip(parsed,combo):
            w = len(str(spec['periodicity']))
            parts.append(f"rotP_{i_step:0{w}d}")
        suffix = '_'+'_'.join(parts)
        outd = args.output_dir or os.path.dirname(args.input_pdb)
        os.makedirs(outd,exist_ok=True)
        outp = os.path.join(outd, f"{base}{suffix}.pdb")
        write_pdb(recs2,outp)
        print(f"Wrote {outp}")

if __name__=='__main__':
    main()
