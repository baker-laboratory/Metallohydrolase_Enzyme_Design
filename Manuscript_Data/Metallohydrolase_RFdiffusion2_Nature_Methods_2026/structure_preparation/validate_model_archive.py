"""Validate the packaged 96 ordered-sequence transition-state models.

Uses only the Python standard library. Sequence, complete ligand atom inventory,
zinc and the three catalytic histidines are checked against the supplied tables
and parameters. The manifest's geometry diagnostics are retained separately;
most designs were not experimental hits, and relaxation is not evidence of
catalytic activity.
"""

import csv
import hashlib
from pathlib import Path
import zipfile


def main():
    base = Path(__file__).resolve().parent
    supplemental = base.parent / "supplemental_data"
    residues = dict(zip(
        "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
        "ACDEFGHIKLMNPQRSTVWY",
    ))
    with (supplemental / "ordered_sequences.csv").open() as handle:
        sequences = list(csv.DictReader(handle))
    with (supplemental / "model_manifest.csv").open() as handle:
        manifest = {row["well"]: row for row in csv.DictReader(handle)}
    archive = supplemental / "Data_S2__4MU_B_Ordered_Models_with_Transition_States.zip"
    with zipfile.ZipFile(archive) as models:
        members = [name for name in models.namelist() if name.endswith(".pdb")]
        assert len(members) == len(sequences) == len(manifest) == 96
        for row in sequences:
            data = models.read("models/" + row["model_file"])
            assert hashlib.sha256(data).hexdigest() == manifest[row["well"]]["sha256"]
            lines = data.decode().splitlines()
            sequence = "".join(residues[line[17:20]] for line in lines
                               if line.startswith("ATOM  ") and line[12:16].strip() == "CA")
            assert sequence == row["ordered_protein_sequence"], row["well"]
            ligand = [line for line in lines if line.startswith("HETATM")]
            assert {line[17:20] for line in ligand} == {row["ligand"]}
            ligand_names = {line[12:16].strip() for line in ligand}
            assert len(ligand) == len(ligand_names), row["well"]
            assert len({line[21:27] for line in ligand}) == 1, row["well"]
            template = (base / "params" / (row["ligand"] + ".pdb")).read_text().splitlines()
            expected = {line[12:16].strip() for line in template
                        if line.startswith(("ATOM  ", "HETATM"))}
            assert ligand_names == expected and "ZN1" in ligand_names, row["well"]
            catalytic = [int(line.split()[11]) for line in lines if line.startswith("REMARK 666")]
            assert len(catalytic) == 3
            for number in catalytic:
                assert any(line.startswith("ATOM  ") and line[17:20] == "HIS"
                           and int(line[22:26]) == number for line in lines), row["well"]
    print("Validated 96/96 sequences, ligand atom sets, zinc, catalytic histidines and SHA-256 checksums.")
    print("Three original publication hits; 93 supplemental reconstructions.")


if __name__ == "__main__":
    main()
