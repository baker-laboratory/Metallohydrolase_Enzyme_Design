"""
Created 2024-02-07 by Seth Woodbury (woodbuse@uw.edu)
This script should do a pyfastrelax & output a new pdb with 
hydrogen bond information about the specified atoms, including # of H-bonds and the residue doing them
"""
"""
Created 2024-02-07 by Seth Woodbury (woodbuse@uw.edu)
This script should do a pyfastrelax & output a new pdb with 
hydrogen bond information about the specified atoms, including # of H-bonds and the residue doing them
"""
from optparse import OptionParser
import os

import pyrosetta
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts

parser = OptionParser(usage="usage: %prog [options] FILE", version="0.1")
parser.add_option("--pdb", type="string", dest="pdb", help="Path to pdb you want to filter")
parser.add_option("--out_dir", type="string", dest="out_dir", help="where to dump the relaxed pdbs")

(opts, args) = parser.parse_args()
parser.set_defaults()
print("Using the following arguments:")
print(opts)

script_dir = os.path.dirname(os.path.abspath(__file__))
dalphaball_path = os.path.join(script_dir, "enzyme_design", "DAlphaBall.gcc")
pyrosetta.init(f"-mute all -beta -dalphaball {dalphaball_path} -mute all")

bn = os.path.basename(opts.pdb)
pose = pyrosetta.pose_from_file(opts.pdb)


def hbond_filter(bn,pose,out_dir,lig_res_num=''):
    '''
    Counts hydrogen bond between a design and its ligand, then does constrained fast relax
    where the key catalytic residues are constrained...hard-coded for glycosidases
    '''   
    xml_script = f"""
    <ROSETTASCRIPTS>

      <SCOREFXNS>
          
          <ScoreFunction name="sfxn_design" weights="beta">
              <Reweight scoretype="arg_cation_pi" weight="3"/>
              <Reweight scoretype="angle_constraint" weight="1.0"/>
              <Reweight scoretype="angle_constraint" weight="1.0"/>
              <Reweight scoretype="coordinate_constraint" weight="1.0"/>
              <Reweight scoretype="dihedral_constraint" weight="1.0"/>
          </ScoreFunction>
          
          <ScoreFunction name="fa_csts" weights="beta">
              <Reweight scoretype="arg_cation_pi" weight="3"/>
              <Reweight scoretype="angle_constraint" weight="1.0"/>
              <Reweight scoretype="coordinate_constraint" weight="1.0"/>
              <Reweight scoretype="angle_constraint" weight="1.0"/>
              <Reweight scoretype="dihedral_constraint" weight="1.0"/>
          </ScoreFunction>
          
          <ScoreFunction name="sfxn" weights="beta" />
      </SCOREFXNS>
      
      <RESIDUE_SELECTORS>
          <Chain name="chainA" chains="A"/>
          <Chain name="chainB" chains="B"/>
      </RESIDUE_SELECTORS>
      
      <SIMPLE_METRICS>
          <TotalEnergyMetric name="total_energy" scorefxn="sfxn_design" />
          <SecondaryStructureMetric name="secondary_structure" dssp_reduced="false"/>
          <SecondaryStructureMetric name="secondary_structure_reduced" dssp_reduced="true"/>
          <SapScoreMetric name="spatial_aggregation_propensity_score"/>
      </SIMPLE_METRICS>
    
      <MOVERS>
          <FastRelax name="FastRelax" scorefxn="sfxn_design" disable_design="1" repeats="1" batch="false" ramp_down_constraints="false" cartesian="False" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" relaxscript="MonomerDesign2019">
              <MoveMap name="MM" bb="1" chi="1" jump="1"/>
          </FastRelax>
          
      </MOVERS>
    
      <FILTERS>
          <ResidueCount name="total_residues_in_design_plus_ligand" max_residue_count="99999" min_residue_count="0" count_as_percentage="0" confidence="0"/>
          <ResidueCount name="hydrophobic_residues_in_design" include_property="HYDROPHOBIC" max_residue_count="99999" min_residue_count="0" count_as_percentage="0" confidence="0"/>
          <ResidueCount name="aliphatic_residues_in_design" include_property="ALIPHATIC" max_residue_count="99999" min_residue_count="0" count_as_percentage="0" confidence="0"/>
          <NetCharge name="net_charge_in_design_NOT_w_HIS" chain="1" confidence="0"/>
          <SecondaryStructureCount name="number_DSSP_helices_in_design" num_helix_sheet="0" num_helix="1" num_sheet="0" num_loop="0" filter_helix_sheet="0" filter_helix="1" filter_sheet="0" filter_loop="0" min_helix_length="3" max_helix_length="9999" min_sheet_length="3" max_sheet_length="9999" min_loop_length="1" max_loop_length="9999" return_total="true" confidence="0"/>
          <SecondaryStructureCount name="number_DSSP_sheets_in_design" num_helix_sheet="0" num_helix="0" num_sheet="1" num_loop="0" filter_helix_sheet="0" filter_helix="0" filter_sheet="1" filter_loop="0" min_helix_length="3" max_helix_length="9999" min_sheet_length="3" max_sheet_length="9999" min_loop_length="1" max_loop_length="9999" return_total="true" confidence="0"/>
          <SecondaryStructureCount name="number_DSSP_loops_in_design" num_helix_sheet="0" num_helix="0" num_sheet="0" num_loop="1" filter_helix_sheet="0" filter_helix="0" filter_sheet="0" filter_loop="1" min_helix_length="3" max_helix_length="9999" min_sheet_length="3" max_sheet_length="9999" min_loop_length="1" max_loop_length="9999" return_total="true" confidence="0"/>
          <Holes name="holes_in_design_lower_is_better" threshold="2" normalize_per_residue="false" exclude_bb_atoms="false" confidence="0"/>
          <ExposedHydrophobics name="hydrophobic_exposure_sasa_in_design" sasa_cutoff="20" threshold="-1" confidence="0"/>
          <TotalSasa name="total_pose_sasa" threshold="800" upper_threshold="1000000000000000" hydrophobic="0" polar="0" confidence="0"/>
          <PreProline name="bad_torsion_preproline" use_statistical_potential="0" confidence="0"/>
          <LongestContinuousPolarSegment name="longest_cont_polar_seg" exclude_chain_termini="false" count_gly_as_polar="false" filter_out_high="false" cutoff="5" confidence="0"/>
          <LongestContinuousApolarSegment name="longest_cont_apolar_seg" exclude_chain_termini="false" filter_out_high="false" cutoff="5" confidence="0"/>

      </FILTERS>

      <PROTOCOLS>
          <Add mover="FastRelax" />
         <Add filter="total_residues_in_design_plus_ligand"/>
         <Add filter="hydrophobic_residues_in_design"/>
         <Add filter="aliphatic_residues_in_design"/>
         <Add filter="net_charge_in_design_NOT_w_HIS"/>
         <Add filter="number_DSSP_helices_in_design"/>
         <Add filter="number_DSSP_sheets_in_design"/>
         <Add filter="number_DSSP_loops_in_design"/>
         <Add filter="holes_in_design_lower_is_better"/>
         <Add filter="hydrophobic_exposure_sasa_in_design"/>
         <Add filter="total_pose_sasa"/>
         <Add filter="bad_torsion_preproline"/>
         <Add filter="longest_cont_polar_seg"/>
         <Add filter="longest_cont_apolar_seg"/>
        

         <Add metrics="total_energy,secondary_structure,secondary_structure_reduced,spatial_aggregation_propensity_score" labels="total_rosetta_energy_metric,secondary_structure,secondary_structure_DSSP_reduced_alphabet,SAP_score"/>

      </PROTOCOLS>
    
    </ROSETTASCRIPTS>
    """
    #         <InterfaceScoreCalculator name="interface_scores" chains="A,B" scorefxn="sfxn_design"/>
    #         <Add filter="interface_scores"/>
    #          <SpecificResiduesNearInterface name="residues_at_interface" task_operation="(&string)" confidence="0"/>
    #         <Add filter="residues_at_interface"/>

    task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml_script)
    task_relax.setup() # syntax check
    packed_pose = task_relax(pose)
    bn = bn.replace(".pdb", "")
    pose.dump_pdb(os.path.join(out_dir, f"{bn}_relaxedAPO.pdb"))

# RUN THE ABOVE FUNCTIONS
hbond_filter(bn, pose, opts.out_dir)