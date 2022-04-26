#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Apr 30 21:35:46 2019

@author: robertharkness
"""

from Bio import SeqIO
from Bio.Seq import Seq
import yaml
import sys

# Calculate primer Tm according to the 2 + 4 rule
def calculate_Tm(sequence):
    
    Tm = sequence.count('A')*2 + sequence.count('T')*2 \
    + sequence.count('G')*4 + sequence.count('C')*4
    
    return Tm

### Set up backbones for making Gibson Assembly primers
# pET-His-SUMO backbone overlap sequences
pHS_overlap_F = 'gctcacagagaacagattggtggt' # AHREQIGG at beginning of pHS vector
pHS_overlap_R = 'ccgaataaatacctaagcttgtct' # reverse complement to RQA*VFIR at end of pHS vector

# pET-His-TEV backbone overlap sequences taken from RH2 hTRF1 in pET29
pHTEV_overlap_F = 'gaccgaaaatctgtacttccagggc' # TENLFYQG at beginning of pHTEV vector
pHTEV_overlap_R = 'ggtgctcgagtgcggccgcaagctt' # reverse complement to KLAAALEH at end of pHTEV vector

# Read file containing desired inserts to Gibson Assemble
GA_parameters = yaml.safe_load(open(sys.argv[1],'r'))

# Generate Gibson Assembly primers based on insert sequence files
primer_dict = {}
with open(GA_parameters['csvname'],"w") as out:
    out.write("OligoName,Sequence\n")
    
    for GA in GA_parameters['Assemblies']:
        
        # Make Gibson Assembly primers for His-SUMO backbone
        if GA_parameters['Assemblies'][GA]['Backbone'] == 'pET-His-SUMO':

            # Read nt sequence from input fasta file
            seq_file = GA_parameters['Assemblies'][GA]['File']
            nt_seq = SeqIO.read(seq_file,'fasta')
            nt_seq = str(nt_seq.seq)
        
            # Five prime sequence is just the first bit of the sense strand
            # Updated code to allow for sequence-specific inserts, ie does not
            # have to start at the very beginning and end of the sequence, can now
            # generate primers for an insert anywhere in a sequence
            fivep_seq = str(nt_seq[(GA_parameters['Assemblies'][GA]['Start']*3 - 3):((GA_parameters['Assemblies'][GA]['Start'] + 1)*3 - 3 + 30)])
            # Start is given in the amino acid number, so multiply by 3 to get the number of the
            # final base in the corresponding codon in the nucleotide sequence, 
            # and subtract 3 to get the number of the first base in the codon 
            # because Python is 0-indexed (must subtract 3 instead of 2)
            # As well, go to :start + 1 because Python list indexing, then subtract
            # 3 for codon start and add 30 to extend out for getting a primer estimate
            fivep_Tm = calculate_Tm(fivep_seq)
            while fivep_Tm > 58:
                fivep_seq = fivep_seq[:-1]
                fivep_Tm = calculate_Tm(fivep_seq)
        
            # Three prime primer sequence is the reverse complement of the last bit of
            # the insert sequence
            threep_seq = str(Seq(nt_seq[(GA_parameters['Assemblies'][GA]['End']*3 - 3 - 30):((GA_parameters['Assemblies'][GA]['End'] + 1)*3 - 3)]).reverse_complement())
            # This does a similar thing to above for the sense primer, dropping back into
            # the 5' part of the fasta sequence from the desired last amino acid/codon, then taking
            # the nucleotide sequence from there to the desired end to get the primer sequence
            # Take the reverse complement of this sequence to get the actual primer
            stop_codon = str(Seq(GA_parameters['Assemblies'][GA]['Stop codon']).reverse_complement()) # Ensure stop codon is there
            # Must supply the stop codon in the parameter file, even if it is already in the fasta file, as a forced check
            threep_Tm = calculate_Tm(threep_seq)
            while threep_Tm > 58:
                threep_seq = threep_seq[:-1]
                threep_Tm = calculate_Tm(threep_seq)
    
            sense_primer = pHS_overlap_F + fivep_seq 
            anti_sense_primer = pHS_overlap_R + stop_codon + threep_seq # Force inclusion of stop codon
        
            print(f"##################### {GA} #####################")
            print(f"Forward backbone overlap: {pHS_overlap_F}")
            print(f"Forward insert sequence: {fivep_seq}")
            print(f"GA Forward Initial Tm: {fivep_Tm}")
            print(f"GA Forward Primer: {sense_primer}\n")
            print(f"Reverse backbone overlap: {pHS_overlap_R}")
            print(f"Reverse insert sequence: {threep_seq}")
            print(f"GA Reverse Initial Tm: {threep_Tm}")
            print(f"GA Reverse Primer: {anti_sense_primer}\n")
            print('Sense order: {}'.format(sense_primer))
            print('Anti-sense order: {} \n'.format(anti_sense_primer))
            print('DID YOU CHECK THAT THE TAG IS CLEAVABLE???\nULP1 CANNOT CLEAVE IF THE +1 AA IS P, V, L, I, D, E, K!!!\n')
    
            # Put GA primers into primer dictionary
            primer_dict[GA] = {'primer_F': sense_primer,\
               'primer_R': anti_sense_primer, 'F_Tm': fivep_Tm, 'R_Tm': threep_Tm}
            
            primer_F_name = GA + '_F'
            primer_R_name = GA + '_R'

            out.write(f"{primer_F_name},{sense_primer}\n")
            out.write(f"{primer_R_name},{anti_sense_primer}\n")
        
        # Make Gibson Assembly primers for His-TEV backbone
        if GA_parameters['Assemblies'][GA]['Backbone'] == 'pET-His-TEV':

            # Read nt sequence from input fasta file
            seq_file = GA_parameters['Assemblies'][GA]['File']
            nt_seq = SeqIO.read(seq_file,'fasta')
            nt_seq = str(nt_seq.seq)
        
            # Five prime sequence is just the first bit of the sense strand
            fivep_seq = str(nt_seq[0:30])
            fivep_Tm = calculate_Tm(fivep_seq)
            while fivep_Tm > 58:
                fivep_seq = fivep_seq[:-1]
                fivep_Tm = calculate_Tm(fivep_seq)
        
            # Three prime sequence is the reverse complement of the last bit of
            # the insert sequence
            threep_seq = str(Seq(nt_seq).reverse_complement())[0:30]
            threep_Tm = calculate_Tm(threep_seq)
            while threep_Tm > 58:
                threep_seq = threep_seq[:-1]
                threep_Tm = calculate_Tm(threep_seq)
    
            sense_primer = pHTEV_overlap_F + fivep_seq 
            anti_sense_primer = pHTEV_overlap_R + threep_seq
        
            print(f"##################### {GA} #####################")
            print(f"Forward backbone overlap: {pHTEV_overlap_F}")
            print(f"Forward insert sequence: {fivep_seq}")
            print(f"GA Forward Initial Tm: {fivep_Tm}")
            print(f"GA Forward Primer: {sense_primer}\n")
            print(f"Reverse backbone overlap: {pHTEV_overlap_R}")
            print(f"Reverse insert sequence: {threep_seq}")
            print(f"GA Reverse Initial Tm: {threep_Tm}")
            print(f"GA Reverse Primer: {anti_sense_primer}\n")
            print('Sense order: {}'.format(sense_primer))
            print('Anti-sense order: {} \n'.format(anti_sense_primer))
            print('DID YOU CHECK THAT THE TAG IS CLEAVABLE???\nCHECK TEV CLEAVAGE SITE!!!\n')
    
            # Put GA primers into primer dictionary
            primer_dict[GA] = {'primer_F': sense_primer,\
               'primer_R': anti_sense_primer, 'F_Tm': fivep_Tm, 'R_Tm': threep_Tm}
            
            primer_F_name = GA + '_F'
            primer_R_name = GA + '_R'

            out.write(f"{primer_F_name},{sense_primer}\n")
            out.write(f"{primer_R_name},{anti_sense_primer}\n")

# Write Gibson Assembly primers to file
out_name = GA_parameters['Outname']    
with open(out_name,'w') as outfile:
    yaml.dump(primer_dict,outfile,default_flow_style=False)    
    
    