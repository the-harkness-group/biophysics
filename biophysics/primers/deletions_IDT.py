#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Apr 30 21:35:46 2019

@author: robertharkness
"""

from Bio.Data import CodonTable
from Bio import SeqIO
from Bio.Seq import Seq
from textwrap import wrap
import yaml
import sys

# Calculate primer Tm according to the 2 + 4 rule
def calculate_Tm(sequence):
    
    Tm = sequence.count('A')*2 + sequence.count('T')*2 \
    + sequence.count('G')*4 + sequence.count('C')*4
    
    return Tm

# Get codon tables for translating nt sequences
bacterial_table = CodonTable.unambiguous_dna_by_name["Bacterial"]
to_prot = bacterial_table.forward_table

# Read file containing desired deletions to make
deletion_parameters = yaml.safe_load(open(sys.argv[1],'r'))
deletions = deletion_parameters['deletions']
primer_dict = {}

with open(f"{deletion_parameters['csv_name']}",'w') as csv_file:
    
    csv_file.write('Name,Sequence,Scale,Purification\n')
    
    for deletion in deletion_parameters['deletions']:

        deletion_parameters['deletions'][deletion]
        seq_file = deletion_parameters['deletions'][deletion]['file_name']
        # Read nt sequence from input fasta file
        nt_seq = SeqIO.read(seq_file,'fasta')
        nt_seq = nt_seq.upper() # Force lowercase nucleotide sequences to be uppercase

        # Split sequence into codons and translate each of these
        nt_seq_codons = wrap(str(nt_seq.seq),3)
        aa_seq = nt_seq.translate()
        aa_seq = str(aa_seq.seq)
        seq_indices = [i+1 for i in range(len(aa_seq))]

        # Check for His-SUMO purification tag, redefine sequence indices to be starting from 1 after the tag
        if deletion_parameters['deletions'][deletion]['tag'] == 'His-SUMO':
            tag_marker = aa_seq.index('AHREQIGG') + 1
            tag_end = tag_marker + len('AHREQIGG') - 1
            seq_indices = [i-tag_end for i in seq_indices]
        
        if deletion_parameters['deletions'][deletion]['tag'] == 'His-TEV':
            tag_marker = aa_seq.index('ENLYFQG') + 1
            tag_end = tag_marker + len('ENLYFQG') - 1
            seq_indices = [i-tag_end for i in seq_indices]

        # Make dictionary where keys are the sequence indices and values are codons and aas
        # Define the start position of the protein of interest to be 1
        seq_dict = {}
        for seq_index in seq_indices:
            seq_dict[int(seq_index)] = [nt_seq_codons[int(seq_index+tag_end)-1], aa_seq[int(seq_index+tag_end)-1]]

        # Go through list of mutations, get mutant codon and aa
        # Can use mutation index directly because the dictionary is numbered
        # according to the start of the protein of interest
        start_del = deletions[deletion]['start']
        end_del = deletions[deletion]['end']
    
        # Generate 5' and 3' primer sequences with specified Tm, then join these
        # and take the reverse complement to get the forward and reverse deletion mutagenesis primers
        try:
            fivep_seq = ''.join([seq_dict[int(start_del)-i][0] for i in reversed(range(1,11))])
        except:
            print('\nYOUR 5 PRIME SEQUENCE IS PROBABLY TOO SHORT, CHECK IT!\n')
            fivep_seq = ''.join([seq_dict[int(start_del)-i][0] for i in reversed(range(1,8))])
            
        fivep_Tm = calculate_Tm(fivep_seq)
        while fivep_Tm > 58:
            fivep_seq = fivep_seq[1:]
            fivep_Tm = calculate_Tm(fivep_seq)
            
        try:
            threep_seq = ''.join([seq_dict[int(end_del)+i][0] for i in range(1,11)])
        except:
            print('\nYOUR 3 PRIME SEQUENCE IS PROBABLY TOO SHORT, CHECK IT!\n')
            threep_seq = ''.join([seq_dict[int(end_del)+i][0] for i in range(1,8)])
            
        threep_Tm = calculate_Tm(threep_seq)
        while threep_Tm > 58:
            threep_seq = threep_seq[:-1]
            threep_Tm = calculate_Tm(threep_seq)
    
        sense_primer = fivep_seq + threep_seq
        anti_sense_primer = str(Seq(sense_primer).reverse_complement())
    
        print(f"#################################### {deletion} ######################################")
        template_plasmid = deletion_parameters['deletions'][deletion]['file_name'].split('.')[0]  
        print(f"Template plasmid: {template_plasmid}")
        print('Deletion: {} {} {}'.format(start_del,'-',end_del))
        print('Sense primer:',fivep_seq + ' ' + threep_seq)
        print('Anti-sense primer:',str(Seq(threep_seq).reverse_complement()) + ' ' \
          +  str(Seq(fivep_seq).reverse_complement()))
        print("5' Tm: {} degrees, 3' Tm: {} degrees".format(fivep_Tm,threep_Tm))
        print('Sense order: {}'.format(sense_primer))
        print('Anti-sense order: {} \n'.format(anti_sense_primer))
        print('DID YOU CHECK THAT THE TAG IS CLEAVABLE???\nULP1 CANNOT CLEAVE IF THE +1 AA IS P, V, L, I, D, E, K!!!\n')
    
        # Get mutation without underscores for writing dictionary
        primer_dict[deletion] = {'primer_F': sense_primer,\
               'primer_R': anti_sense_primer, 'Five_prime_Tm': fivep_Tm, 'Three_prime_Tm': threep_Tm}
        
        csv_file.write(f"{deletion}_F,{sense_primer},25nm,STD\n")
        csv_file.write(f"{deletion}_R,{anti_sense_primer},25nm,STD\n")

out_name = deletion_parameters['yml_name']    
with open(out_name,'w') as outfile:
    yaml.dump(primer_dict,outfile,default_flow_style=False)    
    
    