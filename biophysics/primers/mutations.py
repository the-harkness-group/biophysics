#!/usr/bin/env python3

from Bio.Data import CodonTable
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import CodonUsage as CU
from textwrap import wrap
import yaml
import sys

# Find the mutant codon to be used in the primer sequence
def get_mut_codon(wt_codon, mut_aa, to_prot):
    
    # Get possible codons out of the E. coli codon table
    possible_codons = []
    for k,v in to_prot.items():
        if v == mut_aa:
            possible_codons.append(k)
    
    # Take the mutant codon as the one with the highest usage weight from the
    # Sharpe Index
    possible_codon_usages = {}
    for codon in possible_codons:
        if codon in CU.SharpEcoliIndex:
            possible_codon_usages[codon] = CU.SharpEcoliIndex[codon]
    
    mut_codon = max(possible_codon_usages,key=possible_codon_usages.get)
    
    return mut_codon

# Calculate primer Tm according to the 2 + 4 rule
def calculate_Tm(sequence):
    
    Tm = sequence.count('A')*2 + sequence.count('T')*2 \
    + sequence.count('G')*4 + sequence.count('C')*4
    
    return Tm

# Get codon tables for translating nt sequences
bacterial_table = CodonTable.unambiguous_dna_by_name["Bacterial"]
to_prot = bacterial_table.forward_table

# Read file containing desired mutations to make   
mutation_parameters = yaml.safe_load(open(sys.argv[1],'r'))
seq_file = mutation_parameters['file_name']
mutations = mutation_parameters['mutations']

# Read nt sequence from input fasta file
nt_seq = SeqIO.read(seq_file,'fasta')

# Split sequence into codons and translate each of these
nt_seq_codons = wrap(str(nt_seq.seq),3)
aa_seq = nt_seq.translate()
aa_seq = str(aa_seq.seq)
seq_indices = [i+1 for i in range(len(aa_seq))]

# Check for His-SUMO purification tag, redefine sequence indices to be starting from 1 after the tag
if mutation_parameters['tag'] == 'His-SUMO':
    tag_marker = aa_seq.index('AHREQIGG') + 1
    tag_end = tag_marker + len('AHREQIGG') - 1
    seq_indices = [i-tag_end for i in seq_indices]

if mutation_parameters['tag'] == 'His-TEV':
    tag_marker = aa_seq.index('ENLYFQG') + 1
    tag_end = tag_marker + len('ENLYFQG') - 1
    seq_indices = [i-tag_end for i in seq_indices]

# Make dictionary where keys are the sequence indices and values are codons and aas
# Define the start position of the protein of interest to be 1
seq_dict = {}
for seq_index in seq_indices:
    seq_dict[int(seq_index)] = [nt_seq_codons[int(seq_index+tag_end)-1], aa_seq[int(seq_index+tag_end)-1]]

primer_dict = {}
with open(f"{mutation_parameters['csv_name']}",'w') as csv_file:
    
    csv_file.write('OligoName,Sequence\n')
    
    for mutation in mutations:
    # Go through list of mutations, get mutant codon and aa
    # Can use mutation index directly because the dictionary is numbered
    # according to the start of the protein of interest
        [wt_aa, mut_index, mut_aa] = mutation.split('_')
        wt_codon = seq_dict[int(mut_index)][0]
        if mut_aa == 'Stop': # For stop codon mutagenesis
            mut_codon = 'TGA'
        else: # For every other type of mutation
            mut_codon = get_mut_codon(wt_codon, mut_aa, to_prot)
    
    # Generate 5' and 3' primer sequences with specified Tm, then join these
    # with the mutant codon, and take the reverse complement to get the forward
    # and reverse mutagenesis primers
        fivep_seq = ''.join([seq_dict[int(mut_index)-i][0] for i in reversed(range(1,11))])
        fivep_Tm = calculate_Tm(fivep_seq)
        while fivep_Tm > 58:
            fivep_seq = fivep_seq[1:]
            fivep_Tm = calculate_Tm(fivep_seq)
    
        threep_seq = ''.join([seq_dict[int(mut_index)+i][0] for i in range(1,11)])
        threep_Tm = calculate_Tm(threep_seq)
        while threep_Tm > 58:
            threep_seq = threep_seq[:-1]
            threep_Tm = calculate_Tm(threep_seq)
    
        sense_primer = fivep_seq + mut_codon + threep_seq    
        anti_sense_primer = str(Seq(sense_primer).reverse_complement())
        
        print(f"#################################### MUTATION: {wt_aa} {mut_index} {mut_aa} ######################################")
        #print('Mutation: {} {} {}'.format(wt_aa,mut_index,mut_aa))
        print('Wild-type codon:',wt_codon)
        print('Mutant codon:',mut_codon)
        print('Sense primer:',fivep_seq + ' ' + mut_codon + ' ' + threep_seq)
        print('Anti-sense primer:',str(Seq(threep_seq).reverse_complement()) + ' ' \
          + str(Seq(mut_codon).reverse_complement()) + ' ' \
          + str(Seq(fivep_seq).reverse_complement()))
        print("5' Tm: {} degrees, 3' Tm: {} degrees".format(fivep_Tm,threep_Tm))
        print('Sense order: {}'.format(sense_primer))
        print('Anti-sense order: {} \n'.format(anti_sense_primer))
        print('DID YOU CHECK THAT THE TAG IS CLEAVABLE???\nULP1 CANNOT CLEAVE IF THE +1 AA IS P, V, L, I, D, E, K!!!\n')        

    # Get mutation without underscores for writing dictionary
        concise_mutation = ''.join(mutation.split('_'))
        primer_dict[concise_mutation] = {'wt_codon': wt_codon, 'mut_codon': mut_codon, 'primer_F': sense_primer,\
               'primer_R': anti_sense_primer, 'Five_prime_Tm': fivep_Tm, 'Three_prime_Tm': threep_Tm}

        csv_file.write(f"{seq_file.split('_')[0]}_{concise_mutation}_F,{sense_primer}\n")
        csv_file.write(f"{seq_file.split('_')[0]}_{concise_mutation}_R,{anti_sense_primer}\n")


out_name = mutation_parameters['out_name']    
with open(out_name,'w') as outfile:
    yaml.dump(primer_dict,outfile,default_flow_style=False)    
    
    
    
    
    
    
    




