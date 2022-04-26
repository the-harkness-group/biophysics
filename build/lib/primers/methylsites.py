#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

"""
    Generates a translated protein sequence with corresponding biophysical parameters such as the molecular weight, extinction coefficient at 280 nm, and the isoelectric point. Accepts an input fasta file containing the nucleotide sequence which may or may not have purification tags. The sequence is translated from the first start codon and will be split at the specified tag to generate the mature protein sequence and its characteristic parameters.

Parameters
----------

arg1: script name
arg2: fasta file name for nucleotide sequence
arg3: purification tag (written for either His-TEV (HT), His-SUMO (HS), or no tag (none))

----------

Returns
----------

PROTEIN SEQUENCE TRANSLATED FROM FIRST START CODON (TAG SPECIFIED IN THESE BRACKETS)
PROTEIN SEQUENCE (NO TAG)
MOLECULAR WEIGHT (NO TAG) in Daltons
E280 (NO TAG) in M^-1 cm^-1, with values for reduced and oxidized Cys residues respectively
Isoelectric point (pI, no TAG)

----------
"""

import sys
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

if len(sys.argv) < 3:
    print('Need to specify a total of three arguments: (1) the script name, (2) the fasta file containing the nucleotide sequence to be translated, and (3) the purification tag (HS for His-SUMO, HT for His-TEV, or none for no tag')
    sys.exit()

seq_file = sys.argv[1]
tag = sys.argv[2]

nuc_seq = SeqIO.read(seq_file,'fasta')
nuc_seq = nuc_seq.upper()
nuc_trimmed = nuc_seq.seq[nuc_seq.seq.find('ATG'):]
prot_seq = nuc_trimmed.translate(to_stop=True)

if tag == 'HS':
    prot_noTag = prot_seq.split('QIGG')[1]
    prot_noTag_analysis = ProteinAnalysis(str(prot_noTag))
    print('\nTRANSLATED SEQUENCE FROM FIRST START CODON  (HIS-SUMO TAGGED):',prot_seq)
    print('\nTRANSLATED SEQUENCE (NO TAG):',prot_noTag)
    print('\nMOLECULAR WEIGHT (NO TAG):',prot_noTag_analysis.molecular_weight())
    print('\nE280 (NO TAG, REDUCED, OXIDIZED):',prot_noTag_analysis.molar_extinction_coefficient())
    print('\npI (NO TAG):',prot_noTag_analysis.isoelectric_point())
    print('\n################ AMINO ACID NUMBERING #################')
    Leu = 0
    Val = 0
    Ile = 0
    Met = 0
    for num,aa in enumerate(prot_noTag):
        print(aa,num+1)
        if aa == 'L':
            Leu += 1
        if aa == 'V':
            Val += 1
        if aa == 'I':
            Ile += 1
        if aa == 'M':
            Met += 1
    print(f"\nTotal leucine residues: {Leu}")
    print(f"Total valine residues: {Val}")
    print(f"Total isoleucine residues: {Ile}")
    print(f"Total methionine residues: {Met}")
        
elif tag == 'HT':
    prot_noTag = prot_seq.split('ENLYFQ')[1]
    print('\nTRANSLATED SEQUENCE (NO TAG):',prot_noTag)

else:
    print('\n',prot_seq)

