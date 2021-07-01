import enum
from typing import DefaultDict
from nltk.tokenize import word_tokenize
from numpy import split
import pandas as pd

train_df = pd.read_json("split_dataframe/train_dataset.json", orient="table")
dev_df = pd.read_json("split_dataframe/val_dataset.json", orient="table")
test_df = pd.read_json("split_dataframe/test_dataset.json", orient="table")

train_df = pd.concat([train_df, dev_df], ignore_index=True)

# _ProteinNames.txt: protein_name+\t+type
# protein_name need to be unique
train_name_to_type = {}
test_name_to_type = {}

# _pairs_label.txt 
# {pmid}.pair{pair_idx}+\t+relation_type+Y/N+NA


#_triplets.txt
#OUT.write('--'.join(arr[1:4])+'--'+ENT_dict[arr[0]][e2][0]+'\t'+absid+'|'+pair_id+'\tPROT1_'+ENT_dict[absid][e1][-1]+'\tPROT2_'+ENT_dict[absid][e2][-1]+'\t'+tmp.lower()+'\t'+str(p1[0]+1)+'\t'+str(p2[0]+1)+'\t'+str(piw+1)+' | '+sen_towrite+'\n')
#CPR:10--N--NA--GENE-Y	10224140|10224140.pair12	PROT1_digoxin	PROT2_OAT3	inhibited	33	24	23 | Organic anions such as LUNGCPT02 , LUNGCPT03 , LUNGCPT04 , LUNGCPT05 , LUNGCPT06 , LUNGCPT07 , LUNGCPT08 , LUNGCPT09 , and LUNGCPT10 inhibited PROT2 - mediated LUNGCPT31 uptake , while LUNGCPT12 and PROT1 did not .
#CPR:10--N--NA--GENE-Y	10224140|10224140.pair12	PROT1_digoxin	PROT2_OAT3	mediated	33	24	26 | Organic anions such as LUNGCPT02 , LUNGCPT03 , LUNGCPT04 , LUNGCPT05 , LUNGCPT06 , LUNGCPT07 , LUNGCPT08 , LUNGCPT09 , and LUNGCPT10 inhibited PROT2 - mediated LUNGCPT31 uptake , while LUNGCPT12 and PROT1 did not .
#pair_id = {pmid}.pair{pair_idx}
#{relation_type}--{Y/N}--NA--{GENE-Y/N}+\t+{pmid}+'|'+{pair_id}+'\t'+'PROT1_'+{chemical_name}+'\t'+'PROT2_'+{gene_name}+'\t'+interaction_word.lower()+'\t'+{pos_chemial(start from 1, word_position)}+'\t'+{pos_gene}+'\t'+{pos_interaction_word}+' | '+{sentence}+'\n' 


def remove_subset_entities(entity_list):
    # 2nd filter: remove any entities that is a subset of any entity in entity_list
    ret_entity_list = []
    any_entity_removed = False
    for this_entity in entity_list:
        this_arg, this_start, this_end, this_name = this_entity
        flagged_for_removal = False
        for that_entity in entity_list:
            that_arg, that_start, that_end, that_name = that_entity
            if this_arg == that_arg:
                continue
            if that_start <= this_start and this_end <= that_end:
                flagged_for_removal = True
                any_entity_removed = True
        if not flagged_for_removal:
            ret_entity_list.append(this_entity)
    return any_entity_removed, ret_entity_list

import itertools
import copy
def modify_partial_overlap_entities(sentence, entity_list):
    any_entity_modified = False
    ret_list = copy.deepcopy(entity_list)
    entity_count = len(entity_list)
    for this_idx, that_idx in itertools.combinations(range(entity_count), r=2):
        this_arg, this_start, this_end, this_name = ret_list[this_idx]
        that_arg, that_start, that_end, that_name = ret_list[that_idx]
        if this_start < that_start and that_start < this_end and this_end < that_end:  # THIS is a LHS partial overlap with THAT
            # For convenience, let's just modify THIS
            this_name = sentence[this_start:that_start].strip()
            this_start, this_end = this_start, this_start + len(this_name)
            any_entity_modified = True
        if that_start < this_start and this_start < that_end and that_end < this_end:  # THIS is a RHS partial overlap with THAT
            this_name = sentence[that_end:this_end].strip()
            this_start, this_end = this_end - len(this_name), this_end
            any_entity_modified = True
        ret_list[this_idx] = [this_arg, this_start, this_end, this_name]
    return any_entity_modified, ret_list

import re
def transform_sentence(sentence, in_chemical, in_gene, all_chemicals, all_genes):
    chem_arg, chem_type, chem_start, chem_end, chem_name = in_chemical
    gene_arg, gene_type, gene_start, gene_end, gene_name = in_gene

    chemical = [chem_arg, chem_start, chem_end, chem_name]
    gene = [gene_arg, gene_start, gene_end, gene_name]

    # first filter: deal with entities that (partially) overlaps with THE interacting chemical/gene
    filtered_aux_entities = []  # Auxilary entities means these that are not THE interacting chemical/gene
    for this_entity in all_chemicals + all_genes:
        this_arg, this_start, this_end, this_name = this_entity
        if this_arg in [chem_arg, gene_arg]:
            continue
        # if it's a superset (this_start<=entity_start<entity_end<=this_end) or a subset (entity_start<=this_start<this_end<=this_end) of THE interacting chemical/gene, remove it from the list
        if this_start <= chem_start and chem_end <= this_end:  # superset of chemical
            continue
        elif this_start <= gene_start and gene_end <= this_end:  # superset of gene
            continue
        elif chem_start <= this_start and this_end <= chem_end:  # subset of chemical
            continue
        elif gene_start <= this_start and this_end <= gene_end:  # subset of gene
            continue
        # if it's a LHS partial overlap (this_start<entity_start<this_end<entity_end) or a RHS partial overlap (entity_start<this_start<entity_end<this_end) of THE interacting chemical/gene, change it to the only partial part
        # start another "if" to deal with situations like, gene precessing chemical for example, gene_start<this_start<gene_end<chem_start<this_end<chem_end
        if this_start < chem_start and chem_start < this_end and this_end < chem_end:  # LHS partial overlap with chemical
            this_name = sentence[this_start:chem_start].strip()
            this_start, this_end = this_start, this_start + len(this_name)
        if this_start < gene_start and gene_start < this_end and this_end < gene_end:  # LHS partial overlap with gene
            this_name = sentence[this_start:gene_start].strip()
            this_start, this_end = this_start, this_start + len(this_name)
        if chem_start < this_start and this_start < chem_end and chem_end < this_end:  # RHS partial overlap with chemical
            this_name = sentence[chem_end:this_end].strip()
            this_start, this_end = this_end - len(this_name), this_end
        if gene_start < this_start and this_start < gene_end and gene_end < this_end:  # RHS partial overlap with gene
            this_name = sentence[gene_end:this_end].strip()
            this_start, this_end = this_end - len(this_name), this_end
        if len(this_name) > 0:
            filtered_aux_entities.append([this_arg, this_start, this_end, this_name])

    any_entity_deleted, any_entity_modified = True, True
    while(any_entity_deleted or any_entity_modified):
        any_entity_deleted, filtered_aux_entities = remove_subset_entities(filtered_aux_entities)
        any_entity_modified, filtered_aux_entities = modify_partial_overlap_entities(sentence, filtered_aux_entities)

    split_setence = list(sentence)
    filtered_aux_entities.extend([chemical, gene])
    for entity in filtered_aux_entities:
        entity_arg, entity_start, entity_end, entity_name = entity
        entity_arg_number = int(entity_arg[1:])
        
        masked_entity_name = ""
        if entity_arg == chem_arg:
            masked_entity_name = "PROT1"
        elif entity_arg == gene_arg:
            masked_entity_name = "PROT2"
        else:
            masked_entity_name = "LUNGCPT{:02}".format(entity_arg_number)
        split_setence[entity_start] = " {} ".format(masked_entity_name)
        for pos in range(entity_start+1, entity_end):
            split_setence[pos] = ""
    new_sentence = "".join(split_setence)
    new_sentence = " ".join(new_sentence.split())  # Remove additional white space characters
    return new_sentence


IW = open('./src/IW_CPR_09_2017.txt','r')
IW_list = []
for line in IW:
	if line.startswith('#'):
		continue
	elif line == '\n':
		continue
	else:
		IW_list.append(line.strip().split('\t')[0])


for dataset_type, df in zip(["Train", "Test"], [train_df, test_df]):
    prefix = "DrugProt{}".format(dataset_type)
    OUT = open(prefix+"_triplets.txt", 'w')
    OUT_label = open(prefix+"_pairs_label.txt", 'w')

    pmids = list(set(df["pmid"]))
    for pmid in pmids:
        sub_df = df.loc[df["pmid"] == pmid]
        for sent_idx, (_, entry) in enumerate(sub_df.iterrows()):
            chem_arg, chem_type, chem_start, chem_end, chem_name = entry["chemical"]
            gene_arg, gene_type, gene_start, gene_end, gene_name = entry["gene"]

            sentence = entry["sentence"]
            if chem_start < gene_start and gene_start < chem_end and chem_end < gene_end:
                chem_name = sentence[chem_start:gene_start].strip()
                chem_start, chem_end = chem_start, chem_start + len(chem_name)
            if gene_start < chem_start and chem_start < gene_end and gene_end < chem_end:
                gene_name = sentence[gene_start:chem_start].strip()
                gene_start, gene_end = gene_start, gene_start + len(gene_name)
            entry["chemical"] = [chem_arg, chem_type, chem_start, chem_end, chem_name]
            entry["gene"] = [gene_arg, gene_type, gene_start, gene_end, gene_name]

            transformed_sentence = transform_sentence(entry["sentence"], entry["chemical"], entry["gene"], entry["all_chemicals"], entry["all_geneYs"]+entry["all_geneNs"])
            transformed_sentence = transformed_sentence.replace('down-regulat','downregulat').replace('Down-regulat','Downregulat').replace('up-regulat','upregulat').replace('Up-regulat','Upregulat')
            transformed_sentence = transformed_sentence.replace('/',' / ').replace('(',' ( ').replace(')',' ) ').replace(',',' ,').replace(':',' : ').replace('-',' - ').replace('+',' + ')
            transformed_sentence = " ".join(transformed_sentence.split())  # Remove additional white space characters
            tokens = word_tokenize(transformed_sentence)
            transformed_sentence = " ".join(tokens)

            interaction_word_to_pos = DefaultDict(list)
            for pos, token in enumerate(tokens):
                word = token.lower()
                if word in IW_list:
                    interaction_word_to_pos[word].append(pos + 1)  # Starts from 1

            if len(interaction_word_to_pos) == 0:
                interaction_word_to_pos = {'iw_na': [-1]}

            pos_chem = tokens.index("PROT1") + 1  # Starts from 1
            pos_gene = tokens.index("PROT2") + 1  # Starts from 1

            pair_id = "{}.pair{}".format(pmid, sent_idx)
            relation_type = entry["relation type"]
            Y_N = "N" if relation_type == "NOT" else "Y"
            triplets = ["--".join([relation_type, Y_N, "NA", gene_type])]
            triplets.append("{}|{}".format(pmid, pair_id))
            triplets.append("PROT1_{}".format(chem_name))
            triplets.append("PROT2_{}".format(gene_name))

            # _pairs_label.txt 
            # {pmid}.pair{pair_idx}+\t+relation_type+Y/N+NA
            OUT_label.write(pair_id+'\t'+relation_type+'\t'+Y_N+'\t'+"NA"+'\n')
            for interaction_word, poses_list in interaction_word_to_pos.items():
                for pos_iw in poses_list:
                    this_triplets = copy.deepcopy(triplets)
                    this_triplets.append(interaction_word)
                    this_triplets.extend(map(str,[pos_chem, pos_gene]))
                    this_triplets.append("{} | {}".format(pos_iw, transformed_sentence))
                    OUT.write('\t'.join(this_triplets)+'\n')

            # _ProteinNames.txt: protein_name+\t+type
            target_name_dict = train_name_to_type if dataset_type == "Train" else test_name_to_type
            for entity in entry["all_chemicals"]:
                entity_arg, entity_start, entity_end, entity_name = entity
                target_name_dict[entity_name] = "CHEMICAL"
            for entity in entry["all_geneYs"]:
                entity_arg, entity_start, entity_end, entity_name = entity
                target_name_dict[entity_name] = "GENE-Y"
            for entity in entry["all_geneNs"]:
                entity_arg, entity_start, entity_end, entity_name = entity
                target_name_dict[entity_name] = "GENE-N"

    # _ProteinNames.txt: protein_name+\t+type
    # protein_name need to be unique
    OUT_PN = open(prefix+'_ProteinNames.txt','w')
    for entity_name, entity_type in target_name_dict.items():
        OUT_PN.write(entity_name+'\t'+entity_type+'\n')
    OUT.close()
    OUT_label.close()
    OUT_PN.close()