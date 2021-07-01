import enum
from re import split
from nltk import sent_tokenize
from nltk.inference.discourse import spacer
from nltk.sem.logic import TruthValueType
import pandas as pd

from my_sentence_tokenizer import my_sentence_tokenizer


def gen_dataset(abstract_path, entity_path, relation_path):
    pmid_to_abstract_dict = {}
    with open(abstract_path, "r") as abstract_file:
        for line in abstract_file:
            pmid, text_title, text_abstract = line.strip().split('\t')
            pmid_to_abstract_dict[pmid] = (text_title, text_abstract)

    entities_dict = {}
    entities_pos_dict = {}
    with open(entity_path, "r") as entity_file:
        for line in entity_file:
            pmid, label, entity_type, start_pos, end_pos, name = line.strip().split('\t')
            if pmid not in entities_dict:
                entities_dict[pmid] = {label: (entity_type, start_pos, end_pos, name)}
            else:
                entities_dict[pmid][label] = (entity_type, start_pos, end_pos, name)
            if pmid in entities_pos_dict:
                entities_pos_dict[pmid].append([start_pos, end_pos])
            else:
                entities_pos_dict[pmid] = [[start_pos, end_pos]]

    pmid_to_relations_dict = {}
    with open(relation_path, "r") as relation_file:
        for line in relation_file:
            pmid, relation_type, arg1, arg2 = line.strip().split('\t')
            if pmid in pmid_to_relations_dict:
                pmid_to_relations_dict[pmid].append((relation_type, arg1, arg2))
            else:
                pmid_to_relations_dict[pmid] = [(relation_type, arg1, arg2)]


    output = []
    for pmid, texts in pmid_to_abstract_dict.items():
        text_title, text_abstract = texts
        entity_poses = entities_pos_dict[pmid]
        og_text = text_title + '\t' + text_abstract
        sen_poses = my_sentence_tokenizer(og_text, entity_poses)
        sen_forpair = []
        for start, end in sen_poses:
            sen_forpair.append(og_text[start:end])

        
        if pmid not in entities_dict:
            continue

        sentence_id_to_entities_dict = {}
        entities = entities_dict[pmid]
        for entity_arg, (entity_type, entity_start_pos, entity_end_pos, entity_name) in entities.items():
            entity_arg = entity_arg.strip().split(":")[-1]
            sentence_id = -1
            entity_start_pos, entity_end_pos = int(entity_start_pos), int(entity_end_pos)
            for sent_idx, (sent_start, sent_end) in enumerate(sen_poses):
                if sent_start <= entity_start_pos and entity_end_pos <= sent_end:
                    entity_start_pos -= sent_start
                    entity_end_pos -= sent_start
                    if sent_idx not in sentence_id_to_entities_dict:
                        sentence_id_to_entities_dict[sent_idx] = {"CHEMICAL": [], "GENE-Y": [], "GENE-N": []}
                    sentence_id_to_entities_dict[sent_idx][entity_type].append([entity_arg, entity_start_pos, entity_end_pos, entity_name])

        args_to_relation_dict = {}
        all_relations = pmid_to_relations_dict.get(pmid, [])
        for relation_type, arg1, arg2 in all_relations:
            arg1 = arg1.strip().split(':')[-1]
            arg2 = arg2.strip().split(':')[-1]
            args_to_relation_dict[(arg1, arg2)] = relation_type

        for sent_id, all_sent_entities in sentence_id_to_entities_dict.items():
            all_chemicals = all_sent_entities.get('CHEMICAL', [])
            all_geneYs = all_sent_entities.get('GENE-Y', [])
            all_geneNs = all_sent_entities.get('GENE-N', [])
            all_genes = all_geneYs + all_geneNs

            import itertools
            for chemical, gene in itertools.product(all_chemicals, all_genes):

                chem_arg, chem_start, chem_end, chem_name = chemical
                gene_arg, gene_start, gene_end, gene_name = gene

                if chem_start <= gene_start and gene_end <= chem_end:
                    continue
                if gene_start <= chem_start and chem_end <= gene_end:
                    continue

                if chem_start <= gene_start and gene_end <= chem_end:
                    assert (chem_arg, gene_arg) not in args_to_relation_dict
                    assert (gene_arg, chem_arg) not in args_to_relation_dict
                    continue
                if gene_start <= chem_start and chem_end <= gene_end:
                    if pmid=="23579178" and chem_arg=="T8" and gene_arg=="T20":  # Special case
                        continue
                    if pmid=="23219161" and chem_arg=="T1" and gene_arg=="T9":
                        continue
                    if pmid=="23219161" and chem_arg=="T2" and gene_arg=="T10":
                        continue
                    if pmid=="23548896" and chem_arg=="T6" and gene_arg=="T27":
                        continue
                    if pmid=="23548896" and chem_arg=="T3" and gene_arg=="T11":
                        continue

                relation_type = args_to_relation_dict.get((chem_arg, gene_arg), "NOT")

                this_sentence = sen_forpair[sent_id]
                assert(this_sentence[chem_start:chem_end] == chem_name)
                assert(this_sentence[gene_start:gene_end] == gene_name)

                out_chemical = [chem_arg, "CHEMICAL", chem_start, chem_end, chem_name]
                gene_type = 'GENE-Y' if gene in all_geneYs else 'GENE-N'
                out_gene = [gene_arg, gene_type, gene_start, gene_end, gene_name]
                this_ret = [pmid, out_chemical, out_gene, all_chemicals, all_geneYs, all_geneNs, relation_type, this_sentence]
                output.append(this_ret)

    sv = pd.DataFrame(output, columns=["pmid", "chemical", "gene", "all_chemicals", "all_geneYs", "all_geneNs", "relation type", "sentence"])
    return sv

dataset_training = gen_dataset(
    abstract_path = "train_data/drugprot_training_abstracs.tsv",
    entity_path = "train_data/drugprot_training_entities.tsv",
    relation_path = "train_data/drugprot_training_relations.tsv"
)
dataset_training.to_json("train_dataset.json", orient="table", indent=4)
