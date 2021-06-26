import enum
from nltk import sent_tokenize
from nltk.inference.discourse import spacer
from nltk.sem.logic import TruthValueType
import pandas as pd


def find_all_substring(og_text: str, own_entity: str):
    ret = []
    start_pos = og_text.find(own_entity)
    if start_pos == -1:
        return ret
    else:
        return ret + [[start_pos, start_pos+len(own_entity)]]

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

    import nltk

    extra_abbreviations = ['e.g', 'i.e', 'i.m', 'a.u', 'p.o', 'i.v', 'i.p', 'vivo', 'p.o', 'i.p', 'Vmax', 'i.c.v', ')(', 'E.C', 'sp']
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
    own_entity_list = [
        "no. 4, 1,3,5(10)-estratriene-17alpha-ethynyl-3", 
        "no. 6, 17alpha-ethynyl-androstene-diol", 
        "no. 8, 3beta, 17beta-dihydroxy-androst-5-ene-16-one",
        "no. 10, 3beta-methylcarbonate-androst-5-ene-7,17-dione",
        "4-DAMP. 1.5 +/- 0.4 nM",
        "(Mecp2(tm)(1)(.)(1)(Jae))",
        "(Bioorg. Med. Chem., 2010, 18, 1761-1772., J. Med. Chem., 2011, 54, 2823-2838.)"
    ]

    output = []
    for pmid, texts in pmid_to_abstract_dict.items():
        text_title, text_abstract = texts
        sentence_poses = sentence_tokenizer.span_tokenize(text_title)
        end_title_pos = len(text_title) - 1
        sentence_poses = list(sentence_poses)
        abstract_sentence_poses = sentence_tokenizer.span_tokenize(text_abstract)
        for start_pos, end_pos in abstract_sentence_poses:
            sentence_poses.append((start_pos+end_title_pos+2, end_pos+end_title_pos+2))

        og_text = '\t'.join([text_title, text_abstract])
        entity_poses = entities_pos_dict[pmid]
        for own_entity in own_entity_list:
            entity_poses += find_all_substring(og_text, own_entity)
        is_sentence_break = [True] * (len(sentence_poses) - 1)
        new_sentence_poses = []

        for entity_start, entity_end in entity_poses:
            entity_start = int(entity_start)
            entity_end = int(entity_end)
            for sentence_idx in range(len(sentence_poses)-1):
                sentence_end = sentence_poses[sentence_idx][1]
                if entity_start <= sentence_end and sentence_end < entity_end:
                    is_sentence_break[sentence_idx] = False
        
        sentence_startends = []
        for sent_idx, (sent_start, sent_end) in enumerate(sentence_poses):
            sentence_startends.append(sent_start)
            sentence_startends.append(sent_end)

        is_valid_startends = [True]
        for is_sent_break in is_sentence_break:
            is_valid_startends.append(is_sent_break)
            is_valid_startends.append(is_sent_break)
        is_valid_startends.append(True)
        sentence_startends = [sentence_startends[idx] for idx in range(len(sentence_startends)) if is_valid_startends[idx] == True]
        new_sentence_poses = []
        for idx in range(len(sentence_startends)//2):
            new_sentence_poses.append((sentence_startends[2*idx], sentence_startends[2*idx+1]))
        #--------------------------------------------
        #dealing with some bad separation from tokenizer
        #-------------------------------------------
        # sen_forpair = [tmp_sen_forpair[0]]
        # tmp_sen_forpair = tmp_sen_forpair[1:]
        sen_poses = new_sentence_poses
        sen_forpair = []
        for start, end in new_sentence_poses:
            sen_forpair.append(og_text[start:end])
        # while len(tmp_sen_forpair) > 0:
        #     cond1 = sen_forpair[-1][-4:] in ['e.g.','i.e.','i.m.','a.u.','p.o.','i.v.','i.p.']
        #     cond2 = sen_forpair[-1][-5:] in ['vivo.','p.o.)','i.p.)','Vmax.']
        #     cond3 = sen_forpair[-1][-6:] in ['i.c.v.']
        #     cond4 = sen_forpair[-1][-7:] in ['i.c.v.)']
        #     cond5 = sen_forpair[-1][-3:] in [')(.']
        #     cond6 = sen_forpair[-1][-4:] in ['E.C.']
        #     if not cond1 and not cond2 and not cond3 and not cond4 and not cond5 and not cond6:
        #         sen_forpair.append(tmp_sen_forpair[0])
        #     else:
        #         sen_forpair[-1] = sen_forpair[-1] + ' ' + tmp_sen_forpair[0]
        #     tmp_sen_forpair.remove(tmp_sen_forpair[0])
        
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

        for sent_id, all_sent_entities in sentence_id_to_entities_dict.items():
            all_chemicals_dupes = all_sent_entities.get('CHEMICAL', [])
            all_genes_dupes = all_sent_entities.get('GENE-Y', []) + all_sent_entities.get('GENE-N', [])

            all_chemicals = []
            for i in all_chemicals_dupes:
                if i not in all_chemicals:
                    all_chemicals.append(i)

            all_genes = []
            for i in all_genes_dupes:
                if i not in all_genes:
                    all_genes.append(i)

            import itertools
            for chemical, gene in itertools.product(all_chemicals, all_genes):

                chem_arg, chem_start, chem_end, chem_name = chemical
                gene_arg, gene_start, gene_end, gene_name = gene

                args_to_cpr_label_dict = {}
                all_relations = pmid_to_relations_dict.get(pmid, [])
                for relation_type, arg1, arg2 in all_relations:
                    arg1 = arg1.strip().split(':')[-1]
                    arg2 = arg2.strip().split(':')[-1]
                    if arg1 == chem_arg and arg2 == gene_arg:
                        args_to_cpr_label_dict[(arg1, arg2)] = relation_type
                        args_to_cpr_label_dict[(arg2, arg1)] = relation_type
                    elif arg1 == gene_arg and arg2 == chem_arg:
                        args_to_cpr_label_dict[(arg1, arg2)] = relation_type
                        args_to_cpr_label_dict[(arg2, arg1)] = relation_type

                if chem_start <= gene_start and gene_end <= chem_end:
                    assert (chem_arg, gene_arg) not in args_to_cpr_label_dict
                    assert (gene_arg, chem_arg) not in args_to_cpr_label_dict
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

                    assert (chem_arg, gene_arg) not in args_to_cpr_label_dict
                    assert (gene_arg, chem_arg) not in args_to_cpr_label_dict
                    continue

                relation_type = args_to_cpr_label_dict.get((chem_arg, gene_arg), "NOT")

                this_sentence = sen_forpair[sent_id]
                assert(this_sentence[chem_start:chem_end] == chem_name)
                assert(this_sentence[gene_start:gene_end] == gene_name)
                is_chem_first = (chem_start < gene_start)

                low_start = min([chem_start, gene_start])
                low_end = min([chem_end, gene_end])
                high_start = max([chem_start, gene_start])
                high_end = max([chem_end, gene_end])
                split_sent = [
                    this_sentence[0:low_start], this_sentence[low_start:low_end], this_sentence[low_end:high_start], this_sentence[high_start:high_end],this_sentence[high_end:]
                ]
                split_sent[1] = "chem_name" if is_chem_first else "gene_name"
                split_sent[3] = "gene_name" if is_chem_first else "chem_name"
                
                for token in ["chem_name", "gene_name"]:
                    assert token not in split_sent[0]
                    assert token not in split_sent[2]
                    assert token not in split_sent[4]

                sentence = "".join(split_sent)
                this_ret = [pmid, chem_arg, chem_name, gene_arg, gene_name, relation_type, sentence]
                output.append(this_ret)

    sv = pd.DataFrame(output, columns=["pmid", "chem_arg", "chem_name", "gene_arg", "gene_name", "relation type", "sentence"])
    return sv

dataset_training = gen_dataset(
    abstract_path = "train_data/drugprot_training_abstracs.tsv",
    entity_path = "train_data/drugprot_training_entities.tsv",
    relation_path = "train_data/drugprot_training_relations.tsv"
)
dataset_training.to_csv("train_dataset.csv")
