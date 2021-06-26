import itertools
from types import CoroutineType

class style:
   RED = '\033[91m'
   END = '\033[0m'

def dataset_viewer(pmid, abstracts, entities, relations):

    pmid = int(pmid)
    title, abstract = abstracts
    title_length = len(title)
    entities_title_dict = {}
    entities_abstract_dict = {}
    for arg, (entity_type, start_pos, end_pos, name) in entities.items():
        start_pos, end_pos = int(start_pos), int(end_pos)
        if end_pos <= title_length:
            entities_title_dict[arg] = (entity_type, start_pos, end_pos, name)
        else:
            entities_abstract_dict[arg] = (entity_type, start_pos - title_length - 1, end_pos - title_length - 1, name)
    relations_dict = {}
    for relation_type, arg1, arg2 in relations:
        relations_dict[(arg1, arg2)] = relation_type

    out_title, out_abstract = [], []
    for text, entites_dict, out_text in zip([title, abstract], [entities_title_dict, entities_abstract_dict], [out_title, out_abstract]):

        overlapping_entities = []
        for arg1, arg2 in itertools.combinations(entites_dict, r=2):
            entity1_type, start_pos1, end_pos1, name1 = entites_dict[arg1]
            entity2_type, start_pos2, end_pos2, name2 = entites_dict[arg2]
            if end_pos1 <= start_pos2:
                continue
            elif end_pos2 <= start_pos1:
                continue
            else:
                overlapping_entities.append([arg1, arg2])

        text_idx_has_entity = [False] * len(text)
        for arg, (entity_type, start_pos, end_pos, name) in entites_dict.items():
            for pos in range(start_pos, end_pos):
                text_idx_has_entity[pos] = True
        
        for pos in range(len(text)):
            if pos == 0:
                if text_idx_has_entity[0] == True:
                    out_text.append(style.RED)
                out_text.append(text[0])
            else:
                if text_idx_has_entity[pos-1] == False and text_idx_has_entity[pos] == True:
                    out_text.append(style.RED)
                    out_text.append(text[pos])
                elif text_idx_has_entity[pos-1] == True and text_idx_has_entity[pos] == False:
                    out_text.append(style.END)
                    out_text.append(text[pos])
                else:
                    out_text.append(text[pos])
                    if pos == len(text) - 1 and text_idx_has_entity[pos] == True:
                        out_text.append(style.END)
    out_title = "".join(out_title)
    out_abstract = "".join(out_abstract)

    print("Title\n{}".format(out_title))
    print("Abstract\n{}".format(out_abstract))
    
    print("Entities in Title")
    print("Arg\tType\tSentence")
    for arg, (entity_type, start_pos, end_pos, name) in entities_title_dict.items():
        extended_chars = 10
        prev_chars = "..."+title[start_pos-extended_chars:start_pos] if start_pos >= extended_chars else title[0:start_pos]
        next_chars = title[end_pos:end_pos+extended_chars]+"..." if end_pos + extended_chars < len(title) else title[end_pos:]
        print("{}\t{}\t{}".format(arg, entity_type, prev_chars+style.RED+name+style.END+next_chars))

    print("Entities in Abstract")
    print("Arg\tType\tSentence")
    for arg, (entity_type, start_pos, end_pos, name) in entities_abstract_dict.items():
        extended_chars = 10
        prev_chars = "..."+abstract[start_pos-extended_chars:start_pos] if start_pos >= extended_chars else abstract[0:start_pos]
        next_chars = abstract[end_pos:end_pos+extended_chars]+"..." if end_pos + extended_chars < len(abstract) else abstract[end_pos:]
        print("{}\t{}\t{}".format(arg, entity_type, prev_chars+style.RED+name+style.END+next_chars))

    print("1")

import pandas as pd
from tqdm import tqdm
abstract_path = "train_data/drugprot_training_abstracs.tsv"
entity_path = "train_data/drugprot_training_entities.tsv"
relation_path = "train_data/drugprot_training_relations.tsv"

pmid_to_abstract_dict = {}
with open(abstract_path, "r") as abstract_file:
    for line in abstract_file:
        pmid, text_title, text_abstract = line.strip().split('\t')
        pmid_to_abstract_dict[pmid] = (text_title, text_abstract)

entities_dict = {}
with open(entity_path, "r") as entity_file:
    for line in entity_file:
        pmid, cpr, entity_type, start_pos, end_pos, name = line.strip().split('\t')
        if pmid not in entities_dict:
            entities_dict[pmid] = {cpr: (entity_type, start_pos, end_pos, name)}
        else:
            entities_dict[pmid][cpr] = (entity_type, start_pos, end_pos, name)

pmid_to_relations_dict = {}
with open(relation_path, "r") as relation_file:
    for line in relation_file:
        pmid, relation_type, arg1, arg2 = line.strip().split('\t')
        assert len(arg1.split(":")) == 2
        assert len(arg2.split(":")) == 2
        arg1 = arg1.split(":")[-1]
        arg2 = arg2.split(":")[-1]
        if pmid in pmid_to_relations_dict:
            pmid_to_relations_dict[pmid].append((relation_type, arg1, arg2))
        else:
            pmid_to_relations_dict[pmid] = [(relation_type, arg1, arg2)]

df = pd.read_csv("train_dataset.csv")
pmids = list(set(df["pmid"]))
for pmid in pmids:
    pmid = str(pmid)
    abstracts = pmid_to_abstract_dict.get(pmid, [])
    entities = entities_dict.get(pmid, {})
    relations = pmid_to_relations_dict.get(pmid, {})
    dataset_viewer(pmid, abstracts, entities, relations)
