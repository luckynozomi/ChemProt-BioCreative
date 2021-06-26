# Check tagged position of entities
# Check there aren't any overlapping between tagged entities
# Check all arg1 are chemicals, and all arg2 are genes
import itertools
from types import CoroutineType
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

    for text, entites_dict in zip([title, abstract], [entities_title_dict, entities_abstract_dict]):
        text_idx_has_entity = [False] * len(text)

        overlapping_entities = []
        for arg1, arg2 in itertools.combinations(entites_dict, r=2):
            entity1_type, start_pos1, end_pos1, name1 = entites_dict[arg1]
            entity2_type, start_pos2, end_pos2, name2 = entites_dict[arg2]

            if (arg1, arg2) in relations_dict or (arg2, arg1) in relations_dict:
                if entity1_type != "CHEMICAL":
                    print("PMID: {}, Arg1: {} is not a chemical.".format(pmid, arg1))
                if entity2_type not in ["GENE-Y", "GENE-N"]:
                    print("PMID: {}, Arg2: {} is not a gene.".format(pmid, arg2))

            if end_pos1 <= start_pos2:
                continue
            elif end_pos2 <= start_pos1:
                continue
            else:
                if (arg1, arg2) in relations_dict:
                    relation = relations_dict[(arg1, arg2)]
                    print("Overlapping entities: {} & {}. Relation: {} PMID: {}".format(name1, name2, relation, pmid))
                elif (arg2, arg1) in relations_dict:
                    relation = relations_dict[(arg2, arg1)]
                    print("Overlapping entities: {} & {}. Relation: {} PMID: {}".format(name1, name2, relation, pmid))
                overlapping_entities.append([arg1, arg2])

        for arg, (entity_type, start_pos, end_pos, name) in entites_dict.items():
            try:
                assert text[start_pos:end_pos] == name
            except:
                print("PMID: {}, Arg: {}, name: {} text doesn't match.".format(pmid, arg, name))
            for pos in range(start_pos, end_pos):
                text_idx_has_entity[pos] = True

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
