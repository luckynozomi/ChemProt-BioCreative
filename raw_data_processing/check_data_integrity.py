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

            if (start_pos1 <= start_pos2 and end_pos2 <= end_pos1) or (start_pos2 <= start_pos1 and end_pos1 <= end_pos2):
                relation = relations_dict.get((arg1, arg2), "NOT")
                if relation == "NOT":
                    relation = relations_dict.get((arg2, arg1), "NOT")
                print("Overlapping entities: {} & {}. Relation: {} PMID: {}".format(name1, name2, relation, pmid))

            if end_pos1 <= start_pos2:
                continue
            elif end_pos2 <= start_pos1:
                continue
            else:
                if (arg1, arg2) in relations_dict:
                    relation = relations_dict[(arg1, arg2)]
                    # print("Overlapping entities: {} & {}. Relation: {} PMID: {}".format(name1, name2, relation, pmid))
                elif (arg2, arg1) in relations_dict:
                    relation = relations_dict[(arg2, arg1)]
                    # print("Overlapping entities: {} & {}. Relation: {} PMID: {}".format(name1, name2, relation, pmid))
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

def check_duplicates():
    all_abstracts = pd.read_csv(abstract_path, sep='\t', header=None, names=["pmid", "title", "abstract"], dtype={"pmid": str})
    all_entities = pd.read_csv(entity_path, sep='\t', header=None, names=["pmid", "arg", "type", "start", "end", "name"], dtype={"pmid": str})
    all_relations = pd.read_csv(relation_path, sep='\t', header=None, names=["pmid", "type", "arg1", "arg2"], dtype={"pmid": str})

    duplicated_abstracts = all_abstracts.duplicated(subset=["pmid"], keep=False)
    duplicated_entities = all_entities.duplicated(subset=["pmid", "arg"], keep=False)
    duplicated_relations = all_relations.duplicated(subset=["pmid", "arg1", "arg2"], keep=False)
    pd.set_option("max_rows", None)     

    if any(duplicated_abstracts):
        all_duplicated_abstracts = all_abstracts.loc[duplicated_abstracts, :]
        print("Duplicated Abstracts:")
        print(all_duplicated_abstracts)
        
    if any(duplicated_entities):
        all_duplicated_entities = all_entities.loc[duplicated_entities, :]
        print("Duplicated Entities:")
        print(all_duplicated_entities)
        
    if any(duplicated_relations):
        all_duplicated_relations = all_relations.loc[duplicated_relations, :]
        from collections import defaultdict
        dupe_dict = defaultdict(list)
        for _, relation_dupe in all_duplicated_relations.iterrows():
            this_pmid, this_type, this_arg1, this_arg2 = relation_dupe["pmid"], relation_dupe["type"], relation_dupe["arg1"], relation_dupe["arg2"]
            dupe_dict[(this_pmid, this_arg1, this_arg2)].append(this_type)
        dupe_count_dict = defaultdict(int)
        for this_key, this_relations in dupe_dict.items():
            from itertools import combinations
            for relation1, relation2 in combinations(this_relations, r=2):
                if relation1 > relation2:
                    relation1, relation2 = relation2, relation1
                dupe_count_dict[(relation1, relation2)] += 1
        for (relation1, relation2), dupe_count in dupe_count_dict.items():
            print("{} - {}: {}".format(relation1, relation2, dupe_count))
        print("Duplicated Relations: {} of {}".format(len(all_duplicated_relations), len(all_relations)))
        print(all_duplicated_relations)
    
    totally_duplicated_relations = all_relations.duplicated(keep=False)
    if any(totally_duplicated_relations):
        all_totally_duplicated_relations = all_relations.loc[totally_duplicated_relations, :]
        print("Totally Duplicated Relations: {} of {}".format(len(all_totally_duplicated_relations), len(all_relations)))
        print(all_totally_duplicated_relations)


check_duplicates()
"""
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

df = pd.read_json("train_dataset.json", orient="table")
pmids = list(set(df["pmid"]))
for pmid in pmids:
    pmid = str(pmid)
    abstracts = pmid_to_abstract_dict.get(pmid, [])
    entities = entities_dict.get(pmid, {})
    relations = pmid_to_relations_dict.get(pmid, {})
    dataset_viewer(pmid, abstracts, entities, relations)
"""