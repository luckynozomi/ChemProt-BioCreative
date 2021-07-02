# 1. Abstracts:
### Task 1a. No duplicated PMIDs --> Pass

# 2. Entity Mentions
### Task 2a. No duplicated (pmid, arg) -> Pass
### Task 2b. All entity types are either "CHEMICAL", "GENE-Y" or "GENE-X" -> Pass
### Task 2c. `Title_and_Abstract[start_pos:end_pos] == name` -> **One exception, fixed?**
### Task 2d. The names are not part of bigger word, i.e., `name == "CHEMICA"` when it's actually `"CHEMICAL"` in the original text.

# 3. Relation
### Task 3a. No duplicated (pmid, arg1, arg2) -> **FAILED**
### Task 3b. All arg1 are "CHEMICAL", and all arg2 are either "GENE-Y" or "GENE-N" -> Pass
### Task 3c. All relation types are in the list of the ones we need to predict -> Pass
### Task 3d. All the chemicals and genes are in the same sentence -> We don't have their sentence splitter so we can't check. We use this criteria to validate our own sentence splitter. After its built, we confirm that all the pairs are in the same sentence.
import pandas as pd
import os
import csv

ALL_RELATIONS = [
    "INDIRECT-DOWNREGULATOR", "INDIRECT-UPREGULATOR", "DIRECT-REGULATOR", "ACTIVATOR", "INHIBITOR",
    "AGONIST", "ANTAGONIST", "AGONIST-ACTIVATOR", "AGONIST-INHIBITOR", "PRODUCT-OF",
    "SUBSTRATE", "SUBSTRATE_PRODUCT-OF", "PART-OF"
]

PROJECT_DIR = "/home/manbish/projects/ChemProt-BioCreative/raw_data_processing"

dataset = "training"
abstract_path = "drugprot-gs-training-development/{dataset}/drugprot_{dataset}_abstracs.tsv".format(dataset=dataset)
entity_path = "drugprot-gs-training-development/{dataset}/drugprot_{dataset}_entities.tsv".format(dataset=dataset)
relation_path = "drugprot-gs-training-development/{dataset}/drugprot_{dataset}_relations.tsv".format(dataset=dataset)

abstract_path = os.path.join(PROJECT_DIR, abstract_path)
entity_path = os.path.join(PROJECT_DIR, entity_path)
relation_path = os.path.join(PROJECT_DIR, relation_path)

all_abstracts = pd.read_csv(abstract_path, sep='\t', header=None, names=["pmid", "title", "abstract"], dtype={"pmid": str}, na_filter=False, quoting=csv.QUOTE_NONE)
all_entities = pd.read_csv(entity_path, sep='\t', header=None, names=["pmid", "arg", "type", "start", "end", "name"], dtype={"pmid": str}, na_filter=False, quoting=csv.QUOTE_NONE)
all_relations = pd.read_csv(relation_path, sep='\t', header=None, names=["pmid", "type", "arg1", "arg2"], dtype={"pmid": str}, na_filter=False, quoting=csv.QUOTE_NONE)

# Task 1a, 2a, 3a
def check_duplicates():
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


def check_2b():
    entity_types = all_entities["type"]
    assert all(entity_types.isin(["CHEMICAL", "GENE-Y", "GENE-N"]))

import copy
def check_2c_2d():
    print("Checking 2c and 2d...")
    for _, entity in all_entities.iterrows():
        pmid = entity["pmid"]
        abstracts = all_abstracts[all_abstracts["pmid"]==pmid]
        title, abstract = list(abstracts["title"])[0], list(abstracts["abstract"])[0]
        og_text = title + '\t' + abstract

        entity_name = entity["name"]
        start_pos, end_pos = entity["start"], entity["end"]
        # 2c: passed
        assert og_text[start_pos:end_pos] == entity_name

        prev_char = og_text[start_pos - 1] if start_pos >= 1 else " "
        next_char = og_text[end_pos] if end_pos + 1 < len(og_text) else " "
        if prev_char.isalnum() or next_char.isalnum():
            if all([char.islower() for char in list(entity_name)]):  # Let's now only consider entities that have all lower name chars.
                print("PMID: {}, Entity name: {}, pos: {}-{} is not a full entity.".format(pmid, entity_name, start_pos, end_pos))

def check_3b_3c():
    # 3b
    for _, relation in all_relations.iterrows():
        pmid = relation["pmid"]
        arg1, arg2 = relation["arg1"], relation["arg2"]
        type = relation["type"]

        arg1, arg2 = ":".join(arg1.split(':')[1:]), ":".join(arg2.split(':')[1:])

        entity1 = all_entities.loc[(all_entities["pmid"] == pmid) & (all_entities["arg"] == arg1)]        
        entity2 = all_entities.loc[(all_entities["pmid"] == pmid) & (all_entities["arg"] == arg2)]
        entity1_type = list(entity1["type"])[0]
        entity2_type = list(entity2["type"])[0]

        assert(entity1_type == "CHEMICAL")
        assert(entity2_type in ["GENE-N", "GENE-Y"])
        assert(type in ALL_RELATIONS)

import itertools
def check_overlapping():
    overlapping_entities = {}  # (pmid, small_arg, big_arg) -> (small_name, big_name, relation)
    pmids = set(all_entities["pmid"])
    for pmid in pmids:
        this_entities = all_entities[all_entities["pmid"] == pmid]
        for (_, entity_i), (_, entity_j) in itertools.combinations(this_entities.iterrows(), r=2):
            if entity_i["start"] <= entity_j["start"] and entity_j["end"] <= entity_i["end"]:
                overlapping_entities[(entity_j["pmid"], entity_j["arg"], entity_i["arg"])] = (entity_j["name"], entity_i["name"], "NOT")
            elif entity_j["start"] <= entity_i["start"] and entity_i["end"] <= entity_j["end"]:
                overlapping_entities[(entity_i["pmid"], entity_i["arg"], entity_j["arg"])] = (entity_i["name"], entity_j["name"], "NOT")
    for _, relation in all_relations.iterrows():
        pmid, arg1, arg2, relation_type = relation["pmid"], relation["arg1"], relation["arg2"], relation["type"]
        arg1, arg2 = ":".join(arg1.split(':')[1:]), ":".join(arg2.split(':')[1:])
        if (pmid, arg1, arg2) in overlapping_entities:
            name1, name2, _ = overlapping_entities[(pmid, arg1, arg2)]
            overlapping_entities[(pmid, arg1, arg2)] = (name1, name2, relation_type)
        elif (pmid, arg2, arg1) in overlapping_entities:
            name2, name1, _ = overlapping_entities[(pmid, arg2, arg1)]
            overlapping_entities[(pmid, arg2, arg1)][2] = (name2, name1, relation_type)
    
    for (pmid, arg1, arg2), (name1, name2, relation_type) in overlapping_entities.items():
        print("Overlapping entities: {} & {}. Relation: {} PMID: {}".format(name2, name1, relation_type, pmid))


def check_partial_overlapping():
    overlapping_entities = {}  # (pmid, small_arg, big_arg) -> (small_name, big_name, relation)
    pmids = set(all_entities["pmid"])
    for pmid in pmids:
        this_entities = all_entities[all_entities["pmid"] == pmid]
        for (_, entity_i), (_, entity_j) in itertools.combinations(this_entities.iterrows(), r=2):
            if entity_i["start"] < entity_j["start"] and entity_j["start"] < entity_i["end"] and entity_i["end"] < entity_j["end"]:
                overlapping_entities[(entity_j["pmid"], entity_j["arg"], entity_i["arg"])] = (entity_j["name"], entity_i["name"], "NOT")
            elif entity_j["start"] < entity_i["start"] and entity_i["start"] < entity_j["end"] and entity_j["end"] < entity_i["end"]:
                overlapping_entities[(entity_i["pmid"], entity_i["arg"], entity_j["arg"])] = (entity_i["name"], entity_j["name"], "NOT")
    for _, relation in all_relations.iterrows():
        pmid, arg1, arg2, relation_type = relation["pmid"], relation["arg1"], relation["arg2"], relation["type"]
        arg1, arg2 = ":".join(arg1.split(':')[1:]), ":".join(arg2.split(':')[1:])
        if (pmid, arg1, arg2) in overlapping_entities:
            name1, name2, _ = overlapping_entities[(pmid, arg1, arg2)]
            overlapping_entities[(pmid, arg1, arg2)] = (name1, name2, relation_type)
        elif (pmid, arg2, arg1) in overlapping_entities:
            name2, name1, _ = overlapping_entities[(pmid, arg2, arg1)]
            overlapping_entities[(pmid, arg2, arg1)][2] = (name2, name1, relation_type)
    
    for (pmid, arg1, arg2), (name1, name2, relation_type) in overlapping_entities.items():
        print("Partially overlapping entities: {} & {}. Relation: {} PMID: {}".format(name2, name1, relation_type, pmid))


check_duplicates()
check_2b()
check_2c_2d()
check_3b_3c()
check_overlapping()
check_partial_overlapping()