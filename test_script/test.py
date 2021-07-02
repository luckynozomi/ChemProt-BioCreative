import sys
import itertools
import os

from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

from my_sentence_tokenizer import my_sentence_tokenizer

FALSE_LABEL = "NOT"
TRUE_RELATIONS = [
    "INDIRECT-DOWNREGULATOR", "INDIRECT-UPREGULATOR", "DIRECT-REGULATOR", "ACTIVATOR", "INHIBITOR", 
    "AGONIST", "ANTAGONIST", "AGONIST-ACTIVATOR", "AGONIST-INHIBITOR", "PRODUCT-OF", "SUBSTRATE", 
    "SUBSTRATE_PRODUCT-OF", "PART-OF"
]
ALL_RELATIONS = TRUE_RELATIONS + [FALSE_LABEL]
ALL_RELATIONS_DICT = {relation: index for index, relation in enumerate(ALL_RELATIONS)}

NOT_IN_THE_SAME_SENTENCE = "NOT_SAME_SENTENCE"
ALL_RELATIONS_PLUS = ALL_RELATIONS + [NOT_IN_THE_SAME_SENTENCE]
ALL_RELATIONS_PLUS_DICT = {relation: index for index, relation in enumerate(ALL_RELATIONS_PLUS)}

TRUE_STANDARD_FILE = "true_standard.tsv"


def make_true_standard_df():
    all_abstracts = pd.read_csv("test_data/drugprot_test_abstracts.tsv", sep='\t', header=None, names=["pmid", "title", "abstract"], dtype={"pmid": str})
    all_entities = pd.read_csv("test_data/drugprot_test_entities.tsv", sep='\t', header=None, names=["pmid", "arg", "type", "start", "end", "name"], dtype={"pmid": str})
    all_relations = pd.read_csv("test_data/drugprot_test_relations.tsv", sep='\t', header=None, names=["pmid", "type", "arg1", "arg2"], dtype={"pmid": str})

    true_standard_df = pd.DataFrame(columns=["pmid", "type", "arg1", "arg2"])
    for _, abstract in tqdm(all_abstracts.iterrows()):
        pmid = abstract["pmid"]
        entities = all_entities[all_entities["pmid"] == pmid]
        relations = all_relations[all_relations["pmid"] == pmid]
        
        # key: entity arg. value: the nth sentence it's in.
        entity_to_sent_id_dict = {}
        original_text = abstract["title"] + '\t' + abstract["abstract"]
        entity_poses = [[start, end] for start, end in zip(entities["start"], entities["end"])]
        sentence_poses = my_sentence_tokenizer(original_text, entity_poses)
        for _, this_entity in entities.iterrows():
            this_arg, this_start, this_end = this_entity["arg"], this_entity["start"], this_entity["end"]
            entity_found = False
            for sentence_index, (sentence_start, sentence_end) in enumerate(sentence_poses):
                if sentence_start <= this_start and this_end <= sentence_end:
                    entity_found = True
                    entity_to_sent_id_dict[this_arg] = sentence_index
            if not entity_found:
                raise ValueError("PMID: {}, entity with arg {} not found in any sentence.".format(pmid, this_arg))

        # key: tuple (arg1, arg2). value: relation_type.
        args_to_relation_dict = {}
        for _, this_relation in relations.iterrows():
            this_entity_pair = (this_relation["arg1"], this_relation["arg2"])
            args_to_relation_dict[this_entity_pair] = this_relation["type"]
        
        chemicals = entities[entities["type"] == "CHEMICAL"]
        genes = entities[entities["type"].isin(["GENE-Y", "GENE-N"])]

        for (_, chemical), (_, gene) in itertools.product(chemicals.iterrows(), genes.iterrows()):
            chem_arg, chem_start, chem_end = chemical["arg"], chemical["start"], chemical["end"]
            gene_arg, gene_start, gene_end = gene["arg"], gene["start"], gene["end"]
            args_tuple = ("Arg1:{}".format(chem_arg), "Arg2:{}".format(gene_arg))

            if entity_to_sent_id_dict[chem_arg] != entity_to_sent_id_dict[gene_arg]:
                if args_tuple in args_to_relation_dict:
                    raise ValueError("PMID: {}, chemical-gene ({}, {}) not in the same sentence, but has label in relation file.".format(pmid, chem_arg, gene_arg))
                continue
            elif args_tuple not in args_to_relation_dict:
                args_to_relation_dict[args_tuple] = FALSE_LABEL

        for (chem_arg, gene_arg), relation in args_to_relation_dict.items():
            true_standard_df = true_standard_df.append({"pmid": pmid, "arg1": chem_arg, "arg2": gene_arg, "type": relation}, ignore_index=True)
    true_standard_df.to_csv(TRUE_STANDARD_FILE, header=None, sep='\t', index=False)


def get_test_stats(test_results, true_standard):
    return 0

if not os.path.exists(TRUE_STANDARD_FILE):
    make_true_standard_df()

true_standard = pd.read_csv(TRUE_STANDARD_FILE, sep='\t', header=None, names=["pmid", "type", "arg1", "arg2"], dtype={"pmid": str})
test_file = sys.argv[1]
test_results = pd.read_csv(test_file, sep='\t', header=None, names=["pmid", "type", "arg1", "arg2"], dtype={"pmid": str})

true_standard.drop_duplicates(inplace=True, subset=["pmid", "arg1", "arg2"], keep="first", ignore_index=True)  # Only the first kept.
test_results.drop_duplicates(inplace=True, subset=["pmid", "arg1", "arg2"], keep="first", ignore_index=True)

true_standard.set_index(keys=["pmid", "arg1", "arg2"], inplace=True, verify_integrity=True)
test_results.set_index(keys=["pmid", "arg1", "arg2"], inplace=True, verify_integrity=True)

test_types = test_results["type"]
is_valid_predictions = list(test_types.isin(ALL_RELATIONS))
if not all(is_valid_predictions):
    first_invalid_prediction_index = is_valid_predictions.index(False)
    raise ValueError("Invalid prediction at line: {}".format(first_invalid_prediction_index))

whole_results = true_standard.join(test_results, lsuffix="_true", rsuffix="_pred")
whole_results.fillna({"type_true": NOT_IN_THE_SAME_SENTENCE, "type_pred": "NOT"}, inplace=True)

f1 = f1_score(whole_results["type_true"], whole_results["type_pred"], labels=ALL_RELATIONS_PLUS, average="micro")
confusion_mat = confusion_matrix(whole_results["type_true"], whole_results["type_pred"], labels=ALL_RELATIONS_PLUS)
confusion_df = pd.DataFrame(confusion_mat, columns=ALL_RELATIONS_PLUS, index=ALL_RELATIONS_PLUS)
print("Micro F1 score: {}".format(f1))
confusion_df.to_csv("{}_confusion_matrix.csv".format(test_file.replace(".", "_")))
whole_results.to_csv("{}_whole_results.csv".format(test_file.replace(".", "_")))