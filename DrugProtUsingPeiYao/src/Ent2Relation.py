# This program uses entities files to generate all possible relation pairs.
# Two entities in a pair should be in the same sentence.
# If a pair is not in the relation file, this pair should be CPR10.
# Output: absid+'\t'+CPR+'\t'+NY+'\tNA\tArg1:'+ARG1+'\tArg2:'+ARG2+'\n'

import pandas as pd
import copy

def make_ALLrelations_file(df, output_prefix):
	sub_df = copy.deepcopy(df[["pmid", "chemical", "gene", "relation type"]])
	sub_df.loc[:, "chem_arg"] = ["Arg1:{}".format(item[0]) for item in sub_df.loc[:, "chemical"]]
	sub_df.loc[:, "gene_arg"] = ["Arg2:{}".format(item[0]) for item in sub_df.loc[:, "gene"]]
	sub_df.loc[:, "label"] = sub_df["relation type"] != "NOT"
	sub_df.replace({"label": {True: "Y", False: "N"}}, inplace=True)
	sub_df["NA"] = "NA"
	sub_df = sub_df[["pmid", "relation type", "label", "NA", "chem_arg", "gene_arg"]]
	sub_df.to_csv("{}_ALLrelations.txt".format(output_prefix), index=False, sep='\t', header=False)

train_df = pd.read_json("split_dataframe/train_dataset.json", orient="table")
dev_df = pd.read_json("split_dataframe/val_dataset.json", orient="table")
test_df = pd.read_json("split_dataframe/test_dataset.json", orient="table")
make_ALLrelations_file(pd.concat([train_df, dev_df], ignore_index=True), "DrugProtTrain")
make_ALLrelations_file(test_df, "DrugProtTest")