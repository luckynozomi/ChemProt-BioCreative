cp ../raw_data_processing/my_sentence_tokenizer.py .
rm -r test_data
mkdir test_data
cp ../raw_data_processing/split_original/drugprot_test_abstracts.tsv ../raw_data_processing/split_original/drugprot_test_entities.tsv ../raw_data_processing/split_original/drugprot_test_relations.tsv test_data/.
rm true_standard.tsv