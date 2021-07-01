cat ./data/drugprot_training_abstracts.tsv ./data/drugprot_development_abstracts.tsv > DrugProtTrain_abstracts.tsv
cat ./data/drugprot_training_entities.tsv ./data/drugprot_development_entities.tsv > DrugProtTrain_entities.tsv
cat ./data/drugprot_training_relations.tsv ./data/drugprot_development_relations.tsv > DrugProtTrain_relations.tsv
cp ./data/drugprot_test_abstracts.tsv .
cp ./data/drugprot_test_entities.tsv .

# Input: _abstracts.tsv, _entities.tsv, _relations.tsv, 
# Output: _ALLrelations.txt
python ./src/Ent2Relation.py

# Input: _abstracts.tsv, _entities.tsv, _ALLrelations.tsv, interaction word dict
# Output: _triplets.txt, _pairs_label.txt, _ProteinNames.txt
python ./src/Abs2Triplets.py -a DrugProtTrain_abstracts.tsv -e DrugProtTrain_entities.tsv -r DrugProtTrain_ALLrelations.txt -p DrugProtTrain
python ./src/Abs2Triplets.py -a drugprot_test_abstracts.tsv -e drugprot_test_entities.tsv -r DrugProtTest_ALLrelations.txt -p DrugProtTest

# -------------------------------
# Files to make: _triplets.txt, _pairs_label.txt, _ProteinNames.txt

# Input: _triplets.txt
# Output: _sentence.txt
python ./src/RewriteCorpus.py -i DrugProtTrain_triplets.txt -o DrugProtTrain_sentence.txt
python ./src/RewriteCorpus.py -i DrugProtTest_triplets.txt -o DrugProtTest_sentence.txt

# Input: _sentence.txt 
# Outout: _sentences_sen.txt (self consume), _sentences_sen.dep
python ./src/RunParser.py -i DrugProtTrain_sentence.txt -p DrugProtTrain
python ./src/RunParser.py -i DrugProtTest_sentence.txt -p DrugProtTest

# Input: _triplets.txt, _sentences_sen.dep
# Output: _nnparsed_graph.txt
python ./src/Dep2Graph.py -p DrugProtTrain
python ./src/Dep2Graph.py -p DrugProtTest

# Input: _nnparsed_graph.txt
# Output: _sentFromGraph.txt
python ./src/sentFromGraph.py -i DrugProtTrain_nnparsed_graph.txt -p DrugProtTrain
python ./src/sentFromGraph.py -i DrugProtTest_nnparsed_graph.txt -p DrugProtTest

# Input: _sentFromGraph.txt, _triplets.txt
# Output: _tagged.txt, _trip_label.txt
python ./src/TagTriplets.py -s DrugProtTrain_sentFromGraph.txt -c DrugProtTrain_triplets.txt -p DrugProtTrain
python ./src/TagTriplets.py -s DrugProtTest_sentFromGraph.txt -c DrugProtTest_triplets.txt -p DrugProtTest

# Input: _nnparsed_graph.txt, _tagged.txt, _trip_label.txt
# Output: _shortest_path.txt
python ./src/shortestPath.py -g DrugProtTrain_nnparsed_graph.txt -t DrugProtTrain_tagged.txt -l DrugProtTrain_trip_label.txt -p DrugProtTrain
python ./src/shortestPath.py -g DrugProtTest_nnparsed_graph.txt -t DrugProtTest_tagged.txt -l DrugProtTest_trip_label.txt -p DrugProtTest

# Input: _tagged.txt, _pairs_label.txt, interaction word dict, _ProteinNames.txt, _shortest_path.txt
# Output: _pair_sentences.txt, _labels_pair.txt, _pairID.txt, _Features_pair.txt, _FeatureNames_pair.txt
python ./src/getFeatures_pair.py -p DrugProtTrain
python ./src/getFeatures_pair.py -p DrugProtTest

# Input: _tagged.txt, _pairs_label.txt, interaction word dict, _ProteinNames.txt, _shortest_path.txt
# Output: _labels_triplet.txt, _tripletID.txt, _Features_triplet.txt, _FeatureNames_triplet.txt
python ./src/getFeatures_triplets.py -p DrugProtTrain
python ./src/getFeatures_triplets.py -p DrugProtTest

# Input: _pairID.txt, _labels_triplet.txt, _Features_triplet.txt, _tripletID.txt, _labels_pair.txt, _Features_pair.txt
python ./src/L1_Model.py -r DrugProtTrain_Features_pair.txt -s DrugProtTest_Features_pair.txt -t 0 -a 0
python ./src/L1_Model.py -r DrugProtTrain_Features_triplet.txt -s DrugProtTest_Features_triplet.txt -t 1 -a 0

# Input: _pairID.txt, _labels_pair.txt, _labels_triplet.txt, _Features_triplet.txt, _tripletID.txt, _Features_pair.txt
python ./src/L2_Model.py -r DrugProtTrain_Features_pair.txt -s DrugProtTest_Features_pair.txt -t 0 -l 1 -a 0
python ./src/L2_Model.py -r DrugProtTrain_Features_triplet.txt -s DrugProtTest_Features_triplet.txt -t 1 -l 1 -a 0

# _labels_pair.txt
python ./src/L3_Model.py -r DrugProtTrain -s DrugProtTest
