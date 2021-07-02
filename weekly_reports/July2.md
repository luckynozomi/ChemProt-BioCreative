# Track Objective
Build a model predicting the interaction type (or not interacting) of gene-chemial interactions.

## Training Data
We are given abstracts, entity mentions and relations between these entities.
### Abstracts
Tab-seperated file containing pmid, title and abstract.

Example:
```
23017395	Bioefficacy of EPA-DHA from lipids recovered from fish processing wastes through biotechnological approaches.	The effect of fish oil recovered from fish visceral waste (FVW-FO) on serum and liver lipids, activity of HMG-CoA reductase in liver microsomes and EPA+DHA incorporation in liver, heart and brain were evaluated. Rats were fed different concentrations of FVW-FO providing 1.25%, 2.50%, 5.0% EPA+DHA recovered by either fermentation or enzymatic hydrolysis for 8weeks. Feeding FVW-FO reduce triacylglycerols (5.96-20.3%), total cholesterol (7.9-21.5%) and LDL (7.39-21.7%) cholesterol levels in serum compared to group fed on a control diet (groundnut oil). The activity of HMG-CoA reductase was reduced (p<0.05) in the FVW-FO fed groups compared to the control. EPA+DHA level in serum, liver, brain and heart increased with increments in dietary EPA+DHA. Results show the hypolipidemic property of FVW-FO and reduced HMG-CoA reductase activity which is proportional to the incorporation of EPA+DHA. Recovery of FVW-FO will address the increasing demand for fish oil and reduce pollution problems.
```
Data check:
* No duplicated PMIDs --> Pass

### Entity Mentions
Tab-separated file containing pmid, arg (entity label), entity type, start position, end position and name.

Example:
```
11808879	T12	GENE-Y	1860	1866	KIR6.2
11808879	T13	GENE-N	1993	2016	glutamate dehydrogenase
11808879	T14	GENE-Y	2242	2253	glucokinase
23017395	T1	CHEMICAL	216	223	HMG-CoA
23017395	T2	CHEMICAL	258	261	EPA
```

Data check:
* No duplicated (pmid, arg) -> Pass
* All entity types are either "CHEMICAL", "GENE-Y" or "GENE-X" -> Pass
* `Title_and_Abstract[start_pos:end_pos] == name` -> There was one exception, but fixed in ver. 1.1
* The names are not part of bigger word, i.e., `name == "CHEMICA"` when it's actually `"CHEMICAL"` in the original text. -> 12 entities failed.
    > 23349483 ... suggest that ROS reduction can improve mitochondrial metabolism by suppressing**lactate** overproduction.
    >
    > 10878295 ... who had failed to respond to **methotrexat**e alone.

### Relation
Tab-separated file containing pmid, relation type, arg1 (chemical entity label), and arg2 (gene entity label). 

Example:
```
12488248	INHIBITOR	Arg1:T1	Arg2:T52
12488248	INHIBITOR	Arg1:T2	Arg2:T52
23220562	ACTIVATOR	Arg1:T12	Arg2:T42
23220562	ACTIVATOR	Arg1:T12	Arg2:T43
23220562	INDIRECT-DOWNREGULATOR	Arg1:T1	Arg2:T14
```

Data check:
* No duplicated (pmid, arg1, arg2) -> **FAILED**
* All arg1 are "CHEMICAL", and all arg2 are either "GENE-Y" or "GENE-N" -> Pass
* All relation types are in the list of the ones we need to predict -> Pass
* All the chemicals and genes are in the same sentence -> We don't have their sentence splitter so we can't check. We use this criteria to validate our own sentence splitter. After its built, we confirm that all the pairs are in the same sentence.

Relation type hireachy:
* `INDIRECT-UPREGULATOR`. CEM upregulates GPRO via other target. GPRO is upregulated by CEM as a result from CEMs effect on other target which results in upregulation of GPRO.
* `INDIRECT-DOWNREGULATOR`. CEM downregulates GPRO via other target. GPRO is downregulated by CEM as a result from CEMs effect on other target which results in downregulation of GPRO.
* `DIRECT-REGULATOR`. Binder/Ligand: CEM that directly binds to a GPRO (typically a protein) through a direct physical interaction and changes its activity/function (typically a protein activity/function).
    * `ACTIVATOR`. CEM that binds to a GPRO (typically a protein) and increases its activity. Conceptual synonyms are Stimulator, Inducer, Potentiator and Enhancer.
    * `INHIBITOR`. CEM that binds to a GPRO (typically a protein) and decreases its activity.
    * `AGONIST`. CEM that binds to a receptor and alters the receptor state resulting in a biological response. Conventional agonists increase receptor activity, whereas inverse agonists reduce it. If no information is provided on whether the CEM activates or reduces GPRO activity, this general subclass should be assigned (instead of AGONISTACTIVATOR and AGONIST-INHIBITOR, below).
        * `AGONIST-ACTIVATOR`. Agonists that bind to a **receptor** and increase its biological response. Typically, for full agonists and most partial agonists (depending on concentration).
        * `AGONIST-INHIBITOR`. Agonists that bind to a **receptor** and decrease its biological response. Typically, for inverse agonists.

    * `ANTAGONIST`. CEM that reduces the action of another CEM, generally an agonist. Many antagonists act at the same **receptor** macromolecule as the agonist.
    * `PRODUCT-OF`. CEM that is a product of enzymatic reaction or a transporter
    * `SUBSTRATE`. CEM upon which a GPRO (typically protein) acts. It should be understood as the substrate of a reaction carried out by a protein (“reactant”) or as transporter substrate.
    * `SUBSTRATE_PRODUCT-OF`. CEM that is both, substrate and product of enzymatic reaction.
* `PART-OF`. CEM that are structurally related to a GPRO: e.g. specific amino acid residues of a protein.

## Data Check: Multiple Interaction Labels

Example 1.
```
23064031  INDIRECT-DOWNREGULATOR   Arg1:T6  Arg2:T26
23064031               INHIBITOR   Arg1:T6  Arg2:T26
```
Text:
> **NFD** suppressed EGFT23-mediated protein levels of c-Jun and c-Fos, and reduced **MMP-9** expression and activity, concomitantly with a marked inhibition on cell migration and invasion without obvious cellular cytotoxicity.

Example 2.
```
17114825               SUBSTRATE   Arg1:T5  Arg2:T17
17114825               INHIBITOR   Arg1:T5  Arg2:T17
```
Text:
> Tryptophan-dependent tRNAtrp aminoacylation catalyzed by **TrpRS** can be inhibited by its substrate **tryptophan** at physiological concentrations was demonstrated. 

Statistics
* Duplicated Relations: 430 of 17288
* Totally Duplicated Relations (same interaction type): 28 of 17288
* Categories co-occuring
    ```
    ANTAGONIST - DIRECT-REGULATOR: 38
    ACTIVATOR - DIRECT-REGULATOR: 36
    DIRECT-REGULATOR - INHIBITOR: 34
    ACTIVATOR - INDIRECT-UPREGULATOR: 20
    INDIRECT-DOWNREGULATOR - INHIBITOR: 19
    AGONIST - ANTAGONIST: 10
    AGONIST - DIRECT-REGULATOR: 9
    DIRECT-REGULATOR - SUBSTRATE: 7
    ACTIVATOR - SUBSTRATE: 6
    INHIBITOR - SUBSTRATE: 5
    INHIBITOR - INHIBITOR: 4
    DIRECT-REGULATOR - DIRECT-REGULATOR: 3
    INDIRECT-DOWNREGULATOR - INDIRECT-UPREGULATOR: 3
    ANTAGONIST - INHIBITOR: 3
    ACTIVATOR - INHIBITOR: 2
    INDIRECT-UPREGULATOR - INHIBITOR: 2
    AGONIST - AGONIST-INHIBITOR: 2
    ACTIVATOR - AGONIST: 1
    AGONIST-ACTIVATOR - AGONIST-INHIBITOR: 1
    SUBSTRATE_PRODUCT-OF - SUBSTRATE_PRODUCT-OF: 1
    INDIRECT-UPREGULATOR - INDIRECT-UPREGULATOR: 1
    PRODUCT-OF - PRODUCT-OF: 1
    INDIRECT-DOWNREGULATOR - INDIRECT-DOWNREGULATOR: 1
    DIRECT-REGULATOR - INDIRECT-DOWNREGULATOR: 1
    AGONIST-INHIBITOR - ANTAGONIST: 1
    INDIRECT-DOWNREGULATOR - PART-OF: 1
    ACTIVATOR - ACTIVATOR: 1
    AGONIST - AGONIST: 1
    PART-OF - PART-OF: 1
    ```

Current code does **NOT** deal with this situation. I actually don't know how it's handled in the code.

## Data Check: overlapping entities

There are 2 types of overlapping between entities. Let `chem_start`, `chem_end` be the start/end positions of the chemical (`full_text[chem_start:chem_end]` gives the chemical name in python), and `gene_start`, `gene_end` be those of genes:
* Fully overlapping. 
    * The gene is part of the chemical: `chem_start <= gene_start < gene_end <= chem_end`
    * The chemical is part of the gene: `gene_start <= chem_start < chem_end <= gene_end`
* Partially overlapping.
    * Chemical first, then gene: `chem_start < gene_start < chem_end < gene_end`
    * Gene first, then chemical: `gene_start < chem_start < gene_end < cehm_end`

### Partially Overlapping Entities
We found 11 partially overlapping entities, 1 with a true label.

GST-4 (Gene) vs. 4-HNE (Chemical). Relation: `SUBSTRATE`
> 17553661: ... exhibited high steady-state hGSTA4 mRNA, high **GST-4-HNE** catalytic activities, ...

methyl CpG (Chemial) vs. CpG-binding proteins (Gene). Relation: `NOT`
> 23395981: ... and modulates expression of the DNA methyltransferases (DNMTs) and various **methyl CpG-binding proteins**.

For partially overlapping entities, we modify one of them so that their positions don't overlap.
```python
# Example when chemical first, then gene:
chem_name = original_text[chem_start:gene_start].strip()
chem_start, chem_end = chem_start, chem_start + len(chem_name)
```

### Fully Overlapping Entities
The official annotation guideline states that

> According to the CEM and GPRO guidelines, protein names that contain either the SUBSTRATE or the PRODUCT of the enzymatic reaction or transporter, have double annotation for CEMs and GPROs. For example, the term “Glutamate Receptor” will be labeled as GPRO and also the sub-term “Glutamate” as CEM.
This kind of relationships (either for SUBSTRATE or PRODUCT) must not be annotated.
>
> Example
>
> The phosphorylation of the **glutamate receptor** 1 (GluR1) subunit of **AMPA receptors** by protein kinase A (PKA), protein kinase C (PKC), and Ca2+/calmodulin-dependent protein kinase II (CaMKII) has been characterized extensively.
>
> Annotation
> 
> glutamate : glutamate receptor → NOT LABEL
> AMPA : AMPA receptors → NOT LABEL
>
> However, if there is an independent, separate mention of the same CEM in the same sentence, where it is explicitly mentioned that this CEM acts as a substrate of the GPRO, then that mention (at the given position, separate from the co-mention of the CEM as part of the GPRO mention) should be explicitly labeled.
> 
> Example
>
> It has been postulated that cocaine's modulation of serum **progesterone** levels may in turn alter 
**progesterone receptor** activity, thereby contributing to cocaine-induced alterations of neuronal 
functions and genomic regulations.
> 
> Annotation
> 
> progesterone : progesterone receptor → SUBSTRATE
> 
> But not label if there is not an explicit mention relating both substrate (CEM) and GPRO.
> 
> Example
>
> **Progesterone** serum levels, **progesterone receptor** (PR) protein levels, and PR-DNA binding complexes were measured in the striatum by radioimmunoassay, Western blot, and gel shift analyses, respectively.
>
> Annotation
> progesterone : progesterone receptor → NOT LABEL

We found 5687 fully overlapping chemical-protein pairs in the training data. Examples:
```
Overlapping entities: ATP & K(ATP) channel. Relation: NOT PMID: 23115325
Overlapping entities: carnitine & carnitine acetyltransferase. Relation: NOT PMID: 12582227
Overlapping entities: GnRH & type I and II GnRH receptors. Relation: NOT PMID: 16973761
Overlapping entities: norepinephrine & 5-HT and the norepinephrine transporters. Relation: NOT PMID: 9669506
Overlapping entities: vasopressin & vasopressin. Relation: NOT PMID: 16532916
```

Among them 5 have true relations.
```
Overlapping entities: C & C-terminal Src kinase. Relation: PART-OF PMID: 23548896
Overlapping entities: C & C-terminal Src kinase. Relation: PART-OF PMID: 23548896
Overlapping entities: C & C-peptide. Relation: PART-OF PMID: 23579178
Overlapping entities: N & N-terminal helicase-like domain. Relation: PART-OF PMID: 23219161
Overlapping entities: C & C-terminal DNA polymerase domain. Relation: PART-OF PMID: 23219161
```

Currently we just label all overlapping entities as `NOT`. We may fine-tune this rule based on the names of the gene-chemical pair.

We also found several partially overlapping entities. **EXPAND ON THIS**

## Test Data
For testing, we are only given the abstracts and entities. No relation file is given. We are required to submit a file similar to the relation file, containing all the true interactions found within a sentence.

The official evaluation script has not been released.
> Evaluation will be done using the following micro-averaged scores: f-measure, precision and recall.
> 
> The corresponding DrugProt evaluation library, example predictions and additional details on the prediction format will be released soon. The evaluation will be done by comparting the automatically extracted relations with the previously annotated manual Gold Standard relations.

So we have to write one ourselves when we evaluate our model on the left-out testing set.

### Evaluation Method

Below is the pseudo python code I am currently using for evaluation.
``` python
true_labels = []
pred_labels = []
for article in all_articles_in_test_set:
    for chem, gene in all_possible_pairs_in_article:
        if sentence_its_in[chem] != sentence_its_in[gene]:  # determined by our own sentence splitter
            if (chem, gene) in predicted_labels:
                predicted_label = predicted_labels[(chem, gene)]
                pred_labels.append(predicted_label)
                true_labels.append("NOT_SAME_SENTENCE")
            else:
                continue # we don't add "NOT_SAME_SENTENCE" to both the lists
        else:
            if (chem, gene) in true_annotation:
                this_true_label = true_annotations[(chem, gene)]
            else:
                this_true_label = "NOT"
            true_labels.append(this_true_label)

            if (chem, gene) in predicted_labels:
                this_pred_label = predicted_labels[(chem, gene)]
            else:
                this_pred_label = "NOT"
            pred_labels.append(this_pred_label)

# Then proceeds to get f1-micro score and confusion matrix using true_labels and pred_labels
```

For chemical - gene pairs with multiple labels in the true labels, currently only the one for the last(?) occurance of such pair is used.

# Data Processing

## Sentence Splitter
A sentence splitter splits a paragrah into sentences. It is imporant for this task, because whenever an entity pair with true relations is split into 2 different sentences, we will not be able to predict their interaction, no matter how good our model is.

We are now using the default punkt sentence tokenizer from the nltk package, with a few tweaks.
* Added extra abbreviations, like "e.g.", "i.p.", "vivo.", "sp.", etc. This was first done by PeiYao, and I added some other elements.
* Added an entity list that shouldn't be split. Examples: "no. 4, 1,3,5(10)-estratriene-17alpha-ethynyl-3", "4-DAMP. 1.5 +/- 0.4 nM", and "(Bioorg. Med. Chem., 2010, 18, 1761-1772., J. Med. Chem., 2011, 54, 2823-2838.)".

These extra abbreviations and entities are found by first tokenizing the texts using the default tokenizer, then whenever a chemical - gene pair in the training relation list is split into 2 sentence, examining the split sentences and appending certain items to these lists to fix the problem.

**TODO**: BioCreative organizers have released the development dataset. I can use the current sentence tokenizer on it to see how good its performance is.

# Results
## BERT prediction

Micro F1: 0.80

Confusion Matrix *(The i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.)*



||INDIRECT-DOWNREGULATOR|INDIRECT-UPREGULATOR|DIRECT-REGULATOR|ACTIVATOR|INHIBITOR|AGONIST|ANTAGONIST|AGONIST-ACTIVATOR|AGONIST-INHIBITOR|PRODUCT-OF|SUBSTRATE|SUBSTRATE_PRODUCT-OF|PART-OF|NOT|NOT_SAME_SENTENCE|
|-|----------------------|--------------------|----------------|---------|---------|-------|----------|-----------------|-----------------|----------|---------|--------------------|-------|---|-----------------|
|INDIRECT-DOWNREGULATOR|53|0|0|0|54|0|0|0|0|0|0|0|0|25|0|
|INDIRECT-UPREGULATOR|47|26|0|0|3|0|0|0|0|0|0|0|0|96|0|
|DIRECT-REGULATOR|0|0|65|0|17|0|1|0|0|0|0|0|0|69|0|
|ACTIVATOR|22|14|0|5|6|0|0|0|0|0|1|0|0|119|0|
|INHIBITOR|20|0|2|0|468|0|0|0|0|0|0|0|0|49|0|
|AGONIST|0|0|4|0|2|0|21|0|0|0|0|0|0|29|0|
|ANTAGONIST|0|0|2|0|17|0|45|0|0|0|0|0|0|17|0|
|AGONIST-ACTIVATOR|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
|AGONIST-INHIBITOR|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
|PRODUCT-OF|0|0|0|0|0|0|0|0|0|0|54|0|0|37|0|
|SUBSTRATE|0|0|0|0|3|0|0|0|0|0|93|0|0|103|0|
|SUBSTRATE_PRODUCT-OF|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
|PART-OF|0|0|6|0|1|0|0|0|0|0|1|0|14|62|0|
|NOT|29|3|62|0|200|0|10|0|0|0|52|0|10|4323|0|
|NOT_SAME_SENTENCE|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
