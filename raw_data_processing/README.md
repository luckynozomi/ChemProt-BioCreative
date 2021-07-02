Step 1. Downloadd drugprot-gs-training-development.zip from the BioCreative VII official website and unzip it.

Step 2. Run data_integrity.txt to check integrity of data.

Step 3. Run make_dataset.py to make training data (currently only uses the training sub-dir in the downloaded data)

Step 4. Run split_train_data.py to split training data into training/validation/test sets. (Later this script will be modified as we will use the validation dataset to test model performance.)

Split data are under 2 dirs:
* split_dataframe: pandas dataframes. Read it via `pd.read_json(file_path, orient="table")`
* split_original: .tsv files w/ same format as training dataset