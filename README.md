# SEMEVAL TASK 10 MENDEL 292A

Install the libraries and dependencies first.

```
pip install -r requirements.txt
```

# Guide to how data is loaded and processed for the plotting

# loadData.py

The [loadData.py]() file contains a class [LoadData]() designed to load data from a specified directory and its subdirectories. Here's a detailed breakdown of what it does and how it works:

### Class: [LoadData]()

#### Purpose:

The [LoadData]() class is responsible for loading data from text files located in a given base directory and its subdirectories. The data is then processed and returned as a Pandas DataFrame.

#### Methods:

- **[load_data]()** : This is the primary method of the [LoadData]() class. It takes three parameters:
- [base_dir]() (str): The base directory where the data is stored.
- [txt_file]() (str, default="subtask-1-annotations.txt"): The name of the text file containing the data.
- [subdirs]() (list, default=['EN']): A list of subdirectories to search for the text file.

#### How it works:

1. **Initialization** :

- A list named [dataframes]() is initialized to store DataFrames created from each subdirectory.

1. **Iterating through Subdirectories** :

- The method iterates over each subdirectory specified in the [subdirs]() list.
- For each subdirectory, it constructs the file path by combining the [base_dir](), subdirectory name, and [txt_file](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html).

1. **Reading and Processing the File** :

- The method opens the file at the constructed file path.
- It reads the file line by line, skipping empty or malformed lines.
- Each line is split into parts using tabs as delimiters.
- It ensures that each line has at least 5 parts (article_id, entity_mention, start_offset, end_offset, main_role).
- The required fields are extracted from the parts, and any additional parts are considered as [sub_roles](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html).

1. **Creating DataFrame** :

- A DataFrame is created from the processed rows with columns: `['article_id', 'entity_mention', 'start_offset', 'end_offset', 'main_role', 'sub_roles']`.
- This DataFrame is appended to the [dataframes](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) list.

1. **Combining DataFrames** :

- All DataFrames in the [dataframes](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) list are concatenated into a single DataFrame.
- The combined DataFrame is shuffled and its index is reset.

1. **Returning Data** :

- The final DataFrame, containing all the loaded and processed data, is returned.

# EntityDataset.py

The [EntityDataset.py](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) file defines a class [TrainDataset](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) that is used to prepare and manage a dataset for training a machine learning model, specifically for sentiment analysis and entity feature extraction from articles. Here's a breakdown of its functionality:

### Class: [TrainDataset](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)

This class inherits from [Dataset](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) and is designed to handle the preprocessing and feature extraction of text data for training purposes.

#### Attributes:

- [dataframe](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): A DataFrame containing article information.
- [base_dir](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): The base directory where articles are stored.
- [sen_num](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): The number of sentences to consider for sentiment analysis.
- [sen_lim](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): A flag to limit the number of sentences to [sen_num](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html).
- [language](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): The language of the articles, used to access the correct model and folder.
- [folder](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): The folder containing the articles within the language directory.
- [lang_map](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): A mapping of language codes to their respective models.
- [coref_nlp](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): A Stanza pipeline for coreference resolution.
- [spacy_nlp](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): A SpaCy model for natural language processing.
- [tokenizer](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): A tokenizer for sentiment analysis.
- [model](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): A model for sequence classification (sentiment analysis).

#### Methods:

- [**init**](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Initializes the class with the provided parameters and sets up the NLP pipelines and models.
- [**len**](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Returns the length of the DataFrame.
- [\_sent_score](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Computes sentiment scores for the input text using the sentiment analysis model.
- [\_clean_text](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Cleans the input text by removing unwanted elements like emojis, hyperlinks, and garbage words, and processes it into a clean format.
- [\_coref_text](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Resolves coreferences in the text while preserving specific entity mentions to avoid infinite replacement loops.
- [extract_entity_features](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Extracts entity-related features from the text, such as important words and sentences related to the entity.
- [**getitem**](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Retrieves and processes the data for a given index, including reading the article, cleaning the text, resolving coreferences, extracting features, and computing sentiment scores.

### Workflow:

1. **Initialization** : The class is initialized with a DataFrame of articles, base directory, and other parameters. It sets up the necessary NLP pipelines and models.
2. **Data Length** : The [**len**](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) method returns the number of articles in the DataFrame.
3. **Sentiment Scoring** : The [\_sent_score](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) method computes sentiment scores for given text using a pre-trained sentiment analysis model.
4. **Text Cleaning** : The [\_clean_text](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) method processes the raw text to remove unwanted elements and clean it for further analysis.
5. **Coreference Resolution** : The [\_coref_text](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) method resolves coreferences in the text while preserving specific entity mentions.
6. **Feature Extraction** : The [extract_entity_features](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) method extracts important words and sentences related to the entity from the text.
7. **Data Retrieval** : The [**getitem**](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) method retrieves and processes the data for a given index, including reading the article, cleaning the text, resolving coreferences, extracting features, and computing sentiment scores. It returns a dictionary containing the processed data.

This class is designed to handle the preprocessing and feature extraction of text data, making it ready for training machine learning models for tasks like sentiment analysis and entity recognition.

# Plot.ipynb

Creates a visualization that shows sentiment distribution across three character roles (Protagonist, Antagonist, and Innocent) using the following approach:

1. Iterates through each main role (Protagonist, Antagonist, Innocent)
2. For each role:

- Plots the distribution for each sentiment category (Very Negative to Very Positive)
- Each sentiment is plotted

The visualization shows:

- How sentiment scores are distributed for each character role
- Comparison of sentiment patterns across different roles
- The density of each sentiment category within each role
- Multiple sentiment curves overlaid in each subplot for easy comparison
