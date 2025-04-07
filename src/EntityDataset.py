import os
import re
# import spacy
# from collections import defaultdict
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class DataLoader:
    """
    Loads and cleans text data from given dataframe and directory structure.
    """
    def __init__(self, dataframe, base_dir, language="EN", folder="raw-documents"):
        self.dataframe = dataframe
        self.base_dir = base_dir
        self.folder = folder
        self.language = language

        self.garbage_words = {
            "EN": ["read more", "subscribe now", "follow us", "advertisement", "share your experiences","also read"
                   "more on", "you may like", "subscribe", "unsubscribe", "sign up", "email us", "click here", "whatsapp us"],
        }

    def _clean_text(self, text):

        # Step 1: Clean text from unnecessary symbols and patterns
        lines = [line.strip() for line in re.split(r'\n+', text) if line.strip()]
        cleaned_lines = []

        for line in lines:
            # Remove emoji characters
            line = re.sub(r'[\U0001F600-\U0001FFFF\U00002702-\U000027B0\U000024C2-\U0001F251]', '', line)
            # Remove URLs
            line = re.sub(r'http[s]?://\S+', '', line)
            line = re.sub(r'.*\.co.*[\\].* ','', line)
            # Remove @mentions (if they don't contribute to the content)
            line = re.sub(r'@\w+', '', line)
            # Correct contractions
            line = re.sub(r" n[\u2019']t", ' not', line)
            linr = re.sub(r'it\'s', 'it is', line)
            line = re.sub(r'\'s', '', line)
            line = re.sub(r'\'m', 'am', line)
            line = re.sub(r'\'ve', ' have', line)
            line = re.sub(r'n\'t', " not", line)
            line = re.sub(r'\'re', ' are', line)
            line = re.sub(r'-', ' ', line)  # Remove hyphens
            line = re.sub(r'–', ' ', line)
            line = re.sub(r'(\.)(\.)*', '.', line)  # Clean up ellipses
            line = re.sub(r'^- ', '', line)  # Remove leading dashes
            line = re.sub(r'"|“|”|\'|\‘|\’', '', line)  # Remove quotation marks

            # Step 2: Remove known garbage phrases (ads, self-promotions, etc.)
            if any(garbage in line.lower() for garbage in self.garbage_words[self.language]):
                continue

            # Step 3: Sentence-level cleaning
            sentences = re.split(r'(?<=[.!?])\s+', line)  # Split based on punctuation marks
            valid_sentences = []
            
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                # Ensure the sentence has a valid ending (period, exclamation, or question mark)
                if i == 0 and not re.search(r'[.!?]$', sentence):
                    continue

                # Remove references to other articles or unrelated content
                if 'article' in sentence.lower() or 'read more' in sentence.lower():
                    continue

                valid_sentences.append(sentence)

            # Add cleaned sentences to the final lines
            if valid_sentences:
                cleaned_lines.extend(valid_sentences)

        # Step 4: Join cleaned sentences and remove extra spaces
        cleaned_text = ' '.join(cleaned_lines)
        return re.sub(r'\s+', ' ', cleaned_text).strip()

    

    def _get_text(self):
        for _, row in self.dataframe.iterrows():
            file_path = os.path.join(self.base_dir, self.language, self.folder, row['article_id'])
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                    cleaned_text = self._clean_text(raw_text)

        yield cleaned_text
    
    def _get_ent_role_text(self):
        for _, row in self.dataframe.iterrows():
            entity = row['entity_mention']
            main_role = row['main_role']
            file_path = os.path.join(self.base_dir, self.language, self.folder, row['article_id'])
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                    cleaned_text = self._clean_text(raw_text).lower()
        
        yield {
            'text':cleaned_text,
            'entity':entity,
            'role':main_role
        }

if __name__ == "__main__":

    from LoadData import LoadData

    base_dir = "train"
    txt_file = "subtask-1-annotations.txt"
    subdirs = "EN"
    load_data = LoadData()
    data = load_data.load_data(base_dir, txt_file, subdirs)
    data_loader = DataLoader(data, base_dir)
    
    for data in data_loader._get_text():
        print(data)
        break