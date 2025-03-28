import os
import re
import spacy
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
            "EN": ["read more", "subscribe now", "follow us", "advertisement", 
                   "more on", "you may like", "subscribe", "unsubscribe", "sign up"],
        }

        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("coreferee")

        self.tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-v1_1-base-wikisplit")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("flax-community/t5-v1_1-base-wikisplit")

    def _clean_text(self, text):
        lines = [line.strip() for line in re.split(r'\n+', text) if line.strip()]
        cleaned_lines = []
        for line in lines:
            line = re.sub(r'[\U0001F600-\U0001FFFF\U00002702-\U000027B0\U000024C2-\U0001F251]', '', line)
            line = re.sub(r'http[s]?://\S+', '', line)
            line = re.sub(r'\d+', '', line)
            line = re.sub(r'@\w+', '', line)
            line = re.sub(r".*n[\u2019']t", ' not', line)
            line = re.sub(r'.*\'s', '', line)
            line = re.sub(r'.*\'re', ' are', line)
            line = re.sub(r'.*-.*', ' ', line)
            line = re.sub(r'(\.)(\.)*', '.', line)
            line = re.sub(r'^- ', '', line)
            line = re.sub(r'"', '', line)
            sentences = re.split(r'(?<=[.!?])\s+', line)
            valid_sentences = []
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if i == 0 and not re.search(r'[.!?]$', sentence):
                    continue
                if any(garbage in sentence.lower() for garbage in self.garbage_words[self.language]):
                    continue
                valid_sentences.append(sentence)
            if valid_sentences:
                cleaned_lines.extend(valid_sentences)
        cleaned_text = ' '.join(cleaned_lines)
        return re.sub(r'\s+', ' ', cleaned_text).strip()
    
    def _coreference_resolution(self, text):
        coref_doc = self.nlp(text)

        resolved_text = ""

        for token in coref_doc:
            # Resolve coreference if available
            repres = coref_doc._.coref_chains.resolve(token)
            if repres:
                resolved_text += (
                    " "
                    + " and ".join(
                        [
                            t.text
                            if t.ent_type_ == ""
                            else [e.text for e in coref_doc.ents if t in e][0]
                            for t in repres
                        ]
                    )
                )
            else:
                if token.is_punct or token.text in ["'s", "n't", "'re", "'ve", "'ll", "'d", "\""]:
                    resolved_text += token.text
                else:
                    resolved_text += " " + token.text

        return resolved_text.strip()
        

    def yield_text(self):
        for _, row in self.dataframe.iterrows():
            file_path = os.path.join(self.base_dir, self.language, self.folder, row['article_id'])
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                    cleaned_text = self._clean_text(raw_text)
                    cleaned_text = self._coreference_resolution(cleaned_text)
                    entity_mention = row['entity_mention'].strip().lower().replace('-', '')

                    #sentences = [sent.text for sent in doc.sents]
                    yield entity_mention, cleaned_text

    def yield_entity_sent(self):
        for _, row in self.dataframe.iterrows():
            file_path = os.path.join(self.base_dir, self.language, self.folder, row['article_id'])
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                    start, end = int(row['start_offset']), int(row['end_offset'])
                    
                    while start > 0 and raw_text[start] != "\n":
                        start -= 1
                    while end < len(raw_text) and raw_text[end] != "\n":
                        end += 1
                    
                    sentence = raw_text[start:end].strip()

                    entity_mention = row['entity_mention'].strip().lower().replace('-', '')

                    yield entity_mention, [sentence]

    def yield_NER_sentences(self):
        for _, row in self.dataframe.iterrows():
            file_path = os.path.join(self.base_dir, self.language, self.folder, row['article_id'])
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                    cleaned_text = self._clean_text(raw_text)
                    sentence = self._coreference_resolution(cleaned_text)
                    sample_tokenized = self.tokenizer(cleaned_text, return_tensors="pt")
                    answer = self.model.generate(sample_tokenized['input_ids'], attention_mask = sample_tokenized['attention_mask'], max_length=256, num_beams=5)
                    sentence = self.tokenizer.decode(answer[0], skip_special_tokens=True)
                    

                    #perform NER on cleaned text, get the key names (person, countries, organizations, etc.) and their corresponding sentences
                    #cleaned_text = self._coreference_resolution(cleaned_text)

                    doc = self.nlp(sentence)
                    
                    entities = {}
    
                    for ent in doc.ents:
                        sentence = ent.sent.text  # Get the sentence containing the entity
                        if ent.label_ not in entities:
                            entities[ent.label_] = []
                        entities[ent.label_].append((ent.text, sentence))
                    
                    # Return the entities with only person, countries, organizations
                    filtered_entities = {key: entities[key] for key in ["PERSON", "GPE", "ORG", "LOC"] if key in entities}

                    items = [items for key,items in filtered_entities.items()]

                    all_items = []
                    for item in items:
                        all_items += item

                    yield all_items
                    
                    
                    


if __name__ == "__main__":

    from LoadData import LoadData

    base_dir = "train"
    txt_file = "subtask-1-annotations.txt"
    subdirs = "EN"
    load_data = LoadData()
    data = load_data.load_data(base_dir, txt_file, subdirs)
    data_loader = DataLoader(data, base_dir)

    for file, text in data_loader.yield_text():
        print(file, text)
        break