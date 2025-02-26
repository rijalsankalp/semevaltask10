import os
import re
import torch
import spacy
import stanza
from LoadData import LoadData
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BaseDataset(Dataset):
    """
    Base class for Train and Test datasets

    Args:
    dataframe: pandas DataFrame with data information
    base_dir: base directory for the dataset
    sen_num: number of sentences to extract features from
    sen_lim: boolean to limit the number of sentences
    return_sentiment: boolean to return sentiment scores
    language: language directory of the dataset
    folder: folder with raw documents
    """
    def __init__(self, dataframe, base_dir, sen_num=10, sen_lim=False, return_sentiment = False, language="EN", folder="raw-documents", coref=True):
        self.dataframe = dataframe
        self.base_dir = base_dir
        self.sen_lim = sen_lim
        self.sen_num = sen_num
        self.folder = folder
        self.language = language
        self.coref = coref

        self.lang_map = {
            "EN": ['en', 'en_core_web_sm'],
            "RU": ['ru', 'ru_core_news_sm']
        }
        self.return_sentiment = return_sentiment

        if(self.language in self.lang_map):
            self.coref_nlp = stanza.Pipeline(lang=self.lang_map[self.language][0], processors='tokenize,pos,lemma,ner,depparse,coref', device = 'cpu')
            self.spacy_nlp = spacy.load(self.lang_map[self.language][1])

        if self.return_sentiment:
            self.tokenizer = AutoTokenizer.from_pretrained("tabularisai/multilingual-sentiment-analysis")
            self.model = AutoModelForSequenceClassification.from_pretrained("tabularisai/multilingual-sentiment-analysis")

    def __len__(self):
        """
        Returns the length of the dataset
        
        Returns:
        int: length of the dataset
        """
        return len(self.dataframe)

    def _sent_score(self, text: str) -> list:
        """
        Returns the sentiment scores for the input text
        
        Args:
        text: input text
        
        Returns:
        list: sentiment scores for 5 sentiment classes
        """
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, padding=True, truncation=True, return_overflowing_tokens=False)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.nn.functional.softmax(outputs.logits, dim=1)

    def _clean_text(self, text: str) -> str:
        """
        Cleans the input text by removing URLs, emojis, numbers, and other garbage words
        
        Args:
        text: input text
        
        Returns:
        str: cleaned text
        """
        garbage_words = {
            "EN": ["read more", "subscribe now", "follow us", "advertisement", 
                "more on", "you may like", "subscribe", "unsubscribe", "sign up"],
            "RU": ["читать далее", "подписаться", "следите за нами", "реклама",
                "подробнее", "вам может понравиться", "подпишитесь", "отписаться", "регистрация"]
        }
        lines = [line.strip() for line in re.split(r'\n+', text) if line.strip()]
        cleaned_lines = []
        for line in lines:
            line = re.sub(r'[\U0001F600-\U0001FFFF\U00002702-\U000027B0\U000024C2-\U0001F251]', '', line)
            line = re.sub(r'http[s]?://\S+', '', line)
            line = re.sub(r'\d+', '', line)
            line = re.sub(r'@\w+', '', line)
            sentences = re.split(r'(?<=[.!?])\s+', line)
            valid_sentences = []
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if i == 0 and not re.search(r'[.!?]$', sentence):
                    continue
                if any(garbage in sentence.lower() for garbage in garbage_words[self.language]):
                    continue
                valid_sentences.append(sentence)
            if valid_sentences:
                cleaned_lines.extend(valid_sentences)
        cleaned_text = ' '.join(cleaned_lines)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        return cleaned_text

    def _coref_text(self, text: str, entity_mention: str) -> str:
        """
        Resolves coreferences in the input text
        
        Args:
        text: input text
        entity_mention: entity mention in the text
        
        Returns:
        str: text with resolved coreferences
        """
        doc = self.coref_nlp(text)
        sentences = [sent.text for sent in doc.sentences]
        resolved_sentences = sentences.copy()
        entity_words = set(entity_mention.split())
        replaced_words = set()
        for sent_idx, sentence in enumerate(doc.sentences):
            current_sentence = resolved_sentences[sent_idx]
            replacements = []
            for word in sentence.words:
                if word.text in replaced_words or word.text in entity_words:
                    continue
                if hasattr(word, 'coref_chains') and word.coref_chains:
                    for chain in word.coref_chains:
                        if not chain.is_representative:
                            rep_text = chain.chain.representative_text
                            if (rep_text not in entity_mention and 
                                rep_text not in replaced_words and 
                                rep_text != word.text):
                                replacements.append((word.text, rep_text))
                                replaced_words.add(word.text)
                                replaced_words.add(rep_text)
            for original, replacement in replacements:
                current_sentence = current_sentence.replace(original, replacement)
            resolved_sentences[sent_idx] = current_sentence
        return " ".join(resolved_sentences)

    def extract_entity_features(self, text, entity_mention) -> tuple:
        """
        Extracts features from the input text
        
        Args:
        text: input text
        entity_mention: entity mention in the text
        
        Returns:
        tuple: features extracted from the text
        """
        garbage = {
            "EN": ["say", "deny", "early", "year", "dollar", "currency"],
            "RU": ["сказать", "отрицать", "начало", "год", "доллар", "валюта"]
        }
        doc_coref = self.coref_nlp(text, entity_mention)
        doc = self.spacy_nlp(doc_coref)
        entity_mention = entity_mention.replace("(", "").replace(")", "").replace("»", "").replace("«", "")
        exclude_pos = {"PROPN", "PUNCT", "AUX", "CCONJ", "SCONJ", "DET", "NUM", "SYM", "SPACE", "PART", "ADP"}
        target_entity = entity_mention.split()
        feature_words = []
        feature_sentences = []
        sentence_count = 0
        for sentence in doc.sents:
            for token in sentence:
                if token.text in target_entity:
                    if self.sen_lim and self.sen_num > sentence_count:
                        break
                    sentence_count += 1
                    feature_words += [t.text for t in sentence if not t.is_stop and not t.text in target_entity and not t.lemma_ in garbage[self.language] and t.pos_ not in exclude_pos]
                    feature_sentences.append(sentence.text)
                    break
        return " ".join(feature_words), " ".join(feature_sentences)
    
    def extract_entity_sentence(self, text, entity_offsets) -> str:
        """
        Extracts sentences with the entity mention

        Args:
        text: input text
        entity_mention: entity mention in the text

        Returns:
        str: sentence with the entity mention

        """
        start, end = entity_offsets

        while start > 0 and text[start - 1] != '\n':
            start -= 1
        while end < len(text) and text[end] != '\n':
            end += 1

        return str(text[start:end])


class TrainDataset(BaseDataset):
    """
    Train dataset class
    """
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        article_id = row['article_id']
        entity_mention = row['entity_mention']
        start_offset = row['start_offset']
        end_offset = row['end_offset']
        main_role = row['main_role']
        sub_roles = row['sub_roles']
        article_path = os.path.join(self.base_dir, self.language, self.folder, article_id)
        if not os.path.exists(article_path):
            print(f"Article {article_path} not found")
            return None
        with open(article_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        
        if self.coref:
            text = self._clean_text(text)
            if text == '':
                print(f"Empty text for article {article_id}")
                return None

            text = self._coref_text(text, entity_mention)
            word_features, sent_features = self.extract_entity_features(text, entity_mention)

            if self.return_sentiment:
                sent_sentiments = self._sent_score(sent_features)
                word_sentiments = self._sent_score(word_features)
                return {
                    "article_id": article_id,
                    "word_features": word_features,
                    "sent_features": sent_features,
                    "entity_mention": entity_mention,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "sent_sent": sent_sentiments,
                    "word_sent": word_sentiments,
                    "main_role": main_role,
                    "sub_roles": sub_roles,
                }
            
            return {
                "article_id": article_id,
                "word_features": word_features,
                "sent_features": sent_features,
                "entity_mention": entity_mention,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "main_role": main_role,
                "sub_roles": sub_roles,
            }
        else:
            entity_sentence = self.extract_entity_sentence(text, (int(start_offset), int(end_offset)))

            if self.return_sentiment:
                sent_sentiments = self._sent_score(entity_sentence)
                return {
                    "article_id": article_id,
                    "entity_sentence": entity_sentence,
                    "entity_mention": entity_mention,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "sent_sent": sent_sentiments,
                    "main_role": main_role,
                    "sub_roles": sub_roles,
                }
            
            return {
                "article_id": article_id,
                "entity_sentence": entity_sentence,
                "entity_mention": entity_mention,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "main_role": main_role,
                "sub_roles": sub_roles,
            }

class TestDataset(BaseDataset):
    """
    Test dataset class
    """
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        article_id = row['article_id']
        entity_mention = row['entity_mention']
        start_offset = row['start_offset']
        end_offset = row['end_offset']
        article_path = os.path.join(self.base_dir, self.language, self.folder, article_id)
        if not os.path.exists(article_path):
            print(f"Article {article_path} not found")
            return None
        with open(article_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text = self._clean_text(text)
        if text == '':
            print(f"Empty text for article {article_id}")
            return None
        text = self._coref_text(text, entity_mention)
        word_features, sent_features = self.extract_entity_features(text, entity_mention)

        if self.return_sentiment:
            sent_sentiments = self._sent_score(sent_features)
            word_sentiments = self._sent_score(word_features)
            return {
                "article_id": article_id,
                "word_features": word_features,
                "sent_features": sent_features,
                "entity_mention": entity_mention,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "sent_sent": sent_sentiments,
                "word_sent": word_sentiments,
            }
        
        return {
            "article_id": article_id,
            "entity_mention": entity_mention,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "word_features": word_features,
            "sent_features": sent_features
        }
    

if __name__ == "__main__":
    base_dir = "train"
    txt_file = "subtask-1-annotations.txt"
    subdirs = ['RU']
    ld = LoadData()
    data = ld.load_data(base_dir, txt_file, subdirs)
    train_dataset = TrainDataset(data, base_dir, language="RU")
    print(train_dataset[0])

    train_dataset = TrainDataset(data, base_dir, language="RU", return_sentiment=True)
    print(train_dataset[0])
