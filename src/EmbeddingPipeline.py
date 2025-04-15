from sklearn.cluster import DBSCAN
from bertopic import BERTopic
import spacy
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

class EntityExtractionPipeline:
    def __init__(self, n_clusters=2, n_topics=2):
        # Load pre-trained SpaCy NER model
        self.nlp = spacy.load("en_core_web_trf")
        
        # Load pre-trained BERT model and tokenizer from HuggingFace
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        
        # Set parameters for clustering and topic modeling
        self.n_clusters = n_clusters
        self.n_topics = n_topics

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = [(entity.text, entity.label_) for entity in doc.ents]
        return entities

    def get_entity_embeddings(self, text, entities):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.bert_model(**inputs)
        
        embeddings = {}
        for entity, label in entities:
            entity_tokens = self.tokenizer(entity, return_tensors="pt", truncation=True, padding=True)
            entity_embeddings = self.bert_model(**entity_tokens).last_hidden_state.mean(dim=1).detach().numpy()
            embeddings[entity] = entity_embeddings
        
        return embeddings

    def cluster_entities(self, embeddings):
        entity_list = list(embeddings.keys())
        embedding_matrix = np.vstack(list(embeddings.values()))
        
        # Use DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=1, metric="cosine")
        dbscan_labels = dbscan.fit_predict(embedding_matrix)
        
        clustered_entities = {entity_list[i]: dbscan_labels[i] for i in range(len(entity_list))}
        return clustered_entities

    def topic_modeling(self, text):
        sentences = text.split('.')  
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(sentences)
        topics_words = topic_model.get_topic_info()
        return topics_words

    def extract_relations(self, text):
        doc = self.nlp(text)
        relations = []
        
        for token in doc:
            if token.dep_ in ("nsubj", "dobj") and token.head.pos_ == "VERB":
                subject = token.text
                verb = token.head.text
                obj = [child.text for child in token.head.children if child.dep_ == "dobj"]
                if obj and subject != obj[0]:
                    relations.append((subject, verb, " ".join(obj)))
        
        return relations

    def process_text(self, text):
        entities = self.extract_entities(text)
        entity_embeddings = self.get_entity_embeddings(text, entities)
        clustered_entities = self.cluster_entities(entity_embeddings)
        topics = self.topic_modeling(text)
        relations = self.extract_relations(text)
        
        return {
            "entities": entities,
            "entity_embeddings": entity_embeddings,
            "clustered_entities": clustered_entities,
            "topics": topics,
            "relations": relations
        }

if __name__ == "__main__":
    # Create the pipeline instance
    pipeline = EntityExtractionPipeline(n_clusters=2, n_topics=2)

    # Example cleaned text
    text = """
    David North, chairman of the International Editorial Board of the World Socialist Web Site, warned about the danger of a third world war.
    The Biden administration has escalated tensions with Russia in Ukraine, and its policies may lead to catastrophic global conflict.
    Julian Assange's persecution for exposing U.S. imperialism's crimes continues, despite worldwide protests.
    """

    # Process the text through the pipeline
    result = pipeline.process_text(text)

    # Print the results
    print("Entities:", result["entities"])
    print("Clustered Entities:", result["clustered_entities"])
    print("Topics:", result["topics"])
    print("Relations:", result["relations"])
