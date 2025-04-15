import fasttext
import numpy as np
from collections import defaultdict
import fasttext.util
from torch import cosine_similarity

class FastTextConnotation:
    
    def __init__(self, model_path = 'cc.en.300.bin'):
         
        fasttext.util.download_model('en', if_exists='ignore')  
        self.model = fasttext.load_model(model_path)
        
    
    def text_to_vector(self, text):
        
        words = text.split()
        vectors = [self.model.get_word_vector(word) for word in words if word in self.model]
        
        if len(vectors) == 0:  
            return np.zeros(self.model.get_dimension())
        
        return np.mean(vectors, axis=0)

    
    def find_representative_word(self, sentence_vectors):
       
        avg_vector = np.mean(sentence_vectors, axis=0)
        
        
        max_similarity = -1
        representative_word = None
        
        
        for word in self.model.get_words():
            word_vector = self.model.get_word_vector(word)
            similarity = cosine_similarity([avg_vector], [word_vector])[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                representative_word = word
                
        return representative_word


    def get_entity_connotation(self, entity_sentences_dict):
        entity_connotation = {}
        
        for entity, sentences in entity_sentences_dict.items():
            
            sentence_vectors = np.array([self.text_to_vector(text) for text in sentences])
            
            representative_word = self.find_representative_word(sentence_vectors)
            
            entity_connotation[entity] = representative_word
        
        return entity_connotation