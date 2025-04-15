import requests
import json
from collections import defaultdict
import src.NltkPipe as NltkPipe

class NltkOpenIE(NltkPipe.NltkPipe):
    def __init__(self):
        super().__init__()


    def process_text(self, text):

        sentences = self._resolve_coreferences(text)

        print(text)
        print("\n\n\n")
        print(sentences)
        print("\n\n\n")
        
        entities = self.get_entities(text)

        dict = defaultdict(list)
        
        for entity in entities:
            dict[entity] = [text]
        
        return dict


        triplets = list()
        for sentence in sentences:
            triplets += self.extract_openie(sentence)

        clean_triplets = list()

        for triplet in triplets:
            if triplet not in clean_triplets:
                clean_triplets.append(triplet)

        for sub, rel, obj in clean_triplets:
            if sub == obj:
                clean_triplets.remove((sub, rel, obj))
        
        triplet_dict = defaultdict(list)

        for sub, rel, obj in clean_triplets:
            if rel in {'told', 'said', 'tells', 'says'}:
                break
            for entity in entities:
                for ent in entity.split():
                    if ent in sub.split():
                        triplet_dict[entity].append(" " + sub + " " + rel + " " + obj + ".")
                        break
              
        return triplet_dict
    

    def extract_openie(self, text):
        properties = {
            "annotators": "openie",
            "outputFormat": "json"
        }

        # Send the request
        response = requests.post(self.url, params={"properties": str(properties)}, data=text)

        # Parse the response
        data = response.json()

        result = list()

        # Extract and print the OpenIE triples
        for sentence in data['sentences']:
            for triple in sentence['openie']:
                subject = triple['subject']
                relation = triple['relation']
                object = triple['object']
                #print(f"({subject}; {relation}; {object})")
                result.append((subject, relation, object))
        return result

