import requests
from collections import defaultdict
import src.NltkPipe as NltkPipe

class NltkOpenIE(NltkPipe.NltkPipe):
    def __init__(self):
        super().__init__()
        
    def process_text(self, text):

        sentences = self._resolve_coreferences(text)
        entities = self.get_entities(text)


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
    
    # def filter_triplets(self, triplets, entities_dict):
    #     filtered_triplets = dict(list())

    #     entities = set()
    #     for _, values in entities_dict.items():
    #         for value in values:
    #             entities.add(value)

    #     for sub, obj, rel in triplets:
    #         if sub in entities or obj in entities:
    #             if sub not in filtered_triplets:
    #                 filtered_triplets[sub] = [sub+" "+rel+" "+obj+"."]
    #             else:
    #                 filtered_triplets[sub].append(" " +sub+" "+rel+" "+obj+".")
    #             #filtered_triplets.append((sub, rel, obj))
        
    #     return filtered_triplets