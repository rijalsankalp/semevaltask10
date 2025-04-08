import requests
import json
from collections import defaultdict
import src.NltkPipe as NltkPipe

class NltkOpenIE(NltkPipe.NltkPipe):
    def __init__(self):
        super().__init__()


    def process_dependency(self, text):

        return self.extract_dependency_triplets(text)

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

    def extract_dependency_triplets(self, text):
        
        props = {
            'annotators': 'tokenize,ssplit,pos,lemma,depparse,ner,coref',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }
        response = requests.post(
            self.url,
            params={'properties': json.dumps(props)},
            data=text.encode('utf-8'),
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )

        if response.status_code != 200:
            raise ConnectionError(f"CoreNLP server error {response.status_code}: {response.text}")
        
        result = response.json()

        
        sentences = result['sentences']
        tokenized_text = [[token['word'] for token in s['tokens']] for s in sentences]
        if 'corefs' in result:
            for chain in result['corefs'].values():
                rep_mention = chain[0]
                for mention in chain[1:]:
                    if mention['isRepresentativeMention']:
                        continue
                    sent_idx = mention['sentNum'] - 1
                    start = mention['startIndex'] - 1
                    end = mention['endIndex'] - 1
                    tokenized_text[sent_idx][start:end] = rep_mention['text'].split()
            resolved_sentences = [' '.join(toks) for toks in tokenized_text]
        else:
            resolved_sentences = [' '.join(toks) for toks in tokenized_text]

        
        triples = []

        for s_idx, sentence in enumerate(result['sentences']):
            dependencies = sentence['basicDependencies']
            tokens = sentence['tokens']
            subj_map = {}
            verb_map = {}

            for dep in dependencies:
                dep_type = dep['dep']
                gov_idx = dep['governor'] - 1
                dep_idx = dep['dependent'] - 1

                if gov_idx < 0 or dep_idx < 0:
                    continue

                governor_word = tokens[gov_idx]['word']
                dependent_word = tokens[dep_idx]['word']

                if dep_type in {'nsubj', 'nsubjpass'}:
                    subj_map[gov_idx] = dep_idx
                elif dep_type == 'dobj':
                    verb_map[gov_idx] = dep_idx

            for verb_idx in set(subj_map.keys()) & set(verb_map.keys()):
                subj = tokens[subj_map[verb_idx]]['word']
                verb = tokens[verb_idx]['word']
                obj = tokens[verb_map[verb_idx]]['word']
                triples.append((subj, verb, obj))

        return {
            "coref_resolved_text": resolved_sentences,
            "triplets": triples
        }
