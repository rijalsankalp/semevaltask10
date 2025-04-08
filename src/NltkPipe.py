import nltk
import requests
import json




# Ensure necessary NLTK packages are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('maxent_ne_chunker_tab')
    nltk.download('words')

class NltkPipe:

    def __init__(self, stanford_server_url='http://localhost:9000'):

        self.url = stanford_server_url

        self.pronouns = {"PRP", "PRP$", "WP", "WP$"}
        self.nouns = {"NN", "NNS", "NNP", "NNPS"}
        

    def _resolve_coreferences(self, text):

        # Get coreference info from Stanford CoreNLP
        props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,coref',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }
        response = requests.post(self.url, params={'properties': str(props)}, data=text.encode('utf-8'))
        result = response.json()

        # Extract sentences and tokens
        sentences = result['sentences']
        tokenized_text = [[token['word'] for token in sentence['tokens']] for sentence in sentences]

        # Get coreference chains
        if 'corefs' not in result:
            return [' '.join([' '.join(sent) for sent in tokenized_text])]  # fallback if no coref

        for chain in result['corefs'].values():
            if len(chain) < 2:
                continue
            representative = chain[0]  # usually the most informative noun phrase

            for mention in chain[1:]:
                if mention['isRepresentativeMention']:
                    continue

                sent_index = mention['sentNum'] - 1
                start = mention['startIndex'] - 1
                end = mention['endIndex'] - 1

                # Check POS tags to confirm it's a pronoun
                mention_words = tokenized_text[sent_index][start:end]
                mention_tags = nltk.pos_tag(mention_words)

                if all(tag[1] in self.pronouns for tag in mention_tags):
                    rep_words = representative['text'].split()
                    tokenized_text[sent_index][start:end] = rep_words

        # Reconstruct sentences
        rephrased_text = [' '.join(sentence) for sentence in tokenized_text]
        
        return rephrased_text


    def get_entities(self, text):
        
        params = {"annotators": "ner", "outputFormat": "json"}
        response = requests.post(self.url, params=params, data=text.encode('utf-8'))
        result = json.loads(response.text)
        
        # Process entity mentions to handle multi-word entities
        entities = {"PERSON": [], "ORGANIZATION": [], "LOCATION": []}
        current_entity = {"type": None, "text": []}
        
        for sentence in result["sentences"]:
            for i, token in enumerate(sentence["tokens"]):
                ner_tag = token.get("ner", "O")
                
                # Start of a new entity or continuation
                if ner_tag in entities:
                    if current_entity["type"] != ner_tag:
                        # Save previous entity if exists
                        if current_entity["type"] and current_entity["text"]:
                            full_entity = " ".join(current_entity["text"])
                            if full_entity not in entities[current_entity["type"]]:
                                entities[current_entity["type"]].append(full_entity)
                        # Start new entity
                        current_entity = {"type": ner_tag, "text": [token["word"]]}
                    else:
                        # Continue current entity
                        current_entity["text"].append(token["word"])
                else:
                    # End of entity
                    if current_entity["type"] and current_entity["text"]:
                        full_entity = " ".join(current_entity["text"])
                        if full_entity not in entities[current_entity["type"]]:
                            entities[current_entity["type"]].append(full_entity)
                        current_entity = {"type": None, "text": []}
        
        # Add final entity if exists
        if current_entity["type"] and current_entity["text"]:
            full_entity = " ".join(current_entity["text"])
            if full_entity not in entities[current_entity["type"]]:
                entities[current_entity["type"]].append(full_entity)
        
        entities =  set(item for sublist in entities.values() for item in sublist)

        return entities
        