import nltk
import requests
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.tree import Tree
from collections import defaultdict
from SentenceSimplifier import SentenceSimplifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

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
        
        self.dependency_parser = CoreNLPDependencyParser(url=stanford_server_url)

        self.simplify_sentence = SentenceSimplifier()
        
    def process_text(self, text):


        sentences = self._resolve_coreferences(text)

        relations = []

        result = list()
        for resolved_text in sentences:
            resolved_text = self.simplify_sentence._get_simplified_text(resolved_text)
            context = self.analyze_with_tf(resolved_text)
            relation = self.extract_relations(resolved_text)
            
            relations += relation
            
            for category in context:
                for key, value in context[category].items():
                    if value != '':
                        result.append((key, value))

        return relations, result
        
        
    def _resolve_coreferences(self, text):

        # If text is empty, return it as is
        if not text.strip():
            return text
            
        params = {"annotators": "coref", "outputFormat": "json"}
        
        response = requests.post(self.url, params=params, data=text.encode('utf-8'), timeout=30)
        
        ann = response.json()
        
        sentences = [sentence['tokens'][0]['originalText'] + ''.join(' ' + token['originalText'] 
                        if not token['after'].startswith("'") and token['originalText'] not in [',', '.', '?', '!', ':', ';'] 
                        else token['originalText'] for token in sentence['tokens'][1:]) 
                        for sentence in ann['sentences']]
            
        resolved_sentences = sentences.copy()
        
        # Process coreference chains if they exist
        if 'corefs' in ann:
            for coref_chain_id, mentions in ann['corefs'].items():
                # Find the representative mention (prefer a non-pronominal one)
                rep_mention = None
                for mention in mentions:
                    if mention.get('type') != 'PRONOMINAL':  # Prefer non-pronouns
                        rep_mention = mention
                        break

                # If no non-pronoun mention is found, use the first mention
                if rep_mention is None and mentions:
                    rep_mention = mentions[0]  # Use the first mention in the coref chain

                if rep_mention is None:
                    continue  # Skip if still not found

                rep_text = rep_mention.get('text', '')  # Get the actual name
                
                # Replace all other mentions with the representative mention
                for mention in mentions:
                    if (mention.get('sentNum') != rep_mention.get('sentNum') or 
                        mention.get('startIndex') != rep_mention.get('startIndex')):
                        
                        sent_idx = mention.get('sentNum', 0) - 1
                        if sent_idx < 0 or sent_idx >= len(resolved_sentences):
                            continue
                            
                        # Get the original sentence
                        original_sent = resolved_sentences[sent_idx]
                        
                        # Build token mapping from character positions
                        if sent_idx >= len(ann['sentences']):
                            continue
                            
                        tokens = ann['sentences'][sent_idx].get('tokens', [])
                        token_map = []
                        for token in tokens:
                            # Create span info with character start/end positions
                            char_start = token.get('characterOffsetBegin', -1)
                            char_end = token.get('characterOffsetEnd', -1)
                            if char_start >= 0 and char_end >= 0:
                                token_map.append((char_start, char_end, token.get('originalText', '')))
                        
                        # Find mention span in the sentence
                        start_token_idx = mention.get('startIndex', 0) - 1
                        end_token_idx = mention.get('endIndex', 0) - 2
                        
                        if start_token_idx < 0 or end_token_idx >= len(token_map) or start_token_idx > end_token_idx:
                            continue
                            
                        # Find character positions
                        char_start = token_map[start_token_idx][0]
                        char_end = token_map[end_token_idx][1]
                        
                        # Replace the mention with representative text
                        # Only replace if the mention is a pronoun or shorter than the representative
                        mention_text = mention.get('text', '')
                        if mention.get('type') == 'PRONOMINAL' or len(mention_text) < len(rep_text):
                            # Find the exact position in the sentence text
                            sentence_text = original_sent
                            start_pos = sentence_text.find(mention_text, max(0, char_start - 10))
                            
                            if start_pos >= 0:
                                # Replace the mention with the representative text
                                new_sentence = sentence_text[:start_pos] + rep_text + sentence_text[start_pos + len(mention_text):]
                                resolved_sentences[sent_idx] = new_sentence
        
        return resolved_sentences


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
        
        return entities
    
    def text_filtering(self, text, target):
        
        # Get dependency parse
        parse, = self.dependency_parser.raw_parse(text)
        
        # Find target in dependency graph (handle multi-word targets)
        target_tokens = word_tokenize(target.lower())
        target_node_ids = []
        
        for node_id, node in parse.nodes.items():
            if node_id == 0:  # Skip root
                continue
            if node['word'].lower() in target_tokens:
                target_node_ids.append(node_id)
        
        if not target_node_ids:
            return ""  # Target not found
        
        # Initialize sets for content and function words
        core_words = set()
        func_words = set()
        object_words = set()
        entity_words = set()
        
        # Direct relations for Rule 6
        direct_relations = ['advmod', 'nmod', 'nummod', 'amod', 'cop', 'neg', 
                            'aux', 'case', 'det', 'expl', 'mwe', 'compound']
        
        # Apply Rules 1-5 to find content words
        for node_id, node in parse.nodes.items():
            if node_id == 0:  # Skip root
                continue
            
            for target_id in target_node_ids:
                # Rule 1: If target is subject, add its predicate
                if 'nsubj' in node['deps'] and target_id in node['deps']['nsubj']:
                    core_words.add(node_id)
                
                # Rule 2: If target is object, add its governor
                for obj_rel in ['obj', 'dobj', 'iobj']:
                    if obj_rel in node['deps'] and target_id in node['deps'][obj_rel]:
                        core_words.add(node_id)
                        object_words.add(node_id)
                
                # Rule 3: If target is subject and its content word has xcomp
                if 'nsubj' in node['deps'] and target_id in node['deps']['nsubj']:
                    for rel in ['xcomp', 'ccomp']:
                        if rel in node['deps']:
                            for pred_id in node['deps'][rel]:
                                core_words.add(pred_id)
                
                # Rule 4: Target's content word with object that has ccomp
                if 'nsubj' in node['deps'] and target_id in node['deps']['nsubj']:
                    if 'obj' in node['deps']:
                        for obj_id in node['deps']['obj']:
                            core_words.add(obj_id)
                            object_words.add(obj_id)
                            if obj_id in parse.nodes and 'ccomp' in parse.nodes[obj_id]['deps']:
                                for ccomp_id in parse.nodes[obj_id]['deps']['ccomp']:
                                    core_words.add(ccomp_id)
                
                # Rule 5: If target is subject with advcl or acl modifiers
                if 'nsubj' in node['deps'] and target_id in node['deps']['nsubj']:
                    for mod_rel in ['advcl', 'acl']:
                        if mod_rel in node['deps']:
                            for mod_id in node['deps'][mod_rel]:
                                core_words.add(mod_id)
        
        # Rule 6: Find function words for each content word
        def find_function_words(word_id, visited=None):
            if visited is None:
                visited = set()
            
            if word_id in visited:
                return
            
            visited.add(word_id)
            
            if word_id not in parse.nodes:
                return
                
            for rel, deps in parse.nodes[word_id]['deps'].items():
                if rel in direct_relations:
                    for dep_id in deps:
                        if dep_id not in func_words and dep_id not in core_words:
                            func_words.add(dep_id)
                            find_function_words(dep_id, visited)
        
        # Process each content word to find its function words
        for word_id in core_words:
            find_function_words(word_id)
        
        # Extract entities separately, excluding other entities unless directly affected
        for node_id, node in parse.nodes.items():
            if node['word'] and node['tag'] in ['NNP', 'NNPS']:  # Proper nouns
                if node_id in core_words or node_id in object_words:
                    entity_words.add(node_id)
        
        # Convert node IDs to words and reconstruct context
        relevant_nodes = sorted(list(core_words.union(func_words).union(object_words).union(entity_words)))
        stop_words = set(stopwords.words('english'))
        filtered_words = [parse.nodes[node_id]['word'] for node_id in relevant_nodes if node_id in parse.nodes and parse.nodes[node_id]['word'].lower() not in stop_words]
        
        # Add target's own function words if target itself is a content word
        for target_id in target_node_ids:
            find_function_words(target_id)
        
        return " ".join(filtered_words)

    def analyze_with_tf(self, text):
        
        entities = self.get_entities(text)
        
        results = {}
        for entity_type, entity_list in entities.items():
            results[entity_type] = {}
            for entity in entity_list:
                filtered_context = self.text_filtering(text, entity)
                results[entity_type][entity] = filtered_context
        
        return results

    def extract_relations(self, text):
        relations = defaultdict(list)
        main_author = "Main Author"
        current_speaker = None
        
        
        sentences = nltk.sent_tokenize(text)
        
        for sentence in sentences:
            
            response = requests.post(self.url, params={'annotators': 'depparse', 'outputFormat': 'json'}, data=sentence.encode('utf-8'))
            response_data = response.json()
            
            
            dependencies = response_data['sentences'][0]['basicDependencies']
            
            subject, verb, obj = None, None, None

            for dep in dependencies:
                dep_type = dep['dep']
                gov_word = dep['governorGloss']
                dep_word = dep['dependentGloss']
                
                # Active and passive subject extraction (nsubj, nsubjpass)
                if dep_type in ['nsubj', 'nsubjpass']:  # Subject or passive subject
                    subject = dep_word
                    verb = gov_word
                    
                    # Find the object
                    for sub_dep in dependencies:
                        if sub_dep['governor'] == dep['dependent'] and sub_dep['dep'] in ['dobj', 'prep']:
                            obj = sub_dep['dependentGloss']
                            break
                    
                    # Handle passive voice with nsubjpass
                    if dep_type == 'nsubjpass':  
                        relation_sentence = f"{subject} {verb} {obj}."
                    else:
                        # In active voice, subject verb object
                        relation_sentence = f"{subject} {verb} {obj}" if obj else f"{subject} {verb}."
                    
                    # Append relation to the correct speaker or author
                    if current_speaker:
                        relations[current_speaker].append(relation_sentence)
                    else:
                        relations[main_author].append(relation_sentence)
                
                # Handle direct objects and prep objects (dobj, prep)
                elif dep_type == 'dobj' or dep_type == 'prep':
                    obj = dep_word
                    if subject and verb:
                        relation_sentence = f"{subject} {verb} {obj}."
                        if current_speaker:
                            relations[current_speaker].append(relation_sentence)
                        else:
                            relations[main_author].append(relation_sentence)

            # Handle speaker identification using "said" or "told"
            if 'said' in sentence.lower() or 'told' in sentence.lower():
                words = nltk.word_tokenize(sentence)
                for i, word in enumerate(words):
                    if word.lower() in ['said', 'told']:
                        speaker = words[i-1]  # Assuming the speaker is just before 'said' or 'told'
                        if speaker:
                            current_speaker = speaker
                            break
            
            # Reset current speaker after processing sentence
            current_speaker = None
        
        result = []
        for key, value in dict(relations).items():
            result.append(value)

        return result


if __name__ == "__main__":
    from LoadData import LoadData
    from EntityDataset import DataLoader

    base_dir = "train"
    txt_file = "subtask-1-annotations.txt"
    subdirs = "EN"
    load_data = LoadData()
    data = load_data.load_data(base_dir, txt_file, subdirs)
    data_loader = DataLoader(data, base_dir)
    filter = NltkPipe()

    for text in data_loader._get_text():
        print(filter.process_text(text))
        print("\n\n\n\n")
        break