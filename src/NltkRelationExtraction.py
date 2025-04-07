import requests

from collections import defaultdict

import src.NltkPipe as NltkPipe

import nltk
from nltk.tree import Tree
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.parse.corenlp import CoreNLPDependencyParser


class NltkRelationExtraction(NltkPipe.NltkPipe):

    def __init__(self):
        super().__init__()
        self.dependency_parser = CoreNLPDependencyParser(url=self.url)
        
    def process_text(self, text):

        sentences = self._resolve_coreferences(text)

        relations = []

        result = list()
        for resolved_text in sentences:
            
            context = self.analyze_with_tf(resolved_text)
            relation = self.extract_relations(resolved_text)
            
            relations += relation
            
            for category in context:
                for key, value in context[category].items():
                    if value != '':
                        result.append((key, value))

        return relations, result, sentences
    
    def text_filtering(self, text, target):
        
        parse, = self.dependency_parser.raw_parse(text)
        
        target_tokens = word_tokenize(target.lower())
        target_node_ids = []
        
        for node_id, node in parse.nodes.items():
            if node_id == 0:  # Skip root
                continue
            if node['word'].lower() in target_tokens:
                target_node_ids.append(node_id)
        
        if not target_node_ids:
            return ""  
        
        core_words = set()
        func_words = set()
        object_words = set()
        entity_words = set()
        
        direct_relations = ['advmod', 'nmod', 'nummod', 'amod', 'cop', 'neg', 
                            'aux', 'case', 'det', 'expl', 'mwe', 'compound']
        
        for node_id, node in parse.nodes.items():
            if node_id == 0:  
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
    from src.LoadData import LoadData
    from src.EntityDataset import DataLoader

    base_dir = "train"
    txt_file = "subtask-1-annotations.txt"
    subdirs = "EN"
    load_data = LoadData()
    data = load_data.load_data(base_dir, txt_file, subdirs)
    data_loader = DataLoader(data, base_dir)
    filter = NltkRelationExtraction()

    counter = 0
    with open("resources/nltk_relation_context.txt", "a+") as writer:
        for text in data_loader._get_text():
            relation, result, sentences = filter.process_text(text)
            writer.writelines(f"{sentences}\n{result}\n{relation}\n\n")
            counter += 1
            if(counter > 20):
                break  
    writer.close()