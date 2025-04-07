import spacy
from collections import defaultdict
from src.SentenceSimplifier import SentenceSimplifier

class SpacyPipe:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.nlp.add_pipe("coreferee")

        self.direct_relations = {
            "advmod", "nmod", "nummod", "amod", "cop", "neg", "aux", "case",
            "det", "expl", "mwe", "compound", "obj", "prt"
        }
        self.indirect_relations = {
            "xcomp", "ccomp", "acl", "advcl"
        }

        self.function_pos = {"DET", "AUX", "PART", "ADP", "PRON"}

        self.role_map = {
            "nsubj": "subject",
            "nsubjpass": "passive-subject",
            "dobj": "object",
            "obj": "object",
            "pobj": "prep-object",
            "iobj": "indirect-object"
        }

        self.simplify_sentence = SentenceSimplifier()
    def process_text(self, text):
        """
        Main pipeline function that processes text through all steps:
        1. Coreference resolution
        2. Entity identification
        3. Sentence simplification
        4. Context extraction for each entity
        """
        # Step 1: Coreference resolution
        resolved_text = self._coreference_resolution(text)
        
        # Step 2: Entity identification
        entity_sentence_pairs = self._entity_extraction(resolved_text)
        
        results = []
        
        # Step 3 & 4: For each entity, simplify its sentence and extract context
        for entity, sentence in entity_sentence_pairs:
            # Step 3: Simplify the sentence
            simplified_sentence = self.simplify_sentence._get_simplified_text(sentence)
            
            # Step 4: Extract context using the entity and simplified sentence
            contexts = self.extract_target_context(simplified_sentence, entity)
            results.extend(contexts)
        
        return results
    
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
    
    def _entity_extraction(self, text):
        """
        Extract entities and their containing sentences.
        Returns a list of (entity, sentence) pairs.
        """
        doc = self.nlp(text)
        
        entity_sentence_pairs = []

        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "LOC"]:
                sentence = ent.sent.text  # Get the sentence containing the entity
                entity_sentence_pairs.append((ent.text, sentence))
        
        return entity_sentence_pairs
    
    def get_dependencies(self, sentence):
        doc = self.nlp(sentence)
        deps = [(t.text, t.dep_, t.head.text, t.pos_, t.i, t.head.i) for t in doc]
        return deps, doc

    def build_graph(self, deps):
        graph = defaultdict(list)
        for word, rel, head, _, child_idx, head_idx in deps:
            graph[head_idx].append((child_idx, rel))
        return graph

    def recursive_add_content(self, graph, idx, core_words):
        if idx not in core_words:
            core_words.add(idx)
        for child_idx, rel in graph[idx]:
            if rel in self.indirect_relations or rel in self.direct_relations or rel in {"conj", "cc"}:
                self.recursive_add_content(graph, child_idx, core_words)

    def apply_rules(self, deps, graph, target_idx):
        core_words = set()
        function_words = set()

        def add_content(idx):
            self.recursive_add_content(graph, idx, core_words)

        def add_function(idx):
            if idx not in function_words:
                if deps[idx][3] in self.function_pos or deps[idx][1] in self.direct_relations:
                    function_words.add(idx)

        word, rel, head_word, pos, i, head_idx = deps[target_idx]

        # Rule 0: If target is a modifier, add its head
        if rel in {"compound", "amod", "nmod"}:
            add_content(head_idx)

        # Rule 1: target is subject or passive subject → add head verb
        if rel in {"nsubj", "nsubjpass"}:
            add_content(head_idx)
            core_words.add(target_idx)
        
        # Rule 2: target is object → include verb + subject
        if rel in {"dobj", "obj"}:
            add_content(head_idx)
            for _, r, _, _, idx, h_idx in deps:
                if h_idx == head_idx and r in {"nsubj", "nsubjpass"}:
                    add_content(idx)

        # Rule 3: subject/passive → head has xcomp/ccomp/advcl
        if rel in {"nsubj", "nsubjpass"}:
            for child_idx, child_rel in graph[head_idx]:
                if child_rel in {"xcomp", "ccomp", "advcl"}:
                    add_content(child_idx)

        # Rule 4: head has object → that object has xcomp/ccomp children
        for sib_idx, sib_rel in graph[head_idx]:
            if sib_rel == "obj":
                for child_idx, child_rel in graph[sib_idx]:
                    if child_rel in {"xcomp", "ccomp"}:
                        add_content(child_idx)

        # Rule 5: subject/passive → has acl or advcl modifiers
        if rel in {"nsubj", "nsubjpass"}:
            for child_idx, child_rel in graph[target_idx]:
                if child_rel in {"advcl", "acl"}:
                    add_content(child_idx)

        # Rule Extension: pobj → trace to verb via preposition
        if rel == "pobj":
            # Step 1: Find the preposition that governs this pobj
            for idx, dep, head, _, i, h_idx in deps:
                if i == head_idx and dep == "prep":
                    add_content(i)  # add the preposition (e.g., "by")

                    # Step 2: Try tracing that prep to its verb head
                    for idx2, dep2, _, _, _, h2_idx in deps:
                        if idx2 == h_idx and dep2 in {"agent", "prep", "obl", "nmod"}:
                            add_content(h2_idx)
                    # Fallback: add prep's head anyway (usually a verb)
                    add_content(h_idx)

        if rel in {"pobj"}:  # Prepositional object
            add_content(head_idx)
        if rel == "agent":  # Passive voice
            add_content(head_idx)

        # Always include the target word
        add_content(target_idx)

        # Rule 6: Add function words related to content words
        visited = set()
        queue = list(core_words)
        while queue:
            current = queue.pop()
            for child_idx, rel in graph[current]:
                if rel in self.direct_relations and child_idx not in visited:
                    visited.add(child_idx)
                    add_function(child_idx)
                    queue.append(child_idx)

        return core_words, function_words

    def match_target_indices(self, deps, doc, targets):
        target_indices = []
        seen_spans = set()
        token_texts = [t.text.lower() for t in doc]
        doc_len = len(token_texts)

        for target in targets:
            target_doc = self.nlp(target)
            target_tokens = [t.text.lower() for t in target_doc]
            t_len = len(target_tokens)

            for i in range(doc_len - t_len + 1):
                if token_texts[i:i + t_len] == target_tokens:
                    idxs = tuple(range(i, i + t_len))
                    if idxs not in seen_spans:
                        seen_spans.add(idxs)
                        target_indices.append((target, list(idxs)))
                    break

        return target_indices

    def extract_target_context(self, sentence, targets):
        if isinstance(targets, str):
            targets = [targets]

        deps, doc = self.get_dependencies(sentence)
        deps_graph = self.build_graph(deps)
        all_filtered = []

        target_instances = self.match_target_indices(deps, doc, targets)
        if not target_instances:
            return [f"None of the targets found in: {sentence}"]

        for target_text, indices in target_instances:
            span_tokens = [doc[i] for i in indices]
            head_token = min(span_tokens, key=lambda t: t.head.i if t.head.i in indices else t.i)
            head_idx = head_token.i

            core, func = self.apply_rules(deps, deps_graph, head_idx)
            selected = core | func
            selected.update(indices)
            word_ordered = sorted(selected)
            words = [deps[i][0] for i in word_ordered]
            role = self.role_map.get(head_token.dep_, "other")
            all_filtered.append(f"[{role}] {' '.join(words)}")

        return [f"{sentence}: {all_filtered}"]
    
    def extract_relations(self, text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        relations = defaultdict(list)
        main_author = "Main Author"
        current_speaker = None
        
        sentences = [sent.text for sent in doc.sents]
        
        for sentence in sentences:
            sent_doc = nlp(sentence)
            words = [token.text.lower() for token in sent_doc]
            
            if "said" in words:
                for token in sent_doc:
                    if token.text.lower() == "said":
                        speaker = [child.text for child in token.lefts if child.ent_type_ == "PERSON"]
                        if speaker:
                            current_speaker = speaker[0]
                            sentence = sentence.replace(token.text, "", 1).strip()
                        break
            
            processed_sent = nlp(sentence)
            for token in processed_sent:
                if token.dep_ in ("ROOT", "acl") and token.pos_ == "VERB":
                    subject = [w.text for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                    obj = [w.text for w in token.rights if w.dep_ in ("dobj", "attr", "prep")]
                    
                    if subject and obj:
                        subj = subject[0]
                        verb = token.text
                        obj_text = " ".join(obj)
                        relation_sentence = f"{subj} {verb} {obj_text}."
                        
                        if current_speaker:
                            relations[current_speaker].append(relation_sentence)
                        else:
                            relations[main_author].append(relation_sentence)
        
        return dict(relations)

if __name__ == "__main__":
    from src.LoadData import LoadData
    from src.EntityDataset import DataLoader

    base_dir = "train"
    txt_file = "subtask-1-annotations.txt"
    subdirs = "EN"
    load_data = LoadData()
    data = load_data.load_data(base_dir, txt_file, subdirs)
    data_loader = DataLoader(data, base_dir)
    filter = SpacyPipe()

    for text in data_loader._get_text():
        print(filter.process_text(text))
        break
    
