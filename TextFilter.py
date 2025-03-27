import spacy
from spacy import displacy
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Implementing the TextFilter class, which is responsible for extracting context around target entities in a sentence.
# "A dependency-based hybrid deep learning framework for target-dependent sentiment classification"
# https://doi.org/10.1016/j.patrec.2023.10.026

###Rules###

# Rule 0: If the target is a modifier (compound, amod, nmod), include its head word.
# Rule 1: If the target is a subject (nsubj or nsubjpass), include its head verb.
# Rule 2: If the target is an object (obj or dobj), include its head verb and the subject of that verb.
# Rule 3: If the target has clause children (xcomp or ccomp), include those clauses.
# Rule 4: If the target’s head has clause children (xcomp or ccomp), include those as well.
# Rule 5: If the target is connected to clause modifiers (advcl, acl, advmod), include them.
# Rule 6: For all content words collected from above rules, include function words directly attached to them. 
# Additionally: trace prepositional object (pobj) to its verb via preposition.

class TextFilter:
    def __init__(self, use_pos_check=False):
        self.parser = spacy.load("en_core_web_trf")
        self.use_pos_check = use_pos_check

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
        self.tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-v1_1-base-wikisplit")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("flax-community/t5-v1_1-base-wikisplit")

    def get_dependencies(self, sentence):
        doc = self.parser(sentence)
        displacy.serve(doc, style="dep", auto_select_port=True)
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
                if not self.use_pos_check or deps[idx][3] in self.function_pos or deps[idx][1] in self.direct_relations:
                    function_words.add(idx)

        word, rel, head_word, pos, i, head_idx = deps[target_idx]

        # Rule 0: If target is a modifier, add its head
        if rel in {"compound", "amod", "nmod"}:
            add_content(head_idx)

        # Rule 1: target is subject or passive subject → add head verb
        if rel in {"nsubj", "nsubjpass"}:
            add_content(head_idx)

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
                    # Fallback: add prep’s head anyway (usually a verb)
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
            target_doc = self.parser(target)
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
        
        sample_tokenized = self.tokenizer(sentence, return_tensors="pt")

        answer = self.model.generate(sample_tokenized['input_ids'], attention_mask = sample_tokenized['attention_mask'], max_length=256, num_beams=5)
        sentence = self.tokenizer.decode(answer[0], skip_special_tokens=True)

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


if __name__ == "__main__":
    from LoadData import LoadData
    from EntityDataset import DataLoader

    base_dir = "train"
    txt_file = "subtask-1-annotations.txt"
    subdirs = "EN"
    load_data = LoadData()
    data = load_data.load_data(base_dir, txt_file, subdirs)
    data_loader = DataLoader(data, base_dir)

    filter = TextFilter()
    
    # for target, sentences in data_loader.yield_entity_sent():
        
    #     for sentence in sentences:
    #         print(target)
    #         print(filter.extract_target_context(sentence, target))
        
    #     break

    entities = data_loader.yield_NER_sentences()
    for ent in entities:
        for entity, sentence in ent:
            print(filter.extract_target_context(sentence, entity))

        break
    