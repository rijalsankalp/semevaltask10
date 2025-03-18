import spacy
from collections import defaultdict

# Implementing the TextFilter class, which is responsible for extracting context around target entities in a sentence.
# "A dependency-based hybrid deep learning framework for target-dependent sentiment classification"
# https://doi.org/10.1016/j.patrec.2023.10.026

###Rules###

#If the target is a modifier (compound, amod, nmod), include its head word.
# If the target is a subject (nsubj or nsubjpass), include its head verb.
# If the target is an object (obj or dobj), include its head verb and the subject of that verb.
# If the target has clause children (xcomp or ccomp), include those clauses.
# If the target’s head has clause children (xcomp or ccomp), include those as well.
# If the target is connected to clause modifiers (advcl, acl, advmod), include them.
# For all content words collected from above rules, include function words directly attached to them 


class TextFilter:
    def __init__(self, use_pos_check=False):
        # load the spacy parser
        self.parser = spacy.load("en_core_web_sm")
        self.use_pos_check = use_pos_check

        # dependency relations to extract direct and indirect related words
        self.direct_relations = {
            "advmod", "nmod", "nummod", "amod", "cop", "neg", "aux", "case",
            "det", "expl", "mwe", "compound", "obj", "prt"
        }
        self.indirect_relations = {
            "xcomp", "ccomp", "acl", "advcl"
        }

        # optional POS-based filter for functional words
        self.function_pos = {"DET", "AUX", "PART", "ADP", "PRON"}

        # define role mapping for syntactic role check
        self.role_map = {
            "nsubj": "subject",
            "nsubjpass": "passive-subject",
            "dobj": "object",
            "obj": "object",
            "pobj": "prep-object",
            "iobj": "indirect-object"
        }

    def get_dependencies(self, sentence):
        # get parsed dependencies as list of tuples
        doc = self.parser(sentence)
        deps = [(t.text, t.dep_, t.head.text, t.pos_, t.i, t.head.i) for t in doc]
        return deps, doc

    def build_graph(self, deps):
        # create a dependency graph from parsed relations
        graph = defaultdict(list)
        for word, rel, head, _, child_idx, head_idx in deps:
            graph[head_idx].append((child_idx, rel))
        return graph

    def recursive_add_content(self, graph, idx, core_words):
        # recursively collect all words connected via dependency
        if idx not in core_words:
            core_words.add(idx)
        for child_idx, rel in graph[idx]:
            if rel in self.indirect_relations or rel in self.direct_relations:
                if rel not in {"conj", "cc"}:
                    self.recursive_add_content(graph, child_idx, core_words)

    def apply_rules(self, deps, graph, target_idx):
        # main rule application logic for extracting context
        core_words = set()
        function_words = set()

        def add_content(idx):
            self.recursive_add_content(graph, idx, core_words)

        def add_function(idx):
            if idx not in function_words:
                if not self.use_pos_check or deps[idx][3] in self.function_pos or deps[idx][1] in self.direct_relations:
                    function_words.add(idx)

        word, rel, head_word, pos, i, head_idx = deps[target_idx]

        # rule for modifiers and syntactic attachment
        if rel in {"compound", "amod", "nmod"}:
            add_content(head_idx)

        if rel == "nsubj" or rel == "nsubjpass":
            add_content(head_idx)

        if rel == "pobj":
            add_content(head_idx)

        # rule 2 + extended: subject/object + coordination
        related_head = None

        if rel in {"dobj", "obj"}:
            related_head = head_idx
        else:
            for _, r, _, _, idx, h_idx in deps:
                if idx == head_idx and r in {"dobj", "obj"}:
                    related_head = h_idx
                    break
            if not related_head:
                for _, r, _, _, idx, h_idx in deps:
                    if h_idx == head_idx and r in {"dobj", "obj", "appos"}:
                        related_head = h_idx
                        break

        if related_head is not None:
            add_content(related_head)
            for _, r, _, _, idx, h_idx in deps:
                if h_idx == related_head and r in {"nsubj", "nsubjpass"}:
                    add_content(idx)

        # apply clauses and verbal modifiers
        for child_idx, child_rel in graph[target_idx]:
            if child_rel in {"xcomp", "ccomp", "advcl", "acl", "advmod"}:
                add_content(child_idx)

        for child_idx, child_rel in graph[head_idx]:
            if child_rel in {"advmod", "acl"}:
                add_content(child_idx)

        for sib_idx, sib_rel in graph[head_idx]:
            if sib_rel in {"xcomp", "ccomp"}:
                add_content(sib_idx)

        add_content(target_idx)

        # collect function words attached to core
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
        # match exact token indices for each target phrase
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

        return all_filtered


if __name__ == "__main__":
    
    tests = [
        {
            "sentence": "Russia invaded Ukraine and caused a humanitarian crisis.",
            "targets": ["Russia", "Ukraine", "humanitarian crisis"]
        },
        {
            "sentence": "Ukraine was invaded by Russia, but it is fighting back bravely.",
            "targets": ["Russia", "Ukraine"]
        },
        {
            "sentence": "Russia did not expect Ukraine to resist this strongly.",
            "targets": ["Russia", "Ukraine"]
        },
        {
            "sentence": "Russia attacked Ukraine and annexed Crimea.",
            "targets": ["Russia", "Ukraine", "Crimea"]
        },
    ]

    tf = TextFilter()

    for i, test in enumerate(tests, 1):
        print(f"\nTest {i}:")
        print("Sentence:", test["sentence"])
        print("Targets:", test["targets"])
        result = tf.extract_target_context(test["sentence"], test["targets"])
        for r in result:
            print("-", r)