####  Need to install dependencies for coreferee
### en_core_web_lg is needed even if using en_core_web_trf

# pip install spacy
# python -m spacy download en_core_web_lg
# python -m spacy download en_core_web_trf
# pip install coreferee
# python -m coreferee install en


import spacy
from spacy import displacy
import numpy as np
from LoadData import LoadData
from pathlib import Path
from itertools import chain

np.random.seed(0)


def get_role_labels():
    """
    Returns main roles and fine-grained roles.
    """
    # all_main_roles = ["Protagonist", "Antagonist", "Innocent"]
    # all_sub_roles = ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous", "Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot", "Forgotten", "Exploited", "Victim", "Scapegoat"]
    sub_dict = {
        "Protagonist":["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
        "Innocent":["Forgotten", "Exploited", "Victim", "Scapegoat"],
        "Antagonist":["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"]
    }

    fg_roles = list()
    sub_dict['main_role_list'] = list()
    main_roles_order = ['Protagonist', 'Innocent', 'Antagonist']
    for main_role in main_roles_order:
        fg_roles.extend(sub_dict[main_role])
        sub_dict['main_role_list'].extend([main_role]*len(sub_dict[main_role]))  # creates a list of (repeated) main role labels, redundant, but works for simplicty
     
    sub_dict['role_list'] = fg_roles  # list of fine-grained roles
    fg_to_main = dict()
    # create a dict mapping fine-grained to main role.
    for main_role in sub_dict:
        for fg_role in sub_dict[main_role]:
            fg_to_main[fg_role] = main_role

    return sub_dict

def load_text_file(path):
    """
    Load document in `path` and returns a raw string.
    """

    with open(path) as fin:
        text = fin.read()
    return text

def load_document(base_dir, lang, article_id):
    """
    Load a raw document for given path and `article_id`.
    """

    path = Path(base_dir).joinpath(lang, "raw-documents", article_id)
    text = load_text_file(path)
    return text


def visualize_dependency_parsing(sents, port=5005):
    """
    Visualize the dependency parsing of input sentences.
    More info: https://spacy.io/usage/visualizers/

    """
    displacy.serve(sents, style='dep', port=port)

def find_entity_sentence(doc, start_offset, end_offset):
    """
    Returns the index of the sentence containing the entity between given indices.
    """

    for i, sent in enumerate(doc.sents):
        if sent[0].idx <= start_offset and sent[-1].idx >= end_offset:
            return i


def get_entity_coref_sentences(doc, doc_labels):
    """
    Returns the sentences containing the entity and co-references.
    """
    # Find the actual entity span form the doc
    #print(f"doc text = {doc.text}")
    start_offset = doc_labels['start_offset']
    end_offset = doc_labels['end_offset']

    # Adjust start_offset to the left if it's in the middle of a word
    while start_offset > 0 and not doc.text[start_offset - 1] in {' ', '\n', ',', '-', '.'}:
        start_offset -= 1

    # Adjust end_offset to the right if it's in the middle of a word
    while end_offset < len(doc.text) and not doc.text[end_offset] in {' ', '\n', ',', '-', '.'}:
        end_offset += 1


    # Get the entity span from start to end offsets
    entity_span = doc.char_span(start_offset, end_offset, 
                                label="target_ent",
                                alignment_mode='expand')   # using 'expand' alignment because some labeled indices do not match spacy's char index
    

    # Make dictionary of co-reference chains
    entity_corefs = list()
    for g in doc._.coref_chains.chains:
        #print(f"{g.index=}, {g.mentions}, {g.most_specific_mention_index=}")
        for mention_id in g.mentions:
            # if entity_span.root.i in mention_id:
            if any(token.i in mention_id for token in entity_span):
                entity_corefs.extend(g.mentions)
                continue
    # Unpack coref list
    entity_corefs = list(chain.from_iterable(entity_corefs))
    coref_sents = list({doc[tid].sent for tid in entity_corefs})  # Use a set to avoid repeated sentences
    coref_sents.append(entity_span.root.sent)  # Add the sentence containing the entity

    
    return entity_span, entity_corefs, coref_sents


def iterate_documents(base_dir, labels, subdirs):
    """
    Returns a generator that reads all documents given in input labels_file.
    Yields a tuple (doc, ent_span, ent_corefs, ent_sents).
    doc is the SpaCy document.
    ent_span is the span containing the target entity.
    ent_corefs are co-references to the entity.
    ent_sents are the sentences with co-references to the target.
    """

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe('coreferee')

    for row_id in range(len(labels)):
        doc_labels = labels.iloc[row_id]
        text = load_document(base_dir, 'EN', doc_labels['article_id'])

        doc = nlp(text)
        # Retrieve entity info and coreference
        ent_span, ent_corefs, ent_sents = get_entity_coref_sentences(doc, doc_labels)
        
        yield doc, ent_span, ent_corefs, ent_sents, doc_labels


if __name__ == "__main__":
    base_dir = "train"
    labels_file = "subtask-1-annotations.txt"
    subdirs = ["EN"]

    ld = LoadData()
    labels = ld.load_data(base_dir, labels_file, subdirs)
    # Cast these columns as int
    labels['start_offset'] = labels['start_offset'].astype(int)
    labels['end_offset'] = labels['end_offset'].astype(int)

    for doc, ent_span, ent_corefs, ent_sents, doc_labels in iterate_documents(base_dir, labels, subdirs):
        print(ent_span)
        for tid in ent_corefs:
            tk = doc[tid]
            print(f"{tk} | {tk.head=} | {list(t for t in tk.children)}")

        visualize_dependency_parsing(ent_sents)

