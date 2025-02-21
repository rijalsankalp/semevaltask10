
import numpy as np
from pathlib import Path
from spacy.lang.en.stop_words import STOP_WORDS
import re

from dependency_parsing import iterate_documents, get_role_labels

from gensim.models import KeyedVectors

from LoadData import LoadData


def load_glove_vectors(path):
    """
    Load GloVe embedding vectors in `path`.
    """
    model = KeyedVectors.load_word2vec_format(path, binary=False, no_header=True)

    return model


def get_role_vectors(model, role_words, n_grams='last'):
    """
    For each role word, extract the respective word vector.
    If a role word is an n-gram - return the average vector of the non-stopwords.
    """

    x = list()
    
    for r in role_words:
        tokens = re.split("\s+", r.lower())
        tokens = [t for t in tokens if t not in STOP_WORDS and t in model]
        if n_grams == 'average':
            v = np.mean([model[t] for t in tokens], axis=0)  # take the average vector of n-grams
        elif n_grams == 'last':
            v = np.array(model[tokens[-1]])  # use the last token in the n-gram
        x.append(v)
    
    x = np.row_stack(x)  # Creates an n x d matrix of `n` fine-grained roles and `d`-dimensional vectors
    print(x.shape)
    return x
    

def get_semantic_sim(word, model, role_vectors):
    """
    Returns a vector of cosine similarities between input `word` and role_words using `model`.

    Args:
        role_vectors (2D-array) : 2D matrix of word vectors (n x d)
    """

    # cosine similarity: np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
    x = np.dot(role_vectors, model[word])/(np.linalg.norm(model[word])*np.linalg.norm(role_vectors, axis=1))
    # Apply softmax to turn the similarities into a probability distribution
    # x = np.exp(x)/np.sum(np.exp(x))
    x = np.exp(x)

    return x


def process_sentence(sent, emb_model, role_vectors,
                     ent_span,
                     doc, 
                     c_window=None,
                     reduce='max'):
    """
    Process sentence and the words therein.

    Args:
        sent (spacy.sentence): The parsed sentences.
        emb_model (gensim.word2vecModel): Word embedding.
        role_vectors (np.array): Numpy word vectors for the roles.
        ent_span (spacy.Span): The span containing the entity.
        c_window (int): The size of the context window around the entity.
        reduce (str): Reduction method {'max', 'average', ...}
    """

    x_stack = list()
    x_var = list()

    if c_window is None:
        context_words = sent
    else:
        ent_i = ent_span.root.i
        context_words = [t for t in doc[max(0, ent_i-c_window):ent_i]] + [t for t in doc[ent_i+1:ent_i+1+c_window]]  # get both sides of the entity span
    #print(f"{context_words=}")
    for token in context_words:
        if token.pos_ in {'NOUN', 'VERB', 'ADJ'}:
            if token.text.lower() in emb_model:
                x = get_semantic_sim(token.text.lower(), emb_model, role_vectors)
                x_stack.append(x)
                x_var.append(np.var(x))  # save the in-vector variance to use as weight. Vectors with high variance means some of the classes potentially dominate.
    
    #instance when there no word in the context window
    if not x_stack:
        return np.zeros(role_vectors.shape[0])
    x_stack = np.row_stack(x_stack)

    if reduce == 'max':
        x = np.max(x_stack, axis=0)
    elif reduce == 'average':
        x = np.mean(x_stack, axis=0)
    elif reduce == 'w_average':
        x = np.average(x_stack, axis=0, weights=x_var)

    return x

def process_documents(base_dir, labels, subdirs, role_labels, emb_model, role_vectors, c_window):
    max_match_count = 0
    avg_match_count = 0
    total_instances = 0
    empty_instances = 0

    for doc, ent_span, ent_corefs, ent_sents, doc_labels in iterate_documents(base_dir, labels, subdirs):
        print("=====", ent_span)

        x_sents = list()
        for sent in ent_sents:
            x = process_sentence(sent, emb_model, role_vectors, ent_span, doc, c_window)
            x_sents.append([x])
        

        x_sents = np.array(x_sents)
        x_max = np.max(x_sents, axis=0)
        x_avg = np.mean(x_sents, axis=0)

        max_role = role_labels['role_list'][np.argmax(x_max)]
        avg_role = role_labels['role_list'][np.argmax(x_avg)]

        true_roles = doc_labels['sub_roles']
        
        max_match_count += 1 if [max_role] == true_roles else 0
        avg_match_count += 1 if [avg_role] == true_roles else 0
        total_instances += 1

        print(f"Max role: {max_role}")
        print(f"Average role: {avg_role}")
        print(f"True roles: {true_roles}\n")

    print(f"Total instances: {total_instances}, Empty instances: {empty_instances}")
    print(f"Exact match ratio - Max: {max_match_count / total_instances}, Average: {avg_match_count / total_instances}")

if __name__ == "__main__":
    base_dir = "train"
    labels_file = "subtask-1-annotations.txt"
    subdirs = ["EN"]

    ld = LoadData()
    labels = ld.load_data(base_dir, labels_file, subdirs)
    # Cast these columns as int
    labels['start_offset'] = labels['start_offset'].astype(int)
    labels['end_offset'] = labels['end_offset'].astype(int)

    role_labels = get_role_labels()
    emb_model = load_glove_vectors("embeddings/glove.6B.100d.txt")

    role_vectors = get_role_vectors(emb_model, role_labels['role_list'])
    c_window = 15  # size of context window for model
    

    process_documents(base_dir, labels, subdirs, role_labels, emb_model, role_vectors, c_window)
