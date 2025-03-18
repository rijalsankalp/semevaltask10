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
import stanza
from LoadData import LoadData
from pathlib import Path
from itertools import chain
from TextFilter import TextFilter  # Import the TF class with role-awareness

np.random.seed(0)

def get_role_labels(lang):
    sub_dict_map = {
        'EN':{
            "Protagonist":["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
            "Innocent":["Forgotten", "Exploited", "Victim", "Scapegoat"],
            "Antagonist":["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"]
            },
        'RU':{
            "Protagonist":["Хранитель", "Мученик", "Миротворец", "Бунтарь", "Аутсайдер", "Добродетель"],
            "Innocent":["Забытый", "Эксплуатируемый", "Жертва", "Козел отпущения"],
            "Antagonist":["Подстрекатель", "Заговорщик", "Тиран", "Иностранный противник", "Предатель", "Шпион", "Диверсант", "Коррумпированный", "Неумелый", "Террорист", "Обманщик", "Фанатик"]
        },
        "HI":{
            "Protagonist": ["रक्षक", "शहीद", "शांतिदूत", "विद्रोही", "अंडरडॉग", "सदाचारी"],
            "Innocent": ["भुला दिया गया", "शोषित", "पीड़ित", "बलि का बकरा"],
            "Antagonist": ["उकसाने वाला", "षड्यंत्रकारी", "तानाशाह", "विदेशी विरोधी", "देशद्रोही", "जासूस", "तोड़फोड़ करने वाला", "भ्रष्ट", "अयोग्य", "आतंकवादी", "धोखेबाज", "कट्टरपंथी"]
        },
        "BG":{
            "Protagonist": ["Пазител", "Мъченик", "Миротворец", "Бунтовник", "Аутсайдер", "Добродетел"],
            "Innocent": ["Забравен", "Експлоатиран", "Жертва", "Изкупителна жертва"],
            "Antagonist": ["Подстрекател", "Заговорник", "Тиран", "Чужд противник", "Предател", "Шпионин", "Диверсант", "Корумпиран", "Некомпетентен", "Терорист", "Измамник", "Фанатик"]
        },
        "PT":{
            "Protagonist": ["Guardião", "Mártir", "Pacificador", "Rebelde", "Oprimido", "Virtuoso"],
            "Innocent": ["Esquecido", "Explorado", "Vítima", "Bode expiatório"],
            "Antagonist": ["Instigador", "Conspirador", "Tirano", "Adversário estrangeiro", "Traidor", "Espião", "Sabotador", "Corrupto", "Incompetente", "Terrorista", "Enganador", "Fanático"]
        }
    }

    fg_EN_map = {}
    for ln, sub_dict in sub_dict_map.items():
        for archetype, items in sub_dict.items():
            for item in items:
                if ln == "EN":
                    fg_EN_map[item] = item
                else:
                    if item not in fg_EN_map:
                        fg_EN_map[item] = ""
                    fg_EN_map[item] = sub_dict_map["EN"][archetype][items.index(item)]

    sub_dict = sub_dict_map[lang]

    fg_roles = list()
    sub_dict['main_role_list'] = list()
    main_roles_order = ['Protagonist', 'Innocent', 'Antagonist']
    for main_role in main_roles_order:
        fg_roles.extend(sub_dict[main_role])
        sub_dict['main_role_list'].extend([main_role]*len(sub_dict[main_role]))

    sub_dict['role_list'] = fg_roles
    fg_to_main = dict()
    for main_role in sub_dict:
        for fg_role in sub_dict[main_role]:
            fg_to_main[fg_role] = main_role

    return sub_dict, fg_EN_map

def load_text_file(path):
    with open(path) as fin:
        text = fin.read()
    return text

def load_document(base_dir, lang, article_id):
    path = Path(base_dir).joinpath(lang, "raw-documents", article_id)
    text = load_text_file(path)
    return text

def visualize_dependency_parsing(sents, port=5005):
    displacy.serve(sents, style='dep', port=port)

def find_entity_sentence(doc, start_offset, end_offset):
    for i, sent in enumerate(doc.sents):
        if sent[0].idx <= start_offset and sent[-1].idx >= end_offset:
            return i

def get_entity_coref_sentences(doc, doc_labels, use_corefs=False):
    start_offset = doc_labels['start_offset']
    end_offset = doc_labels['end_offset']
    if start_offset > 1:
        while start_offset > 0 and not doc.text[start_offset - 1] in {' ', '\n', ',', '-', '.'}:
            start_offset -= 1
            if start_offset == 0:
                break
    while end_offset < len(doc.text) and not doc.text[end_offset] in {' ', '\n', ',', '-', '.'}:
        end_offset += 1
    entity_span = doc.char_span(start_offset, end_offset, label="target_ent", alignment_mode='expand')
    entity_corefs = list()
    if use_corefs:
        for g in doc._.coref_chains.chains:
            for mention_id in g.mentions:
                if any(token.i in mention_id for token in entity_span):
                    entity_corefs.extend(g.mentions)
                    continue
        entity_corefs = list(chain.from_iterable(entity_corefs))
        coref_sents = list({doc[tid].sent for tid in entity_corefs})
        coref_sents.append(entity_span.root.sent)
    else:
        while start_offset > 0 and doc.text[start_offset - 1] != '\n':
            start_offset -= 1
        while end_offset < len(doc.text) and doc.text[end_offset] != '\n':
            end_offset += 1
        coref_sents = [doc.char_span(start_offset, end_offset, alignment_mode='expand').sent]
    return entity_span, entity_corefs, coref_sents

def iterate_documents(base_dir, labels, subdir='EN', use_corefs=False):
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe('coreferee')
    for row_id in range(len(labels)):
        doc_labels = labels.iloc[row_id]
        text = load_document(base_dir, subdir, doc_labels['article_id'])
        doc = nlp(text)
        ent_span, ent_corefs, ent_sents = get_entity_coref_sentences(doc, doc_labels, use_corefs)
        yield doc, ent_span, ent_corefs, ent_sents, doc_labels

def apply_role_aware_extraction(ent_sents, target):
    tf = TextFilter()
    for sent in ent_sents:
        results = tf.extract_target_context(sent.text if hasattr(sent, 'text') else str(sent), target)
        print("Target:", target)
        for r in results:
            print("-", r)

def visualize_role_output(ent_sents):
    tf = TextFilter()
    for sent in ent_sents:
        if hasattr(sent, 'text'):
            text = sent.text
        else:
            text = str(sent)
        print("\nOriginal:", text)
        doc = tf.parser(text)
        displacy.serve(doc, style="dep", port=5005)

if __name__ == "__main__":
    base_dir = "train"
    labels_file = "subtask-1-annotations.txt"
    subdir = "EN"  # fixed: this should be a string, not a list

    ld = LoadData()
    labels = ld.load_data(base_dir, labels_file, subdir)
    labels['start_offset'] = labels['start_offset'].astype(int)
    labels['end_offset'] = labels['end_offset'].astype(int)

    for doc, ent_span, ent_corefs, ent_sents, doc_labels in iterate_documents(base_dir, labels, subdir):
        print("\nEntity Span:", ent_span)
        target = ent_span.text if hasattr(ent_span, 'text') else ent_span
        apply_role_aware_extraction(ent_sents, target)
        visualize_role_output(ent_sents)
        for tid in ent_corefs:
            tk = doc[tid]
            print(f"{tk} | {tk.head=} | {list(t for t in tk.children)}")
        break