import pandas as pd

from src.LoadData import LoadData
from transformers import pipeline
from src.NltkOpenIE import NltkOpenIE
from src.EntityDataset import DataLoader
from src.Similarity import FastTextConnotation
from src.EntityContextNltk import EntityContextExtractor


base_dir = "data/train"
txt_file = "subtask-1-annotations.txt"
subdir = "EN"
load_data = LoadData()
data = load_data.load_data(base_dir, txt_file, subdir)
data_loader = DataLoader(data, base_dir)

entity_and_context = EntityContextExtractor()
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

candidate_main_labels = ['protagonist', 'innocent', 'antagonist']

candidate_sub_roles = ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous", "Instigator", 
                       "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", "Spy", "Saboteur", "Corrupt", 
                       "Incompetent", "Terrorist", "Deceiver", "Bigot", "Forgotten", "Exploited", "Victim", "Scapegoat"]


# Output file
# output = pd.DataFrame(columns=["ARTICLE_ID", "ENTITY", "GIVEN_MAIN_ROLE", "GIVEN_SUB_ROLES", "PRED_M","PRED_S", "CONTEXT"])
rows = []
for text, article_id, given_entity, given_role, g_sub_roles in data_loader._get_text():
    relations = entity_and_context.extract_entity_contexts(text)

    for entity, context in relations.items():

        if entity == given_entity:

            main_result = classifier(
                context,
                candidate_main_labels,
                #hypothesis_template=f"The stance of {entity} in this text is {{}}." #First template
                hypothesis_template=f"{entity} has a role of {{}} in this text."    #Second template
            )

            sub_result = classifier(
                context,
                candidate_sub_roles,
                #hypothesis_template=f"The role of {entity} in this text is {{}}."
                hypothesis_template=f"{entity} has a role of {{}} in this text."
            )

            main_pred = main_result['labels'][0]
            sub_pred = sub_result['labels'][0]

            rows.append({
                "ARTICLE_ID": article_id,
                "ENTITY": entity,
                "GIVEN_MAIN_ROLE": given_role,
                "GIVEN_SUB_ROLES": g_sub_roles,
                "PRED_M": main_pred,
                "PRED_S": sub_pred,
                "CONTEXT": context
            })

            print(
                f"Article ID: {article_id}\n"
                f"Entity: {entity}\n"
                f"Given Main Role: {given_role}\n"
                f"Given Sub Roles: {g_sub_roles}\n"
                f"Predicted Main Role: {main_pred}\n"
                f"Predicted Sub Role: {sub_pred}\n"
                f"Context: {context}\n"
                f"----------------------------------------"
            )

output = pd.DataFrame(rows)
output.to_csv("resources/bart_main_sub_second_template.csv", index=False)