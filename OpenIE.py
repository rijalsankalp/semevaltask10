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

candidate_labels = ['protagonist', 'innocent', 'antagonist']

total_instances = 0
truths = 0

# Output file
with open("resources/comparitive.txt", "a+", encoding="utf-8") as writer:
    for text, article_id, given_entity, given_role in data_loader._get_text():
        relations = entity_and_context.extract_entity_contexts(text)

        for entity, context in relations.items():
            result = classifier(
                context,
                candidate_labels,
                hypothesis_template=f"This stance of {entity} in this text is {{}}."
            )

            top_label = result['labels'][0]
            score = result['scores'][0]

            

            print(f"ARTICLE_ID: {article_id}")
            print(f"ENTITY: {entity}")
            print(f"LABEL: {top_label} (score: {score:.4f})")
            if entity == given_entity:
                print(f"Given Role: {given_role}")
            print(f"CONTEXT: {context}")
            print("-" * 80)

            writer.write(f"ARTICLE_ID: {article_id}\n")
            writer.write(f"ENTITY: {entity}\n")
            writer.write(f"LABEL: {top_label} (score: {score:.4f})\n")

            if(entity == given_entity):
                total_instances += 1
                writer.write(f"Given Role: {given_role}\n")
                if top_label == given_role:
                    truths += 1
                    
            writer.write(f"CONTEXT: {context}\n")
            writer.write("-" * 80 + "\n")
        
writer.close()

print(f"Total instances: {total_instances}")
print(f"Correctly classified: {truths}")
print(f"Accuracy: {truths / total_instances:.4f}")