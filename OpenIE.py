from src.NltkOpenIE import NltkOpenIE
from src.LoadData import LoadData
from src.EntityDataset import DataLoader
from transformers import pipeline
import numpy as np

base_dir = "data/train"
txt_file = "subtask-1-annotations.txt"
subdir = "EN"
load_data = LoadData()
data = load_data.load_data(base_dir, txt_file, subdir)
data_loader = DataLoader(data, base_dir)
filter = NltkOpenIE()

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", weights_only=True)

candidate_labels = [
    'victim', 'virtue', 'innocent','agitator'
    'mentor', 'sidekick', 'rebel', 'leader', 'deceiver', 'savior', 
    'martyr', 'bystander', 'enforcer', 'manipulator','prejudiced'
]

counter = 0
with open("resources/nltk_openie.txt", "a+") as writer:
    for text,article_id in data_loader._get_text():
        relations = filter.process_text(text)
        for entity, sentences in relations.items():
            label = classifier(" ".join(sentences), candidate_labels)
            writer.write(f"{article_id}\t{entity}\t{label['labels'][np.argmax(label['scores'])]}\n")
        counter += 1
        if(counter > 20):
            break  
writer.close()