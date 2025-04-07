from src.NltkOpenIE import NltkOpenIE
from src.LoadData import LoadData
from src.EntityDataset import DataLoader

base_dir = "data/train"
txt_file = "subtask-1-annotations.txt"
subdir = "EN"
load_data = LoadData()
data = load_data.load_data(base_dir, txt_file, subdir)
data_loader = DataLoader(data, base_dir)
filter = NltkOpenIE()

counter = 0
with open("resources/nltk_openie.txt", "a+") as writer:
    for text in data_loader._get_text():
        relation, result, sentences = filter.process_text(text)
        writer.writelines(f"{sentences}\n{result}\n{relation}\n\n")
        counter += 1
        if(counter > 20):
            break  
writer.close()