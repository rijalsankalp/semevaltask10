from transformers import pipeline
import numpy as np

class BartClassifier:
    def __init__(self):
        # Initialize the zero-shot classification pipeline
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", weights_only=True)

        self.candidate_labels = [
    'victim', 'virtue', 'innocent','agitator'
    'mentor', 'sidekick', 'rebel', 'leader', 'deceiver', 'savior', 
    'martyr', 'bystander', 'enforcer', 'manipulator','prejudiced'
]
    
    def _getlabel_(self, text):
        return self.classifier(text, self.candidate_labels)


if __name__ == "__main__":
    from src.LoadData import LoadData
    from src.EntityDataset import DataLoader
    from src.NltkPipe import NltkPipe
    import StanfordCoreNlpServer

    #StanfordCoreNlpServer.startServer()

    base_dir = "train"
    txt_file = "subtask-1-annotations.txt"
    subdirs = "EN"
    load_data = LoadData()
    data = load_data.load_data(base_dir, txt_file, subdirs)
    data_loader = DataLoader(data, base_dir)
    filter = NltkPipe()
    bart = BartClassifier()

    with open("BARTOutput.txt", "a+") as file:
        for text in data_loader._get_text():
            file.write(f"{text}\n\n")
            relations, result, sentences = filter.process_text(text)

            entities = set()
            for ent,_ in result:
                entities.add(ent)
            
            outputs = list()

            for ent in entities:
                ent_sentences = list()
                for sentence in sentences:
                    if ent in sentence:
                        ent_sentences.append(sentence)

                if not len(ent_sentences) < 1:
                    label = bart._getlabel_("".join(ent_sentences))
                    idx = np.argmax(label['scores'])
                    outputs.append((ent, label['labels'][idx]))
            
            print(outputs)
            for entity, character in outputs:
                file.write(f"{entity}\t\t{character}\n\n\n")

            print("\n\n\n\n")
            break
        
        file.close()