from transformers import pipeline

class BartClassifier:
    def __init__(self):
        # Initialize the zero-shot classification pipeline
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        self.candidate_labels = ['protagonist', 'antagonist', 'victim', 'vice', 'virtue', 'innocent']
    
    def _getlabel_(self, text):
        return self.classifier(text, self.candidate_labels)


if __name__ == "__main__":
    from LoadData import LoadData
    from EntityDataset import DataLoader
    from NltkPipe import NltkPipe

    base_dir = "train"
    txt_file = "subtask-1-annotations.txt"
    subdirs = "EN"
    load_data = LoadData()
    data = load_data.load_data(base_dir, txt_file, subdirs)
    data_loader = DataLoader(data, base_dir)
    filter = NltkPipe()
    bart = BartClassifier()

    for text in data_loader._get_text():
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

                outputs.append((ent, label))
        
        print(outputs)
        print("\n\n\n\n")
        break