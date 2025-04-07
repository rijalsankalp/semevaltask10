import requests
import src.NltkPipe as NltkPipe

class NltkOpenIE(NltkPipe.NltkPipe):
    def __init__(self):
        super().__init__()
        
    def process_text(self, text):

        sentences = self._resolve_coreferences(text)

        for resolved_text in sentences:
            
            self.extract_openie(resolved_text)

    def extract_openie(self, text):
        properties = {
            "annotators": "openie",
            "outputFormat": "json"
        }

        # Send the request
        response = requests.post(self.url, params={"properties": str(properties)}, data=text)

        # Parse the response
        data = response.json()

        # Extract and print the OpenIE triples
        for sentence in data['sentences']:
            for triple in sentence['openie']:
                subject = triple['subject']
                relation = triple['relation']
                object = triple['object']
                print(f"({subject}; {relation}; {object})")

if __name__ == "__main__":
    from LoadData import LoadData
    from EntityDataset import DataLoader

    base_dir = "train"
    txt_file = "subtask-1-annotations.txt"
    subdirs = "EN"
    load_data = LoadData()
    data = load_data.load_data(base_dir, txt_file, subdirs)
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