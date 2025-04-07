from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SentenceSimplifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-v1_1-base-wikisplit")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("flax-community/t5-v1_1-base-wikisplit")
    
    def _get_simplified_text(self, sentence):
        sample_tokenized = self.tokenizer(sentence, return_tensors="pt")
        answer = self.model.generate(sample_tokenized['input_ids'], attention_mask = sample_tokenized['attention_mask'], max_length=256, num_beams=5)
        sentences = self.tokenizer.decode(answer[0], skip_special_tokens=True)
        return sentences