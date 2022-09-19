from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class IntentClassificator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-e2m-intent")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-e2m-intent")

    def predict(self, text):
        inputs = self.tokenizer.encode(text, return_tensors='pt', max_length=512)
        outputs = self.model.generate(inputs, max_length=512)
        text_output = self.tokenizer.decode(outputs[0])
        return text_output
