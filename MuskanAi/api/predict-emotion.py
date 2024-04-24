import torch
from transformers import SqueezeBertTokenizer
import torch.nn as nn
import torch.nn.functional as F


class GoEmotionClassifier(nn.Module):
    def __init__(self, n_train_steps, n_classes, do_prob, bert_model):
        super(GoEmotionClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(do_prob)
        self.out = nn.Linear(768, n_classes)
        self.n_train_steps = n_train_steps
        self.step_scheduler_after = "batch"

    def forward(self, ids, mask):
        output_1 = self.bert(ids, attention_mask=mask)["pooler_output"]
        output_2 = self.dropout(output_1)
        output = self.out(output_2)
        return output
    

# Load the saved model
model = GoEmotionClassifier()  # Initialize your model
model.load_state_dict(torch.load("F:/backup-kali/codeFiles/projects/MuskanAi/models/emotion_model_epoch_5.pt"))  # Load the saved model weights

# Load the tokenizer
tokenizer = SqueezeBertTokenizer.from_pretrained("squeezebert/squeezebert-uncased")

# Given text for prediction
text = "Hi, how are you, I love you too."

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

# Perform prediction
outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
predictions = torch.sigmoid(outputs)  # Apply sigmoid activation if needed

print("Predictions:", predictions)

