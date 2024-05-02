import torch
from transformers import SqueezeBertTokenizer, SqueezeBertModel
from torch.nn.functional import softmax
from tqdm import tqdm

def load_model(model_path):
    model = SqueezeBertModel.from_pretrained("squeezebert/squeezebert-uncased")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_emotions(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=40)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.last_hidden_state.mean(dim=1)
        probabilities = softmax(logits, dim=1)
    return probabilities

def main():
    model_path = "models\emotion_model_epoch_5.pt"
    model = load_model(model_path)
    tokenizer = SqueezeBertTokenizer.from_pretrained("squeezebert/squeezebert-uncased")

    while True:
        text = input("Enter text to predict emotions (type 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        probabilities = predict_emotions(text, model, tokenizer)
        print(probabilities)

if __name__ == "__main__":
    main()
