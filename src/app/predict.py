
import torch
from transformers import AutoTokenizer
from app.NeuralNet.bert import MultiTaskNN

from app.config.config import config
from app.schema.schema import *


loaded_model = MultiTaskNN(bert_model_name=config["BERT_MODEL_NAME"])
device = torch.device(config['DEVICE'])

loaded_model.load_state_dict(torch.load(
    config['MODEL_PATH']))

loaded_model = loaded_model.to(device)
loaded_model.eval()
loaded_model.freeze()

tokenizer = AutoTokenizer.from_pretrained(config["BERT_MODEL_NAME"])


def preprocess(text: TextInput) -> dict:

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=config['MAX_TOKEN_LEN'],
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    return dict(
        input_ids=encoding['input_ids'].flatten(),
        attention_mask=encoding['attention_mask'].flatten()
    )


def predict(text: TextInput) -> ModelResponse:

    item = preprocess(text)
    with torch.no_grad():
        prediction = loaded_model(
            item['input_ids'].unsqueeze(dim=0).to(device),
            item['attention_mask'].unsqueeze(dim=0).to(device))

        all_task_classes_score = []
        for i in range(3):
            task_classes_score = []
            for idx, score in enumerate(prediction[i].flatten().tolist()):
                task_class_score = TaskClassScore(
                    class_value=config[f"LABELS_{i+1}"][idx], score=score)
                task_classes_score.append(task_class_score)

            all_task_classes_score.append(TaskClassesScore(
                task_id=i+1, scores=task_classes_score))

        return ModelResponse(response=all_task_classes_score)
