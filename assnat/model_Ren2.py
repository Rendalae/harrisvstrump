
from assnat.clean import complete_preproc
from assnat.models_base import load_X_y, sample
from transformers import CamembertTokenizer, CamembertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import pandas as pd
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments

def prepare_data(df):
    return Dataset.from_pandas(df)

# Tokenizer et modèle
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=7)

# Tokenization
def tokenize_function(texts):
    return tokenizer(texts["Texte"], truncation=True, padding=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.from_numpy(logits), dim=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Entraînement
def train_model(X_train, X_test, y_train, y_test):
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    train_dataset = pd.concat([X_train, pd.Series(y_train)], axis=1)
    train_dataset.columns = ['Texte', 'labels']
    val_dataset = pd.concat([X_test, pd.Series(y_test)], axis=1)
    val_dataset.columns = ['Texte', 'labels']


    train_dataset = prepare_data(train_dataset)
    val_dataset = prepare_data(val_dataset)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    #train_dataset = train_dataset.rename_column("famille", "labels")
    #val_dataset = val_dataset.rename_column("famille", "labels")

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()

def predict(text):
    # Check if MPS is available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Make sure the model is in evaluation mode
    model.eval()

    # Perform the prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # The model outputs logits, we need to convert them to probabilities and then to predicted labels
    logits = outputs.logits
    return torch.argmax(logits, dim=1).item()

X_train, X_test, y_train, y_test =load_X_y('leg15', min_n_words=30)
X_train, X_test, y_train, y_test=sample(100, X_train, X_test, y_train, y_test)

train_model(X_train, X_test, y_train, y_test)

text = "Nous devons augmenter les taxes pour les riches"
print(f'{predict(text)} : {text}')

text = "Nous devons augmenter le SMIC"
print(f'{predict(text)} : {text}')

text = "Nous devons remettre les chômeurs au travail"
print(f'{predict(text)} : {text}')
