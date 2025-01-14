# https://huggingface.co/google-bert/bert-base-multilingual-uncased

# --------------------------------------------------------

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import evaluate
import pandas as pd
from datasets import Dataset
from enelvo.normaliser import Normaliser
from transformers import Trainer
import logging
from transformers import Trainer, TrainingArguments, TrainerCallback  # Ensure TrainerCallback is imported

from transformers import TrainingArguments

# Normalizador de texto
# --------------------------------------------------------

normalizador = Normaliser(tokenizer='readable', capitalize_inis=True, 
                          capitalize_pns=True, capitalize_acs=True, 
                          sanitize=True)

def normalize_text(input_text):

    #final_text = normalizador.normalise(text)
    final_text = " ".join(word.strip() for word in input_text.split())
    final_text = re.sub(r'(?<!\*)\*\*(?!\*)', '', final_text)
    return final_text

# --------------------------------------------------------

etl = True

if etl:

    print("Loading csv")

    # Load the CSV file
    df_gpt4 = pd.read_csv("../DATASETS_NOTICIAS/GPT4o/noticias_gpt4_n2000.csv")

    df_original = pd.read_csv("../DATASETS_NOTICIAS/G1_Metropoles_MinhaExtracaoFinal/dataset_noticias_final.csv")
    df_original = df_original.sample(n=2000, replace=False)

    print("Loaded")

    # AI MADE = LABEL 1
    df_gpt4['label'] = 1

    # HUMAN MADE = LABEL 0
    df_original['label'] = 0

    print("Labels made")

    #df_gpt4['text'] = df_gpt4['text'].apply(normalize_text)

    print("df_gpt4 normalizado")

    #df_original['text'] = df_original['text'].apply(normalize_text)

    print("df_original normalizado")

    # Concatenate both datasets
    df = pd.concat([df_gpt4, df_original], ignore_index=True)

    print("concat complete")

    #df.to_csv("df_fine_tune_gpt40_original.csv", index=False)

    print("df saved")

    # Convert to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    print("limpeza feita")

else:

    df_gpt4 = pd.read_csv("df_fine_tune_gpt40_original.csv")
    dataset = Dataset.from_pandas(df)

# --------------------------------------------------------

# Split the data
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

# Convert splits to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

print("Split dados")

# Load the tokenizer
model_checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print("Tokenize dados")

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns and set the format
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])
train_dataset.set_format("torch")
test_dataset.set_format("torch")

print("Map dados")

# Load model

num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

print("Loaded madel")

# Argments

# baseline
training_args_baseline = TrainingArguments(
    output_dir="./results_gpt4_original_baseline",
    evaluation_strategy="steps",
    eval_steps=500,  # Evaluate every 500 steps
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs_gpt4_original_baseline",
    warmup_steps=500,
)

# data collector and metrics
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load accuracy metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

print("data collector and metrics")

# ------------------------------------------------------------------------------------------------

Train1 = True 
Train2 = True
Train3 = True 

# PARAMS 1
# ------------------------------------------------------------------------------------------------

# Train

if Train1 == True:
    
    print("Traning - Params 1")

    trainer1 = Trainer(
        model=model,
        args=training_args_baseline,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer1.train()

    trainer1.save_model("./gpt4_original_bert_multilingual_params_base")
    trainer1.push_to_hub("gpt4_original_bert_multilingual_params_base")

    results1 = trainer1.evaluate()

    print(results1)

    with open("logfile_params_base_gpt4_original.txt", "w") as file:  # Use "a" to append instead of "w" to overwrite
        file.write(str(results1))

# PARAMS 2
# ------------------------------------------------------------------------------------------------

# Train

if Train2 == True:

    print("Traning - Params 2")

    # Configure logging
    log_filename = "training_log_gpt4_original_aggressive.txt"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Add logging callback during training
    class LogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                logging.info(f"Step {state.global_step}: {logs}")

    training_args_aggressive = TrainingArguments(
        output_dir="./results_gpt4_original_aggressive",
        evaluation_strategy="steps",
        eval_steps=500,  # Evaluate every 500 steps
        learning_rate=5e-5,  # Higher learning rate
        per_device_train_batch_size=32,  # Larger batch size
        per_device_eval_batch_size=128,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs_gpt4_original_aggressive",
        warmup_steps=200,  # Fewer warmup steps
        logging_steps=100,  # Log every 100 steps
        log_level="info",  # Set logging level
    )

    trainer2 = Trainer(
        model=model,
        args=training_args_aggressive,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[LogCallback()],
    )

    trainer2.train()

    trainer2.save_model("./gpt4_original_bert_multilingual_params_aggressive")
    trainer2.push_to_hub("gpt4_original_bert_multilingual_params_aggressive")

    results2 = trainer2.evaluate()

    print(results2)

    with open("logfile_params_aggressive_gpt4_original.txt", "w") as file:  # Use "a" to append instead of "w" to overwrite
        file.write(str(results2))

# PARAMS 3
# ------------------------------------------------------------------------------------------------

if Train3 == True:
    
    print("Traning - Params 3")

    training_args_careful = TrainingArguments(
        output_dir="./results_gpt4_original_careful",
        evaluation_strategy="steps",
        eval_steps=500,  # Evaluate every 500 steps
        learning_rate=1e-5,  # Lower learning rate
        per_device_train_batch_size=8,  # Smaller batch size
        per_device_eval_batch_size=32,
        num_train_epochs=5,  # More epochs
        weight_decay=0.1,  # Higher weight decay
        save_total_limit=3,
        logging_dir="./logs_gpt4_original_careful",
        warmup_steps=500,
    )

    trainer3 = Trainer(
        model=model,
        args=training_args_careful,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer3.train()

    trainer3.save_model("./gpt4_original_bert_multilingual_params_careful")
    trainer3.push_to_hub("gpt4_original_bert_multilingual_params_careful")

    results3 = trainer3.evaluate()

    print(results3)

    with open("logfile_params_careful_gpt4_original.txt", "w") as file:  # Use "a" to append instead of "w" to overwrite
        file.write(str(results3))

# ------------------------------------------------------------------------------------------------



