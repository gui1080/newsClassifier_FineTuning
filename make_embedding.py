from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertTokenizer, BertModel
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import evaluate
import pandas as pd
import numpy as np
from datasets import Dataset
from enelvo.normaliser import Normaliser
from transformers import Trainer
import re
import os
from tqdm import tqdm 
import torch

# ---------------------------------------------------------------------------------------------------

# Função para gerar embeddings com um modelo BERT-like
def get_embeddings(text, model, tokenizer, hidden_size):
    # encode the sentence
    encoded = tokenizer(
        text=text,
        truncation = True,
        return_attention_mask = True,  
        return_tensors = 'pt'
    )
    
    # get contextual embeddings
    with torch.no_grad():
        last_hidden_states = model(**encoded)['last_hidden_state']

    # pooled embedding of shape <1, hidden_size>
    pooled_embedding = last_hidden_states.mean(axis=1)
    
    return pooled_embedding[0][:hidden_size].tolist()

##funcao para dividir o df para melhorar o desempenho do processos dos embeddings BERT-like
def get_dfs_splited(df: pd.DataFrame):
    num_cpus = os.cpu_count()
    
    return np.array_split(df, num_cpus)

# ---------------------------------------------------------------------------------------------------

# The hidden size is 768. This means the pooled embedding will have 768 dimensions regardless of the input text's length.
def set_bert(df: pd.DataFrame) -> pd.Series:
    # BERT - vetor dinamico
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    bert_model = BertModel.from_pretrained('dominguesm/legal-bert-base-cased-ptbr', output_hidden_states = False)
    bert_model.eval()
    
    dfs_split = get_dfs_splited(df)
    results = list()
    for df_split in tqdm(dfs_split, desc='Gerando o vetor semantico BERT'):
        results.append(
            df_split['text'].apply(lambda prompt: get_embeddings(prompt, bert_model, bert_tokenizer, 768))
        )

    return pd.concat(results)

# The hidden size for BART is 1024, meaning the pooled embedding from BART will have 1024 dimensions.
def set_bart(df: pd.DataFrame) -> pd.Series:
    # BART - vetor dinamico
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    bart_model = BartModel.from_pretrained('adalbertojunior/bart-base-portuguese', output_hidden_states = False)
    bart_model.eval()
    
    dfs_split = get_dfs_splited(df)
    results = list()
    for df_split in tqdm(dfs_split, desc='Gerando o vetor semantico BART'):
        results.append(
            df_split['text'].apply(lambda prompt: get_embeddings(prompt, bart_model, bart_tokenizer, 1024))
        )

    return pd.concat(results)

# The hidden size is 768. This means the pooled embedding will have 768 dimensions regardless of the input text's length.
def set_roberta(df: pd.DataFrame) -> pd.Series:
    # RoBERTa - vetor dinamico
    roberta_tokenizer = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base')

    roberta_model = RobertaModel.from_pretrained('josu/roberta-pt-br', output_hidden_states = False)
    roberta_model.eval()
    
    dfs_split = get_dfs_splited(df)
    results = list()
    for df_split in tqdm(dfs_split, desc='Gerando o vetor semantico ROBERTA'):
        results.append(
            df_split['text'].apply(lambda prompt: get_embeddings(prompt, roberta_model, roberta_tokenizer, 768))
        )

    return pd.concat(results)


# Normalizador de texto
# --------------------------------------------------------
'''
normalizador = Normaliser(tokenizer='readable', capitalize_inis=True, 
                          capitalize_pns=True, capitalize_acs=True, 
                          sanitize=True)
'''

def normalize_text(input_text):

    #final_text = normalizador.normalise(text)

    #final_text = normalizador.normalise(text)
    final_text = " ".join(word.strip() for word in input_text.split())
    final_text = re.sub(r'(?<!\*)\*\*(?!\*)', '', final_text)
    return final_text

# --------------------------------------------------------

print("Loading csv")

# Load the CSV file
df_gpt4 = pd.read_csv("../DATASETS_NOTICIAS/GPT4o/noticias_gpt4_n2000.csv").head(2000)
df_original = pd.read_csv("../DATASETS_NOTICIAS/G1_Kaggle/Sample_Noticias_Originais.csv").sample(n=2000, random_state=32)

print("Loaded")

# AI MADE = LABEL 1
df_gpt4['label'] = 1
# HUMAN MADE = LABEL 0
df_original['label'] = 0

print("Labels made")

df_gpt4['text'] = df_gpt4['text'].apply(normalize_text)

print("df_gpt4 normalizado")

df_original['text'] = df_original['text'].apply(normalize_text)

print("df_original normalizado")

# Concatenate both datasets
df = pd.concat([df_gpt4, df_original], ignore_index=True)

print("concat complete")

#df.to_csv("df_fine_tune_gpt4sss0_original.csv", index=False)
print("df saved")

# Convert to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)


print("limpeza feita")
print("\n\n")
print(df["text"][0])

'''
print(df_bert)
print(df_bert[0])
print(len(df_bert[0]))
print(type(df_bert[0]))
print(type(df_bert))
'''

# ---------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
# ---------------------------------------------------------------------------------------------------
# TESTE

df['text'] = set_bert(df)

# Ensure 'text' is a numpy array
df['text'] = df['text'].apply(np.array)

# Prepare features (X) and labels (y)
X = np.vstack(df['text'].values)  # Stack arrays into 2D array
y = df['label'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and train the classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

'''
Accuracy: 0.94
Classification Report:
               precision    recall  f1-score   support

           0       0.94      0.93      0.94       476
           1       0.94      0.95      0.94       524

    accuracy                           0.94      1000
   macro avg       0.94      0.94      0.94      1000
weighted avg       0.94      0.94      0.94      1000

'''

# Save the model to a file

folder = 'sklearn_models'
file_name = 'randomForest_bert_gpt4_sem_finetuning.pkl'
file_path = os.path.join(folder, file_name)

# Create the folder if it doesn't exist
os.makedirs(folder, exist_ok=True)
with open(file_path, 'wb') as file:
    pickle.dump(clf, file)
    
'''
# Later, load the model from the file
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Use the loaded model to make predictions
y_pred_loaded = loaded_model.predict(X_test)
print("Predictions from loaded model:", y_pred_loaded)
'''