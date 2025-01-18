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
import logging
import os
from tqdm import tqdm 
import torch
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import sys
import time
import pickle

# -----------------------------------------------------------------------------------------

from make_embedding import set_bert, set_bart, set_roberta, normalize_text

# -----------------------------------------------------------------------------------------

start = time.time()

''' Passando as combinações de dataset e embedding por parâmetro
Ex:

python main_ml.py "gpt4" "bert"

'''

if len(sys.argv) > 2:
    dados_treino = sys.argv[1]  # Qual caso eu vou treinar
    vetor_semantico = sys.argv[2]  # Qual embedding eu vou usar

''' dados_treino
Opções
- "gpt4"
- "gpt4_finetuned"
- "llama_8b" <- modelo local sem fine-tunning
- "llama_8b_finetuned" <- meu modelo local do hugging face

'''

''' vetor_semantico
Opções
- "bert" 
- "bart" 
- "roberta" 

'''

print("\n\n")
print("Parâmetros")
print(dados_treino)
print(vetor_semantico)
print("\n\n")

# -----------------------------------------------------------------------------------------

if dados_treino == "gpt4":

    print("Loading csv\n")
    print("gpt4\n")

    # Load the CSV file
    df_ai_made = pd.read_csv("../DATASETS_NOTICIAS/GPT4o/noticias_gpt4_n2000.csv").head(2000)
    df_original = pd.read_csv("../DATASETS_NOTICIAS/G1_Kaggle/Sample_Noticias_Originais.csv").sample(n=2000, random_state=32)

    print("Loaded\n")

elif dados_treino == "gpt4_finetuned":

    print("Loading csv\n")
    print("gpt4_finetuned\n")

    # Load the CSV file
    df_ai_made = pd.read_csv("../DATASETS_NOTICIAS/GPT4o_Finetuning/noticias_gpt4_n3000.csv")
    df_original = pd.read_csv("../DATASETS_NOTICIAS/G1_Kaggle/Sample_Noticias_Originais.csv").sample(n=3000)

    print("Loaded\n")

elif dados_treino == "llama_8b":

    print("Loading csv\n")
    print("llama_8b\n")

    # Load the CSV file
    df_ai_made1 = pd.read_csv("../DATASETS_NOTICIAS/Llama8B_Original/noticias_llama3_head1000.csv")
    df_ai_made2 = pd.read_csv("../DATASETS_NOTICIAS/Llama8B_Original/noticias_llama3_last1000.csv")
    df_ai_made = pd.concat([df_ai_made1, df_ai_made2], ignore_index=True)
    df_original1 = pd.read_csv("../DATASETS_NOTICIAS/G1_Kaggle/Sample_Noticias_Originais.csv").head(1000)
    df_original2 = pd.read_csv("../DATASETS_NOTICIAS/G1_Kaggle/Sample_Noticias_Originais.csv").tail(1000)
    df_original = pd.concat([df_original1, df_original2], ignore_index=True)

    print("Loaded\n")

elif dados_treino == "llama_8b_finetuned":

    print("Loading csv\n")
    print("llama_8b_finetuned\n")

    # Limpeza de dados básica
    


    print("Loaded\n")

elif dados_treino == "tudo":

    # gpt4
    df_ai_gpt4 = pd.read_csv("../DATASETS_NOTICIAS/GPT4o/noticias_gpt4_n2000.csv")

    # gpt4_finetuned
    df_ai_gpt4_finetuned = pd.read_csv("../DATASETS_NOTICIAS/GPT4o_Finetuning/noticias_gpt4_n3000.csv")

    # llama 8b
    df_ai_llama8b1 = pd.read_csv("../DATASETS_NOTICIAS/Llama8B_Original/noticias_llama3_head1000.csv")
    df_ai_llama8b2 = pd.read_csv("../DATASETS_NOTICIAS/Llama8B_Original/noticias_llama3_last1000.csv")
    df_ai_llama8b = pd.concat([df_ai_llama8b1, df_ai_llama8b2], ignore_index=True)

    # llama 8b finetuned

    df_ai_made = pd.concat([df_ai_gpt4, df_ai_gpt4_finetuned, df_ai_llama8b], ignore_index=True)

    quantidade_noticias_IA = len(df_ai_made)

    if quantidade_noticias_IA > 20000:

        df_ai_made = df_ai_made.sample(n=20000)
        df_original = df_original = pd.read_csv("../DATASETS_NOTICIAS/G1_Metropoles_MinhaExtracaoFInal/dataset_noticias_final.csv").sample(n=20000)
    
    else:

        df_original = df_original = pd.read_csv("../DATASETS_NOTICIAS/G1_Metropoles_MinhaExtracaoFInal/dataset_noticias_final.csv").sample(n=quantidade_noticias_IA)


# AI MADE = LABEL 1
df_ai_made['label'] = 1
# HUMAN MADE = LABEL 0
df_original['label'] = 0

df_ai_made['text'] = df_ai_made['text'].apply(normalize_text)
df_original['text'] = df_original['text'].apply(normalize_text)

print("Dados normalizados")

# Concatenate both datasets
df = pd.concat([df_ai_made, df_original], ignore_index=True)

print("Dados concatenados")

# EMBEDDINGS
# -----------------------------------------------------------------------------------------

print("Fazendo embedding!\n")

if vetor_semantico == "bert":

    df['text'] = set_bert(df)

if vetor_semantico == "bart":

    df['text'] = set_bart(df)

if vetor_semantico == "roberta":

    df['text'] = set_roberta(df)

# Ensure 'text' is a numpy array
df['text'] = df['text'].apply(np.array)

# Prepare features (X) and labels (y)
X = np.vstack(df['text'].values)  # Stack arrays into 2D array
y = df['label'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# LOGGING
# -----------------------------------------------------------------------------------------

# Configure logging
log_file_name = vetor_semantico + "_" + dados_treino + "_results.log"

log_folder = "logs_sklearn"
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, log_file_name)

logging.basicConfig(filename=log_file, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# Pasta dos modelos
folder = 'sklearn_models'

end = time.time()
execution_time = end - start
minutes = int(execution_time // 60)
seconds = execution_time % 60

logging.info("Embedding feito e dados carregados!\n")
logging.info(f"Checkpoint: {minutes} minutos e {seconds:.2f} segundos!")

logging.info("Dataset - ")
logging.info(str(dados_treino))
logging.info("Modelo usado para Embedding:\n")
logging.info(str(vetor_semantico))

# CLASSIFICADORES BINÁRIOS
# -----------------------------------------------------------------------------------------
# RANDOM FOREST

# Initialize and train the classifier
clf_RF = RandomForestClassifier(random_state=42)

# Perform cross-validation
cv_scores_RF = cross_val_score(clf_RF, X_train, y_train, cv=5)
logging.info("Cross-Validation Scores: %s", cv_scores_RF)
logging.info("Mean CV Accuracy: %f", cv_scores_RF.mean())

clf_RF.fit(X_train, y_train)

# Predict on the test set
y_pred_RF = clf_RF.predict(X_test)

auc = roc_auc_score(y_test, clf_RF.predict_proba(X_test)[:, 1]) if hasattr(clf_RF, "predict_proba") else np.nan
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_RF, average="weighted")
conf_matrix = confusion_matrix(y_test, y_pred_RF)

# Evaluate the model
logging.info("\nRandom Forest")
logging.info("Accuracy")
logging.info(str(accuracy_score(y_test, y_pred_RF)))
logging.info("Classification Report:\n")
logging.info(str(classification_report(y_test, y_pred_RF)))
logging.info("Area Under ROC:\n")
logging.info(str(auc))
logging.info("Confusion Matrix:\n")
logging.info(str(conf_matrix))
logging.info("Weighted Precision:\n")
logging.info(str(precision))
logging.info("Weighted Recall:\n")
logging.info(str(recall))
logging.info("Weighted F1 Score:\n")
logging.info(str(f1))
logging.info("-----------------------------------------------")

# Save the model to a file
file_name = "RandomForest_" + vetor_semantico + "_" + dados_treino + "_resultsModel.pkl"

file_path = os.path.join(folder, file_name)

# Create the folder if it doesn't exist
os.makedirs(folder, exist_ok=True)
with open(file_path, 'wb') as file:
    pickle.dump(clf_RF, file)

logging.info("Modelo salvo como: ", file_name)

end = time.time()
execution_time = end - start
minutes = int(execution_time // 60)
seconds = execution_time % 60

logging.info(f"Checkpoint: {minutes} minutos e {seconds:.2f} segundos!")

# -----------------------------------------------------------------------------------------
# GradientBoostingClassifier

# Initialize the classifier
clf_GBC = GradientBoostingClassifier(random_state=42)

# Perform cross-validation
cv_scores_GBC = cross_val_score(clf_GBC, X_train, y_train, cv=5)
logging.info("Cross-Validation Scores: %s", cv_scores_GBC)
logging.info("Mean CV Accuracy: %f", cv_scores_GBC.mean())

# Train the classifier on the full training set
clf_GBC.fit(X_train, y_train)

# Predict on the test set
y_pred_GBC = clf_GBC.predict(X_test)

auc = roc_auc_score(y_test, clf_GBC.predict_proba(X_test)[:, 1]) if hasattr(clf_GBC, "predict_proba") else np.nan
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_GBC, average="weighted")
conf_matrix = confusion_matrix(y_test, y_pred_GBC)

# Evaluate the model
logging.info("\nGradient Boosting Classifier")
logging.info("Accuracy")
logging.info(str(accuracy_score(y_test, y_pred_GBC)))
logging.info("Classification Report:\n")
logging.info(str(classification_report(y_test, y_pred_GBC)))
logging.info("Area Under ROC")
logging.info(str(auc))
logging.info("Confusion Matrix:\n")
logging.info(str(conf_matrix))
logging.info("Weighted Precision:\n")
logging.info(str(precision))
logging.info("Weighted Recall:\n")
logging.info(str(recall))
logging.info("Weighted F1 Score:\n")
logging.info(str(f1))
logging.info("-----------------------------------------------")

# Save the model to a file
file_name = "GradientBoostingClassifier_" + vetor_semantico + "_" + dados_treino + "_resultsModel.pkl"

file_path = os.path.join(folder, file_name)

# Create the folder if it doesn't exist
os.makedirs(folder, exist_ok=True)
with open(file_path, 'wb') as file:
    pickle.dump(clf_GBC, file)

logging.info("Modelo salvo como: ", file_name)

end = time.time()
execution_time = end - start
minutes = int(execution_time // 60)
seconds = execution_time % 60

logging.info(f"Checkpoint: {minutes} minutos e {seconds:.2f} segundos!")

# -----------------------------------------------------------------------------------------
# LogisticRegression

# Initialize the classifier
clf_LR = LogisticRegression(random_state=42, max_iter=1000)

# Perform cross-validation
cv_scores_LR = cross_val_score(clf_LR, X_train, y_train, cv=5)
logging.info("Cross-Validation Scores: %s", cv_scores_LR)
logging.info("Mean CV Accuracy: %f", cv_scores_LR.mean())

# Train the classifier on the full training set
clf_LR.fit(X_train, y_train)

# Predict on the test set
y_pred_LR = clf_LR.predict(X_test)

auc = roc_auc_score(y_test, clf_LR.predict_proba(X_test)[:, 1]) if hasattr(clf_LR, "predict_proba") else np.nan
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_LR, average="weighted")
conf_matrix = confusion_matrix(y_test, y_pred_LR)

# Evaluate the model
logging.info("\nLogistic Regression")
logging.info("Accuracy")
logging.info(str(accuracy_score(y_test, y_pred_LR)))
logging.info("Classification Report:\n")
logging.info(str(classification_report(y_test, y_pred_LR)))
logging.info("Area Under ROC")
logging.info(str(auc))
logging.info("Confusion Matrix:\n")
logging.info(str(conf_matrix))
logging.info("Weighted Precision:\n")
logging.info(str(precision))
logging.info("Weighted Recall:\n")
logging.info(str(recall))
logging.info("Weighted F1 Score:\n")
logging.info(str(f1))
logging.info("-----------------------------------------------")

# Save the model to a file
file_name = "LogisticRegression_" + vetor_semantico + "_" + dados_treino + "_resultsModel.pkl"

file_path = os.path.join(folder, file_name)

# Create the folder if it doesn't exist
os.makedirs(folder, exist_ok=True)
with open(file_path, 'wb') as file:
    pickle.dump(clf_LR, file)

logging.info("Modelo salvo como: ", file_name)

end = time.time()
execution_time = end - start
minutes = int(execution_time // 60)
seconds = execution_time % 60

logging.info(f"Checkpoint: {minutes} minutos e {seconds:.2f} segundos!")

# -----------------------------------------------------------------------------------------
# NaiveBayes

# Initialize the classifier
clf_NB = GaussianNB()

# Perform cross-validation
cv_scores_NB = cross_val_score(clf_NB, X_train, y_train, cv=5)
logging.info("Cross-Validation Scores: %s", cv_scores_NB)
logging.info("Mean CV Accuracy: %f", cv_scores_NB.mean())

# Train the classifier on the full training set
clf_NB.fit(X_train, y_train)

# Predict on the test set
y_pred_NB = clf_NB.predict(X_test)

auc = roc_auc_score(y_test, clf_NB.predict_proba(X_test)[:, 1]) if hasattr(clf_NB, "predict_proba") else np.nan
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_NB, average="weighted")
conf_matrix = confusion_matrix(y_test, y_pred_NB)

# Evaluate the model
logging.info("\nNaive Bayes")
logging.info("Accuracy")
logging.info(str(accuracy_score(y_test, y_pred_NB)))
logging.info("Classification Report:\n")
logging.info(str(classification_report(y_test, y_pred_NB)))
logging.info("Area Under ROC")
logging.info(str(auc))
logging.info("Confusion Matrix:\n")
logging.info(str(conf_matrix))
logging.info("Weighted Precision:\n")
logging.info(str(precision))
logging.info("Weighted Recall:\n")
logging.info(str(recall))
logging.info("Weighted F1 Score:\n")
logging.info(str(f1))
logging.info("-----------------------------------------------")

# Save the model to a file
file_name = "NaiveBayes_" + vetor_semantico + "_" + dados_treino + "_resultsModel.pkl"

file_path = os.path.join(folder, file_name)

# Create the folder if it doesn't exist
os.makedirs(folder, exist_ok=True)
with open(file_path, 'wb') as file:
    pickle.dump(clf_NB, file)

logging.info("Modelo salvo como: ", file_name)

end = time.time()
execution_time = end - start
minutes = int(execution_time // 60)
seconds = execution_time % 60

logging.info(f"Checkpoint: {minutes} minutos e {seconds:.2f} segundos!")

# -----------------------------------------------------------------------------------------
# DecisionTreeClassifier

# Initialize the classifier
clf_DTC = GaussianNB()

# Perform cross-validation
cv_scores_DTC = cross_val_score(clf_DTC, X_train, y_train, cv=5)
logging.info("Cross-Validation Scores: %s", cv_scores_DTC)
logging.info("Mean CV Accuracy: %f", cv_scores_DTC.mean())

# Train the classifier on the full training set
clf_DTC.fit(X_train, y_train)

# Predict on the test set
y_pred_DTC = clf_DTC.predict(X_test)

auc = roc_auc_score(y_test, clf_DTC.predict_proba(X_test)[:, 1]) if hasattr(clf_DTC, "predict_proba") else np.nan
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_DTC, average="weighted")
conf_matrix = confusion_matrix(y_test, y_pred_DTC)

# Evaluate the model
logging.info("\nDecision Tree Classifier")
logging.info("Accuracy")
logging.info(str(accuracy_score(y_test, y_pred_DTC)))
logging.info("Classification Report:\n")
logging.info(str(classification_report(y_test, y_pred_DTC)))
logging.info("Area Under ROC")
logging.info(str(auc))
logging.info("Confusion Matrix:\n")
logging.info(str(conf_matrix))
logging.info("Weighted Precision:\n")
logging.info(str(precision))
logging.info("Weighted Recall:\n")
logging.info(str(recall))
logging.info("Weighted F1 Score:\n")
logging.info(str(f1))
logging.info("-----------------------------------------------")

# Save the model to a file
file_name = "DecisionTreeClassifier_" + vetor_semantico + "_" + dados_treino + "_resultsModel.pkl"

file_path = os.path.join(folder, file_name)

# Create the folder if it doesn't exist
os.makedirs(folder, exist_ok=True)
with open(file_path, 'wb') as file:
    pickle.dump(clf_DTC, file)

logging.info("Modelo salvo como: ", file_name)

end = time.time()
execution_time = end - start
minutes = int(execution_time // 60)
seconds = execution_time % 60

logging.info(f"Checkpoint: {minutes} minutos e {seconds:.2f} segundos!")

# -----------------------------------------------------------------------------------------

print("\n\n\n")
print(f"Execução completa em {minutes} minutos e {seconds:.2f} segundos!")
print("\n\n\n")