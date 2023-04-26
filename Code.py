# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 09:46:40 2023

@author: Helen
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import seaborn as sns
import matplotlib.pyplot as plt


from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')

# Read dataset
dataset = pd.read_pickle("data_seperated_gbt.pkl")

#Drop date for now - perhaps in the real one, loot out all the ones after 2022? 
dataset = dataset.drop(['update_date',], axis=1)

# Create a binary label for AI text (1) and human text (0)
dataset["is_ai_generated"] = dataset["ai_generated"].apply(lambda x: 1 if x else 0)


#------------------------------------------ Perplexity ----------------------------------------------------#


#https://huggingface.co/docs/transformers/perplexity 
def calculate_perplexity_gpt2(text, tokenizer, model):
    tokens = tokenizer.encode(text, return_tensors="pt")
    
    with torch.no_grad(): #Makes it process faster and with less memory  (does not track gradients)
        outputs = model(tokens, labels=tokens)
        loss = outputs.loss #Returns the loss function, (a measure of how well it predicted the text)
        perplexity = torch.exp(loss).item() 
    return perplexity


def add_perplexity_column_huggingface(df):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    perplexities = [calculate_perplexity_gpt2(text, tokenizer, model) for text in df["abstract"]]
    df["Perplexity"] = perplexities

    return df

def add_perplexity_column_huggingface_1(df):
    print("Loading models")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    print("Finished Loading models")
    perplexities = []
    for i, text in enumerate(df["abstract"]):
        perplexity = calculate_perplexity_gpt2(text, tokenizer, model)
        perplexities.append(perplexity)
        print(f"Finished processing text {i+1}/{len(df)}")

    df["Perplexity"] = perplexities

    return df

dataset_with_perplexity = add_perplexity_column_huggingface_1(dataset)

print(dataset_with_perplexity.head())



# Create a box plot to visualize the distribution of perplexity scores
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x="is_ai_generated", y="Perplexity", data=dataset_with_perplexity,showfliers = False) ##REMOVED OUTLIERS!! 
ax.set_xticklabels(["Human-generated", "AI-generated"])
ax.set_xlabel("Text")
ax.set_ylabel("Perplexity")
ax.set_title("Perplexity Distribution for Human-generated vs. AI-generated Text")

plt.show()



#---------------------------------- Stratistics -------------------------------------------#
def ngram_distribution(texts, max_ngram_length):
    all_ngrams = []
    for text in texts:
        tokens = word_tokenize(text)
        for n in range(1, max_ngram_length + 1):
            text_ngrams = list(ngrams(tokens, n))
            all_ngrams.extend(text_ngrams)
    return Counter(all_ngrams)

def descriptive_statistics(df, max_ngram_length=8):
    human_texts = df[df["is_ai_generated"] == 0]["abstract"]
    ai_texts = df[df["is_ai_generated"] == 1]["abstract"]
    
    return {
        "Human N-gram Distribution": ngram_distribution(human_texts, max_ngram_length),
        "AI N-gram Distribution": ngram_distribution(ai_texts, max_ngram_length),
        "Human Average Text Length": np.mean([len(word_tokenize(text)) for text in human_texts]),
        "AI Average Text Length": np.mean([len(word_tokenize(text)) for text in ai_texts]),
        "Human Average Text character Length": np.mean([len([*text]) for text in human_texts]),
        "AI Average Text character Length": np.mean([len([*text]) for text in ai_texts]),
    }

def plot_ngram_distribution(statistics, n):
    data = []
    for category in ["Human", "AI"]:
        ngrams = statistics[f"{category} N-gram Distribution"]
        counts = [count for ngram, count in ngrams.items() if len(ngram) == n]
        data.append(counts)

    fig, ax = plt.subplots()
    ax.boxplot(data, labels=[f"{category} {n}-grams" for category in ["Human", "AI"]])
    ax.set_title(f"{n}-gram Frequency Distribution")
    plt.show()

def plot_average_text_lengths(statistics):
    data = [statistics[f"{category} Average Text Length"] for category in ["Human", "AI"]]

    fig, ax = plt.subplots()
    ax.bar(["Human", "AI"], data)
    ax.set_ylabel("Average Text Length")
    ax.set_title("Comparison of Average Text Lengths")
    plt.show()
    
def plot_average_text_Character_lengths(statistics):
    data = [statistics[f"{category} Average Text character Length"] for category in ["Human", "AI"]]

    fig, ax = plt.subplots()
    ax.bar(["Human", "AI"], data)
    ax.set_ylabel("Average Text character Length")
    ax.set_title("Comparison of Average Text character Lengths")
    plt.show()

def create_top_grams_dataframes(stat, n):
    data = []
    for category in ["Human", "AI"]:
        ngrams = stat[f"{category} N-gram Distribution"]
        top_ngrams = [item for item in ngrams.items() if len(item[0]) == n]
        top_ngrams.sort(key=lambda x: x[1], reverse=True)
        top_ngrams = top_ngrams[:10]
        data.append(pd.DataFrame(top_ngrams, columns=['ngram', 'count']))
        data[-1]['type'] = category

    return data[0], data[1]

statistics = descriptive_statistics(dataset, max_ngram_length=8)

for n in range(1, 8):
    plot_ngram_distribution(statistics, n)

plot_average_text_lengths(statistics)
plot_average_text_Character_lengths(statistics)


for n in range(1, 4):
    human_df, ai_df = create_top_grams_dataframes(statistics, n)
    print(f"Top 10 {n}-grams for Human Text:")
    print(human_df)
    print(f"\nTop 10 {n}-grams for AI-generated Text:")
    print(ai_df)
    print("\n" + "="*80 + "\n")


def add_ngram_columns(dataset, max_ngram_length=3):
    for n in range(1, max_ngram_length + 1):
        dataset[f'{n}-gram Distribution'] = dataset['abstract'].apply(lambda text: ngram_distribution(text, n))
    return dataset

# -----------------------------------------------------Grammer -------------------------------------------------------#

# https://pypi.org/project/language-tool-python/
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
#https://spacy.io/usage/models#download-pip
#import spacy #Tried to get it into a spacy pipline for efficiency but could not get it to work. 
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

def grammar_score(text):
    matches = tool.check(text)
    errors = len(matches)
    return errors / len(text.split())

def calculate_grammar_scores(df, text_column="abstract"):
    texts = df[text_column].tolist()
    grammar_scores = [grammar_score(text) for text in texts]
    df["Grammar Score"] = grammar_scores
    return df

df_with_grammar_scores = calculate_grammar_scores(dataset)



plt.figure(figsize=(10, 6))
ax = sns.boxplot(x="is_ai_generated", y="Grammar Score", data=df_with_grammar_scores,showfliers = False) ##REMOVED OUTLIERS!! 
ax.set_xticklabels(["Human-generated", "AI-generated"])
ax.set_xlabel("Text Type")
ax.set_ylabel("Grammar Score")
ax.set_title("Grammar Score for Human-generated vs. AI-generated Text")

plt.show()


#-------------------------------------------- TTR -------------------------------------#

##TTR is the ratio obtained by dividing the types (the total number of different words) 
##occurring in a text or utterance by its tokens (the total number of words). 
##A high TTR indicates a high degree of lexical variation while a low TTR indicates the opposite.
##
##
# ------------------------------------------- SpaCy -------------------------------------------------------------#
#cython: auto_pickle=False 
#import sys; print(sys.implementation)

#import spacy

#nlp = spacy.load("en_core_web_sm")
#doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
#for token in doc:
#    print(token.text, token.pos_, token.dep_)

#------------------------------------------------------ Modelling ---------------------------------------------------#

import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer


# Function for text normalization
def text_normalizer(texts):
    normalized_texts = []
    for text in texts:
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        normalized_texts.append(text)
    return normalized_texts

# Split the data into training, validation, and testing sets
X = dataset["abstract"]
y = dataset["is_ai_generated"]

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('text_normalizer', FunctionTransformer(text_normalizer, validate=False)),
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Define the parameter grid for GridSearchCV
param_grid = [
    {
        'tfidf__max_df': [0.9, 1.0],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'classifier': [LogisticRegression()],
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2']
    },
    {
        'tfidf__max_df': [0.9, 1.0],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20]
    }
]

# Perform a grid search with cross-validation #high presision
grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best combination of hyperparameters
print("Best parameters: ", grid_search.best_params_)

# Evaluate the model on Train data
y_train_pred = grid_search.predict(X_train)
print("Validation Classification Report:\n", classification_report(y_train, y_train_pred))

# Evaluate the model on  data
y_val_pred = grid_search.predict(X_val)
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))


from sklearn.metrics import classification_report
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from datasets import Dataset


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
# Train-test-validation split

train_df, test_val_df = train_test_split(dataset, test_size=0.3, stratify=dataset["is_ai_generated"], random_state=42)
val_df, test_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df["is_ai_generated"], random_state=42)

# Convert the DataFrames to HuggingFace's Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

print(train_dataset.column_names)
print(train_dataset)

# Tokenize the input text
def tokenize(batch):
    return tokenizer(batch['abstract'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')


train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['abstract', 'ai_generated', 'is_ai_generated'])
val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=['abstract', 'ai_generated', 'is_ai_generated'])
test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=['abstract', 'ai_generated', 'is_ai_generated'])

# Set the dataset format to PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'is_ai_generated'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'is_ai_generated'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'is_ai_generated'])

# Create the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Evaluate the model on validation data
val_predictions = trainer.predict(val_dataset)
val_preds = np.argmax(val_predictions.predictions, axis=1)
print("Validation Classification Report:\n", classification_report(y_val, val_preds))

# Create a classification pipeline for inference
text_classification_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

# Evaluate the model on test data
y_pred = [np.argmax(pred['score']) for pred in text_classification_pipeline(list(X_test))]
print("Test Classification Report:\n", classification_report(y_test, y_pred))



