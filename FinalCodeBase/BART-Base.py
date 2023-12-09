#!/usr/bin/env python
# coding: utf-8
import re
import os
import pandas as pd
import numpy as np
import chardet
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from contractions import contractions_dict

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

import warnings
warnings.filterwarnings("ignore")

def expand_contractions(text, contraction_map=None):
    """
    Purpose:
    Expand contractions in the given text using a contraction map.

    Args:
    - text (str): The input text with contractions.
    - contraction_map (dict): A dictionary mapping contractions to their expanded form.

    Returns:
    - str: The text with expanded contractions.
    """

    if contraction_map is None:
        contraction_map = contractions_dict

    contractions_keys = '|'.join(re.escape(key) for key in contraction_map.keys())
    contractions_pattern = re.compile(f'({contractions_keys})', flags=re.DOTALL)

    expanded_text = contractions_pattern.sub(lambda match: contraction_map.get(match.group(0), match.group(0)), text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def detect_encoding(file_path):
    """
    Purpose:
    Detect the encoding of a file using the chardet library.

    Args:
    - file_path (str): The path to the file.

    Returns:
    - str: The detected encoding.
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# Read individual File of News articles of each class and load them into respective Dataframes
print("Loading Data....")
base_dir = "data/BBCNewsSummary/News Articles"
output_dir = "data/BBCNewsSummaryCSV"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

classes = os.listdir(base_dir)
dfs = {}
# Create dataframes and write to CSV files for each class
for class_name in classes:
    news_articles_path = os.path.join(base_dir, class_name)
    summaries_path = os.path.join("data/BBCNewsSummary/Summaries", class_name)
    news_articles_files = os.listdir(news_articles_path)
    summaries_files = os.listdir(summaries_path)
    common_files = set(news_articles_files) & set(summaries_files)
    df_list = []

    # Read content from files and populate dataframe
    for filename in sorted(common_files):
        news_article_file_path = os.path.join(news_articles_path, filename)
        summary_file_path = os.path.join(summaries_path, filename)
        # Detect encoding
        news_encoding = detect_encoding(news_article_file_path)
        summary_encoding = detect_encoding(summary_file_path)

        try:
            with open(news_article_file_path, 'r', encoding=news_encoding) as news_file, \
                    open(summary_file_path, 'r', encoding=summary_encoding) as summary_file:
                news_content = news_file.read()
                summary_content = summary_file.read()
                file_name_without_extension = os.path.splitext(filename)[0]
                new_filename = f'{class_name}_{file_name_without_extension}'
                df_list.append({'filename': new_filename, 'newsarticle': news_content, 'summary': summary_content})

        except UnicodeDecodeError:
            print(f"UnicodeDecodeError: Could not read {filename}. Skipping this file.")

    # Create dataframe from the list and sort by filename
    df = pd.DataFrame(df_list).sort_values(by='filename')

    # Save dataframe to the dictionary
    dfs[class_name] = df

    # Write dataframe to CSV in the output directory
    # csv_filename = os.path.join(output_dir, f'{class_name}_data.csv')
    # df.to_csv(csv_filename, index=False)
    # print(f'Dataframe for {class_name} written to {csv_filename}')

business = dfs['business']
entertainment = dfs['entertainment']
sport = dfs['sport']
politics = dfs['politics']
tech = dfs['tech']

sample_text = business['newsarticle'][242]
print(sample_text)

# entertainment = pd.read_csv('data/BBCNewsSummaryCSV/entertainment_data.csv') 
# sport = pd.read_csv('data/BBCNewsSummaryCSV/sport_data.csv')
# tech = pd.read_csv('data/BBCNewsSummaryCSV/tech_data.csv')
# business = pd.read_csv('data/BBCNewsSummaryCSV/business_data.csv')
# politics = pd.read_csv('data/BBCNewsSummaryCSV/politics_data.csv')

# Combine dataframes for training and testing datasets
# - Combine Articles of Business, Politics and Technology for Training
# - Combine Articles of Entertainment and Sport for Testing
training_dataset = pd.concat([business,politics,tech], ignore_index=True)
testing_dataset = pd.concat([entertainment,sport], ignore_index=True)

print("Training size:",training_dataset.size)
print("Testing size:",testing_dataset.size)

print("Preprocessing Data...")
# Shuffle the training and testing datasets
training_dataset = training_dataset.sample(frac=1).reset_index(drop=True)
testing_dataset = testing_dataset.sample(frac=1).reset_index(drop=True)

# Apply contraction expansion to news articles in both datasets
training_dataset['newsarticle'] = training_dataset['newsarticle'].apply(expand_contractions)
testing_dataset['newsarticle'] = testing_dataset['newsarticle'].apply(expand_contractions)


def getSenLen(sentence):
    """
    Purpose:
    Get the length of a sentence in words.

    Args:
    - sentence (str): The input sentence.

    Returns:
    - int: The length of the sentence in words.
    """
    return len(sentence.split())


# Apply the getSenLen function to calculate article and summary lengths
training_dataset['article_length'] = training_dataset['newsarticle'].apply(getSenLen)
training_dataset['summary_length'] = training_dataset['summary'].apply(getSenLen)
testing_dataset['article_length'] = testing_dataset['newsarticle'].apply(getSenLen)
testing_dataset['summary_length'] = testing_dataset['summary'].apply(getSenLen)

print(training_dataset.head())

print("Remove Outliers based on Text Length")
# Plot boxplots for article and summary lengths
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

sns.boxplot(training_dataset["article_length"], ax=axes[0])
axes[0].set_ylabel("Number of Words")
axes[0].set_title("Boxplot of Article Lengths")

sns.boxplot(training_dataset["summary_length"], ax=axes[1])
axes[1].set_ylabel("Number of Words")
axes[1].set_title("Boxplot of Summary Lengths")
plt.ion()
plt.show()

print(training_dataset.describe())

# Get statistics for the articles boxplot
lines_articles = axes[0].lines[:6]
articles_stats = [line.get_ydata()[0] for line in lines_articles]
# print(articles_stats)
Q1_articles, Q3_articles, lower_whisker_articles, upper_whisker_articles, median_articles = articles_stats[:5]

# Get statistics for the summaries boxplot
lines_summaries = axes[1].lines[:6]
summaries_stats = [line.get_ydata()[0] for line in lines_summaries]
# print(summaries_stats)
Q1_summaries, Q3_summaries, lower_whisker_summaries, upper_whisker_summaries, median_summaries = summaries_stats[:5]

#Alternate Approach for getting UpperWhiskers
def getUpperWhiskers(description):
    Q1 = description["25%"]
    Q3 = description["75%"]
    IQR = Q3-Q1
    upperWhisker = int(Q3+1.5*IQR)
    return upperWhisker

upper_whisker_articles = getUpperWhiskers(training_dataset.describe()['article_length'])
upper_whisker_summaries = getUpperWhiskers(training_dataset.describe()['summary_length'])
print(upper_whisker_articles)
print(upper_whisker_summaries)

# Remove outliers based on upper whisker values
training_dataset = training_dataset[(training_dataset['summary_length'] <= upper_whisker_summaries) & (training_dataset['article_length'] <= upper_whisker_articles)]
testing_dataset = testing_dataset[(testing_dataset['summary_length'] <= upper_whisker_summaries) & (testing_dataset['article_length'] <= upper_whisker_articles)]

# Create boxplots again after removing outliers
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot for the articles' number of words
sns.boxplot(training_dataset["article_length"], ax=axes[0])
axes[0].set_ylabel("Number of Words")
axes[0].set_title("Boxplot of Article Lengths")

# Plot for the summaries' number of words
sns.boxplot(training_dataset["summary_length"], ax=axes[1])
axes[1].set_ylabel("Number of Words")
axes[1].set_title("Boxplot of Summary Lengths")
plt.ion()
plt.show()

print(training_dataset.head())
print(training_dataset.describe())

# df = training_dataset[0:10]
df = training_dataset


# Determine the available device (CPU, GPU, or MPS)
if torch.backends.mps.is_available():
    arch = "mps"
elif torch.cuda.is_available():
    arch = "cuda"
else:
    arch = "cpu"

device = torch.device(arch)
# device = torch.device("cpu")
print(f"Device Set to {arch}")

# Tokenize and preprocess the text data
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')  # Initialize BART tokenizer
max_length = 512  # Maximum sequence length
print("Loaded Tokeniser")

def tokenize_text(text):
    """
    Purpose:
    Tokenize the input text using the BART tokenizer, adding special tokens and attention masks.

    Args:
    - text (str): The input text to be tokenized.

    Returns:
    - torch.Tensor: Tokenized input text as a PyTorch tensor.
    """
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True, padding='max_length', return_attention_mask=True)
    return inputs.to(device)


def tokenize_summary(text):
    """
    Purpose:
    Tokenize the input summary using the BART tokenizer, adding special tokens and attention masks.

    Args:
    - text (str): The input summary to be tokenized.

    Returns:
    - torch.Tensor: Tokenized input summary as a PyTorch tensor.
    """
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=280, truncation=True, padding='max_length', return_attention_mask=True)
    return inputs.to(device)


def tokenize_and_stack(df, text_column, summary_column):
    """
    Purpose:
    Tokenize the text and summary columns in the given dataframe and stack the tokenized sequences into PyTorch tensors.

    Args:
    - df (pd.DataFrame): The input dataframe containing text and summary columns.
    - text_column (str): The name of the column containing the text to be tokenized.
    - summary_column (str): The name of the column containing the summary to be tokenized.

    Returns:
    - tuple: A tuple containing three elements - X (tokenized text as PyTorch tensor),
             Y (tokenized summary as PyTorch tensor), and dataloader (DataLoader for the tokenized data).
    """
    df['TokenizedText'] = df[text_column].apply(tokenize_text)
    df['TokenizedSummary'] = df[summary_column].apply(tokenize_summary)

    # Convert tokenized data to PyTorch tensors
    X = torch.stack([seq.squeeze() for seq in df['TokenizedText']])
    Y = torch.stack([seq.squeeze() for seq in df['TokenizedSummary']])

    # Define a DataLoader for batching data
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    return X, Y, dataloader


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
test_df = testing_dataset[0:len(val_df)]

print("Tokenising and Stacking datasets..")
# Tokenize and stack the input datasets
X_train, Y_train, train_dataloader = tokenize_and_stack(train_df, 'newsarticle', 'summary')
X_val, Y_val, val_dataloader = tokenize_and_stack(val_df, 'newsarticle', 'summary')
X_test, Y_test, test_dataloader = tokenize_and_stack(test_df, 'newsarticle', 'summary')


def genSummaryAndEvaluate(model, dataloader):
    """
    Purpose:
    Generate summaries using the provided model and evaluate the generated summaries against ground truth.

    Args:
    - model (BartForConditionalGeneration): The pre-trained BART model.
    - dataloader (DataLoader): DataLoader containing the validation/test dataset.

    Returns:
    - pd.DataFrame: DataFrame containing evaluation results, including actual summaries, predicted summaries,
                    ROUGE-1 Precision scores, and BERT scores.
    """
    print("Evaluation takes time as it involves Generating Text!!!")
    model.eval()

    test_articles = []
    actual_summaries = []
    predicted_summaries = []
    rouge1_precision_scores = []
    bert_scores = []

    # Initialize Rouge scorer for evaluation
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    # Disable gradient computation during evaluation
    with torch.no_grad():
        # Iterate through the validation dataloader
        for batch in tqdm(dataloader, desc="Evaluating Model"):
            inputs = batch[0].to(device)
            attention_mask = (inputs != 0).float().to(device)
            targets = batch[1].to(device)
            max_length = 280  # Maximum length for generated summaries
            # Generate summaries using the model
            outputs = model.generate(input_ids=inputs, attention_mask=attention_mask, max_length=280, num_beams=17, length_penalty=2.0, early_stopping=False)

            for output, target, input_text in zip(outputs, targets, inputs):
                #  ROUGE-1 precision
                output_text = tokenizer.decode(output, skip_special_tokens=True)
                target_text = tokenizer.decode(target, skip_special_tokens=True)
                target_text = ' '.join(target_text.split()[:max_length])
                rouge_scores = scorer.score(output_text, target_text)
                rouge1_precision_scores.append(rouge_scores['rouge1'].precision)

                # # BERTScore
                _, _, bert_score_f1 = bert_score([output_text], [target_text], lang='en', model_type='bert-base-uncased')
                bert_scores.append(bert_score_f1.item())

                # Append tokenized text, actual summaries, and predicted summaries
                test_articles.append(tokenizer.decode(input_text, skip_special_tokens=True))
                actual_summaries.append(target_text)
                predicted_summaries.append(output_text)

    # Evaluation Results
    data = {
        'Article': test_articles,
        'Actual Summary': actual_summaries,
        'Predicted Summary': predicted_summaries,
        'ROUGE-1 Precision': rouge1_precision_scores,
        'BERT Score': bert_scores,

    }
    results_df = pd.DataFrame(data)
    return results_df


# Load the pre-trained BART model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# Initialize gradient scaler for mixed-precision training
scaler = GradScaler()

# Define hyperparameters
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)  # Optimiser
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=len(train_dataloader) * 10)  # Add learning rate scheduler
accumulation_steps = 40  # Number of steps before performing gradient update

train_losses = []
rouge_scores = []
bert_scores = []
# Training loop
print("Starting training the Model... Training takes Time 4-6 hours.")
for epoch in range(3):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    # Training loss calculation and gradient accumulation
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{2}")):
        inputs = batch[0].to(device)  # Extract inputs, attention masks, and targets from the batch
        attention_mask = (inputs != 0).float().to(device)  # Create attention mask
        targets = batch[1].to(device)

        # Enable auto-casting for mixed-precision training
        with autocast():
            outputs = model(input_ids=inputs, attention_mask=attention_mask, decoder_input_ids=targets, labels=targets)
            loss = outputs.loss

        # Perform gradient accumulation
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        # Update gradients and optimizer once every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping to prevent exploding gradients
            # Update model parameters using the optimizer and # Reset gradients in the optimizer
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

    train_loss = total_loss / len(train_dataloader)
    train_losses.append(train_loss)
    # Evaluation on validation dataset
    print("Model is being continuously Evaluated on Eval Dataset...")
    evalResult = genSummaryAndEvaluate(model,val_dataloader)
    rouge = evalResult['ROUGE-1 Precision']
    bert = evalResult['BERT Score']
    rouge_scores.append(rouge.mean())
    bert_scores.append(bert.mean())
    print(f"Epoch {epoch+1}/{3}, Train Loss: {train_loss:.4f}, Mean Evaluation Rouge-1 Score: {rouge.mean()}, Mean Evaluation BERT Score: {bert.mean()}")

sameCategoryData = evalResult

# Save the model
model.save_pretrained("KYS_BART_BASE")
# model.save_pretrained("TEST")

print(train_losses)
print(rouge_scores)
print(bert_scores)

# Plot the Learning Curve for training losses
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot learning curves for training losses, Rouge scores, and BERT scores
epochs = list(range(1, len(train_losses) + 1))
# Create subplots
fig, ax1 = plt.subplots(figsize=(10, 6))
# Plot train losses on the first y-axis
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color='tab:blue')
ax1.plot(epochs, train_losses, color='tab:blue', label='Train Loss')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')
# Create a second y-axis for ROUGE and BERT scores
ax2 = ax1.twinx()
ax2.set_ylabel('ROUGE / BERT Scores', color='tab:red')
# Plot ROUGE scores on the second y-axis
ax2.plot(epochs, rouge_scores, color='tab:red', linestyle='dashed', label='ROUGE Scores')
ax2.tick_params(axis='y', labelcolor='tab:red')
# Plot BERT scores on the second y-axis
ax2.plot(epochs, bert_scores, color='tab:orange', linestyle='dashed', label='BERT Scores')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.legend(loc='upper right')
plt.title('Learning Curves')
plt.show()

print(sameCategoryData[['ROUGE-1 Precision','BERT Score']])
print(sameCategoryData['ROUGE-1 Precision'].mean(),sameCategoryData['BERT Score'].mean())

# Testing Model Performance on Different Domain News
print('Applying the Trained model on News of different Domain....')
diffCategoryData = genSummaryAndEvaluate(model,test_dataloader)
print(diffCategoryData[['ROUGE-1 Precision','BERT Score']])
print(diffCategoryData['ROUGE-1 Precision'].mean(),diffCategoryData['BERT Score'].mean())


# Means of Evaluation Metrics
same_category_mean_rouge1 = sameCategoryData['ROUGE-1 Precision'].mean()
same_category_mean_bert = sameCategoryData['BERT Score'].mean()

diff_category_mean_rouge1 = diffCategoryData['ROUGE-1 Precision'].mean()
diff_category_mean_bert = diffCategoryData['BERT Score'].mean()

categories = ['ROUGE-1 Precision', 'BERT Score']
mean_scores_same_category = [same_category_mean_rouge1, same_category_mean_bert]
mean_scores_diff_category = [diff_category_mean_rouge1, diff_category_mean_bert]

# Plot Graphs of Evaluation Metrics - Mean and Median

bar_width = 0.35
index = np.arange(len(categories))

fig, ax = plt.subplots()
bar1 = ax.bar(index, mean_scores_same_category, bar_width, label='Same Category')
bar2 = ax.bar(index + bar_width, mean_scores_diff_category, bar_width, label='Different Category')

ax.set_xlabel('Metrics')
ax.set_ylabel('Mean Score')
ax.set_title('Mean ROUGE-1 Precision and BERT Score Comparison between Categories')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(categories)
ax.legend()

plt.show()


same_category_median_rouge1 = sameCategoryData['ROUGE-1 Precision'].median()
same_category_median_bert = sameCategoryData['BERT Score'].median()

diff_category_median_rouge1 = diffCategoryData['ROUGE-1 Precision'].median()
diff_category_median_bert = diffCategoryData['BERT Score'].median()

categories = ['ROUGE-1 Precision', 'BERT Score']
median_scores_same_category = [same_category_median_rouge1, same_category_median_bert]
median_scores_diff_category = [diff_category_median_rouge1, diff_category_median_bert]

bar_width = 0.35
index = np.arange(len(categories))

fig, ax = plt.subplots()
bar1 = ax.bar(index, median_scores_same_category, bar_width, label='Same Category')
bar2 = ax.bar(index + bar_width, median_scores_diff_category, bar_width, label='Different Category')

ax.set_xlabel('Metrics')
ax.set_ylabel('median Score')
ax.set_title('median ROUGE-1 Precision and BERT Score Comparison between Categories')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(categories)
ax.legend()

plt.show()


# Calculate mean values
sameCat_mean_rouge = np.mean(sameCategoryData['ROUGE-1 Precision'])
sameCat_mean_bert = np.mean(sameCategoryData['BERT Score'])

# Create separate histogram plots for Rouge and BERT Scores
plt.figure(figsize=(12, 6))
# plt.title('Same Category Data')

# Rouge Score Histogram
plt.subplot(1, 2, 1)
plt.hist(sameCategoryData['ROUGE-1 Precision'], bins=15, color='#1f77b4')
plt.axvline(sameCat_mean_rouge, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {sameCat_mean_rouge:.3f}')
plt.title('Same Category Data Rouge Score Histogram')
plt.xlabel('Rouge Score')
plt.ylabel('Frequency')

# BERT Score Histogram
plt.subplot(1, 2, 2)
plt.hist(sameCategoryData['BERT Score'], bins=15, color='darkorange')
plt.axvline(sameCat_mean_bert, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {sameCat_mean_bert:.3f}')
plt.title('Same Category Data BERT Score Histogram')
plt.xlabel('BERT Score')
plt.ylabel('Frequency')

plt.tight_layout()

plt.show()


def generate_summary(model, text, max_length=280, num_beams=17, length_penalty=2.0, early_stopping=False):
    """
    Purpose:
    Reusable Method Generate a summary for a given input text using the trained Summarizer.

    Args:
    - model (BartForConditionalGeneration): The pre-trained trained Summarizer.
    - text (str): The input text for which the summary is to be generated.
    - max_length (int): Maximum length of the generated summary.
    - num_beams (int): Number of beams for beam search.
    - length_penalty (float): Length penalty for beam search.
    - early_stopping (bool): Whether to stop generation when all beam hypotheses have reached the maximum length.

    Returns:
    - str: Generated summary for the input text.
    """
    model.eval()

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)['input_ids'].to(device)

    # Generate summary
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs, max_length=max_length, num_beams=num_beams, length_penalty=length_penalty, early_stopping=early_stopping)

    # Decode the generated summary
    summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary_text


# Example usage:
text_to_summarize = testing_dataset['newsarticle'][43]

summary = generate_summary(model, text_to_summarize)
# print(f"Text:\n{text_to_summarize}")
print(f"Summary:\n{summary}")

# get_ipython().system('jupyter nbconvert --to script BART-Base.ipynb')
