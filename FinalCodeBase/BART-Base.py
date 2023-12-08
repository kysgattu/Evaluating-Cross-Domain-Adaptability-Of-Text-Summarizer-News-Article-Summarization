#!/usr/bin/env python
# coding: utf-8
import re
import os
import pandas as pd
import chardet
import matplotlib.pyplot as plt
import seaborn as sns
from contractions import contractions_dict

def expand_contractions(text, contraction_map=None):
    if contraction_map is None:
        contraction_map = contractions_dict

    # Using regex for getting all contracted words
    contractions_keys = '|'.join(re.escape(key) for key in contraction_map.keys())
    contractions_pattern = re.compile(f'({contractions_keys})', flags=re.DOTALL)

    expanded_text = contractions_pattern.sub(lambda match: contraction_map.get(match.group(0), match.group(0)), text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# Set the base directory
base_dir = "data/BBCNewsSummary/News Articles"
output_dir = "data/BBCNewsSummaryCSV"  # Output directory

# Check if the output directory exists, and create it if not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the list of classes (subfolder names)
classes = os.listdir(base_dir)
dfs = {}
# Create dataframes and write to CSV files for each class
for class_name in classes:
    # Define the paths for news articles and summaries
    news_articles_path = os.path.join(base_dir, class_name)
    summaries_path = os.path.join("data/BBCNewsSummary/Summaries", class_name)

    # Get the list of file names in both directories
    news_articles_files = os.listdir(news_articles_path)
    summaries_files = os.listdir(summaries_path)

    # Match file names
    common_files = set(news_articles_files) & set(summaries_files)

    # Create dataframe
    df_list = []

    # Read content from files and populate dataframe
    for filename in sorted(common_files):  # Sort by filename
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

                # Extract file name without extension
                file_name_without_extension = os.path.splitext(filename)[0]

                # Generate the new entry in the filename
                new_filename = f'{class_name}_{file_name_without_extension}'

                df_list.append({'filename': new_filename, 'newsarticle': news_content, 'summary': summary_content})

        except UnicodeDecodeError:
            print(f"UnicodeDecodeError: Could not read {filename}. Skipping this file.")

    # Create dataframe from the list and sort by filename
    df = pd.DataFrame(df_list).sort_values(by='filename')
    
    # Save dataframe to the dictionary
    dfs[class_name] = df

    # # Write dataframe to CSV in the output directory
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

training_dataset = pd.concat([business,politics,tech], ignore_index=True)
testing_dataset = pd.concat([entertainment,sport], ignore_index=True)

print("Training size:",training_dataset.size)
print("Testing size:",testing_dataset.size)

training_dataset = training_dataset.sample(frac=1).reset_index(drop=True)
testing_dataset = testing_dataset.sample(frac=1).reset_index(drop=True)

training_dataset['newsarticle'] = training_dataset['newsarticle'].apply(expand_contractions)
testing_dataset['newsarticle'] = testing_dataset['newsarticle'].apply(expand_contractions)

def getSenLen(sentence):
    return len(sentence.split())

training_dataset['article_length'] = training_dataset['newsarticle'].apply(getSenLen)
training_dataset['summary_length'] = training_dataset['summary'].apply(getSenLen)
testing_dataset['article_length'] = testing_dataset['newsarticle'].apply(getSenLen)
testing_dataset['summary_length'] = testing_dataset['summary'].apply(getSenLen)

print(training_dataset.head())

# Boxplots for Article and Summary Lengths
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

sns.boxplot(training_dataset["article_length"], ax=axes[0])
axes[0].set_ylabel("Number of Words")
axes[0].set_title("Boxplot of Article Lengths")

sns.boxplot(training_dataset["summary_length"], ax=axes[1])
axes[1].set_ylabel("Number of Words")
axes[1].set_title("Boxplot of Summary Lengths")

print(training_dataset.describe())

# Get statistics for the articles boxplot
lines_articles = axes[0].lines[:6]
articles_stats = [line.get_ydata()[0] for line in lines_articles]
Q1_articles, Q3_articles, lower_whisker_articles, upper_whisker_articles, median_articles = articles_stats[:5]

# Get statistics for the summaries boxplot
lines_summaries = axes[1].lines[:6]
summaries_stats = [line.get_ydata()[0] for line in lines_summaries]
Q1_summaries, Q3_summaries, lower_whisker_summaries, upper_whisker_summaries, median_summaries = summaries_stats[:5]

print(upper_whisker_articles)
print(upper_whisker_summaries)

training_dataset = training_dataset[(training_dataset['summary_length'] <= upper_whisker_summaries) & (training_dataset['article_length'] <= upper_whisker_articles)]
testing_dataset = testing_dataset[(testing_dataset['summary_length'] <= upper_whisker_summaries) & (testing_dataset['article_length'] <= upper_whisker_articles)]

# Create a subplot with 1 row and 2 columns
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot for the articles' number of words
sns.boxplot(training_dataset["article_length"], ax=axes[0])
axes[0].set_ylabel("Number of Words")
axes[0].set_title("Boxplot of Article Lengths")

# Plot for the summaries' number of words
sns.boxplot(training_dataset["summary_length"], ax=axes[1])
axes[1].set_ylabel("Number of Words")
axes[1].set_title("Boxplot of Summary Lengths")

print(training_dataset.head())
print(training_dataset.describe())

# df = training_dataset[0:100]
df = training_dataset

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

# Define the device for GPU usage (if available)
if torch.backends.mps.is_available():
    arch = "mps"
elif torch.cuda.is_available():
    arch = "cuda"
else:
    arch = "cpu"

device = torch.device(arch)
# device = torch.device("cpu")

# Tokenize and preprocess the text data
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
max_length = 512  # Maximum sequence length

def tokenize_text(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True, padding='max_length', return_attention_mask=True)
    return inputs.to(device)  # Move the tokenized inputs to the GPU

def tokenize_summary(text):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=280, truncation=True, padding='max_length', return_attention_mask=True)
    return inputs.to(device)  # Move the tokenized summaries to the GPU

def tokenize_and_stack(df, text_column, summary_column):
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

# Tokenize and stack for training set
X_train, Y_train, train_dataloader = tokenize_and_stack(train_df, 'newsarticle', 'summary')

# Tokenize and stack for validation set
X_val, Y_val, val_dataloader = tokenize_and_stack(val_df, 'newsarticle', 'summary')

# Tokenize and stack for validation set
X_test, Y_test, test_dataloader = tokenize_and_stack(test_df, 'newsarticle', 'summary')

def genSummaryAndEvaluate(model, dataloader):
    model.eval()
    
    test_articles = []
    actual_summaries = []
    predicted_summaries = []
    rouge1_precision_scores = []
    bert_scores = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Test"):
            inputs = batch[0].to(device)
            attention_mask = (inputs != 0).float().to(device)
            targets = batch[1].to(device)
            max_length=280
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

    data = {
        'Article': test_articles,
        'Actual Summary': actual_summaries,
        'Predicted Summary': predicted_summaries,
        'ROUGE-1 Precision': rouge1_precision_scores,
        'BERT Score': bert_scores,

    }
    results_df = pd.DataFrame(data)
    return results_df

# Define the BART model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# Create a GradScaler for mixed-precision training
scaler = GradScaler()

# Define hyperparameters
model.to(device)  # Move the model to the GPU
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)  
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=len(train_dataloader) * 10)  # Add learning rate scheduler

# Define gradient accumulation steps
accumulation_steps = 40  # You can adjust this number

train_losses = []
rouge_scores = []
bert_scores = []
# Training loop
for epoch in range(3):  # Change the number of epochs as needed
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{2}")):
        inputs = batch[0].to(device)  # Move the input batch to the GPU
        attention_mask = (inputs != 0).float().to(device)  # Create attention mask
        targets = batch[1].to(device)  # Move the target batch to the GPU

        with autocast():
            outputs = model(input_ids=inputs, attention_mask=attention_mask, decoder_input_ids=targets, labels=targets)
            loss = outputs.loss

        # Perform gradient accumulation
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            # Update gradients and optimizer once every accumulation_steps
            clip_grad_norm_(model.parameters(), max_norm=1.0)  # Optional gradient clipping
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

    train_loss = total_loss / len(train_dataloader)
    train_losses.append(train_loss)
    evalResult = genSummaryAndEvaluate(model,val_dataloader)
    rouge = evalResult['ROUGE-1 Precision']
    bert = evalResult['BERT Score']
    rouge_scores.append(rouge.mean())
    bert_scores.append(bert.mean())
    print(f"Epoch {epoch+1}/{2}, Train Loss: {train_loss:.4f}, Mean Evaluation Rouge-1 Score: {rouge.mean()}, Mean Evaluation BERT Score: {bert.mean()}")

sameCategoryData = evalResult

# Save the model after training
model.save_pretrained("KYS_BART_BASE")

print(train_losses)
print(rouge_scores)
print(bert_scores)

# Plot the Learning Curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

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

# Add legend
ax2.legend(loc='upper right')

# Show the plot
plt.title('Learning Curves')
plt.show()

sameCategoryData[['ROUGE-1 Precision','BERT Score']]
print(sameCategoryData['ROUGE-1 Precision'].mean(),sameCategoryData['BERT Score'].mean())

diffCategoryData = genSummaryAndEvaluate(model,test_dataloader)

diffCategoryData[['ROUGE-1 Precision','BERT Score']]
print(diffCategoryData['ROUGE-1 Precision'].mean(),diffCategoryData['BERT Score'].mean())

import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual data)
same_category_mean_rouge1 = sameCategoryData['ROUGE-1 Precision'].mean()
same_category_mean_bert = sameCategoryData['BERT Score'].mean()

diff_category_mean_rouge1 = diffCategoryData['ROUGE-1 Precision'].mean()
diff_category_mean_bert = diffCategoryData['BERT Score'].mean()

categories = ['ROUGE-1 Precision', 'BERT Score']
mean_scores_same_category = [same_category_mean_rouge1, same_category_mean_bert]
mean_scores_diff_category = [diff_category_mean_rouge1, diff_category_mean_bert]

bar_width = 0.35
index = np.arange(len(categories))

fig, ax = plt.subplots()
bar1 = ax.bar(index, mean_scores_same_category, bar_width, label='Same Category')
bar2 = ax.bar(index + bar_width, mean_scores_diff_category, bar_width, label='Different Category')

# Add labels, title, and legend
ax.set_xlabel('Metrics')
ax.set_ylabel('Mean Score')
ax.set_title('Mean ROUGE-1 Precision and BERT Score Comparison between Categories')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(categories)
ax.legend()

# Show the plot
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual data)
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

# Add labels, title, and legend
ax.set_xlabel('Metrics')
ax.set_ylabel('median Score')
ax.set_title('median ROUGE-1 Precision and BERT Score Comparison between Categories')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(categories)
ax.legend()

# Show the plot
plt.show()


import matplotlib.pyplot as plt

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

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


def generate_summary(model, text, max_length=280, num_beams=17, length_penalty=2.0, early_stopping=False):
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