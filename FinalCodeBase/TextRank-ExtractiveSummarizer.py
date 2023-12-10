#!/usr/bin/env python
# coding: utf-8

import re
import os
import chardet
import numpy as np
import pandas as pd

import nltk
import textwrap
nltk.download("stopwords")
nltk.download("punkt")
from contractions import contractions_dict
from sklearn.model_selection import train_test_split

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx

from rouge_score import rouge_scorer
from bert_score import score as bert_score

import warnings
warnings.filterwarnings("ignore")

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

domain1 = pd.concat([business,tech], ignore_index=True)
domain2 = pd.concat([entertainment,politics], ignore_index=True)

# domain1 = domain1[0:10]
# domain2 = domain2[0:10]

print("Training size:",len(domain1))
print("Testing size:",len(domain2))

domain1 = domain1.sample(frac=1).reset_index(drop=True)
domain2 = domain2.sample(frac=1).reset_index(drop=True)

domain1['newsarticle'] = domain1['newsarticle'].apply(expand_contractions)
domain2['newsarticle'] = domain2['newsarticle'].apply(expand_contractions)

def wrap(x):
    return textwrap.fill(x,replace_whitespace=False,fix_sentence_endings=True)

stop_words = stopwords.words("english")

def textRank(text):
    # Tokenize sentences
    sentences=sent_tokenize(text)
    # Clean and lowercase sentences
    sentences_clean=[re.sub(r'[^\w\s]','',sentence.lower()) for sentence in sentences]
     # Remove stopwords
    stop_words = stopwords.words('english')
    sentence_tokens=[[words for words in sentence.split(' ') if words not in stop_words] for sentence in sentences_clean]
    # Create Word2Vec model
    w2v=Word2Vec(sentence_tokens,vector_size=1,min_count=1,epochs=1000)
    # Get sentence embeddings
    sentence_embeddings=[[w2v.wv[word][0] for word in words] for words in sentence_tokens]
    # Pad embeddings to the maximum sentence length
    max_len=max([len(tokens) for tokens in sentence_tokens])
    # Calculate similarity matrix
    sentence_embeddings=[np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]
    similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])
    for i,row_embedding in enumerate(sentence_embeddings):
        for j,column_embedding in enumerate(sentence_embeddings):
            similarity_matrix[i][j]=1-spatial.distance.cosine(row_embedding,column_embedding)
    # Create a graph from the similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)
    # Apply PageRank algorithm
    scores = nx.pagerank(nx_graph, max_iter = 600)
    # Get top-ranked sentences
    top_sentence={sentence:scores[index] for index,sentence in enumerate(sentences)}
    top=dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:4])
    # Generate the summary
    summary = ''
    for sent in sentences:
        if sent in top.keys():
            summary = summary+sent
            # print(summary)
    return summary

# Function to apply summarizer and calculate scores for each row
def tr_summarize(row):
    news_article = row['newsarticle']
    actual_summary = row['summary']

    # Generate extractive summary
    extractive_summary_str = textRank(news_article)
    
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    rouge_scores = scorer.score(extractive_summary_str, actual_summary)

    # Compute BERT scores
    _, _, bert_f1 = bert_score([extractive_summary_str], [actual_summary], lang='en', model_type='bert-base-uncased')

    return pd.Series({
        'predicted_summary': extractive_summary_str,
        'rouge1_precision': rouge_scores['rouge1'].precision,
        'bert_score': bert_f1.item()
    })


textRank(domain1['newsarticle'][1])

textRank(domain2['newsarticle'][1])

# Apply the process_row function to each row in the DataFrame and concatenate the result_df with the original DataFrame
trOnDomain1 = pd.concat([domain1, domain1.apply(tr_summarize, axis=1)], axis=1)

trOnDomain2 = pd.concat([domain2, domain2.apply(tr_summarize, axis=1)], axis=1)

trOnDomain1.head()

trOnDomain2.head()

import matplotlib.pyplot as plt
# Calculate mean values
sameCat_mean_rouge = np.mean(trOnDomain1['rouge1_precision'])
sameCat_mean_bert = np.mean(trOnDomain1['bert_score'])
# Create separate histogram plots for Rouge and BERT Scores
plt.figure(figsize=(12, 6))
# plt.title('Same Category Data')

# Rouge Score Histogram
plt.subplot(1, 2, 1)
plt.hist(trOnDomain1['rouge1_precision'], bins=15, color='#1f77b4')
plt.axvline(sameCat_mean_rouge, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {sameCat_mean_rouge:.3f}')
plt.title('Domain1 Data Rouge Score Histogram')
plt.xlabel('Rouge Score')
plt.ylabel('Frequency')

# BERT Score Histogram
plt.subplot(1, 2, 2)
plt.hist(trOnDomain1['bert_score'], bins=15, color='darkorange')
plt.axvline(sameCat_mean_bert, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {sameCat_mean_bert:.3f}')
plt.title('Domain1 Data BERT Score Histogram')
plt.xlabel('BERT Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Calculate mean values
sameCat_mean_rouge = np.mean(trOnDomain2['rouge1_precision'])
sameCat_mean_bert = np.mean(trOnDomain2['bert_score'])
# Create separate histogram plots for Rouge and BERT Scores
plt.figure(figsize=(12, 6))
# plt.title('Same Category Data')

# Rouge Score Histogram
plt.subplot(1, 2, 1)
plt.hist(trOnDomain2['rouge1_precision'], bins=15, color='#1f77b4')
plt.axvline(sameCat_mean_rouge, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {sameCat_mean_rouge:.3f}')
plt.title('Domain2 Data Rouge Score Histogram')
plt.xlabel('Rouge Score')
plt.ylabel('Frequency')

# BERT Score Histogram
plt.subplot(1, 2, 2)
plt.hist(trOnDomain2['bert_score'], bins=15, color='darkorange')
plt.axvline(sameCat_mean_bert, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {sameCat_mean_bert:.3f}')
plt.title('Domain2 Data BERT Score Histogram')
plt.xlabel('BERT Score')
plt.ylabel('Frequency')

plt.tight_layout()

plt.show()

import matplotlib.pyplot as plt
import numpy as np

same_category_mean_rouge1 = trOnDomain1['rouge1_precision'].mean()
same_category_mean_bert = trOnDomain1['bert_score'].mean()

diff_category_mean_rouge1 = trOnDomain2['rouge1_precision'].mean()
diff_category_mean_bert = trOnDomain2['bert_score'].mean()

categories = ['rouge1_precision', 'bert_score']
mean_scores_same_category = [same_category_mean_rouge1, same_category_mean_bert]
mean_scores_diff_category = [diff_category_mean_rouge1, diff_category_mean_bert]

bar_width = 0.35
index = np.arange(len(categories))

fig, ax = plt.subplots()
bar1 = ax.bar(index, mean_scores_same_category, bar_width, label='Same Category')
bar2 = ax.bar(index + bar_width, mean_scores_diff_category, bar_width, label='Different Category')

ax.set_xlabel('Metrics')
ax.set_ylabel('Mean Score')
ax.set_title('Mean rouge1_precision and bert_score Comparison between Categories')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(categories)
ax.legend()

plt.show()


import matplotlib.pyplot as plt
import numpy as np

same_category_median_rouge1 = trOnDomain1['rouge1_precision'].median()
same_category_median_bert = trOnDomain1['bert_score'].median()

diff_category_median_rouge1 = trOnDomain2['rouge1_precision'].median()
diff_category_median_bert = trOnDomain2['bert_score'].median()

categories = ['rouge1_precision', 'bert_score']
median_scores_same_category = [same_category_median_rouge1, same_category_median_bert]
median_scores_diff_category = [diff_category_median_rouge1, diff_category_median_bert]

bar_width = 0.35
index = np.arange(len(categories))

fig, ax = plt.subplots()
bar1 = ax.bar(index, median_scores_same_category, bar_width, label='Same Category')
bar2 = ax.bar(index + bar_width, median_scores_diff_category, bar_width, label='Different Category')

ax.set_xlabel('Metrics')
ax.set_ylabel('median Score')
ax.set_title('median rouge1_precision and bert_score Comparison between Categories')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(categories)
ax.legend()

plt.show()

# get_ipython().system('jupyter nbconvert --to script TextRank-ExtractiveSummarizer.ipynb')