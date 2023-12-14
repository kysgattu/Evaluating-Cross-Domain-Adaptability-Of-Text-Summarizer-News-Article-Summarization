# Evaluating Cross-domain Adaptability Of Text Summarizer: News Article Summarization
Text summarization condenses lengthy documents while retaining important information and meaning. It assists users in efficiently processing large amounts of information in a variety of industries such as news, education, business, and healthcare. Despite their benefits, text summarization models frequently struggle with generalizing to new domains, with differing styles and vocabularies causing domain shifts and negatively impacting performance. This limitation limits the models' practical use and scalability in real-world scenarios where users encounter texts from various domains. 
 
To address this issue, we propose assessing the cross-domain adaptability of text summarizer models, with a particular focus on news article summarization. With their diverse topics, genres, and perspectives, news articles present challenges for summarization models. We compare the performance of the TextRank summarizer, an Extractive Summarization model, and the BARTBase Model, an Abstractive Summarization model, using the BBC News Summary dataset, which contains articles and summaries from five domains. Our evaluation involves training and comparing these models on the domain set on which they were developed as well as an unknown different domain set, to assess their adaptability to diverse domains. This study aims to help users who want high-quality summaries in any domain, as well as developers who want to create adaptable text summarization models without the need for additional labeled data. 

## Table of Contents
- [Prerequisites](#prerequisites)
    - [Environment](#environment)
    - [Dataset Description](#dataset-description)
- [Summarizers](#summarizers)
    - [Extractive Text Summarization](#ETS)
    - [Abstractive Text Summarization](#ATS)
- [Implementation of Summarizers](#implementation)
    - [TextRank](#textrankImp)
    - [BART-Base](#bartImp)
- [Evaluation of the models](#model_eval)
  - [ROUGE-1 Precision Score](#rouge1)
  - [BERTScore](#bert)
- [Results](#results)
- [Analysis](#analysis)
- [Developers](#developers)
- [Links](#links)
- [References](#references)            

## Prerequisites <a name='prerequisites'></a>

### Environment <a name='environment'></a>

1. Python 3 Environment (Anaconda preferred)
2. Python modules required: os, re, chardet, numpy, pandas, nltk, textwrap, contractions, sklearn, gensim, scipy, networkx, matplotlib, seaborn, torch, transformers, tqdm, rouge_score, bert_score, warnings

OR
- Any Python3 IDE installed with the above modules. (Pycharm is used in the development of this project)

### Dataset Description <a name='dataset-description'></a>

The dataset we use for this project is from Kaggle and was created using a dataset used for data categorization that consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004 to 2005. This dataset contains texts from five different domains, making it appropriate for assessing cross-domain adaptability. Articles and summaries are stored as individual text files in separate folders for each domain. For the next steps, the article and summary files are combined into five dataframes, each containing the filename, news article, and summary for each domain.

<img width="468" alt="image" src="https://github.com/kysgattu/Evaluating-Cross-Domain-Adaptability-Of-Text-Summarizer-News-Article-Summarization/assets/42197976/119b493e-b396-49e8-98cd-29707853fa35">

 
The datasets are shuffled and Text Contractions are expanded across the datasets. We analyze the sentence lengths and remove the records with outliers.

<img width="1143" alt="image" src="https://github.com/kysgattu/Evaluating-Cross-Domain-Adaptability-Of-Text-Summarizer-News-Article-Summarization/assets/42197976/91746b1e-283b-4616-bf80-b2b689595d06">


## Summarizers<a name='summarizers'></a>

Text summarization can be divided into two categories: Extractive and Abstractive. 

> ### Extractive Text Summarization <a name = 'ETS'></a>

- Extractive summarization involves determining and combining the most important sentences or phrases from the source text, forming the summary without introducing new words or sentences. This procedure is similar to highlighting key sections of the text. 
- Our study focuses on extractive summarization using the TextRank Algorithm, which ranks sentences based on their frequency and similarity to other texts. TextRank, which is based on the PageRank algorithm used in online search results, treats sentences as nodes in a graph, connected by edges that represent their similarity. 
Various similarity metrics, such as cosine similarity, Jaccard similarity, and word overlap, quantify this similarity. TextRank assigns scores to sentences based on the quality and quantity of their links, with the highest-scoring sentences chosen for the summary. 

> ### Abstractive Text Summarization <a name = 'ATS'></a>
- Abstractive summarization methods create new words or phrases to capture the essence of the original text rather than simply copying and pasting. Instead, they paraphrase or rewrite the content, providing a condensed rewrite.
- Our method makes use of the BART algorithm, which has been optimized for abstractive summarization. BART is a Facebook AI-developed sequence-to-sequence transformer based model that consists of an encoder and a decoder supported by the Transformer architecture. The encoder creates hidden states from the input text, and the decoder generates the output text from these hidden states. BART-base excels in abstractive summarization due to its bidirectional context in the encoder and autoregressive generation in the decoder, which was trained on a large text corpus with a denoising autoencoder objective.

## Implementation of Summarizers<a name='implementation'></a>

> ### TextRank <a name = 'textrankImp'></a>
We group the dataset into two categories: business, technology and entertainment, politics, and run the TextRank algorithm on each domain separately to observe if there is a difference in the results.

<img width="1143" alt="image" src="https://github.com/kysgattu/Evaluating-Cross-Domain-Adaptability-Of-Text-Summarizer-News-Article-Summarization/assets/42197976/0b11367f-0196-4167-9464-92bac87454d2">


#### Text Preprocessing: 
To maintain consistency and focus on important content, the input text is tokenized, converted to lowercase, and punctuation is removed. Common stopwords are excluded. 
 
#### Constructing the Graphs: 
Word2Vec generates word embeddings, which are then combined to form sentence embeddings for semantic meaning. The similarity matrix summarizes sentence relationships using cosine similarity, providing a comprehensive overview of sentence connections. 
 
#### Weighting Edges: 
The edges of the graph are weighted according to the calculated similarity matrix, emphasizing strong connections between sentences, and establishing the importance of each sentence with others. 
 
#### Applying PageRank Algorithm: 
Originally designed for web page ranking, the algorithm is applied to evaluate and rank sentences based on importance. Higher rankings are assigned to sentences connected to other important ones, conveying essential information. 
 
#### Sentence Extraction: 
Top-ranked sentences are extracted using Ranking-Based Selection, ensuring key information retention. This procedure collects the most important and relevant information for the summary. 

> ### BART-base <a name = 'bartImp'></a>
We divide the dataset into two different sets – business, politics, technology and entertainment, sport. We fine-tune and train the BART model on one set and continuously evaluate the model. The model thus created is tested on the other set to determine how it adapts to a different domain.

<img width="468" alt="image" src="https://github.com/kysgattu/Evaluating-Cross-Domain-Adaptability-Of-Text-Summarizer-News-Article-Summarization/assets/42197976/9a1cd6d4-a971-441e-b5b7-932c7a5e5cde">



#### Model Definition and Initialization:
We use the BART-Base model from transformers for transfer learning to leverage saved weights of the model pre-trained on a large corpus of data.

#### Data Preprocessing:
BART Tokenizer is used to tokenize input text. Tokenized sequences are stacked into PyTorch Tensors, resulting in the creation of a Tensor Dataset for articles and summaries. Sequence lengths are adjusted to accommodate model input size constraints. Shuffling prevents the model from learning the same examples in the same order each epoch, and batching is done via DataLoaders for faster model updates.

#### Fine-tuning for Summarization:
The model undergoes fine-tuning for summarization using a sequence-to-sequence learning approach. AdamW optimizer with a 1e-4 learning rate and 0.001 weight decay is employed. A Linear Schedule with a Warm-up learning rate reducer is used, gradually increasing the learning rate at the training start. A warm-up period and linear decay stabilize training. Beam search decoding captures diverse text possibilities, enhancing summary quality.

#### Training Loop:
Training steps are determined by batches in the dataloader and epochs. The training loop spans multiple epochs, with gradient accumulation for less frequent weight updates and Mixed-precision training for efficiency. Continuous evaluation is also done on a validation set to record Training loss, Rouge-1, and BERT Scores. The model, trained for three epochs without improvement, is saved after completion.


## Evaluation of models <a name = 'model_eval'></a>

> ### ROUGE-1 Precision Score <a name = 'rouge'></a>

ROUGE, is a set of metrics used to evaluate the quality of text summarization by comparing the overlap between the generated summary and reference summaries.
- We use Rouge-1 which specifically measures unigram overlap between the generated and reference summaries.
- It considers the recall of unigrams, capturing how many reference unigrams are also present in the generated summary.
- It is especially useful for assessing the informativeness and content overlap.

> ### BERTScore <a name = 'bert'></a>
BERT Score utilizes contextual embeddings from BERT to measure the similarity between the generated and reference summaries - It is designed to capture semantic similarity.
- It considers the surrounding context of each word, providing a more nuanced understanding of semantics.
- It focuses on measuring semantic similarity, and evaluating how well the generated summary captures the meaning of the reference summaries.
- This goes beyond lexical matching, emphasizing a deeper assessment of the underlying semantic content in the text.

By evaluating these both we get insights of both informativeness and semantic efficiency of the generated summaries.

## Results <a name = 'results'></a>

> - When the TextRank model is applied to data from two different domains, findings match those of [Wang et al.,](https://arxiv.org/abs/1908.11664) who used the BERT model as a basis for extractive summarization. On the two domains we trained, our implementation has a Mean ROUGE-1 precision of 0.35 and 0.37 and a Mean BERTScore of 0.67 and 0.65.

<img width="1143" alt="image" src="https://github.com/kysgattu/Evaluating-Cross-Domain-Adaptability-Of-Text-Summarizer-News-Article-Summarization/assets/42197976/df75610c-eda8-4d0d-85cf-2c0c4c02993f">


> 
> 
> - The BART model developed in this work performs better than the model in the paper where it was first introduced as well as the model trained on the BBC News dataset in the work done by [Anushka Gupta et al.,](https://link.springer.com/chapter/10.1007/978-981-16-9012-9_21)
> - We used one set of domains to train the BART Model, and then we applied it to a different set of domains. The domain data used for model testing is a different dataset that is unknown, whereas the dataset used for validation during training is the same. The model's ROUGE-1 precision and BERTScore are 0.67 and 0.71 for the same domain data, respectively, and 0.71 and 0.69 when tested on a different domain.

<img width="468" alt="image" src="https://github.com/kysgattu/Evaluating-Cross-Domain-Adaptability-Of-Text-Summarizer-News-Article-Summarization/assets/42197976/36161bc4-1949-456d-90c1-26d950da1a5b">




<img width="468" alt="image" src="https://github.com/kysgattu/Evaluating-Cross-Domain-Adaptability-Of-Text-Summarizer-News-Article-Summarization/assets/42197976/be97313b-0798-46b8-9a82-518ff548e83d">


## Analysis <a name='analysis'></a>
This work provides an extensive investigation of cross-domain adaptability in text summarization, most notably demonstrating the improved performance of the BART model over its initial implementation and a model optimized for the BBC News dataset. The BART model's excellent ROUGE scores demonstrate how well it summarizes news articles from various industries. Prospective research directions entail expanding the scope of adaptability testing to encompass larger text domains, including dialogues and research articles, which should yield additional insights into the practicality of the model. Notably, the observed limitations point to areas that need to be improved upon in subsequent research, such as the reliance on a small amount of labeled data and the requirement for a more comprehensive set of evaluation metrics.

## Developers <a name='developers'></a>
* [Kamal Yeshodhar Shastry Gattu](https://github.com/kysgattu)

## Links <a name='links'></a>

GitHub:     [G K Y SHASTRY](https://github.com/kysgattu)

Contact me:     <gkyshastry0502@gmail.com> , <kysgattu0502@gmail.com>

## References <a name='references'></a>

[BBC News Summary Dataset](https://www.kaggle.com/datasets/pariza/bbc-news-summary)

