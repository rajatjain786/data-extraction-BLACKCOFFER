import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import re
import pyphen


nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


dic = pyphen.Pyphen(lang='en')


stop_words = set(stopwords.words('english'))


input_data = pd.read_excel('input.xlsx')


script_dir = os.path.dirname(os.path.abspath(__file__))
master_dict_dir = os.path.join(script_dir, 'MasterDictionary')

positive_words_path = os.path.join(master_dict_dir, 'positive-words.txt')
negative_words_path = os.path.join(master_dict_dir, 'negative-words.txt')

with open(positive_words_path, 'r') as file:
    positive_words = set(file.read().split())
with open(negative_words_path, 'r') as file:
    negative_words = set(file.read().split())


stopwords_dir = os.path.join(script_dir, 'StopWords')
custom_stopwords = set()
for filename in os.listdir(stopwords_dir):
    with open(os.path.join(stopwords_dir, filename), 'r') as file:
        custom_stopwords.update(file.read().split())


stop_words.update(custom_stopwords)


def extract_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
   
    title_tag = soup.find('h1')
    title = title_tag.get_text() if title_tag else 'No Title'
    
  
    paragraphs = soup.find_all('p')
    article_text = ' '.join([para.get_text() for para in paragraphs]) if paragraphs else 'No Content'
    
    return title, article_text


def save_text(url_id, title, article_text):
    with open(f'{url_id}.txt', 'w', encoding='utf-8') as file:
        file.write(f"{title}\n{article_text}")


def count_syllables(word):
    syllables = dic.inserted(word)
    return len(syllables.split('-'))


def clean_text(text):
    words = word_tokenize(text)
    cleaned_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    return cleaned_words


def text_analysis(text):
  
    cleaned_words = clean_text(text)
    word_count = len(cleaned_words)
    
  
    sentences = sent_tokenize(text)
    sentence_count = len(sentences)
    
   
    avg_sentence_length = word_count / sentence_count if sentence_count != 0 else 0
    
    
    complex_word_count = 0
    syllable_count = 0
    for word in cleaned_words:
        syllables = count_syllables(word)
        syllable_count += syllables
        if syllables >= 3:
            complex_word_count += 1
    syllable_per_word = syllable_count / word_count if word_count != 0 else 0
    
   
    percentage_complex_words = complex_word_count / word_count * 100 if word_count != 0 else 0
    
   
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    
   
    avg_word_length = sum(len(word) for word in cleaned_words) / word_count if word_count != 0 else 0
    
   
    positive_score = sum(1 for word in cleaned_words if word.lower() in positive_words)
    negative_score = sum(1 for word in cleaned_words if word.lower() in negative_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (word_count + 0.000001)
    
   
    personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', text, re.I))
    
    return {
        'Positive Score': positive_score,
        'Negative Score': negative_score,
        'Polarity Score': polarity_score,
        'Subjectivity Score': subjectivity_score,
        'Avg Sentence Length': avg_sentence_length,
        'Percentage of Complex Words': percentage_complex_words,
        'Fog Index': fog_index,
        'Avg Number of Words per Sentence': avg_sentence_length,  
        'Complex Word Count': complex_word_count,
        'Word Count': word_count,
        'Syllable per Word': syllable_per_word,
        'Personal Pronouns': personal_pronouns,
        'Avg Word Length': avg_word_length
    }


output_data = []

for index, row in input_data.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    
    title, article_text = extract_article_text(url)
    save_text(url_id, title, article_text)
    
    analysis_results = text_analysis(article_text)
    output_row = [url_id, url] + list(analysis_results.values())
    output_data.append(output_row)


output_columns = ['URL_ID', 'URL'] + list(analysis_results.keys())
output_df = pd.DataFrame(output_data, columns=output_columns)
output_df.to_excel('output.xlsx', index=False)
