import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load the LLM for natural language processing and add the pipe for
# sentiment analysis.

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Read in the three CSV files from the folder provided and 
# combine them into one DataFrame

df = pd.concat(
    map(pd.read_csv, ['Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv',
                      'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv', 
                      '1429_1.csv'],))

# Look up the column names and date types present in the DataFrame
# to get an overview of the structure and contents of the data.

print(df.info(),
     df.head())

# Clean the column containing the reviews by getting rid of null
# values and standardising string formats.

reviews_data = df['reviews.text'].dropna()
clean_reviews = reviews_data.str.lower()
clean_reviews = clean_reviews.str.lstrip()
clean_reviews = clean_reviews.str.rstrip()

# Remove all stop words from the reviews for best results from 
# natural language processing and then apply the loaded LLM to the
# reviews. In this case apply the LLM only to the first five reviews
# in the DataFrame because of the large number of observations present
# in the data.

all_stopwords = nlp.Defaults.stop_words
cleanest_reviews = clean_reviews.apply(
    lambda x: ' '.join([word for word in x.split() if word not in (all_stopwords)]))
cleanest_reviews = cleanest_reviews.head().apply(nlp)

# Define a function which takes a selection of reviews from the
# DataFrame and the index positions of those reviews and returns the
# sentiment scores for each review with the review they belong to.


def sentiment_analysis(reviews, i_start, i_end):
    scores = []
    orig_reviews = []
    for i in clean_reviews[i_start:i_end]:
        orig_reviews.append(i)
    for review in reviews:
        scores.append(review._.blob.sentiment)
    scores_dict = dict(zip(scores, orig_reviews))
    return scores_dict


# Assign scores to a variable and then to a new column in dataframe
sentiment_scores = sentiment_analysis(cleanest_reviews, 0, None).keys()

df['Scores (Polarity, Subjectivity)'] = sentiment_scores