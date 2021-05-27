# %%
import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random
import re
import string
from nltk.corpus import stopwords          # stopword with NLTK
from nltk.stem import PorterStemmer        # stemming module
from nltk.tokenize import TweetTokenizer  # string tokenizer


nltk.download('twitter_samples')
nltk.download('stopwords')


# Examination of NLT twitter datasets

all_positive_tweets = twitter_samples.strings("positive_tweets.json")
all_negative_tweets = twitter_samples.strings("negative_tweets.json")

print('Number of positive tweets', len(all_positive_tweets))
print('Number of negative tweets ', len(all_negative_tweets))
print("\nThe type of all_positive_tweets is: ", type(all_positive_tweets))
print("The type of all_negative_tweets is: ", type(all_negative_tweets))

fig = plt.figure(figsize=(5, 5))
labels = "Positives", "Negative"
sizes = [len(all_positive_tweets), len(all_negative_tweets)]
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
plt.axis("equal")
plt.show


# Observing Raw texts

print("Positive tweet = " + all_positive_tweets[random.randint(0, 5000)])
print("Negative tweet = " + all_negative_tweets[random.randint(0, 5000)])

# Preprocess raw text for Sentiment analysis
tweet = all_positive_tweets[2277]
print("Initial Tweet: ")
print(tweet)

tokenizer = TweetTokenizer(
    preserve_case=False, strip_handles=True, reduce_len=True)
tweet_tokens = tokenizer.tokenize(tweet)
print()
print("Tokenized string: ")
print(tweet_tokens)

# Removing stop words and punctuations
stopwords_english = stopwords.words('english')


# Import the english stop words list from NLTK
stopwords_english = stopwords.words('english')


tweets_clean = []
for word in tweet_tokens:
    if(word not in stopwords_english and
            word not in string.punctuation):
        tweets_clean.append(word)

print("Removed stop words and punctuations: ")
print(tweets_clean)

# Stemming

# Instantiate stemming class
stemmer = PorterStemmer()

# Create an empty list to store the stems
tweets_stem = []

for word in tweets_clean:
    stem_word = stemmer.stem(word)
    tweets_stem.append(stem_word)

print('\nstemmed words:')
print(tweets_stem)


# %%
