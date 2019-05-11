import re
from sklearn.externals import joblib
from abcd import clean_str
from textblob import Word

vect = joblib.load('vectorizer.pkl')   #tfidf vectorizer
model = joblib.load('classifier.pkl') #classifier

def check_news_type(news_article):  # function to accept raw string and to provide news type(class)
    news_article = [' '.join([Word(word).lemmatize() for word in clean_str(news_article).split()])]
    features = vect.transform(news_article)
    return str(model.predict(features)[0])

article = """Ad sales boost Time Warner profit

Quarterly profits at US media giant TimeWarner jumped 76% to $1.13bn (?600m) for the three months to December, from $639m year-earlier.

The firm, which is now one of the biggest investors in Google, benefited from sales of high-speed internet connections and higher advert sales. TimeWarner said fourth quarter sales rose 2% to $11.1bn from $10.9bn. Its profits were buoyed by one-off gains which offset a profit dip at Warner Bros, and less users for AOL.

Time Warner said on Friday that it now owns 8% of search-engine Google. But its own internet business, AOL, had has mixed fortunes. It lost 464,000 subscribers in the fourth quarter profits were lower than in the preceding three quarters. However, the company said AOL's underlying profit before exceptional items rose 8% on the back of stronger internet advertising revenues. It hopes to increase subscribers by offering the online service free to TimeWarner internet customers and will try to sign up AOL's existing customers for high-speed broadband. TimeWarner also has to restate 2000 and 2003 results following a probe by the US Securities Exchange Commission (SEC), which is close to concluding.

Time Warner's fourth quarter profits were slightly better than analysts' expectations. But its film division saw profits slump 27% to $284m, helped by box-office flops Alexander and Catwoman, a sharp contrast to year-earlier, when the third and final film in the Lord of the Rings trilogy boosted results. For the full-year, TimeWarner posted a profit of $3.36bn, up 27% from its 2003 performance, while revenues grew 6.4% to $42.09bn. "Our financial performance was strong, meeting or exceeding all of our full-year objectives and greatly enhancing our flexibility," chairman and chief executive Richard Parsons said. For 2005, TimeWarner is projecting operating earnings growth of around 5%, and also expects higher revenue and wider profit margins.

TimeWarner is to restate its accounts as part of efforts to resolve an inquiry into AOL by US market regulators. It has already offered to pay $300m to settle charges, in a deal that is under review by the SEC. The company said it was unable to estimate the amount it needed to set aside for legal reserves, which it previously set at $500m. It intends to adjust the way it accounts for a deal with German music publisher Bertelsmann's purchase of a stake in AOL Europe, which it had reported as advertising revenue. It will now book the sale of its stake in AOL Europe as a loss on the value of that stake."""

article = """ computer technology of intel chipsets"""
article = """ Lok sabha elections in india will held in month of april. hope the parties are ready to face the competition"""
article = """ tom cruise will appear in a hollywood movie name avengers """

check_news_type(article)


# Writing the Summary

# Preprocessing the text
article = re.sub(r'\[[0-9]*\]', ' ', article)
article = re.sub(r'\s+', ' ', article)
clean_text = article.lower()
clean_text = re.sub(r'\W', ' ', clean_text)
clean_text = re.sub(r'\d', ' ', clean_text)
clean_text = re.sub(r'\s+', ' ', clean_text)

# Making the sentences out of text
import nltk 
nltk.download('stopwords')
sentences = nltk.sent_tokenize(article)
stop_words = nltk.corpus.stopwords.words('english')

# Creating the histogram and weighted histogram
word2count = {}   #dictionary
for word in nltk.word_tokenize(clean_text):
    if word not in stop_words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
            
# weighted Histogram
for key in word2count.keys():
     word2count[key] = word2count[key]/max(word2count.values())

# calculating the sentence scores
sent2score = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count.keys():
            if len(sentence.split(' ')) < 25:
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]
                    
# Getting the Summary
import heapq
best_sentences = heapq.nlargest(5, sent2score, key = sent2score.get)
print('-----------------------------------------------------------')
for sentence in best_sentences:
    print(sentence)
                    
               
     