
from textblob import TextBlob
import nltk
from newspaper import Article
import sys
import os.path
from os import path

#Get the article
url = str(sys.argv[1])
mFilePath = str(sys.argv[2])
article = Article(url)

# Do some NLP
article.download() #Downloads the link’s HTML content
article.parse() #Parse the article
nltk.download('punkt')#1 time download of the sentence tokenizer
article.nlp()#  Keyword extraction wrapper

text = article.summary
print(text)

obj = TextBlob(text)
#returns the sentiment of text
#by returning a value between -1.0 and 1.0
sentiment = obj.sentiment.polarity
  
if path.exists(mFilePath):
    Html_file=open(mFilePath, "a", newline='')
    Html_file.write('\n')
    Html_file.write(str(sentiment))
    Html_file.close()
else:
    Html_file= open(mFilePath,"w")
    Html_file.write(str(sentiment))
    Html_file.close()
