# File Imports
import fitz
import nltk
import textstat
import pandas as pd
import PyPDF2 as pdf
import matplotlib.pyplot as plt

from nltk.sentiment import SentimentIntensityAnalyzer

# File Settings
nltk.download('vader_lexicon')
nltk.download('punkt')

EXTRACTED_OUTPUT_FILENAME = 'research_txt_output.txt'
RESEARCH_FILENAME = '2024-01-08_AAPL.pdf'

# Get all the text
pdfFile = fitz.open(RESEARCH_FILENAME)
pages = pdfFile.page_count

# Write the PDF file text to a .txt file
with open(EXTRACTED_OUTPUT_FILENAME, 'w', encoding='utf-8') as outputFile:
    for pageNum in range(pages):
        l_page = pdfFile[pageNum]
        text = l_page.get_text()
        outputFile.write(text)

pdfFile.close()
outputFile.close()

# Open generated .txt file
inputTxt = open(EXTRACTED_OUTPUT_FILENAME, 'r').read()

# TODO: Create custom Financial and Economic based lexicon
# Create the sentiment intensity analyzer with the newly created lexicon

# Analyze the sentiment of the document
sa = SentimentIntensityAnalyzer()
sentimentScores = sa.polarity_scores(inputTxt)

print(f'\n{RESEARCH_FILENAME} Sentiment Scores:')
print(f"Document Rated as Positive: {round((sentimentScores['pos']*100), 4)}%")
print(f"Document Rated as Negative: {round((sentimentScores['neg']*100), 4)}%")
print(f"Document Rated as Neutral : {round((sentimentScores['neu']*100), 4)}%")
print(f"Overall Document Sentiment: {sentimentScores['compound']}")

# Find posititve and negative sentiment words
words = nltk.word_tokenize(inputTxt)

# Get sentiment score for each word
posWords = []
negWords = []

for word in words:
    wordScore = sa.polarity_scores(word)['compound']

    if wordScore > 0:
        posWords.append(word)

    if wordScore < 0:
        negWords.append(word)

print(posWords)
print(negWords)
