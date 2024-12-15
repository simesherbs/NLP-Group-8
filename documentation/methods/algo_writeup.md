# Algorithm Explanation

## Corpus processing:

The corpus is stemmed and has its stop words deleted. Then it is put through a program that extracts bigrams and trigrams.
Those bigrams and trigrams are then append to the end of their respective documents.

## Candadicy Score

The processed corpus is then run through a program that calculate the Candadicy Score (CS) for each token of each document that belongs to a given genre.
CS is calculated from a mixture of features. Currently we've added support for:
- TFIDF
- Chi-squared

Currently we're working on:

- C-Value
- Domain Relevance
- Domain Consensus

Each of these feature scores are then normalized to a value between 0-1. Then each a assigned a weight and added together to get the CS for a given token in a given genre.

## Term Extraction

After finding the CS for all tokens, for each genre we take the all tokens above a certain CS threshold to create our list of extracted terminology.
To evalulate this system, we check against other popular term extractors via inter-annontator agreement tests.

## Genre Classification 

The other aspect of our project is classifying unseen synopses. To do this we will be using an SVM model. To train this model, we take every synopsis 
and transform the tokens within it to a vector of its CS. For instance if we have a synopsis that looks like the following (after processing):

 - synopsis: [war, agent, axe, battle], tags: [War, Action, Horror]

We would transform this into the following:

 - CS: [.9, .7, .3, .8], tags:[War]
 - CS: [.5, .8, .3, .4], tags:[Action]
 - CS: [0, .1, .7, 0], tags:[Horror]

After doing this for every synopsis, we would train the model and then use it for classifying unsen synopses.

### OOV Processing
Currently, we don't have a strategy for dealing with OOV tokens in unseen synopses, but it is something we are thinking about.