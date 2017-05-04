# gloveEmbeddingSentimentAnalysis

you will need to change the location of the word embeddings file to get this to run properly.

I downloaded the GloVe vectors from https://nlp.stanford.edu/projects/glove/#discuss and then used gensims conversion script (https://radimrehurek.com/gensim/scripts/glove2word2vec.html) to convert the GloVe vectors to word2vec format, which gensim can handle. For reference, gensim was used as it is library that provides an extremely efficient implementation for working with these word embeddings.
