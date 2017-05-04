# gloveEmbeddingSentimentAnalysis

you will need to change the location of the word embeddings file and update the "embedSize" variable to match the number of dimensions in the word embeddings you are using to get this to run properly.

I downloaded the GloVe vectors from https://nlp.stanford.edu/projects/glove/#discuss and then used gensims conversion script (https://radimrehurek.com/gensim/scripts/glove2word2vec.html) to convert the GloVe vectors to word2vec format, which gensim can handle. For reference, gensim was used as it is library that provides an extremely efficient implementation for working with these word embeddings.

Data was pulled from the v2.0 of the Pang and Lee movie review dataset here https://www.cs.cornell.edu/people/pabo/movie-review-data/. I then pulled 100 files from the positive and negative sets to give a 90-10 split on training and test.
