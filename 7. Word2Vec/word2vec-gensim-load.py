import gensim


model = gensim.models.Word2Vec.load("word2vec.model")
w1 = "heart"
print("Most similar to {0}".format(w1), model.wv.most_similar(positive=w1))

# look up top 6 words similar to 'polite'
w1 = ["disease"]
print("Most similar to {0}".format(w1),
      model.wv.most_similar(
        positive=w1,
        topn=6))

w1 = ["heart", 'disease']
w2 = ['middle']
print("Most similar to {0}".format(w1),
      model.wv.most_similar(
        positive=w1,
        negative=w2,
        topn=10))

# similarity between two different words
print("Similarity between 'dirty' and 'smelly'",
      model.wv.similarity(w1="dirty", w2="smelly"))

# similarity between two different words
print("Similarity between 'heart' and 'love'",
      model.wv.similarity(w1="heart", w2="love"))
