from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
import argparse
import gensim, logging
from scipy import spatial
import sys

def LDAModeling(fileName):

    def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
        for topic_idx, topic in enumerate(H):
            print("Topic %d:" % (topic_idx))
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))
            top_doc_indices = np.argsort(W[:, topic_idx])[::-1][0:no_top_documents]
            for doc_index in top_doc_indices:
                print(documents[doc_index])

    documents = []
    with open(fileName,'r')as filename:
        for i in filename:
            documents.append(i)

    customList = []
    with open('stopWordList.txt', 'r') as stopFile:
        for words in stopFile:
            Words = words.strip()
            customList.append(Words)
    print(customList)

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=customList)
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print(tf_feature_names)
    no_topics = 35

    # Run LDA
    lda_model = LatentDirichletAllocation(n_topics=no_topics, max_iter=100, learning_method='online',
                                          learning_offset=50., random_state=0).fit(tf)
    lda_W = lda_model.transform(tf)
    lda_H = lda_model.components_
    # print(lda_H)
    # print(lda_W)

    no_top_words = 10
    no_top_documents = 200
    display_topics(lda_H, lda_W, tf_feature_names, documents, no_top_words, no_top_documents)


def Word2Vector():
    sentences = [['smart', 'home', 'able', 'order', 'delivery', 'food', 'simple', 'voice', 'command'],
                 ['smart', 'home', 'turn', 'certain', 'lights', 'dusk'],
                 ['smart', 'home', 'sync', 'biorhythm', 'app', 'turn', 'music', 'might', 'suit', 'mood', 'arrive',
                  'home', 'work'],
                 ['smart', 'home', 'ring', 'favorite', 'shows', 'start'],
                 ['children', 'surveilled', 'go', 'bathroom'],
                 ['smart', 'home', 'send', 'text', 'kid', 'gets', 'home', 'school'],
                 ['smartphone', 'create', 'environment', 'lighting', 'sound', 'temperature'],
                 ['open', 'front', 'gate', 'automatically', 'vehicle', 'approach', 'gate', 'close', 'vehicle', 'pass',
                  'gate'],
                 ['smarthome', 'automatically', 'turn', 'lights', 'whenever', 'leave'],
                 ['motion', 'sensors'],
                 ['light', 'kick', 'someone', 'approaches', 'door'],
                 ['moisture', 'sensors', 'vegetable', 'garden', 'regulate', 'drip', 'irrigation'],
                 ['lawn', 'mower', 'device', 'automatically', 'cut', 'grass', 'detects', 'certain', 'height', 'similar',
                  'roomba', 'vacuum', 'robot'],
                 ['automatically', 'turn', 'sprinklers', 'water', 'lawn', 'conjuction', 'mobile', 'devices']]

    # Initialize model with sentences
    model = gensim.models.Word2Vec(sentences, min_count=1, iter=15)

    example1 = ['motion', 'sensors']
    example2 = ['light', 'kick', 'someone', 'approaches', 'door']


    # Calculating the word vectors of all the words from each sentence
    listModelval = []
    for lines in sentences:
        modelsum = 0
        for word in lines:
            modelsum += model[word]
        listModelval.append(modelsum)
    # Storing the resulting vector of each sentence in a list
    print(len(listModelval))

    # Computing the cosine distance between two consecutive sentences
    cosineDistance = []
    sentenceVector = 0
    for eachSum in listModelval:
        if sentenceVector == 13:
            break
        else:
            result = 1 - spatial.distance.cosine(listModelval[sentenceVector], listModelval[sentenceVector + 1])
            cosineDistance.append(result)
            sentenceVector = sentenceVector + 1
    print('The cosine distance between consecutive sentences is {0}'.format(cosineDistance))



def main():

    fileName = sys.argv[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("data",help="String options:\n W2V - Implementation of Word2Vector,\n FileName/FilePath - Implementation of LDA on dataset provided in the argument ")
    args = parser.parse_args()
    if(args.data == 'W2V'):
        Word2Vector()
    else:
        LDAModeling(fileName)


if __name__ == "__main__":
    main()