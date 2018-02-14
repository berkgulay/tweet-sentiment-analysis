from sklearn.feature_extraction.text import CountVectorizer

#Bag of Words(Bow) model implementation using sklearn-CountVectorizer
class BAG:

    def __init__(self):
        self.word_dict = {}
        self.num_of_words = 0


    #adds new tweets to BAG and vocabulary also
    def add_tweet(self,tweet,vocabulary_num,bi_gram):
        vectorizer = CountVectorizer()
        analyze = vectorizer.build_analyzer()
        words = []
        if bi_gram == False:
            words = analyze(tweet)
        else:
            temp = analyze(tweet)
            for t in range(0,len(temp)-1):
                words.append((temp[t],temp[t+1]))

        for w in words:
            if w in self.word_dict.keys():
                self.word_dict[w] += 1
            else:
                self.word_dict[w] = 1
                vocabulary_num += 1

            self.num_of_words += 1

        return vocabulary_num