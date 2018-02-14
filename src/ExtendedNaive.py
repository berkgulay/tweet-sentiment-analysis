import NaiveBayes

#Naive Bayes class which can use uni-gram and bi-gram together(so extended) , it has same and essential methods as Naive Bayes but extended versions
class ExtendedNaive:

    def __init__(self,train_tweets,uni_gram=True,bi_gram=True):
        self.train_tweets = train_tweets
        self.uni_gram = uni_gram
        self.bi_gram = bi_gram
        self.nb_uni = None
        self.nb_bi = None


    def train(self):
        if self.uni_gram == True:
            self.nb_uni = NaiveBayes.NaiveBayes(self.train_tweets)
            self.nb_uni.train()
        if self.bi_gram==True:
            self.nb_bi = NaiveBayes.NaiveBayes(self.train_tweets, bi_gram=True)
            self.nb_bi.train()

    def test(self,test_tweets):
        if self.uni_gram == True and self.bi_gram==False:
            return self.nb_uni.test(test_tweets)
        elif self.uni_gram == False and self.bi_gram == True:
            return self.nb_bi.test(test_tweets)
        elif self.uni_gram == True and self.bi_gram == True:
            uni_nb_res = sorted(self.nb_uni.test(test_tweets),key=lambda x:x[0])
            bi_nb_res = sorted(self.nb_bi.test(test_tweets),key=lambda x:x[0])

            extended_nb_res = [] #re-calculate probability for bi-gram and uni-gram usage together
            for elm in range(0,3):
                pos = uni_nb_res[elm][1] + bi_nb_res[elm][1] #prob. are in log form, just take sum of them
                extended_nb_res.append((uni_nb_res[elm][0],pos))

            return sorted(extended_nb_res,key=lambda x:x[1],reverse=True) #return newly calculated (extended) probabilities
        else:
            return 0