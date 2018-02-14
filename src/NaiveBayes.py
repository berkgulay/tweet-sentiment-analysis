import BAG
from sklearn.feature_extraction.text import CountVectorizer
import math

#Naive Bayes implementation
class NaiveBayes:

    def __init__(self,train_tweets,bi_gram=False):
        self.train_tweets = train_tweets
        self.bi_gram = bi_gram
        self.vocabulary_num = 0
        self.negative_bag = BAG.BAG()
        self.neutral_bag = BAG.BAG()
        self.positive_bag = BAG.BAG()
        self.negative_num = 0
        self.neutral_num = 0
        self.positive_num = 0
        self.neg_prob_dict={}
        self.neut_prob_dict = {}
        self.pos_prob_dict = {}

    #train this model with given tweets in constructor
    def train(self):
        for t in self.train_tweets:
            if t[0]==0: #negative
                self.negative_num += 1
                self.vocabulary_num = self.negative_bag.add_tweet(t[1],self.vocabulary_num,self.bi_gram)
            elif t[0]==2: #neutral
                self.neutral_num += 1
                self.vocabulary_num = self.neutral_bag.add_tweet(t[1], self.vocabulary_num,self.bi_gram)
            elif t[0]==4: #positive
                self.positive_num += 1
                self.vocabulary_num = self.positive_bag.add_tweet(t[1], self.vocabulary_num,self.bi_gram)
            else:
                continue


    #test given tweet and predict a label for it
    def test(self,test_tweet):
        vectorizer = CountVectorizer()
        analyze = vectorizer.build_analyzer()
        t_words = []
        if self.bi_gram==False:
            t_words = analyze(test_tweet)
        else:
            temp = analyze(test_tweet)
            for t in range(0, len(temp) - 1):
                t_words.append((temp[t], temp[t + 1]))

        neg_prob = 1
        neut_prob = 1
        pos_prob = 1
        for w in t_words:

            if w in self.neg_prob_dict.keys():
                neg_prob += math.log10(self.neg_prob_dict[w])
            else:
                w_prob = self.find_prob(w,0)
                self.neg_prob_dict[w] = w_prob
                neg_prob += math.log10(w_prob)

            if w in self.neut_prob_dict.keys():
                neut_prob += math.log10(self.neut_prob_dict[w])
            else:
                w_prob = self.find_prob(w, 2)
                self.neut_prob_dict[w] = w_prob
                neut_prob += math.log10(w_prob)

            if w in self.pos_prob_dict.keys():
                pos_prob += math.log10(self.pos_prob_dict[w])
            else:
                w_prob = self.find_prob(w, 4)
                self.pos_prob_dict[w] = w_prob
                pos_prob += math.log10(w_prob)

        total_tweet_val=self.negative_num+self.neutral_num+self.positive_num
        neg_prob += math.log10(self.negative_num/total_tweet_val)
        neut_prob += math.log10(self.neutral_num/total_tweet_val)
        pos_prob += math.log10(self.positive_num/total_tweet_val)

        res_list = [(0,neg_prob),(2, neut_prob),(4, pos_prob)]
        res_list = sorted(res_list,key=lambda x:x[1],reverse=True)

        return res_list

    #find likelihood of given word in given class
    def find_prob(self,word,class_flag):

        if class_flag==0:
            if word in self.negative_bag.word_dict.keys():
                likelyhood_prob = (self.negative_bag.word_dict[word] + 1) / (self.negative_bag.num_of_words + self.vocabulary_num)
            else:
                likelyhood_prob = 1 / (self.negative_bag.num_of_words + self.vocabulary_num)
        elif class_flag==2:
            if word in self.neutral_bag.word_dict.keys():
                likelyhood_prob = (self.neutral_bag.word_dict[word] + 1) / (self.neutral_bag.num_of_words + self.vocabulary_num)
            else:
                likelyhood_prob = 1 / (self.neutral_bag.num_of_words + self.vocabulary_num)
        else:
            if word in self.positive_bag.word_dict.keys():
                likelyhood_prob = (self.positive_bag.word_dict[word] + 1) / (self.positive_bag.num_of_words + self.vocabulary_num)
            else:
                likelyhood_prob = 1 / (self.positive_bag.num_of_words + self.vocabulary_num)

        return likelyhood_prob