import numpy as np
import ExtendedNaive

tweets= np.load("train_tweets.npy")
corpus=[]
for i in tweets: #take train tweets to corpus list
    corpus.append((int(str(i[0])[2]) , str(i[1]).split(str(i[1])[1])[1]))


ex_nb = ExtendedNaive.ExtendedNaive(corpus) #Naive Bayes model, which is able to use uni-gram model and bi-gram model together
ex_nb.train()

##############################################################################################################################
# ex_nb = ExtendedNaive.ExtendedNaive(corpus,bi_gram=False) #Naive Bayes model which use uni-gram word model only
# ex_nb.train()
#
# ex_nb = ExtendedNaive.ExtendedNaive(corpus,uni_gram=False) #Naive Bayes model which use bi-gram word model only
# ex_nb.train()
##############################################################################################################################

#accuracy calculation
total = 0
true_count = 0

test = np.load('validation_tweets.npy') #load test tweets
for j in test:
    total += 1

    real_res = int(str(j[0])[2]) #take result(true label) of validation tweet
    prediction= ex_nb.test(str(j[1]).split(str(j[1])[1])[1]) #predict validation tweet's label by Naive Bayes model
    if real_res==prediction[0][0]:
        true_count+=1 #if prediction is true increase true_count by 1

print('Naive Bayes accuracy of uni-gram & bi-gram together:')
print(true_count/total*100) #accuracy score (%)





