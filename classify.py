from collections import defaultdict
from nltk.corpus import stopwords
import os,sys
import json
import re
import time
import nltk 
import numpy as np
import pickle
import simplejson
from sklearn.metrics import f1_score

from functions import get_filenames
from functions import get_clean_tokens
from timeit import default_timer as timer

start = timer() 

train_categories =[]
train_categories_doc_count =[]
train_documents_count = 0
train_cat_docslist = []


train_rootdir = 'train'
train_categories ,train_categories_doc_count,train_cat_docslist = get_filenames(train_rootdir)


test_rootdir = 'test'
test_categories ,test_categories_doc_count,test_cat_docslist = get_filenames(test_rootdir)


train_category_docs = []
train_total_docs = []
train_category_wcount = []
vocab_list =[]

''' Tokenizing train and test documents '''

vocab_list,training_docs,train_total_docs,train_category_wcount = get_clean_tokens(train_cat_docslist,train_rootdir,train_categories)
test_vocab_list,test_docs,test_total_docs,test_category_wcount = get_clean_tokens(test_cat_docslist,test_rootdir,test_categories)


politics_vocab_dict = defaultdict(int)
graphics_vocab_dict = defaultdict(int)
autos_vocab_dict = defaultdict(int)
for x in vocab_list:
	politics_vocab_dict[x] = 0
	graphics_vocab_dict[x] = 0
	autos_vocab_dict[x] = 0


''' Building training vectors '''
for n in range(0,3):
	for docs in training_docs[n]:
		for w in docs:		
			if n == 0:
				politics_vocab_dict[w] += 1
		
			if  n == 1:
				graphics_vocab_dict[w] += 1
	
			if n == 2:
				autos_vocab_dict[w] += 1
	

''' Prior calculation'''

prior = np.zeros(3)
for i in range(3):
	prior[i] = float(train_total_docs[i])/sum(train_total_docs)

ground_y = []
y_predict = []



''' Calucalating likelihood for each test document '''

for n in range(0,3):	
	for docs in test_docs[n]:
		ground_y.append(n)
		likelihood = [0,0,0] # politics = 0 , graphics = 1, autos = 2
		for fw in docs:

			if politics_vocab_dict[fw] != 0:
				likelihood[0] +=  np.log(float(politics_vocab_dict[fw]) + 1/(train_category_wcount[n] + len(vocab_list)))
		
			if graphics_vocab_dict[fw] != 0:
				likelihood[1] +=  np.log(float(graphics_vocab_dict[fw]) + 1/(train_category_wcount[n] + len(vocab_list)))

			if autos_vocab_dict[fw] != 0:
				likelihood[2] +=  np.log(float(autos_vocab_dict[fw])+ 1/(train_category_wcount[n] + len(vocab_list)))
		

		'''allocating document to category'''
		posterior = [0,0,0]
		for j in range(3):
			posterior[j] = float(likelihood[j] + np.log(prior[j]) )
		
		print '\nTesting : ground truth',n,'prediction',posterior.index(max(posterior))
		y_predict.append(posterior.index(max(posterior)))


''' Calucalating total accuracy '''
count = 0
for i in range(len(y_predict)):
	if y_predict[i] == ground_y[i]:
		count = count + 1

F1_score = f1_score(ground_y, y_predict, average='macro')  

end = timer()
print '\nRunning Time :',(end - start) 

print '\n Vocabulary Size', len(vocab_list)

print '\n Overall Accuracy =', float(count)/ len(y_predict)
print '\n F1 score', F1_score
					


