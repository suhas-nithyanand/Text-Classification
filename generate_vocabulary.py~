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


from functions import get_filenames
from functions import get_clean_tokens

train_categories =[]
train_categories_doc_count =[]
train_documents_count = 0
train_cat_docslist = []


train_rootdir = '/home/suhas/ucf/2ndsem/nlp/1st_assign/workspace/train'
train_categories ,train_categories_doc_count,train_cat_docslist = get_filenames(train_rootdir)


test_rootdir = '/home/suhas/ucf/2ndsem/nlp/1st_assign/workspace/test'
test_categories ,test_categories_doc_count,test_cat_docslist = get_filenames(test_rootdir)


#category_wcount_dict = {}
train_category_docs = []
train_total_docs = []
train_category_wcount = []
vocab_list =[]

vocab_list,training_docs,train_total_docs,train_category_wcount = get_clean_tokens(train_cat_docslist,train_rootdir,train_categories)
test_vocab_list,test_docs,test_total_docs,test_category_wcount = get_clean_tokens(test_cat_docslist,test_rootdir,test_categories)


politics_vocab_dict = defaultdict(int)
graphics_vocab_dict = defaultdict(int)
autos_vocab_dict = defaultdict(int)
for x in vocab_list:
	politics_vocab_dict[x] = 0
	graphics_vocab_dict[x] = 0
	autos_vocab_dict[x] = 0

for n in range(0,3):
	for docs in training_docs[n]:
		#print n,'doc len',len(training_docs[n])
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

for n in range(0,3):	
	for docs in test_docs[n]:
		ground_y.append(n)
		#print n,'doc len',len(training_docs[n])
		likelihood = [0,0,0] # politics = 0 , graphics = 1, autos = 2
		for fw in docs:

			if politics_vocab_dict[fw] != 0:
				#print '\n val',politics_vocab_dict[fw], 'wcount', category_wcount[index]
				likelihood[0] +=  np.log(float(politics_vocab_dict[fw]) + 1/(train_category_wcount[n] + len(vocab_list)))
		
			if graphics_vocab_dict[fw] != 0:
				likelihood[1] +=  np.log(float(graphics_vocab_dict[fw]) + 1/(train_category_wcount[n] + len(vocab_list)))

			if autos_vocab_dict[fw] != 0:
				likelihood[2] +=  np.log(float(autos_vocab_dict[fw])+ 1/(train_category_wcount[n] + len(vocab_list)))
		

		'''allocating document to category'''
		posterior = [0,0,0]
		for j in range(3):
			#print '\nj',j,'likelihood',likelihood[j]
			#print '\nprior',prior
			posterior[j] = float(likelihood[j] + np.log(prior[j]) )
		
		print '\nground truth',n,'likelihood',likelihood 
		#print '\nprior',prior
		print 'prediction',posterior,'decision',posterior.index(max(posterior))
		y_predict.append(posterior.index(max(posterior)))


print 'y_predict', len(y_predict),'ground_y',len(ground_y)

count = 0
for i in range(len(y_predict)):
	if y_predict[i] == ground_y[i]:
		count = count + 1
print 'accuracy', float(count)/ len(y_predict)
					


