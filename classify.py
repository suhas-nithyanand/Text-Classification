import os,sys
import json
import re
import time
import nltk 
import numpy as np
import pickle
import simplejson
from collections import defaultdict
import pprint
from nltk.corpus import stopwords





with open('vocab.txt') as data_file:    
	vocab_list = json.load(data_file)

with open('training_docs.txt') as data_file:    
	training_docs = json.load(data_file)

with open('total_docs.txt') as data_file:    
	total_docs = json.load(data_file)


category_wcount = []

with open('category_wcount.txt') as data_file:    
	category_wcount = json.load(data_file)

vocab_list = np.asarray(vocab_list)
#training_docs = np.asarray(training_docs)


politics_vocab_dict = defaultdict(int)
for x in vocab_list:
	politics_vocab_dict[x] = 0
'''remove words in vocab which occurs less than 5 times'''

graphics_vocab_dict = defaultdict(int)
for x in vocab_list:
	graphics_vocab_dict[x] = 0

autos_vocab_dict = defaultdict(int)
for x in vocab_list:
	autos_vocab_dict[x] = 0

#print 'shape',training_docs.shape

print len(training_docs[0])

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
 		
#print 'p',politics_vocab_dict['andy'],'auto',autos_vocab_dict['andy']
#pprint.pprint(graphics_vocab_dict)		
print 'shape of vocab_list',vocab_list.shape
#print 'shape of training docs',training_docs.shape



categories =[]
categories_doc_count =[]
d = 0
documents_count = 0

cat_docslist = []
k = 0
rootDir = '/home/suhas/ucf/2ndsem/nlp/1st_assign/workspace/test'
dirnamelist = []
for dirName, subdirList, fileList in os.walk(rootDir):

	if len(subdirList) >0:
		categories = subdirList
		
	if len(subdirList) == 0:
		documents_count = len(fileList)
		categories_doc_count.append(documents_count)
	
	if len(subdirList) ==0:
		cat_docslist.append(fileList)
		dirnamelist.append(dirName)
		

#print 'categories:',categories
#print 'categories count:',categories_doc_count
#print 'category docs list',cat_docslist

global_dict = {}
doc_name_list = []
vocab_list =[]
vocab_list =[]
words_docs = []
category_docs = []
ground_y = []


''' Prior calculation'''

prior = np.zeros(3)

for i in range(3):
	prior[i] = float(total_docs[i])/sum(total_docs)

y_predict = []

for index,dlist in enumerate(cat_docslist):
	print '\nindex:',index, 'value',dlist
	
	for d in dlist:
		ground_y.append(index)
		#print '\ndocument:',d
		f=open('{0}/{1}/{2}'.format(rootDir,categories[index],d),'r')
		wlist = f.read()
		wlist = re.sub('[^A-Za-z]+', ' ', wlist)
		wlist = re.sub(r'\b\w{1,2}\b', '', wlist)
		wlist = re.sub(r'\w*\d\w*', '', wlist).strip()
		word_list = re.findall(r"[\w']+", wlist)
		tokens = [token.lower() for token in word_list]
		filtered_words = [word for word in tokens if word not in stopwords.words('english')]
		for lword in filtered_words:
			if len(lword) < 3:
        			filtered_words.remove(lword)

		#print 'filtered words',filtered_words
		
		likelihood = [0,0,0] # politics = 0 , graphics = 1, autos = 2
		for fw in filtered_words:
			#print 'word',fw
		
			if politics_vocab_dict[fw] != 0:
				#print '\n val',politics_vocab_dict[fw], 'wcount', category_wcount[index]
				likelihood[0] +=  np.log(float(politics_vocab_dict[fw]) + 1/(category_wcount[index] + len(vocab_list)))
		
			if graphics_vocab_dict[fw] != 0:
				likelihood[1] +=  np.log(float(graphics_vocab_dict[fw]) + 1/(category_wcount[index] + len(vocab_list)))

			if autos_vocab_dict[fw] != 0:
				likelihood[2] +=  np.log(float(autos_vocab_dict[fw])+ 1/(category_wcount[index] + len(vocab_list)))
		
		'''allocating document to category'''
		posterior = [0,0,0]
		for j in range(3):
			#print '\nj',j,'likelihood',likelihood[j]
			#print '\nprior',prior
			posterior[j] = float(likelihood[j] + np.log(prior[j]) )
		
		print '\nindex',index,'likelihood',likelihood 
		#print '\nprior',prior
		print 'prediction',posterior,'decision',posterior.index(max(posterior))
		y_predict.append(posterior.index(max(posterior)))


print 'y_predict', len(y_predict),'ground_y',len(ground_y)

count = 0
for i in range(len(y_predict)):
	if y_predict[i] == ground_y[i]:
		count = count + 1
print 'accuracy', float(count)/ len(y_predict)
					
			

