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
#nltk.download('punkt')
categories =[]
categories_doc_count =[]
d = 0
documents_count = 0

cat_docslist = []
k = 0
rootDir = '/home/suhas/ucf/2ndsem/nlp/1st_assign/workspace/train'
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
		

print 'categories:',categories
#print 'categories count:',categories_doc_count
print 'category docs list',cat_docslist
counter = 0

global_dict = {}
doc_name_list = []
vocab_list =[]
vocab_list =[]
words_docs = []
category_docs = []

category_wcount_dict = {}

category_word_count = 0

total_docs = []


category_wcount = []
for index,dlist in enumerate(cat_docslist):
	print '\nindex:',index, 'value',dlist
	category_word_count = 0
	total_docs_cat = 0
	words_docs = []
	for d in dlist:
		total_docs_cat = len(dlist) 
		print '\ndocument:',d
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
#		str1 = ' '.join(filtered_words)
#		tokens = nltk.word_tokenize(str1)
#		print 'tokens',tokens
		#word_list = re.findall(r"[\w']+", tokens)
		#print 'word list',word_list
		#print 'filtered words',filtered_words

		category_word_count += len(filtered_words) 

		for uword in filtered_words:
			if uword not in vocab_list:
				vocab_list.append(uword)
	
		words_docs.append(filtered_words)
	total_docs.append(total_docs_cat)
	category_docs.append(words_docs)
	category_wcount_dict[categories[index]] = category_word_count
	category_wcount.append(category_word_count)

print 'len of vocab list', len(vocab_list)
print 'category docs', category_docs
#np.savetxt('vocab_list', vocab_list, delimiter=' ')
#pickle.dump(vocab_list, 'vocab_list.txt')
#outfile.write("\n".join(vocab_list))
#f = open('vocab.txt', 'w')
#simplejson.dump(vocab_list, f)
#f.close()
# 
#f = open('training_docs.txt', 'w')
#simplejson.dump(category_docs, f)
#f.close()		

with open('total_docs.txt', 'w') as doutfile:
    json.dump(total_docs, doutfile)

with open('vocab.txt', 'w') as outfile:
    json.dump(vocab_list, outfile)

with open('training_docs.txt', 'w') as toutfile:
    json.dump(category_docs, toutfile)

with open('category_wcount.txt', 'w') as coutfile:
    json.dump(category_wcount, coutfile)
	
#		for uword in filtered_words:
#			if uword not in global_dict:
#				global_dict[uword] = 1
#			else:
#				global_dict[uword] +=1

#print 'global_dict',global_dict	

