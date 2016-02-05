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



def get_filenames(rootDir):

	categories =[]
	categories_doc_count =[]
	documents_count = 0	
	cat_docslist = []
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
	
	return categories ,categories_doc_count,cat_docslist


def get_clean_tokens(cat_docslist,rootDir,categories):

	category_wcount_dict = {}
	total_docs = []
	category_wcount = []
	vocab_list = []
	category_docs = []

	for index,dlist in enumerate(cat_docslist):
		print '\n Reading document',dlist
		category_word_count = 0
		total_docs_cat = 0
		words_docs = []
		for d in dlist:
			total_docs_cat = len(dlist) 
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

			category_word_count += len(filtered_words) 

			for uword in filtered_words:
				if uword not in vocab_list:
					vocab_list.append(uword)
	
			words_docs.append(filtered_words)
		total_docs.append(total_docs_cat)
		category_docs.append(words_docs)
		category_wcount_dict[categories[index]] = category_word_count
		category_wcount.append(category_word_count)

	return vocab_list,category_docs,total_docs,category_wcount
	
