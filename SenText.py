#SenText - A Text sentiment analyser

'''
*Author: Adarsh Kumar
*Language: Python
*External Modules required: nltk (including the "movie_reviews" corpus)
*Dataset used: ntltk.corpus.movie_reviews - Contains 1000 negative and 1000 positive movie reviews

*Important note:
If you want to pickle the trained classifier, uncomment the line 151 and run once. Once run, comment it again
and uncomment line 78. Pickle loading may take longer time than training everytime. This happens because we're 
loading the classifier from the hard disk. It totally depends on the disk speed. While the training depends on
the CPU Time.
'''


import nltk.classify.util
from nltk.metrics import precision
from nltk.metrics import recall
from nltk.metrics import f_measure
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.classify.svm import SvmClassifier
import random
import pickle
import collections
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from Tkinter import * #For Gui
import ttk
import nltk.classify.api

#Root element for the GUI
top = Tk()
#Set title for the GUI
top.wm_title("SenText - A Text Sentiment Analyser")
#Set default frame size
top.geometry("500x500")

#Takes a while to do all the below stuff
print("Please wait...")

#A set describing all english stopwords
stopwordset = set(stopwords.words('english'))

#This feature extractor will return the dictionary even with the bigram collocations
def word_feats(words):
	bigram_score=BigramAssocMeasures.chi_sq
	n=200
	bigram_finder = BigramCollocationFinder.from_words(words)
	bigrams = bigram_finder.nbest(bigram_score, n)
	return dict([(str(ngram).lower(), True) for ngram in itertools.chain(words, bigrams)])

#Negative sentiment files 
negids = movie_reviews.fileids('neg')
#Positive sentiment files
posids = movie_reviews.fileids('pos')

#Generate negative and positive features
negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

#Cutoff training data
negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4

#Equally likely distributed features
random.shuffle(negfeats)
random.shuffle(posfeats)

#Final training data: 3/4 of the whole corpora
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
#Final test data: 1/4 of the whole corpora
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print '\nTraining on %d instances, testing on %d instances' % (len(trainfeats), len(testfeats))
 
classifier = NaiveBayesClassifier.train(trainfeats)

#To load the classifier once trained,
#We'll load from pickle

'''
#"Sentiment.pkl" is the saved file in the current directory after running pickle.dump()
clf_f = open("Sentiment.pkl","rb") 
classifier = pickle.load(clf_f)
clf_f.close
'''

#Precison and recall calculation
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
 
for i, (feats, label) in enumerate(testfeats):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

#35% false positives for the pos label.
print 'Positive precision:', precision(refsets['pos'], testsets['pos'])
#98% recall, so very few false negatives
print 'Positive recall:', recall(refsets['pos'], testsets['pos'])

print 'Positive F-measure:', f_measure(refsets['pos'], testsets['pos'])
print 'Negative precision:', precision(refsets['neg'], testsets['neg'])
print 'Negative recall:', recall(refsets['neg'], testsets['neg'])
print 'Negative F-measure:', f_measure(refsets['neg'], testsets['neg'])

#Accuracy
print '\nAccuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()
#print classifier.classify(word_feats('bad'))

#Remove stopwords and present the paragraph words in order to make sense
def summary(words):
    return collections.OrderedDict([(word, True) for word in words if word not in stopwordset])


#while(True):
def analyse():
	st = E1.get() #get text from entry box
	p=st.split() #split to list
	context = classifier.classify(word_feats(p))
	if(str(context)=='pos'):
		finalstr = "Positive Statement\n*Tags: "
		l = summary(p)
		for k in l.keys():
			finalstr += k+", "
		var.set(finalstr)
		label.pack()

	elif(str(context)=='neg'):
		finalstr = "Negative Statement\n*Tags: "
		l = summary(p)
		for k in l.keys():
			finalstr += k+", "
		var.set(finalstr)
		label.pack()
	#B.destroy() #TODO: remove this line

#Gui start
B = Button(top, text ="Get sentiment", command = analyse)

L1 = Label(top, text="Enter text to analyse:")

var = StringVar()
label = Label(top, textvariable=var, relief=RIDGE, fg= "blue", bg="yellow", wraplength=490, font="Verdana 10 bold")

E1 = Entry(top, bd =5)
L1.pack()
E1.pack()
B.pack()
#Gui ends



#Run once, to store the trained classifier
'''
save_clf = open("Sentiment.pkl","wb")
pickle.dump(classifier, save_clf)
save_clf.close()
'''
top.mainloop()