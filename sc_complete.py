# Shakespeare play classification
# Naive Bayes algorithm originally from scikitlearn:
# http://scikit-learn.org/stable/modules/naive_bayes.html

# import all the naive bayes stuff
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import re
import urllib2
import os.path
import datetime

# import the data -- the plays I'm training the model on

# tragedies
hamlet = open ('shakes_data/hamlet.txt').read().lower()
macbeth = open ('shakes_data/macbeth.txt').read().lower()
othello = open ('shakes_data/othello.txt').read().lower()
king_lear = open ('shakes_data/king_lear.txt').read().lower()
r_and_j = open ('shakes_data/r_and_j.txt').read().lower()
titus = open ('shakes_data/titus.txt').read().lower()
julius_caesar = open ('shakes_data/julius_caesar.txt').read().lower()
coriolanus = open ('shakes_data/coriolanus.txt').read().lower()

timon = open ('shakes_data/timon.txt').read().lower()
a_and_c = open ('shakes_data/a_and_c.txt').read().lower()

# comedies
midsummer = open ('shakes_data/midsummer.txt').read().lower()
much_ado = open ('shakes_data/much_ado.txt').read().lower()
twelfth_night = open ('shakes_data/twelfth_night.txt').read().lower()
as_you_like_it = open ('shakes_data/as_you_like_it.txt').read().lower()
comedy_of_errors = open ('shakes_data/comedy_of_errors.txt').read().lower()
alls_well = open ('shakes_data/alls_well.txt').read().lower()
loves_labors = open ('shakes_data/loves_labors.txt').read().lower()
merry_wives = open ('shakes_data/merry_wives.txt').read().lower()

two_gentlemen = open ('shakes_data/two_gentlemen.txt').read().lower()
tempest = open ('shakes_data/tempest.txt').read().lower()
cymbeline = open ('shakes_data/cymbeline.txt').read().lower()
pericles = open ('shakes_data/pericles.txt').read().lower()
merchant = open ('shakes_data/merchant.txt').read().lower()
measure = open ('shakes_data/measure.txt').read().lower()
shrew = open ('shakes_data/shrew.txt').read().lower()
winters_tale = open ('shakes_data/winters_tale.txt').read().lower()
t_and_c = open ('shakes_data/t_and_c.txt').read().lower()

# histories
henry_v = open ('shakes_data/henry_v.txt').read().lower()
richard_iii = open ('shakes_data/richard_iii.txt').read().lower()
henry_iv_1 = open ('shakes_data/henry_iv_1.txt').read().lower()
henry_iv_2 = open ('shakes_data/henry_iv_2.txt').read().lower()
henry_vi_1 = open ('shakes_data/henry_vi_1.txt').read().lower()
henry_vi_2 = open ('shakes_data/henry_vi_2.txt').read().lower()
henry_vi_3 = open ('shakes_data/henry_vi_3.txt').read().lower()

richard_ii = open ('shakes_data/richard_ii.txt').read().lower()
john = open ('shakes_data/john.txt').read().lower()
henry_viii = open ('shakes_data/henry_viii.txt').read().lower()


class_names = {0:'tragedy', 1:'comedy', 2:'history'}


htmlplays = {"Hamlet":[0, 'http://shakespeare.mit.edu/hamlet/full.html'],
    "Macbeth":[0, 'http://shakespeare.mit.edu/macbeth/full.html'],
    "Othello":[0, 'http://shakespeare.mit.edu/othello/full.html'],
    "King Lear":[0, 'http://shakespeare.mit.edu/lear/full.html'],
    "Romeo and Juliet":[0, 'http://shakespeare.mit.edu/romeo_juliet/full.html'],
    "Titus Andronicus":[0, 'http://shakespeare.mit.edu/titus/full.html'],
    "Julius Caesar":[0, 'http://shakespeare.mit.edu/julius_caesar/full.html'],
    "Coriolanus":[0, 'http://shakespeare.mit.edu/coriolanus/full.html'],
    "Midsummer Night's Dream":[1, 'http://shakespeare.mit.edu/midsummer/full.html'],
    "Much Ado About Nothing":[1, 'http://shakespeare.mit.edu/much_ado/full.html'],
    "Twelfth Night":[1, 'http://shakespeare.mit.edu/twelfth_night/full.html'],
    "As You Like It":[1, 'http://shakespeare.mit.edu/asyoulikeit/full.html'],
    "Comedy of Errors":[1, 'http://shakespeare.mit.edu/comedy_errors/full.html'],
    "All's Well that Ends Well":[1, 'http://shakespeare.mit.edu/allswell/full.html'],
    "Love's Labors Lost":[1, 'http://shakespeare.mit.edu/lll/full.html'],
    "Merry Wives of Windsor":[1, 'http://shakespeare.mit.edu/merry_wives/full.html'],
    "Henry V":[2, 'http://shakespeare.mit.edu/henryv/full.html'],
    "Richard III":[2, 'http://shakespeare.mit.edu/richardiii/full.html'],
    "Henry IV part 1":[2, 'http://shakespeare.mit.edu/1henryiv/full.html'],
    "Henry IV part 2":[2, 'http://shakespeare.mit.edu/2henryiv/full.html'],
    "Henry VI part 1":[2, 'http://shakespeare.mit.edu/1henryvi/full.html'],
    "Henry VI part 2":[2, 'http://shakespeare.mit.edu/2henryvi/full.html'],
    "Henry VI part 3":[2, 'http://shakespeare.mit.edu/3henryvi/full.html'],
    "Timon of Athens":[0, 'http://shakespeare.mit.edu/timon/full.html'],
    "Antony and Cleopatra":[0, 'http://shakespeare.mit.edu/cleopatra/full.html'],
    "Two Gentlemen of Verona":[1, 'http://shakespeare.mit.edu/two_gentlemen/full.html'],
    "The Tempest":[1, 'http://shakespeare.mit.edu/tempest/full.html'],
    "Cymbeline":[1, 'http://shakespeare.mit.edu/cymbeline/full.html'],
    "Pericles":[1, 'http://shakespeare.mit.edu/pericles/full.html'],
    "Merchant of Venice":[1, 'http://shakespeare.mit.edu/merchant/full.html'],
    "Measure for Measure":[1, 'http://shakespeare.mit.edu/measure/full.html'],
    "Taming of the Shrew":[1, 'http://shakespeare.mit.edu/taming_shrew/full.html'],
    "Winter's Tale":[1, 'http://shakespeare.mit.edu/winters_tale/full.html'],
    "Troilus and Cressida":[1, 'http://shakespeare.mit.edu/troilus_cressida/full.html'],
    "Richard II":[2, 'http://shakespeare.mit.edu/richardii/full.html'],
    "King John":[2, 'http://shakespeare.mit.edu/john/full.html'],
    "Henry VIII":[2, 'http://shakespeare.mit.edu/henryviii/full.html']}

playtxts = {"Hamlet":[hamlet, 0], "Macbeth":[macbeth, 0], "Othello":[othello, 0], "King Lear":[king_lear, 0],
	"Romeo and Juliet":[r_and_j, 0], "Titus Andronicus":[titus, 0], "Julius Caesar":[julius_caesar, 0],
	"Coriolanus":[coriolanus, 0], "Midsummer Night's Dream":[midsummer, 1], "Much Ado About Nothing":[much_ado, 1],
	"Twelfth Night":[twelfth_night, 1], "As You Like It":[as_you_like_it, 1], "Comedy of Errors":[comedy_of_errors, 1],
	"All's Well that Ends Well":[alls_well, 1], "Love's Labors Lost":[loves_labors, 1],
	"Merry Wives of Winsor":[merry_wives, 1], "Henry V":[henry_v, 2], "Richard II":[richard_iii, 2],
	"Henry IV part 1":[henry_iv_1, 2], "Henry IV part 2":[henry_iv_2, 2], "Henry VI part 1":[henry_vi_1, 2],
	"Henry VI part 2":[henry_vi_2, 2], "Henry VI part 3":[henry_vi_3, 2]}

new_lines = {
	'few love to hear the sins they love to act': 0,
	'we are such stuff as dreams are made on': 1,
	'i will make the statue move indeed': 1,
	'what have we here? a man, or a fish?': 1,
	'the queen, the queen! the sweetest, dearest creature is dead!': 2,
	'or to take arms against a sea of troubles': 2,
	'there is special providence in the fall of a sparrow': 0,
	'i am justly killed with my own treachery': 0,
	'o that this too too sullied flesh would melt': 0,
	'the time is out of joint, o cursed spite': 0,
	'words without thoughts never to heaven go': 2,
	'give me that man that is not passions slave': 1,
	'i have a speech of fire that fain would blaze': 0,
	'show thy valor and put up thy sword': 2,
	'or close the wall up with our english dead': 2,
	'there is very excellent services committed at the bridge': 2,
	'an absolute gentleman, full of most excellent differences': 1,
	'ay, i praise god, and i have merited some love at his hands': 1}

word_dict = {'dead': 0, 'love': 1, 'crown': 2, 'king': 2, 'laugh': 1, 'drunk': 1, 'stab': 0, 
	'blood': 0, 'die': 0, 'battle': 2, 'kill': 0, 'rich': 1, 'magic': 1, 'mad': 1, 'wine': 1}

def txttrain():

	play_data = [playtxts[key][0] for key in playtxts]
	classes = [playtxts[key][1] for key in playtxts]

	word_vector = CountVectorizer()
	word_vector_counts = word_vector.fit_transform(play_data)

	term_freq_transformer = TfidfTransformer()
	term_freq = term_freq_transformer.fit_transform(word_vector_counts)

	model = MultinomialNB().fit(term_freq, classes)

def secttrain():

	play_data = []
	classes = []

	for key in htmlplays:
		url = htmlplays[key][1]
		if key + ".txt" in os.listdir("cache"):
			raw_data = open("cache/" + key + ".txt").read().lower()
		else:
			print "Getting file " + url + " ..."
			raw_data = urllib2.urlopen(url).read().lower()
			open(os.path.join("cache", key) + ".txt", "w").write(raw_data)
		raw_data = raw_data.split("<h3>")
		processed = [re.sub('(\\n)', ' ', section) for section in raw_data]
		processed = [re.sub('(<.+?>)', ' ', section) for section in processed]
		processed = [re.sub('(\s+)', ' ', section) for section in processed]
		htmlplays[key].append(processed)
		play_data.extend(htmlplays[key][2])
		classes.extend([htmlplays[key][0] for section in processed])

	endscript = datetime.datetime.now()

	difference = endscript - startscript
	print "It took " + str(difference.seconds) + " seconds to load the data."

	word_vector = CountVectorizer()
	word_vector_counts = word_vector.fit_transform(play_data)

	term_freq_transformer = TfidfTransformer()
	term_freq = term_freq_transformer.fit_transform(word_vector_counts)

	model = MultinomialNB().fit(term_freq, classes)

def play():

	test_play_dict = {"Timon of Athens":[timon, 0], "Antony and Cleopatra":[a_and_c, 0],
		"Two Gentlemen of Verona":[two_gentlemen, 1], "The Tempest":[tempest, 1],
		"Cymbeline":[cymbeline, 1], "Pericles":[pericles, 1], "Merchant of Venice":[merchant, 1],
		"Measure for Measure":[measure, 1], "Taming of the Shrew":[shrew, 1],
		"Winter's Tale":[winters_tale, 1], "Troilus and Cressida":[t_and_c, 1],
		"Richard II":[richard_ii, 2], "King John":[john, 2], "Henry VIII":[henry_viii, 2]}

	test_play = [test_play_dict[key][0] for key in test_play_dict]
	new_counts = word_vector.transform(test_play)
	new_term_freq = term_freq_transformer.transform(new_counts)

	predicted = model.predict(new_term_freq)
	print ' '
	print 'Predictions:'
	
	for key, prediction in zip(test_play_dict, predicted): 
		test_play_dict[key].append(prediction)

	for item in test_play_dict:
		predicted_play_class = test_play_dict[item][2]
		print item + " => " + class_names[predicted_play_class]

	probabilities = model.predict_proba(new_term_freq)
	print ' '
	print 'Probabilities:'
	print probabilities
	print ' '

	print 'Validation:'
	ncorrect = 0

	correct_play_classes = [test_play_dict[key][1] for key in test_play_dict]

	for prediction, truth in zip(predicted, correct_play_classes):
		print "Prediction: {}, Truth: {}".format(prediction, truth)
		if prediction == truth:
			ncorrect = ncorrect + 1

	pcorrect = (ncorrect / float(len(correct_play_classes))) * 100

	print ' '
	print "The model got " + str(pcorrect) + "% of its predictions correct."
	print ' '

def lines():

	test_line = new_lines.keys()
	new_counts = word_vector.transform(test_line)
	new_term_freq = term_freq_transformer.transform(new_counts)

	predicted = model.predict(new_term_freq)
	print ' '
	print 'Predictions:'
	
	for key, prediction in zip(new_lines, predicted): 
		new_lines[key] = [new_lines[key], prediction]

	for item in new_lines:
		predicted_play_class = new_lines[item][1]
		print item + " => " + class_names[predicted_play_class]

	probabilities = model.predict_proba(new_term_freq)
	print ' '
	print 'Probabilities:'
	print probabilities
	print ' '

	print 'Validation:'
	ncorrect = 0

	correct_play_classes = [new_lines[key][0] for key in new_lines]

	for prediction, truth in zip(predicted, correct_play_classes):
		print "Prediction: {}, Truth: {}".format(prediction, truth)
		if prediction == truth:
			ncorrect = ncorrect + 1

	pcorrect = (ncorrect / float(len(correct_play_classes))) * 100

	print ' '
	print "The model got " + str(pcorrect) + "% of its predictions correct."
	print ' '

def words():

	test_line = word_dict.keys()
	new_counts = word_vector.transform(test_line)
	new_term_freq = term_freq_transformer.transform(new_counts)

	predicted = model.predict(new_term_freq)
	print ' '
	print 'Predictions:'
	
	for key, prediction in zip(word_dict, predicted): 
		word_dict[key] = [word_dict[key], prediction]

	for item in word_dict:
		predicted_play_class = word_dict[item][1]
		print item + " => " + class_names[predicted_play_class]

	probabilities = model.predict_proba(new_term_freq)
	print ' '
	print 'Probabilities:'
	print probabilities
	print ' '

	print 'Validation:'
	ncorrect = 0

	correct_play_classes = [word_dict[key][0] for key in word_dict]

	for prediction, truth in zip(predicted, correct_play_classes):
		print "Prediction: {}, Truth: {}".format(prediction, truth)
		if prediction == truth:
			ncorrect = ncorrect + 1

	pcorrect = (ncorrect / float(len(correct_play_classes))) * 100

	print ' '
	print "The model got " + str(pcorrect) + "% of its predictions correct."
	print ' '



