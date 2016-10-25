# Shakespeare play classification
# Naive Bayes algorithm originally from scikitlearn:
# http://scikit-learn.org/stable/modules/naive_bayes.html

# import all the naive bayes stuff
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# import regex and url reader
import re
import urllib2


hamlet_data = urllib2.urlopen('http://shakespeare.mit.edu/hamlet/full.html').read().lower().split("<h3>")
hamlet = [re.sub('(\\n)', ' ', section) for section in hamlet_data]
hamlet = [re.sub('(<.+?>)', ' ', section) for section in hamlet]
hamlet = [re.sub('(\s+)', ' ', section) for section in hamlet]

henryv_data = urllib2.urlopen('http://shakespeare.mit.edu/henryv/full.html').read().lower().split("<h3>")
henryv = [re.sub('(\\n)', ' ', section) for section in henryv_data]
henryv = [re.sub('(<.+?>)', ' ', section) for section in henryv]
henryv = [re.sub('(\s+)', ' ', section) for section in henryv]

# import the data -- the plays I'm training the model on
# tell the model which play is in which class

plays = {"Hamlet":[hamlet, 0, 'http://shakespeare.mit.edu/hamlet/full.html'],
	"Macbeth":[macbeth, 0, 'http://shakespeare.mit.edu/macbeth/full.html'],
	"Othello":[othello, 0, 'http://shakespeare.mit.edu/othello/full.html'],
	"King Lear":[king_lear, 0, 'http://shakespeare.mit.edu/lear/full.html'],
	"Romeo and Juliet":[r_and_j, 0, 'http://shakespeare.mit.edu/romeo_juliet/full.html'],
	"Titus Andronicus":[titus, 0, 'http://shakespeare.mit.edu/titus/full.html'],
	"Julius Caesar":[julius_caesar, 0, 'http://shakespeare.mit.edu/julius_caesar/full.html'],
	"Coriolanus":[coriolanus, 0, 'http://shakespeare.mit.edu/coriolanus/full.html'],
	"Midsummer Night's Dream":[midsummer, 1, 'http://shakespeare.mit.edu/midsummer/full.html'],
	"Much Ado About Nothing":[much_ado, 1, 'http://shakespeare.mit.edu/much_ado/full.html'],
	"Twelfth Night":[twelfth_night, 1, 'http://shakespeare.mit.edu/twelfth_night/full.html'],
	"As You Like It":[as_you_like_it, 1, 'http://shakespeare.mit.edu/asyoulikeit/full.html'],
	"Comedy of Errors":[comedy_of_errors, 1, 'http://shakespeare.mit.edu/comedy_errors/full.html'],
	"All's Well that Ends Well":[alls_well, 1, 'http://shakespeare.mit.edu/allswell/full.html'],
	"Love's Labors Lost":[loves_labors, 1, 'http://shakespeare.mit.edu/lll/full.html'],
	"Merry Wives of Windsor":[merry_wives, 1, 'http://shakespeare.mit.edu/merry_wives/full.html'],
	"Henry V":[henry_v, 2, 'http://shakespeare.mit.edu/henryv/full.html'],
	"Richard III":[richard_iii, 2, 'http://shakespeare.mit.edu/richardiii/full.html'],
	"Henry IV part 1":[henry_iv_1, 2, 'http://shakespeare.mit.edu/1henryiv/full.html'],
	"Henry IV part 2":[henry_iv_2, 2, 'http://shakespeare.mit.edu/2henryiv/full.html'],
	"Henry VI part 1":[henry_vi_1, 2, 'http://shakespeare.mit.edu/1henryvi/full.html'],
	"Henry VI part 2":[henry_vi_2, 2, 'http://shakespeare.mit.edu/2henryvi/full.html'],
	"Henry VI part 3":[henry_vi_3, 2, 'http://shakespeare.mit.edu/3henryvi/full.html'],
	"Timon of Athens":[timon, 0, 'http://shakespeare.mit.edu/timon/full.html'],
	"Antony and Cleopatra":[a_and_c, 0, 'http://shakespeare.mit.edu/cleopatra/full.html'],
	"Two Gentlemen of Verona":[two_gentlemen, 1, 'http://shakespeare.mit.edu/two_gentlemen/full.html'],
	"The Tempest":[tempest, 1, 'http://shakespeare.mit.edu/tempest/full.html'],
	"Cymbeline":[cymbeline, 1, 'http://shakespeare.mit.edu/cymbeline/full.html'],
	"Pericles":[pericles, 1, 'http://shakespeare.mit.edu/pericles/full.html'],
	"Merchant of Venice":[merchant, 1, 'http://shakespeare.mit.edu/merchant/full.html'],
	"Measure for Measure":[measure, 1, 'http://shakespeare.mit.edu/measure/full.html'],
	"Taming of the Shrew":[shrew, 1, 'http://shakespeare.mit.edu/taming_shrew/full.html'],
	"Winter's Tale":[winters_tale, 1, 'http://shakespeare.mit.edu/winters_tale/full.html'],
	"Troilus and Cressida":[t_and_c, 1, 'http://shakespeare.mit.edu/troilus_cressida/full.html'],
	"Richard II":[richard_ii, 2, 'http://shakespeare.mit.edu/richardii/full.html'],
	"King John":[john, 2, 'http://shakespeare.mit.edu/john/full.html'],
	"Henry VIII":[henry_viii, 2, 'http://shakespeare.mit.edu/henryviii/full.html']}

play_data = [plays[key][0] for key in plays]
classes = [plays[key][1] for key in plays]
class_names = {0:'tragedy', 1:'comedy', 2:'history'}

word_vector = CountVectorizer()
word_vector_counts = word_vector.fit_transform(play_data)

# Account for the length of the plays:
#   get the frequency with which the word occurs instead of the raw number of times
term_freq_transformer = TfidfTransformer()
term_freq = term_freq_transformer.fit_transform(word_vector_counts)

# Train the Naive Bayes model
model = MultinomialNB().fit(term_freq, classes)

# tests: classify lines/words as comedy/history/tragedy
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
    'too much of water hast thou, poor ophelia': 0,
    'show thy valor and put up thy sword': 2,
    'or close the wall up with our english dead': 2,
    'there is very excellent services committed at the bridge': 2,
    'an absolute gentleman, full of most excellent differences': 1,
    'ay, i praise god, and i have merited some love at his hands': 1,}

word_dict = {'dead': 0, 'love': 1, 'crown': 2, 'king': 2, 'laugh': 1, 'drunk': 1, 'stab': 0, 
	'blood': 0, 'die': 0, 'battle': 2, 'kill': 0, 'rich': 1, 'magic': 1, 'mad': 1, 'wine': 1}

# dictionary of the plays I'm trying to classify: their name, content, and correct class
test_play_dict = {"Timon of Athens":[timon, 0], "Antony and Cleopatra":[a_and_c, 0],
	"Two Gentlemen of Verona":[two_gentlemen, 1], "The Tempest":[tempest, 1],
	"Cymbeline":[cymbeline, 1], "Pericles":[pericles, 1], "Merchant of Venice":[merchant, 1],
	"Measure for Measure":[measure, 1], "Taming of the Shrew":[shrew, 1],
	"Winter's Tale":[winters_tale, 1], "Troilus and Cressida":[t_and_c, 1],
	"Richard II":[richard_ii, 2], "King John":[john, 2], "Henry VIII":[henry_viii, 2]}

# take the texts and figure out the frequencies of the words
test_play = [test_play_dict[key][0] for key in test_play_dict]
new_counts = word_vector.transform(test_play)
new_term_freq = term_freq_transformer.transform(new_counts)

# based on that, predict their classes and print that
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

# Validation!!

print 'Validation:'
ncorrect = 0

# take the correct play classes from the dictionary
correct_play_classes = [test_play_dict[key][1] for key in test_play_dict]

# for each predicted class, compare it to the correct class
# count the number of predictions that the model got correct
for prediction, truth in zip(predicted, correct_play_classes):
	print "Prediction: {}, Truth: {}".format(prediction, truth)
	if prediction == truth:
		ncorrect = ncorrect + 1

# based on the number of correct guesses and the number of plays overall, print the
#  percentage the model got correct
print ' '
print "The model got " + str((ncorrect / float(len(correct_play_classes))) * 100) + "% of its predictions correct."
print ' '

