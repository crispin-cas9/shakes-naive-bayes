# Shakespeare play classification
# Naive Bayes algorithm originally from scikitlearn

# import all the naive bayes stuff
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

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

# tell the model which play is in which class
plays = {"Hamlet":[hamlet, 0], "Macbeth":[macbeth, 0], "Othello":[othello, 0], "King Lear":[king_lear, 0],
	"Romeo and Juliet":[r_and_j, 0], "Titus Andronicus":[titus, 0], "Julius Caesar":[julius_caesar, 0],
	"Coriolanus":[coriolanus, 0], "Midsummer Night's Dream":[midsummer, 1], "Much Ado About Nothing":[much_ado, 1],
	"Twelfth Night":[twelfth_night, 1], "As You Like It":[as_you_like_it, 1], "Comedy of Errors":[comedy_of_errors, 1],
	"All's Well that Ends Well":[alls_well, 1], "Love's Labors Lost":[loves_labors, 1],
	"Merry Wives of Winsor":[merry_wives, 1], "Henry V":[henry_v, 2], "Richard II":[richard_iii, 2],
	"Henry IV part 1":[henry_iv_1, 2], "Henry IV part 2":[henry_iv_2, 2], "Henry VI part 1":[henry_vi_1, 2],
	"Henry VI part 2":[henry_vi_2, 2], "Henry VI part 3":[henry_vi_3, 2]}

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

