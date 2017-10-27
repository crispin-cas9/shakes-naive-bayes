# Shakespeare play classification
# Naive Bayes algorithm originally from scikitlearn:
# http://scikit-learn.org/stable/modules/naive_bayes.html

# import all the naive bayes stuff
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# import the data -- the plays I'm training the model on

# dictionary of plays: their name, file name, and correct class
plays = {"Hamlet":['hamlet', 0], "Macbeth":['macbeth', 0], "Othello":['othello', 0], "King Lear":['king_lear', 0], "Romeo and Juliet":['r_and_j', 0], "Titus Andronicus":['titus', 0], "Julius Caesar":['julius_caesar', 0], "Coriolanus":['coriolanus', 0], "Midsummer Night's Dream":['midsummer', 1], "Much Ado About Nothing":['much_ado', 1], "Twelfth Night":['twelfth_night', 1], "As You Like It":['as_you_like_it', 1], "Comedy of Errors":['comedy_of_errors', 1], "All's Well that Ends Well":['alls_well', 1], "Love's Labors Lost":['loves_labors', 1], "Merry Wives of Windsor":['merry_wives', 1], "Henry V":['henry_v', 2], "Richard II":['richard_iii', 2], "Henry IV part 1":['henry_iv_1', 2], "Henry IV part 2":['henry_iv_2', 2], "Henry VI part 1":['henry_vi_1', 2], "Henry VI part 2":['henry_vi_2', 2], "Henry VI part 3":['henry_vi_3', 2]}

# dictionary of the plays I'm trying to classify
test_plays = {"Timon of Athens":['timon', 0], "Antony and Cleopatra":['a_and_c', 0], "Two Gentlemen of Verona":['two_gentlemen', 1], "The Tempest":['tempest', 1], "Cymbeline":['cymbeline', 1], "Pericles":['pericles', 1], "Merchant of Venice":['merchant', 1], "Measure for Measure":['measure', 1], "Taming of the Shrew":['shrew', 1], "Winter's Tale":['winters_tale', 1], "Troilus and Cressida":['t_and_c', 1], "Richard II":['richard_ii', 2], "King John":['john', 2], "Henry VIII":['henry_viii', 2], "King Leir":['leir', 1]}

for play in plays:
	shortname = plays[play][0]
	plays[play].append(open ('shakes_data/' + shortname + '.txt').read().lower())

for play in test_plays:
	shortname = test_plays[play][0]
	test_plays[play].append(open ('shakes_data/' + shortname + '.txt').read().lower())

play_data = [plays[key][2] for key in plays]
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

# take the texts and figure out the frequencies of the words
test_play = [test_plays[key][2] for key in test_plays]
new_counts = word_vector.transform(test_play)
new_term_freq = term_freq_transformer.transform(new_counts)

# based on that, predict their classes and print that
predicted = model.predict(new_term_freq)
print ' '
print 'Predictions:'
	
for key, prediction in zip(test_plays, predicted): 
	test_plays[key].append(prediction)

for play in test_plays:
	predicted_play_class = test_plays[play][3]
	print play + " => " + class_names[predicted_play_class]

probabilities = model.predict_proba(new_term_freq)
print ' '
print 'Probabilities:'
print probabilities
print ' '

# Validation!!

print 'Validation:'
ncorrect = 0

# take the correct play classes from the dictionary
correct_play_classes = [test_plays[key][1] for key in test_plays]

# for each predicted class, compare it to the correct class
# count the number of predictions that the model got correct
for prediction, truth in zip(predicted, correct_play_classes):
	print "Prediction: {}, Truth: {}".format(prediction, truth)
	if prediction == truth:
		ncorrect = ncorrect + 1

# based on the number of correct guesses and the number of plays overall, print the
#  percentage the model got correct

pcorrect = (ncorrect / float(len(correct_play_classes))) * 100

print ' '
print "The model got " + str(pcorrect) + "% of its predictions correct."
print ' '

