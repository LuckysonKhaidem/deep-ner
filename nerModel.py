import os
from collections import defaultdict
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from nltk.tokenize import word_tokenize
import pickle
import gensim

# word2vecModel = gensim.models.KeyedVectors.load_word2vec_format("/remote/users/lkhaidem/nlu/data_generation/GoogleNews-vectors-negative300-SLIM.bin.gz", binary = True)

def build_vocabulary(corpus,top_num_words = None):
	word_counter = defaultdict(int)
	for document in corpus:
		for token in document:
			word_counter[token] += 1
	word_counter = sorted(word_counter.items(), key = lambda x: x[1], reverse = True)
	vocabulary = ["<PAD>","<UNK>"]
	for word,count in word_counter[:top_num_words]:
		if word not in vocabulary:
			vocabulary.append(word)
	return vocabulary

def get_embedding_matrix(vocabulary):
	embedding_matrix = np.zeros((len(vocabulary),300))
	for i,word in enumerate(vocabulary):
		try:
			embedding_matrix[i] = word2vecModel.wv[word]
		except:
			unk_vector = np.random.normal(size =(300,))
			unk_vector /= np.linalg.norm(unk_vector)
			embedding_matrix[i] = unk_vector
	return embedding_matrix

def pad(corpus, padding_token = "<PAD>", max_length = None):
	if max_length is None:
		max_length = max(map(len, corpus))
	for i in xrange(len(corpus)):
		if len(corpus[i]) < max_length:
			n_padding = max_length - len(corpus[i])
			corpus[i] = [padding_token]*n_padding + corpus[i]
	return corpus


def read_parse_training_data(train_file):
	with open(train_file, "r") as f:
		text = f.read().strip()

	samples = text.split("\n\n")
	sentences = []
	entities = []
	entity_set = []
	for sample in samples:
		sentence = []
		tags = []
		for tokens in sample.split("\n"):
			word, tag = tokens.split("\t")
			sentence.append(word.lower())
			tags.append(tag)
		entity_set += tags
		sentences.append(sentence)
		entities.append(tags)


	entity_set = list(set(entity_set))
	vocabulary = build_vocabulary(sentences, top_num_words = 5000)
	sentences = pad(sentences, padding_token = "<PAD>")
	entities = pad(entities, padding_token = "O")
	max_sentence_length = max(map(len,sentences))

	train_x = np.zeros((len(sentences), max_sentence_length))
	train_y = np.zeros((len(entities), max_sentence_length, len(entity_set)))

	for i in xrange(len(sentences)):
		for j in xrange(len(sentences[i])):
			word = sentences[i][j]
			tag = entities[i][j]
			tag_index = entity_set.index(tag)
			if word not in vocabulary:
				word_index = vocabulary.index("<UNK>")
			else:
				word_index = vocabulary.index(word)
			train_x[i][j] = word_index
			train_y[i][j][tag_index] = 1

	return train_x, train_y, vocabulary, entity_set

def build_model(vocabulary_size, sequence_length, num_entities, embedding_size = 32, embedding_matrix = None):
	model = Sequential()
	if embedding_matrix is not None:
		model.add(Embedding(vocabulary_size, embedding_matrix.shape[1], input_length = sequence_length, weights = [embedding_matrix], trainable = False))
	else:
		model.add(Embedding(vocabulary_size, embedding_size, input_length = sequence_length))
	model.add(Bidirectional(LSTM(64, activation = "tanh", return_sequences=True )))
	# model.add(Dropout(0.5))
	# model.add(Bidirectional(LSTM(150, activation = "relu", return_sequences=True )))
	# model.add(Dropout(0.5))
	model.add(TimeDistributed(Dense(num_entities, activation = "softmax")))
	optimizer = Adam(lr = 0.001, clipvalue = 0.001)
	model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
	return model

def get_sentence_representation(sentence,max_length, vocabulary):
	tokens = [word_tokenize(sentence)]
	tokens = pad(tokens, padding_token = "<PAD>", max_length = max_length)
	x = np.zeros((1,max_length))
	for i,token in enumerate(tokens[0]):
		if token not in vocabulary:
			x[0][i] = vocabulary.index("<UNK>")
		else:
			x[0][i] = vocabulary.index(token)
	return x

def extract_entities(model,vocabulary, entity_set, max_length):
	while True:
		sentence = raw_input("Enter a setence: ").lower()
		x = get_sentence_representation(sentence,max_length,vocabulary)
		y_pred = model.predict(x)[0]
		y_pred = np.argmax(y_pred, axis = -1).ravel()
		tokens = word_tokenize(sentence)
		tokens = pad([tokens], max_length = max_length)
		for i in xrange(y_pred.shape[0]):
			word = tokens[0][i]
			tag = entity_set[int(y_pred[i])]
			if word != "<PAD>":
				print word,tag
	
def test(model,X, vocabulary, entity_set):
	for i in xrange(len(X)):
		y_pred = model.predict(X[i].reshape((1,-1)))[0]
		y_pred = np.argmax(y_pred, axis = -1).ravel()
		for j in xrange(y_pred.shape[0]):
			word = vocabulary[int(X[i][j])]
			tag = entity_set[int(y_pred[j])]
			if word != "<PAD>":
				print word, tag
		print ""

def save_model(model, vocabulary, entity_set, max_length):
	print ""
	print "Saving model to disk....."
	model_json = model.to_json()
	with open("entity_extraction_model.json", "w") as f:
		f.write(model_json)
	model.save_weights("entity_extraction_model_weights.h5")
	auxillary_information = {"vocabulary": vocabulary, "entity_set": entity_set, "max_length": max_length}
	with open("auxillary_information.pickle", "wb") as f:
		pickle.dump(auxillary_information, f)

def main():
	X,y, vocabulary, entity_set = read_parse_training_data("ner_dataset.txt")
	# embedding_matrix = get_embedding_matrix(vocabulary)
	vocabulary_size = len(vocabulary)
	sequence_length = X.shape[1]
	num_entities = y.shape[2]

	model = build_model(vocabulary_size,sequence_length,num_entities, 500)
	print "Model Summary........"
	print model.summary()

	train_x, test_x, train_y, test_y = train_test_split(X,y, random_state = 600)
	train_x, train_y = resample(train_x, train_y, replace=  True, random_state = 600)

	model.fit(train_x,train_y, epochs = 5, verbose = 1, shuffle = True, batch_size = 512, validation_data = (test_x, test_y))
	save_model(model,vocabulary, entity_set, sequence_length)
	# test(model, test_x[:1000], vocabulary, entity_set)

if __name__ == "__main__":
	main()
