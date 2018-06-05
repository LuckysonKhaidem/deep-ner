from flask import Flask, request, jsonify, render_template
from nltk.tokenize import word_tokenize
import numpy as np 
import os
import json
import pickle
from keras.models import model_from_json

def load_model(model_name, auxillary_filename):
	with open("{}.json".format(model_name)) as f:
		model_json = f.read()
	model = model_from_json(model_json)
	model.load_weights("{}_weights.h5".format(model_name))
	with open(auxillary_filename, "rb") as f:
		aux = pickle.load(f)
	return model,aux

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(32)
model, aux = load_model("entity_extraction_model", "auxillary_information.pickle")

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

def pad(corpus, padding_token = "<PAD>", max_length = None):
	if max_length is None:
		max_length = max(map(len, corpus))
	for i in xrange(len(corpus)):
		if len(corpus[i]) < max_length:
			n_padding = max_length - len(corpus[i])
			corpus[i] = [padding_token]*n_padding + corpus[i]
	return corpus

def process_sentence(sentence):
	vocabulary = aux["vocabulary"]
	entity_set = aux["entity_set"]
	max_length = aux["max_length"]
	x = get_sentence_representation(sentence,max_length,vocabulary)
	y_pred = model.predict(x)[0]
	y_pred = np.argmax(y_pred, axis = -1).ravel()
	tokens = word_tokenize(sentence)
	tokens = pad([tokens], max_length = max_length)
	result = []
	for i in xrange(y_pred.shape[0]):
		word = tokens[0][i]
		tag = entity_set[int(y_pred[i])]
		if word != "<PAD>":
			result.append((word,tag))
	return result

@app.route("/extract_entities", methods = ["GET"])
def extract_entities():
	sentence = request.args["sentence"].lower()
	result = process_sentence(sentence)
	return jsonify(result = result)

@app.route("/", methods = ["GET"])
def index():
	return render_template("index.html")

if __name__ == "__main__":
	app.run("localhost",8080, debug = False)
