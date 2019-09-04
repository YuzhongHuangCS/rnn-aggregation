import os.path
import pickle

def is_ordered(opt):
	keywords = ['Less', 'Between', 'More', 'inclusive','less', 'between', 'more']
	if len(opt) <= 2: #binary
		return False
	for o in opt:
		if any(x in o for x in keywords):
			return True

	return False

def initialize_embedding():
	if os.path.exists('deps.words.pickle'):
		print('Loading embedding from cache')
		with open('deps.words.pickle', 'rb') as fin:
			return pickle.load(fin)
	else:
		word_embedding = {}
		print('Loading embedding from text')
		for line in open('deps.words.txt', encoding='utf-8'):
			parts = line.rstrip().split(' ')
			word = parts[0]
			embs = [float(x) for x in parts[1:]]
			word_embedding[word] = embs
		with open('deps.words.pickle', 'wb') as fout:
			pickle.dump(word_embedding, fout, pickle.HIGHEST_PROTOCOL)
			return word_embedding

def embedding_lookup(t, e):
	return e.get(t, e.get(t.lower(), e.get(t.upper(), e.get(t.capitalize(), None))))
