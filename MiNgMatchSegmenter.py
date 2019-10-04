
"""
***MiNgMatch Segmenter***

Data-driven word segmentation algorithm searching for the
shortest sequence of n-grams needed to match the input string.

Copyright (c) 2019 by Karol Nowakowski
"""


import argparse
import io
import math

import time  # for measuring execution time


class Model(object):
	
	def __init__(self, filename):
		self.ngrams = self.parse(filename)

	def is_ngram(self, string):
		"""
		Checks if there is an n-gram in the
		training data, such that matches
		'string' if we ignore whitespaces.
		"""
		return string in self.ngrams

	
	def get_segmentation(self, string):
		"""
		Returns the most probable segmentation
		of a string, if it exists in the training
		data. Otherwise, returns the string without
		modifying it.
		This version uses n-gram data with indices
		of whitespaces for each n-gram, rather than
		actual segmented strings.
		"""
		if self.is_ngram(string):
			indices = self.get_whitespace_indices(string)
			segmented = string
			for i in indices:
				segmented = segmented[:i] + ' ' + segmented[i:]
			return segmented
		else:
			return string
	
	
	def get_whitespace_indices(self, ngram):
		indices = self.ngrams[ngram][0]
		if indices:
			return (int(i) for i in indices.split(','))
		else:
			return ()


	def calc_probability(self, ngrams):
		"""
		Sums log probabilities of
		a sequence of n-grams
		"""
		return sum((self.ngrams[n][1] for n in ngrams))

	
	@staticmethod
	def parse(filename):
		"Read `filename` and parse tab-separated file with n-grams and probabilities."
		with io.open(filename, encoding='utf-8') as reader:
			lines = (line.split('\t') for line in reader)
			return dict((unsegmented, (indices , float(probability)))
						for unsegmented, indices, probability in lines)



class Segmenter(object):
	
	def __init__(self, model, max):
		self.model = model
		self.limit = max  # limit of n-grams per input string
		self.processor = Processor(to_lowercase = True)
	
	
	def find_candidates(self, text):
		candidates = []  # candidate segmentations
		# 2 dicts for memoization; keys represent the remaining parts of text, yet to be segmented
		partial_segmentations = {text:[]}
		current_step_partial = {text:[]}
		limit = min(self.limit, len(text))
		
		for i in range(0, limit):
			# get the parts yet to be segmented and check if there are any full matches
			remaining = current_step_partial.keys()
			for string in remaining:
				if self.model.is_ngram(string):
					ngram_seq = partial_segmentations[string] + [string]
					candidates.append(ngram_seq)
			if candidates:
				break
			# if no n-grams could be matched to the given
			# input segment or its substring, treat it as OoV
			if not current_step_partial:
				break
			# reset the dictionary
			current_step_partial = dict()
			# find further partial segmentations
			for string in remaining:
				for prefix, suffix in Segmenter.divide(string):
					# if a partial segmentation resulting in the same
					# suffix has already been recorded, ignore it
					if (suffix not in partial_segmentations
							and self.model.is_ngram(prefix)):
						ngram_seq = partial_segmentations[string] + [prefix]
						# if a partial segmentation resulting in the same
						# suffix has already been recorded in the current step,
						# choose the sequence with higher probability
						if suffix in current_step_partial:
							prob = self.model.calc_probability(ngram_seq)
							prob_old = self.model.calc_probability(
										current_step_partial[suffix])
							if prob > prob_old:
								current_step_partial[suffix] = ngram_seq	
						else:
							current_step_partial[suffix] = ngram_seq
			# update the list of best partial segmentations
			for key in current_step_partial:
				partial_segmentations[key] = current_step_partial[key]
		
		if not candidates:
			return [[text]]
		
		return candidates
	
	
	def choose_best_sequence(self, candidates):
		if len(candidates) == 1:
			return candidates[0]
		best_seq = []
		highest_prob = -math.inf
		for seq in candidates:
			prob = self.model.calc_probability(seq)
			if prob > highest_prob:
				best_seq = seq
				highest_prob = prob
		return best_seq
	
	
	def segment(self, text):
		preprocessed = self.processor.preprocess(text)
		candidates = self.find_candidates(preprocessed)
		best = self.choose_best_sequence(candidates)
		segmented = " ".join([self.model.get_segmentation(string)
										for string in best])
		return self.processor.handle_punctuation(segmented)
	
	
	@classmethod
	def divide(cls, text):
		"Yield `(prefix, suffix)` pairs from `text`."
		for pos in range(1, len(text)+1):
			yield (text[:pos], text[pos:])



class Processor(object):
	
	def __init__(self, to_lowercase = False):
		self.to_lowercase = to_lowercase  # if set to true, all strings will be converted to lowercase
		self.transformation_rules = {}
		#self.punc = [',', '"', '\'', '?', '!', ':', ';', '/']
		self.punc_left = ['"', '\'', '(', '[']  # punctuation marks that stand before a word
		self.punc_right = ['.', ',', '"', '\'', '?', '!', ':', ';', '/', ')', ']']  # punctuation marks that stand after a word
		# The following punctuation marks are expected to appear in pairs.
		# Such symbol shall not be separated from a token if its pair stands
		# in the middle of that token (abc [de] -> abc [ de ], but abc[de] -> abc[de]).
		#self.punc_pairs = {'(':')', ')':'(', '[':']', ']':'['}
		
		
	def preprocess(self, string):
		if string is not None:
			if self.to_lowercase:
				string = string.lower()
			# apply transformation rules
			for r in self.transformation_rules:
				if r in string:
					string = string.replace(r, self.transformation_rules[r])
			return string
	
	
	def handle_punctuation(self, string):
		"""
		Separates punctuation marks from alpha-numeric
		strings.
		"""
		tokens_in = string.split()
		tokens_out = []
		for t in tokens_in:
			punc_left = []  # for punctuation marks standing before the word
			punc_right = []  # for punctuation marks standing after the word
			while len(t) > 1 and t[0] in self.punc_left:
				punc_left.append(t[0])
				t = t[1:]
			while len(t) > 1 and t[-1] in self.punc_right:
				punc_right.append(t[-1])
				t = t[:-1]
			tokens_out.extend(punc_left + [t] + punc_right)
		return self.handle_ellipses(" ".join(tokens_out))
	
	def handle_ellipses(self, string):
		"""
		Merge back together the ellipses split in the
		process of handling punctuation.
		"""
		while ". ." in string:
			string = string.replace(". .", "..")
		return string
		

def main():
	start = time.time()
	args = initialize()
	if args.trainset is None:
		print("Specify training data file!")
	elif args.input is None:
		print("Specify input file!")
	elif args.output is None:
		print("Specify output file!")
	else:
		model = Model(args.trainset)
		segmenter = Segmenter(model, args.limit)
		lines = read_lines(args.input)
		output = []
		for ln in lines:
			tokens = ln.split()
			line_segmented = []
			for t in tokens:
				tok_segmented = segmenter.segment(t)
				line_segmented.append(tok_segmented)
			output.append(" ".join(line_segmented))
		write_to_file(output, args.output)
	end = time.time()
	print("Execution time: {}".format(end - start))


def initialize():
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--trainset', default=None, help='Path to training data')
	parser.add_argument('-i', '--input', default=None, help='Path to input file')
	parser.add_argument('-o', '--output', default=None, help='Output path')
	parser.add_argument('-l', '--limit', default=math.inf, type=int, help='Limit of n-grams per input segment')
	args = parser.parse_args()
	return args


def read_lines(filename):
	with io.open(filename, 'r', encoding='utf-8') as file:
		return [line.strip() for line in file]

def write_to_file(output, filename):
	with io.open(filename, 'w', encoding='utf-8') as out_file:
		for line in output:
			out_file.write(line + "\n")


main()
