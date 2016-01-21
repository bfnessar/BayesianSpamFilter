#!/usr/bin/env python

from __future__ import division
import string
import glob
import os
import ast

class BayesFilter(object):
	# Belol note: type 'chmod u+x bayesian_filter.py', then can run with just './bayesian_filter.py'

	# Consider using pairs of words instead of single words. Wikipedia says this might be a good idea to filter noise.

	# USER SETTINGS: =====================
	verbosity_settings = {} # dict[string] => boolean	# 	show_loading, show_probs_each_word, show_probs_each_email, show_tests_live, export
	training_data_settings = {}	# Keys: Full, Decap, Prepackaged, Priority

	# PARSING: ===========================
	american_dict = {}
	header_fields_dict = {}
	html_tags_dict = {}
	# Make a dict of ordinary words
	ham_dict = {"!": 0, "?": 0, "$": 0}
	spam_dict = {"!": 0, "?": 0, "$": 0}

	# CALIBRATION/ TRAINING: =============
	vocabulary = {}
	hams = 0 # Number of sample ham emails.
	spams = 0 # Number of sample spam emails.
	ham_probs = {}
	spam_probs = {}
	ham_text = []
	spam_text = []
	priority_dict = {} # Also is used to check for "priority" words
	# Set properties of mcw_ham/spam:
	#	union(spam, ham) is what I'm currently using.
	#	intersect(spam, ham) contains noise words, probably.
	#	set_diff(spam, ham) contains spammy words.
	#	set_diff(ham, spam) contains especially mundane words, I guess.
	# mcw_ham = set()
	# mcw_spam = set()
	trash_dict = {}

	# EVALUATING ACCURACY/ TESTING: ======
	testing_data_dict = {} # Stores just the testing results, no actual robot work.

	# Takes as argument a dict of settings.
	def __init__(self, settings_dict):

		# Determine decap vs. full, 
		self.apply_settings(settings_dict)
		
		# Build dictionaries for use later.
		self.learn_american()
		self.learn_headers()
		self.learn_html()
		self.learn_priority_words() # Currently reduces accuracy by like, 0.6%
		self.learn_trash_words()
		self.trash_dict.update(self.header_fields_dict)
		self.trash_dict.update(self.html_tags_dict)

		# Study (Read) Ham archives, Spam archives, build dictionaries accordingly.
		if self.training_data_settings["prepackaged"]:
			self.study_from_files()
		else:
			self.study_easy_ham()
			self.study_hard_ham()
			self.study_spam()

		# At this point, dictionaries have been built (all archives have been parsed). Begin "shaving" the dictionaries now.
		self.shave_dict(self.ham_dict)
		self.shave_dict(self.spam_dict)

		# Take the m most common words from both dictionaries. Do AFTER dictionaries have been "shaved".
		# If you have anything else to add to the priority list, add it here.
		if (self.training_data_settings.get("priority")):
			mcw_ham = set(self.get_most_common_words(100, self.ham_dict))
			mcw_spam = set(self.get_most_common_words(100, self.spam_dict))
			mundane_set = mcw_ham.intersection(mcw_spam)
			spammy_set = mcw_spam - mcw_ham
			for item in mcw_ham | mcw_spam:
				self.priority_dict[item] = 0

			outstring = "" + "{} mcw's, plus priority words:\n".format(len(self.priority_dict.keys()))
			i = 0
			for key in sorted(self.priority_dict.keys()):
				if len(key) >= 8:
					outstring += "{}\t".format(key)
				else:
					outstring += "{}\t\t".format(key)
				if (i % 4 == 0):
					outstring += "\n"
				i += 1
			# print outstring

			if self.verbosity_settings.get("export"):
				print "Exporting shaved dictionaries to exported/full/"
				self.export_text(self.ham_dict, "exported/full/shaved_ham_dict.out")
				self.export_text(self.spam_dict, "exported/full/shaved_spam_dict.out")

		# Begin mathing, training.
		self.build_vocab()
		self.train_ham()
		self.train_spam()

		# Begin testing.
		self.init_testing_dict()
		self.test_model_easyham()
		self.test_model_hardham()
		self.test_model_spam()
		# Results of testing:
		self.print_results()
		if (self.verbosity_settings["export"]):
			self.export_all()
		return

# USER SETTINGS ==================================================================

	def apply_settings(self, set_dict):
		# Apply quiet if requested.
		if (set_dict.get("quiet")):
			self.verbosity_settings["show_loading"] = False
			self.verbosity_settings["show_studying"] = False
			self.verbosity_settings["show_probs_each_word"] = False
			self.verbosity_settings["show_probs_each_email"] = False
			self.verbosity_settings["show_tests_live"] = False
			self.verbosity_settings["minimum_progress_reports"] = True
		else: # Use default settings
			self.verbosity_settings["show_loading"] = True
			self.verbosity_settings["show_studying"] = True
			self.verbosity_settings["show_probs_each_word"] = False
			self.verbosity_settings["show_probs_each_email"] = False
			self.verbosity_settings["show_tests_live"] = True
			self.verbosity_settings["minimum_progress_reports"] = True

		# Check if export results.
		if (set_dict.get("export")):
			self.verbosity_settings["export"] = True
		else:
			self.verbosity_settings["export"] = False
		if (set_dict.get("priority")):
			self.training_data_settings["priority"] = True
		else:
			self.training_data_settings["priority"] = False

		# Check data source settings (decap or full, or preloaded dict.)
		if (set_dict.get("full") == True): # Use full emails
			self.training_data_settings["full"] = True
			self.training_data_settings["decap"] = False
			self.training_data_settings["prepackaged"] = False			
		elif (set_dict.get("decap") == True): # Default to decap emails.
			self.training_data_settings["full"] = False
			self.training_data_settings["decap"] = True
			self.training_data_settings["prepackaged"] = False
		if (set_dict.get("prepackaged") == True):
			self.training_data_settings["prepackaged"] = True
		return

	def print_results(self):
		total_emails = self.testing_data_dict["test_hams_total"] + self.testing_data_dict["test_spams_total"]
		print "Final Results: "
		print "Whole numbers"
		print "\tTrue Positives: {}/{}".format(self.testing_data_dict["true_positives"], total_emails)
		print "\tTrue Negatives: {}/{}".format(self.testing_data_dict["true_negatives"], total_emails)
		print "\tFalse Positives: {}/{}".format(self.testing_data_dict["false_positives"], total_emails)
		print "\tFalse Negatives: {}/{}".format(self.testing_data_dict["false_negatives"], total_emails)

		print "Percents: "
		print "\tTrue Positives: {}".format(self.testing_data_dict["true_positives"]/total_emails)
		print "\tTrue Negatives: {}".format(self.testing_data_dict["true_negatives"]/total_emails)
		print "\tFalse Positives: {}".format(self.testing_data_dict["false_positives"]/total_emails)
		print "\tFalse Negatives: {}".format(self.testing_data_dict["false_negatives"]/total_emails)
		return

	def export_all(self):
		if self.verbosity_settings["export"]:
			if self.training_data_settings["decap"]:
				self.export_text(self.ham_dict, "exported/decap/ham_dict_decap.out")
				self.export_text(self.spam_dict, "exported/decap/spam_dict_decap.out")
				self.export_text(self.ham_text, "exported/decap/ham_text_decap.out")
				self.export_text(self.spam_text, "exported/decap/spam_text_decap.out")
				self.export_text(self.ham_probs, "exported/decap/ham_probs_decap.out")
				self.export_text(self.spam_probs, "exported/decap/spam_probs_decap.out")
				self.export_text(self.vocabulary, "exported/decap/vocabulary_decap.out")
			elif self.training_data_settings["full"]:
				self.export_text(self.ham_dict, "exported/full/ham_dict_full.out")
				self.export_text(self.spam_dict, "exported/full/spam_dict_full.out")
				self.export_text(self.ham_text, "exported/full/ham_text_full.out")
				self.export_text(self.spam_text, "exported/full/spam_text_full.out")
				self.export_text(self.ham_probs, "exported/full/ham_probs_full.out")
				self.export_text(self.spam_probs, "exported/full/spam_probs_full.out")
				self.export_text(self.vocabulary, "exported/full/vocabulary_full.out")
		return

	def export_text(self, obj, to_file):
		# Write data to a file.
		fp = open(to_file, "w+")
		if (isinstance(obj, dict)):
			for key in obj.keys():
				fp.write("{}: {}\n".format(key, obj[key]))
		else:
			fp.write(str(obj))
		fp.close()


# PARSING PHASE =====================================================================
# Builds a base of knowledge for later heuristic use.
# Parses ham/spam archives and builds ham_dict and spam_dict, which contain word counts for all words.
# MAYBE shaves the dictionaries so they don't have a count for irrelevant words.

	def learn_american(self):
		path_american = os.getcwd() + "/assets/american.txt"
		with open(path_american, "r") as fp:
			american_words = fp.read().split()
		for word in american_words:
			self.american_dict[word] = 0
	def learn_headers(self):
		path_headers = os.getcwd() + "/assets/header_field_names.txt"
		with open(path_headers, "r") as fp:
			header_tokens = fp.read().split()
		for token in header_tokens:
			self.header_fields_dict[token] = 0
	def learn_html(self):
		path_html = os.getcwd() + "/assets/html_words.txt"
		with open(path_html, "r") as fp:
			html_tokens = fp.read().split()
		for token in html_tokens:
			self.html_tags_dict[token] = 0
	def learn_trash_words(self):
		path_trash = os.getcwd() + "/assets/trash_words.txt"
		with open(path_trash, "r") as fp:
			trash_tokens = fp.read().split()
		for token in trash_tokens:
			self.trash_dict[token] = 0
	def learn_priority_words(self):
		path_priority = os.getcwd() + "/assets/priority_words.txt"
		with open(path_priority, "r") as fp:
			priority_tokens = fp.read().split()
		for token in priority_tokens:
			self.priority_dict[token] = 0

	# Tokenization happens here. Also happens in the Testing phase, but without this specific function.
	def process_line(self, line, into_dict, into_text):
		words = line.split()
		# Visit each word.
		for word in words:
			# Count number of special symbols, update into_dict accordingly.
			into_dict["!"] += word.count("!")
			into_dict["?"] += word.count("?")
			into_dict["$"] += word.count("$")
			# Strip the word of uninteresting punctuation.
			for letter in word:
				if letter not in (string.letters + "!?$-'"):
					word = word.replace(letter,"")
			# Eliminate, shorten, or standardize words.
			# If word is excessively long or short, skip it.
			if len(word) > 10:	
				continue
			if len(word) < 3:
				continue
			if word.isdigit():
				continue
			if word.isupper() and word.lower() in self.american_dict: #Ping! Caps-locked word.
				if word in into_dict:
					into_dict[word] += 1
					into_text.append(word)
					continue
				else:
					into_dict[word] = 1
					self.vocabulary[word] = 1
					into_text.append(word)
					continue
			if word.lower() in self.american_dict: # Ping! Ordinary word.
				if word in into_dict:
					into_dict[word] += 1
					into_text.append(word)
					continue
				else:
					into_dict[word] = 1
					self.vocabulary[word] = 1
					into_text.append(word)
					continue
			if word.istitle(): # Ping! Proper Noun. Is implicitly not in american_dict.
				if word in into_dict:
					into_dict[word] += 1
					into_text.append(word)
					continue
				else:
					into_dict[word] = 1
					self.vocabulary[word] = 1
					into_text.append(word)
					continue

		return

	def study_easy_ham(self):
		print("Studying easy ham")
		# Iterate through easy_ham.dir. For each word, either add it to the dictionary or increment its entry.
		if (self.training_data_settings.get("full") == True):
			path_easy_ham = os.getcwd() + "/training/full/easy_ham"
		else: # Else is decap
			path_easy_ham = os.getcwd() + "/training/decap/easy_ham"
		files_visited = 0
		# Check each file in ./training/*/easy_ham
		for filename in glob.glob(os.path.join(path_easy_ham, '*')):
			files_visited += 1
			self.hams += 1
			out_of = len(glob.glob(os.path.join(path_easy_ham, '*')))
			with open(filename, "r") as fp:
				if (self.verbosity_settings["show_studying"]):
					if (files_visited % 100 == 0):
						print("studying easy_ham: File {}/{}".format(files_visited, out_of))
				# Visit each line in the file, split line into words.
				for line in fp:
					self.process_line(line, self.ham_dict, self.ham_text) # Add words in the line to dict and text.
		return

	def study_hard_ham(self):
		print("Studying hard ham")
		# Iterate through hard_ham.dir. For each word, either add it to the dictionary or increment its entry.
		if (self.training_data_settings.get("full") == True):
			path_hard_ham = os.getcwd() + "/training/full/hard_ham"
		else: # Else is decap
			path_hard_ham = os.getcwd() + "/training/decap/hard_ham"
		files_visited = 0
		# Check each file in ./training/*/hard_ham
		for filename in glob.glob(os.path.join(path_hard_ham, '*')):
			self.hams += 1
			files_visited += 1
			out_of = len(glob.glob(os.path.join(path_hard_ham, '*')))
			with open(filename, "r") as fp:
				if (self.verbosity_settings["show_studying"]):
					if (files_visited % 100 == 0):
						print("studying hard_ham: File {}/{}".format(files_visited, out_of))
				# Visit each line in the file, split line into words.
				for line in fp:
					self.process_line(line, self.ham_dict, self.ham_text)			
		return

	def study_spam(self):
		print("Studying spam")
		if (self.training_data_settings.get("full") == True):
			path_spam = os.getcwd() + "/training/full/spam"
		else: # Else is decap
			path_spam = os.getcwd() + "/training/decap/spam"
		files_visited = 0
		# Check each file in ./training/decap/spam
		for filename in glob.glob(os.path.join(path_spam, '*')):
			self.spams += 1
			files_visited += 1
			out_of = len(glob.glob(os.path.join(path_spam, '*')))
			with open(filename, "r") as fp:
				if (self.verbosity_settings["show_studying"]):
					if (files_visited % 100 == 0):
						print("studying spam: File {}/{}".format(files_visited, out_of))
				# Visit each line in the file, split line into words.
				for line in fp:
					self.process_line(line, self.spam_dict, self.spam_text)
		return

	def study_from_files(self):
		if self.training_data_settings.get("full"):
			ham_path = os.getcwd() + "/exported/ham_dict_full.out"
			spam_path = os.getcwd() + "/exported/spam_dict_full.out"
			text_path_ham = os.getcwd() + "/exported/ham_text_full.out"
			text_path_spam = os.getcwd() + "/exported/spam_text_full.out"
			vocabulary_path = os.getcwd() + "/exported/vocabulary_full.out"
		else:
			ham_path = os.getcwd() + "/exported/ham_dict_decap.out"
			spam_path = os.getcwd() + "/exported/spam_dict_decap.out"
			text_path_ham = os.getcwd() + "/exported/ham_text_decap.out"
			text_path_spam = os.getcwd() + "/exported/spam_text_decap.out"
			vocabulary_path = os.getcwd() + "/exported/vocabulary_decap.out"

		with open(ham_path, "r") as fp:
			self.ham_dict.update(ast.literal_eval(fp.read()))
		with open(spam_path, "r") as fp:
			self.spam_dict.update(ast.literal_eval(fp.read()))
		with open(vocabulary_path, "r") as fp:
			self.vocabulary.update(ast.literal_eval(fp.read()))
		with open(text_path_ham, "r") as fp:
			self.ham_text = fp.read()
		with open(text_path_spam, "r") as fp:
			self.spam_text = fp.read()

		self.hams = 4400
		self.spams = 1600
		return

	# Returns the m most common words.
	def get_most_common_words(self, m, the_dict):
		tmp_dict = the_dict.copy()
		if (self.verbosity_settings["show_loading"]):
			print "Assessing most common words"
		maximum_words = []
		occurrences = max(tmp_dict.values())
		for i in range(0,m):
			for key in tmp_dict.keys():
				if (tmp_dict.get(key) == occurrences):
					maximum_words.append(key)
					del tmp_dict[key]
			occurrences = max(tmp_dict.values())

		# If "export", then export max words list.
		if self.verbosity_settings.get("export"):
			if (cmp(the_dict, self.ham_dict) == 0):
				if (self.training_data_settings.get("full")):
					self.export_text(maximum_words, "exported/full/mcw_ham.out")
				else:
					self.export_text(maximum_words, "exported/decap/mcw_ham.out")
			elif (cmp(the_dict, self.spam_dict) == 0):
				if (self.training_data_settings.get("full")):
					self.export_text(maximum_words, "exported/full/mcw_spam.out")
				else:
					self.export_text(maximum_words, "exported/decap/mcw_spam.out")
		return maximum_words

	# Removes all entries that are too-common words, header info, or stray html tags (ie, "noise").
	def shave_dict(self, the_dict):
		for key in self.trash_dict.keys():
			if key in the_dict.keys():
				del the_dict[key]
		return

	def refine_prio_dict(self):

		return

# TRAINING PHASE =====================================================================

	def build_vocab(self):
		for word in self.ham_dict.keys():
			self.vocabulary[word] = 1
		for word in self.spam_dict.keys():
			self.vocabulary[word] = 1
		return

	def train_ham(self):
		prob_ham = self.hams / (self.hams + self.spams)
		n = 0 # The number of words from Vocab in ham_text
		m = len(self.vocabulary.keys()) # The total number of words in Vocab.
		for word in self.ham_text:
			if word in self.ham_dict:
				n += 1
		for word_k in self.vocabulary.keys():
			if word_k in self.ham_dict:
				n_k = self.ham_dict[word_k]
			else:
				n_k = 0
			prob_word_k_given_ham = (n_k + 1) / (n + m)
			# print "P('{}' given ham): {}".format(word_k, prob_word_k_given_ham)
			self.ham_probs[word_k] = prob_word_k_given_ham
		return

	def train_spam(self):
		prob_spam = self.spams / (self.hams + self.spams)
		n = 0 # = The number of words from Vocab in ham_text
		m = len(self.vocabulary.keys()) # Total number of words in vocab.
		for word in self.spam_text:
			if word in self.spam_dict:
				n += 1
		for word_k in self.vocabulary.keys():
			if word_k in self.spam_dict:
				n_k = self.spam_dict[word_k]
			else:
				n_k = 0
			prob_word_k_given_spam = (n_k + 1) / (n + m)
			# print "P('{}' given spam): {}".format(word_k, prob_word_k_given_spam)
			self.spam_probs[word_k] = prob_word_k_given_spam
		return

# TESTING PHASE =====================================================================

	def init_testing_dict(self):
		self.testing_data_dict["test_hams_total"] = 0; self.testing_data_dict["test_spams_total"] = 0;
		self.testing_data_dict["easy_hams_total"] = 0; self.testing_data_dict["hard_hams_total"] = 0; self.testing_data_dict["spams_total"] = 0;
		self.testing_data_dict["true_positives"] = 0; self.testing_data_dict["true_negatives"] = 0;
		self.testing_data_dict["false_positives"] = 0; self.testing_data_dict["false_negatives"] = 0;

	def test_model_easyham(self):
		print("Testing: easy_ham ")
		if (self.training_data_settings.get("full") == True):
			path_easy_ham = os.getcwd() + "/test/full/easy_ham"
		else: # Else is decap
			path_easy_ham = os.getcwd() + "/test/decap/easy_ham"
		tests_run = 0
		for filename in glob.glob(os.path.join(path_easy_ham, '*')):
			prob_this_ham = 1
			prob_this_spam = 1
			with open(filename, "r") as fp:
				for line in fp:
					words = line.split()
					for word in words:

						for letter in word:
							if letter not in (string.letters + "!?$-'"):
								word = word.replace(letter,"")

						# If we're using a subset of important words.
						if self.training_data_settings.get("priority"):
							if (word not in self.priority_dict):
								continue

						if len(word) > 10:
							continue
						if len(word) < 3:
							continue
						if word in self.ham_probs:
							prob_this_ham *= self.ham_probs[word]
						if word in self.spam_probs:
							prob_this_spam *= self.spam_probs[word]
						if self.verbosity_settings["show_probs_each_word"]:
							print "{}: p_h={} vs p_s{}".format(word, prob_this_ham, prob_this_spam)
			if (prob_this_ham >= prob_this_spam):
				result = "Ham"
				self.testing_data_dict["true_negatives"] += 1
			else:
				result = "Spam"
				self.testing_data_dict["false_positives"] += 1
			if self.verbosity_settings["show_probs_each_email"]:
				print "Easy_Ham_Mail {}: ".format(tests_run), result
			tests_run += 1
			self.testing_data_dict["easy_hams_total"] += 1
			self.testing_data_dict["test_hams_total"] += 1
		return

	def test_model_hardham(self):
		print("Testing: hard_ham ")
		if (self.training_data_settings.get("full") == True):
			path_hard_ham = os.getcwd() + "/test/full/hard_ham"
		elif (self.training_data_settings.get("decap")): # Else is decap
			path_hard_ham = os.getcwd() + "/test/decap/hard_ham"
		else:
			print("whoa wth, settings are wrong.")
			exit()
		tests_run = 0
		for filename in glob.glob(os.path.join(path_hard_ham, '*')):
			prob_this_ham = 1
			prob_this_spam = 1
			with open(filename, "r") as fp:
				for line in fp:
					words = line.split()
					for word in words:
						for letter in word:
							if letter not in (string.letters + "!?$-'"):
								word = word.replace(letter,"")

						if self.training_data_settings.get("priority"):
							if (word not in self.priority_dict):
								continue

						if len(word) > 10:
							continue
						if len(word) < 3:
							continue
						if word in self.ham_probs:
							prob_this_ham *= self.ham_probs[word]
						if word in self.spam_probs:
							prob_this_spam *= self.spam_probs[word]
						if self.verbosity_settings["show_probs_each_word"]:
							print "{}: p_h={} vs p_s{}".format(word, prob_this_ham, prob_this_spam)
			if (prob_this_ham >= prob_this_spam):
				result = "Ham"
				self.testing_data_dict["true_negatives"] += 1
			else:
				result = "Spam"
				self.testing_data_dict["false_positives"] += 1
			if self.verbosity_settings["show_probs_each_email"]:
				print "Hard_Ham_Mail {}: ".format(tests_run), result
			tests_run += 1
			self.testing_data_dict["hard_hams_total"] += 1
			self.testing_data_dict["test_hams_total"] += 1
		return

	def test_model_spam(self):
		print("Testing: spam ")
		if (self.training_data_settings.get("full") == True):
			path_spam = os.getcwd() + "/test/full/spam"
		else: # Else is decap
			path_spam = os.getcwd() + "/test/decap/spam"
		tests_run = 0
		for filename in glob.glob(os.path.join(path_spam, '*')):
			prob_this_ham = 1
			prob_this_spam = 1
			with open(filename, "r") as fp:
				for line in fp:
					words = line.split()
					for word in words:
						for letter in word:
							if letter not in (string.letters + "!?$'-"):
								word = word.replace(letter,"")
						if self.training_data_settings.get("priority"):
							if (word not in self.priority_dict):
								continue
						if len(word) > 10:
							continue
						if len(word) < 3:
							continue
						if word in self.ham_probs:
							prob_this_ham *= self.ham_probs[word]
						if word in self.spam_probs:
							prob_this_spam *= self.spam_probs[word]
						if self.verbosity_settings["show_probs_each_word"]:
							print "{}: p_h={} vs p_s{}".format(word, prob_this_ham, prob_this_spam)
			if (prob_this_ham >= prob_this_spam):
				result = "Ham"
				self.testing_data_dict["false_negatives"] += 1
			else:
				result = "Spam"
				self.testing_data_dict["true_positives"] += 1
			if self.verbosity_settings["show_probs_each_email"]:
				print "Spam_Mail {}: ".format(tests_run), result
			tests_run += 1
			self.testing_data_dict["test_spams_total"] += 1
		return

if __name__ == "__main__":
	bf = BayesFilter({'full': True, 'export': True, 'priority': True, 'quiet': True})

