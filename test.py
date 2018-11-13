# -*- coding: utf-8 -*-

#from __future__ import unicode_literals, print_function, division
from __future__ import unicode_literals, division
from io import open
import glob
import unicodedata
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import random
from collections import Counter

all_letters = string.ascii_letters + " .,;'-" # Plus * for missing character
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): return glob.glob(path)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [line for line in lines]
#    return [unicodeToAscii(line) for line in lines]

category_lines = {}
all_categories = []
#for filename in findFiles('data/names/*.txt'):
for filename in findFiles('data/names/training.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
#        self.dropout = nn.Dropout(0.1)
	self.dropout = nn.Dropout(0)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)



criterion = nn.NLLLoss()

#learning_rate = 0.0005
learning_rate = 0.0010

def train(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()
    
    loss = 0
    
    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)

        loss += criterion(output, target_line_tensor[i])

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output

# All trained
#rnn = RNN(n_letters, 128, n_letters)
rnn = RNN(n_letters, 1400, n_letters)


for category in all_categories:
	for line in category_lines[category]:
		category_tensor = Variable(categoryTensor(category))
		input_line_tensor = Variable(inputTensor(line))
		target_line_tensor = Variable(targetTensor(line))
		train(category_tensor, input_line_tensor, target_line_tensor)
#	print (category)

#rnn = torch.load('rnn.pt')
#torch.save(rnn, 'rnn.pt')

def sample(category, plain_word):
	category_tensor = Variable(categoryTensor(category))
	rand = random.randrange(1, len(plain_word)-1)
	word = plain_word[:rand] + "*" + plain_word[rand+1:]

#	print "--------------------"
#	print plain_word
#	print word   
	word_len = len(word)
	index = word.find('*')
#	print "length of word = " + (str(word_len)) 
#	print "index for iteration = " +  (str(index))
#	print "--------------------"
	
	alphabet = "abcdefghijklmnopqrstuvwxyz"
	result_dict = {}
	final_prob = 1
	final_word = ""
	sum = 0
	dict = {}
	for alpha in alphabet:
		hidden = rnn.initHidden()
		prob = 0
		result_word = ""
		for i in range(word_len-1):
			if i == index:
				result_word += alpha
				input = Variable(inputTensor(alpha))
			else:
				result_word += word[i]
				input = Variable(inputTensor(word[i]))
			output, hidden = rnn(category_tensor, input[0], hidden)
			
			if i + 1 == index:
				next_index = all_letters.find(alpha)
			else:
				next_index = all_letters.find(word[i+1])
			prob += output.data[0][next_index]

		result_word += word[i+1]
		sum += math.exp(prob)
		dict[result_word] = prob

#	print "##### RESULT #####"
#	print plain_word
	max_word = max(dict, key = dict.get)
#	print max_word
	if plain_word == max_word:
		return 1
	else:
		return 0

#	for k, v in dict.iteritems():
#		print k + ", log prob = " + str(v) + ", normalized prob after exp = ", str(math.exp(v)/sum)

#	print str(sum)
#	for k, v in result_dict.iteritems():
#		print k
#		print v


#print(sample('English', 'Benjam*n'))
#print(sample('English', 'Benj*min'))
#print(sample('English', 'Abr*ham'))
#print(sample('English', 'Abrah*m'))
#print(sample('training', 'Adam'))

total = 0
correct = 0
with open("data/names/testing.txt", "r") as read_file:
	for i in read_file:
		correct += sample('training', i)
		total += 1

print str(correct)
print str(total)
print str(correct/total)
