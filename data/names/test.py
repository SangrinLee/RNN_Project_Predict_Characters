from random import shuffle

data = []
with open("English.txt", "r") as read_file:
	for line in read_file:
		data.append(line.rstrip())

shuffle(data)

with open("training.txt", "w") as training_file:
	for line in data[:3300]:
		training_file.write(line + "\n")

with open("testing.txt", "w") as testing_file:
	for line in data[3300:]:
		testing_file.write(line + "\n")
