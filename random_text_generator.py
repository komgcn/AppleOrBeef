import random

# Generate a .txt file that contains lines of random number of words
# Words are chosen "randomly" from the list of words(80% as keywords, 20% as other words)

# Number of lines
num_lines = 10000;

# Maximum number of words in a line
num_words = 20;

# Proportion of keywords and others
key = 7

apple_words = ['apple', 'orange','banana','pear','cherry']

beef_words = ['fish', 'beef','lamb','chicken','duck']

other_words = ['i','have','eat','like','some','today','school']

def generate(text):
    with open(text,'w') as f:
        for _ in range(num_lines):
            sentence = []
            for _ in range(random.randint(1,num_words)):
                if random.randint(0,10) < key :
                    if text == 'apple.txt':
                        sentence.append(apple_words[random.randint(0,len(apple_words)-1)])
                    elif text == 'beef.txt':
                        sentence.append(beef_words[random.randint(0, len(beef_words)-1)])
                else:
                    sentence.append(other_words[random.randint(0,len(other_words)-1)])
            sentence = ' '.join(sentence)
            print(sentence)
            f.write(sentence+'\n')

#generate('apple.txt')
#generate('beef.txt')