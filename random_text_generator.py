import random

apple_words = ['apple', 'orange','banana','pear','cherry','i','have','eat','like','some','today','school']

beef_words = ['fish', 'beef','lamb','chicken','duck','i','have','eat','like','some','today','school']


def generate(text):
    with open(text,'w') as f:
        for _ in range(10000):
            sentence = []
            for _ in range(random.randint(1,20)):
                if text == 'apple.txt':
                    sentence.append(apple_words[random.randint(0,11)])
                elif text == 'beef.txt':
                    sentence.append(beef_words[random.randint(0, 11)])
            sentence = ' '.join(sentence)
            print(sentence)
            f.write(sentence+'\n')


#generate('apple.txt')
#generate('beef.txt')