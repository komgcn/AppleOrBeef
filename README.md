# AppleOrBeef

**Introduction**<br />
This is a basic deep neural network python NLP project.<br />
The purpose is to learn python and tensorflow.<br />
The goal is to train the neural network to be able to distinguish a statement whether it belongs to 'APPLE' or 'BEEF'<br />
'APPLE' has lexicon full of fruits, eg: apple, orange, banana, etc<br />
'BEEF' has lexicon full of meat, eg: beef, lamb, chicken, etc<br />
<br />

**Files**<br />
<br />
random_text_generator.py - generate 'apple.txt' and 'beef.txt' file, each file has specified number of lines of 
randomly generated strings<br />
<br />
create_feature_sets.py - takes in two .txt file and vectorise the strings, also classified by [1,0] for 'APPLE' 
and [0,1] for 'BEEF', returns lists ready for training and testing<br />
<br />
neural_network.py - the actual deep neural network model with 2 hidden layers. For training, testing and using.
