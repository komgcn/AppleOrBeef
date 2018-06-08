import tensorflow as tf
import numpy as np
from create_feature_sets import create_sets
import nltk
from nltk import word_tokenize
from nltk import WordNetLemmatizer

train_x , train_y, test_x, test_y = create_sets('apple.txt','beef.txt')

lexicon = ['fish','apple', 'orange', 'beef','lamb','banana','chicken','pear','cherry','duck']
lemmatizer = WordNetLemmatizer()
node_hl1 = 500
node_hl2 = 500
classes = 2
epochs = 20

x = tf.placeholder('float',[None, len(lexicon)])
y = tf.placeholder('float')

#The neural network structure: 2 hidden layers with 500 nodes each
def neural_network_model(x):

    hidden_layer_1 = {'weight':tf.Variable(tf.random_normal([len(lexicon),node_hl1])),
                      'bias':tf.Variable(tf.random_normal([node_hl1]))}

    hidden_layer_2 = {'weight': tf.Variable(tf.random_normal([node_hl1, node_hl2])),
                      'bias': tf.Variable(tf.random_normal([node_hl2]))}

    output_layer = {'weight': tf.Variable(tf.random_normal([node_hl2, classes])),
                    'bias': tf.Variable(tf.random_normal([classes]))}

    l1 = tf.add(tf.matmul(x,hidden_layer_1['weight']),hidden_layer_1['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_layer_2['weight']),hidden_layer_2['bias'])
    l2 = tf.nn.relu(l2)

    output = tf.add(tf.matmul(l2,output_layer['weight']),output_layer['bias'])

    return output

#Train the network for 20 epochs, then run test set to see accuracy, then run user actual input
def train_neural_network(x, input):

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0
            _ ,c = sess.run([optimizer,cost],feed_dict={x:np.array(train_x),y:np.array(train_y)})
            epoch_loss += c
            print('Epoch ',epoch,' out of ',epochs,' Loss: ',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:test_x,y:test_y}))

        input_words = word_tokenize(input.lower())
        input_words = [lemmatizer.lemmatize(i) for i in input_words]
        features = np.zeros(len(lexicon))
        for word in input_words:
            if word.lower() in lexicon:
                index = lexicon.index(word.lower())
                features[index] += 1
        print(features)
        result = sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 0))
        print(result)
        if result[0] == 0:
            print('apple:', input)
        elif result[0] == 1:
            print('beef:', input)
        result = sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1))
        print(result)
        if result[0] == 0:
            print('apple:', input)
        elif result[0] == 1:
            print('beef:', input)


train_neural_network(x,'I have eaten beef, lamb and chicken yesterday, also an apple today')