# coding: utf-8

# # 3 Data preparation
#
# You can use the helper class (05 book-helper.py) to read-in the data. The script creates a tokenized version of the book once you create an instance of it. Since the tf.nn.embedding lookup function expects word ids, we need to map the words in the tokenized version of the book to ids. The create dictionaries method does exactly this. You need to pass the size of your vocabulary to the method. The method will then select the most common words and creates a unique id each for them, all other words are mapped to id 0, which is equivalent to ”unknown”.

# In[8]:


import numpy as np
import tensorflow as tf
from collections import Counter
from nltk.tokenize import RegexpTokenizer


class Book:
    def __init__(self, file):
        with open(file, encoding="utf-8") as fd:
            book = fd.read()
            book = book.lower()
            tokenizer = RegexpTokenizer(r'\w+')
            book = tokenizer.tokenize(book)

        print("Unique words: " + str(len(Counter(book))))
        self._book_text = book

    def create_dictionaries(self, vocabulary_size):
        words_and_count = Counter(self._book_text).most_common(vocabulary_size - 1)

        word2id = {word: word_id for word_id, (word, _) in enumerate(words_and_count, 1)}
        word2id["UNKNOWN"] = 0

        id2word = dict(zip(word2id.values(), word2id.keys()))

        # Map words to ids
        self._book = [word2id.get(word, 0) for word in self._book_text]

        self._word2id = word2id
        self._id2word = id2word

    def words2ids(self, words):
        if type(words) == list or type(words) == range or type(words) == np.ndarray:
            return [self._word2id.get(word, 0) for word in words]
        else:
            return self._word2id.get(words, 0)

    def ids2words(self, ids):
        if type(ids) == list or type(ids) == range or type(ids) == np.ndarray:
            return [self._id2word.get(wordid, "UNKNOWN") for wordid in ids]
        else:
            return self._id2word.get(ids, 0)

    def get_training_batch(self, batch_size, skip_window):
        valid_indices = range(skip_window, len(self._book) - (skip_window + 1))
        context_range = [x for x in range(-skip_window, skip_window + 1) if x != 0]
        wordid_contextid_pairs = [(word_id, word_id + shift) for word_id in valid_indices for shift in context_range]

        np.random.shuffle(wordid_contextid_pairs)

        counter = 0
        words = np.zeros((batch_size), dtype=np.int32)
        contexts = np.zeros((batch_size, 1), dtype=np.int32)

        for word_index, context_index in wordid_contextid_pairs:
            words[counter] = self._book[word_index]
            contexts[counter, 0] = self._book[context_index]
            counter += 1

            if counter == batch_size:
                yield words, contexts
                counter = 0


# # 4 Test the mapping
#
# Test the words2ids and ids2words methods by ﬁrst converting a list of words into ids and then back to words.

#### other stuff #####
bible = Book("pg10.txt")

bible.create_dictionaries(1000)

list1 = ["one", "two", "three", "four", "five", "god", "christ"]
ids = bible.words2ids(list1)
print(ids)
print(bible.ids2words(ids))

# # 5 Embedding
# 
# 
def layer(input, input_size, output_size, layerName):
    with tf.variable_scope(layerName):
        weights = tf.Variable(tf.truncated_normal(dtype = tf.float32, shape = [input_size,output_size], stddev = 0.1))
        
        activation = tf.matmul(input,weights)
        return activation


##### hyperparameters #####
number_words = 10000
embedding_size = 128

batchSize = 128
epochs = 5
stepSize = 1
skipWindows = 2
noiseSamples = 64

##### dataflowgraph #####
#inputs and expected outputs
x = tf.placeholder(dtype = tf.float32, shape = [None, number_words])
targets = tf.placeholder(dtype = tf.float32, shape = [None, number_words])

### compute first layer
first_layer = layer(x, number_words, embedding_size, 'l_1')

### compute the output layer
out_layer = layer(first_layer, embedding_size, number_words, 'out_layer')
prediction = tf.nn.softmax(out_layer)


# embedding
with tf.variable_scope("embedding"):
    # Create a word-embedding of size vocabulary size x embedding size
    initializer = tf.random_uniform_initializer(-1.0, 1.0)
    embeddings = tf.get_variable("embedding", [number_words, embedding_size], initializer = initializer)
    # Given a tensor of word ids, retrieve the respective embedding
    embed = tf.nn.embedding_lookup(embeddings, tf.cast(x, tf.int32))

with tf.variable_scope("nce-loss"):
    # Create weights and biases for NCE
    initializer = tf.truncated_normal_initializer(stddev = 1.0 / np.sqrt(embedding_size))
    nce_weights = tf.get_variable("weights", [number_words, embedding_size], tf.float32, initializer)
    nce_biases = tf.get_variable("biases", [number_words], initializer = tf.zeros_initializer())
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=targets, inputs=embed, num_sampled=3000, num_classes=number_words))

with tf.variable_scope("optimizer"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_step = optimizer.minimize(nce_loss)

#now run the network:
with tf.Session() as session:
    for inputs, labels in get_training_batch(self, batch_size, skip_window):
        feed_dict = {x: inputs, targets: labels}
        _, current_loss = session.run([optimizer, loss], feed_dict=feed_dict)

