# Dependency-Parser-With-Neural-Networks

## Abstract
This work proposes a way of learning a neural network classifier for use in a greedy, transition-based dependency parser. This classifier learns and uses a relatively small number of dense features and achieves an about 2% improvement in unlabeled and labeled attachment scores on both English and Chinese datasets compared to dependency parsers that classify based on millions of sparse indicator features.

## Introduction
As the basis of this parser, we employ the arc-standard system. We will train a feed-forward neural network to predict the transitions of an arc-standard dependency parser. The input to this network will be a representation of the current state (including words on the stack and buffer). The output will be a transition (shift, left_arc, right_arc), together with a dependency relation label. 


## Prequisites: Installing necessary packages
You will use the Keras package to construct the neural net. Keras is a high-level Python API that allows you to easily construct, train, and apply neural networks. However, Keras is not a neural network library itself and depends on one of several neural network backends. We will use the Tensorflow backend. TensorFlow is an open-source library for neural networks (and other mathematical models based on sequences of matrix and tensor computations), originally developed by Google.

Run the following command to install all necessary packages. 
```
$pip3 install -r requirements.txt
```

Note that this will install the CPU version of TensorFlow that does not use the GPU to speed up neural network training. For this assignment, training on the CPU will be sufficient, but if your computer has a GPU (or you want to try running the assignment in the cloud), follow the installation instructions on the tensorflow page.

You could also try
```
$ pip install tensorflow
$ pip install keras
```

but only if your system has an nvidia GPU.
If you get stuck during the installation, you can find installation instructions for each package here:

Tensorflow: https://www.tensorflow.org/install/

Keras: https://keras.io/#installation


## Testing your Setup:
To test your setup, run a Python interpreter and type the following:
```
$ python
Python 3.6.1 |Anaconda 4.4.0 (x86_64)| (default, May 11 2017, 13:04:09) [GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_t ensor_util' does not match runtime version 3.6
 return f(*args, **kwds)
>>> import keras
Using TensorFlow backend. >>>
```


This repo contains the following python files:

- **conll_reader

** - This file defines data structures that represent a dependency tree. It also includes methods that read and write trees in the CoNLL-X format (explained below).
- **get_vocab.py** - This file contains methods that extract a set of words and POS tags that appear in the training data. This is necessary to format the input to the neural net (the dimensionality of the input vectors depends on the number of words).
- **extract_training_data.py** - This file extracts two numpy matrices representing input output pairs (as described below). You will have to modify this file to change the input representation to the neural network.
- **train_model.py** - This file pecifies and trains the neural network model. This script writes a file containing the model architecture and trained weights.
- **decoder.py** - This file uses the trained model file to parse some input. For simplicity, the input is a CoNLL-X formatted file, but the dependency structure in the file is ignored. Prints the parser output for each sentence in CoNLL-X format.
- **evaluate.py** - This works like **decoder.py**, but instead of neglecting the input dependencies, it uses them to compare the parser output. It also prints evaluation results.

There are also some data files, corresponding to a standard split of the WSJ part of the Penn Treebank. The original Penn Treebank contains constituency parses, but these were converted automatically to dependencies.
- **data/train.conll** - Training data that contains around 40k sentences
- **data/dev.conll** - Development data.  Around 5k sentences. Can be used to experiment and tune the model.
- **data/sec0.conll** - section 0 of the Penn Treebank. Around 2k sentence. Can be good for quick initial testing.
- **data/test.conll** - Test data. Around 2.5k sentences. 



## Dependency Format
The files are annotated using a modified CoNLL-X format (CoNLL is the conference on Computational Natural Language learning -- this format was first used for shared tasks at this conference). Each sentences corresponds to a number of lines, one per word. Sentences are separated with a blank line. You wil need to be able to read these annotations and draw dependency trees (by hand) in order to debug your parser.
Each line contains fields, seperated by a single tab symbol. The fields are, in order, as follows:
- word ID (starting at 1)
- word form
- lemma
- universal POS tag
- corpus-specific POS tag (for our purposes the two POS annotations are always the same)
- features (unused)
- word ID of the parent word ("head"). 0 if the word is the root of the dependency tree.
- dependency relation between the parent word and this word.
- deps (unused)
- misc annotations (unused)
Any field that contains no entry is replaced with a _.
For example, consider the following sentence annotation:

```
1 The _ DT DT _ 2 dt _ _
2 cat _ NN NN _ 3 nsubj _ _ 
3 eats _ VB VB _ 0 root _ _ 
4 tasty _ JJ JJ _ 5 amod _ _ 
5 fish _ NN NN _ 3 dobj _ _ 
6 . _ . . _ 3 punct _ _
```
Take a look at **data/sec0.conll** for more examples.
The file **conll_reader.py** contains classes for representing dependency trees and reading in a CoNLL-X formatted data files.
The class DependencyEdge represents a singe word and its incoming dependency edge. It includes the attribute variables id, word, pos, head, deprel. Id is just the position of the word in the sentence. Word is the word form and pos is the part of speech. Head is the id of the parent word in the tree. Deprel is the dependency label on the edge pointing to this label. Note that the information in this class is a subset of what is represented in the CoNLL format.
The class DependencyStructure represents a complete dependency parse. The attribute deprels is a dictionary that maps integer word ids to DependencyEdge instances. The attribute root contains the integer id of the root note. The method print_conll returns a string representation for the dependency structure formatted in CoNLL format (including line breaks).



Because we will use one-hot representations for words and POS tags, we will need to know which words appear in the data, and we will need a mapping from words to indices.
Run the following to generate an index of words and POS indices.
```
$python get_vocab.py data/train.conll data/words.vocab data/pos.vocab
```

This contains all words that appear more than once in the training data. The words file will look like this:


```
<CD> 0
<NNP> 1
<UNK> 2
<ROOT> 3
<NULL> 4
blocking 5
crossing 6
hello 7
```

The first 5 entries are special symbols. <CD> stands for any number (anything tagged with the POS tag CD), <NNP> stands for any proper name (anything tagged with the POS tag NNP). <UNK> stands for unknown words (in the training data, any word that appears only once). <ROOT> is a special root symbol (the word associated with the word 0, which is initially placed on the stack of the dependency parser). <NULL> is used to pad context windows.
  
  
  
To train the neural network we first need to obtain a set of input/output training pairs. More specifically, each training example should be a pair (x,y), where x is a parser state and y is the transition the parser should make in that state.


Take a look at the file **extract_training_data.py**
- States: The input will be an instance of the class State, which represents a parser state. The attributes of this class consist of a stack, buffer, and partially built dependency structure deps. stack and buffer are lists of word ids (integers).
The top of the stack is the last word in the list stack[-1]. The next word on the buffer is also the last word in the list, buffer[-1].
Deps is a list of (parent, child, relation) triples, where parent and child are integer ids and relation is a string (the dependency label).
- Transitions: The output is a pair (transition, label), where the transition can be one of "shift", "left_arc", or "right_arc" and the label is a dependency label. If the transition is "shift", the dependency label is None. Since there are 45 dependency relations (see list deps_relations), there are 45*2+1 possible outputs.

**Obtaining oracle transitions and a sequence of input/output examples.**
We cannot observe the transitions directly from the treebank. We only see the resulting dependency structures. We therefore need to convert the trees into a sequence of (state, transition) pairs that we use for training. This part is implemented in the function get_training_instances(dep_structure). Given a DependencyStructure instance, this method returns a list of (State, Transition) pairs in the format described above.


**get_input_representation(self, words, pos, state)** takes as parameters a list of words in the input sentence, a list of POS tags in the input sentence and an instance of class State. It should return an encoding of the input to the neural network, i.e. a single vector.
To represent a state, we will use the top-three words on the buffer and the next-three word on the stack, i.e. stack[-1], stack[-2], stack[-3] and buffer[-1], buffer[-2], buffer[-3]. We could use embedded representations for each word, but we would like the network to learn these representations itself. Therefore, the neural network will contain an embedding layer and the words will be represented as a one-hot representation. The actual input will be the concatenation of the one-hot vectors for each word.

So for example, if the next words on the buffer is "dog eats a" and the top word on the stack is "the", the return value should be a numpy array numpy.array([4047, 4, 4, 8346, 8995, 14774]) where 4 is the index for the <NULL> symbol and 8346, 8995, 14774 are the indices for "dog", "eats" and "a".
  
This representation is a subset of the features in the Chen & Manning (2014) paper. Feel free to experiment with the complete feature set once you got the basic version running.

**get_output_representation(self, output_pair)** takes a (transition, label) pair as its parameter and return a one-hot representation of these actions. Because there are 45*2+1 = 91 possible outputs, the output should be represented as a one-hot vector of length 91.


## Saving training matrices
The neural network will take two matrices as its input, a matrix of training data (in the basic case a N x 6 matrix, where N is the number of training instances) and an output matrix (an Nx91 matrix).
The function get_training_matrices(extractor, in_file) will take a FeatureExtractor instance and a file object (a CoNLL formatted file) as its input. It will then extract state-transition sequences
 
and call the input and output representation methods on each to obtain input and output vectors. Finally it will assemble the matrices and return them.
The main program in **extract_training_data.py** calls get_training_matrices to obtain the matrices and then writes them to two binary files (encoded in the numpy array binary format). You can call it like this:

```
$python extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy
```

You can also obtain matrices for the development set, which is useful to tune network parameters.

```
$python extract_training_data.py data/dev.conll data/input_dev.npy data/target_dev.npy
```


## Designing and Training the network 
Now that we have training data, we can build the actual neural net. In the file **train_model.py**, the function *build_model(word_types, pos_types, outputs)* defines the network model that we are going to use. Here, we are simply using a shallow network and categorical crossentropy as the loss and the Adam optimizer with a learning rate of 0.01.


The main function of **train_model.py** will load in the input and output matrices and then train the network. We will train the network for 10 epochs with a batch_size of 100. You can also implement early stopping with the validation set and other addtional features provided by Keras. Training will take a while on a CPU-only setup.
```
python train_model.py data/input_train.npy data/target_train.npy data/model.h5
```

## Greedy Parsing Algorithm - Building and Evaluating the Parser
We will now use the trained model to construct a parser. In the file **decoder.py**, the class constructor takes the name of a keras model file, loads the model and stores it in the attribute model. It also uses the feature extractor defined previously.

The method **parse_sentence(self, words, pos)** takes as parameters a list of words and POS tags in the input sentence. The method will return an instance of DependencyStructure.

The function first creates a State instance in the initial state, i.e. only word 0 is on the stack, the buffer contains all input words (or rather, their indices) and the deps structure is empty.

The algorithm is the standard transition-based algorithm. As long as the buffer is not empty, we use the feature extractor to obtain a representation of the current state. We then predict and retrieve a softmax actived vector of possible actions. In principle, we would only have to select the highest scoring transition and update the state accordingly.

Unfortunately, it is possible that the highest scoring transition is not possible. arc-left or arc-right are not permitted when the stack is empty, or when the buffer is empty.  Shifting the only word out of the buffer is invalid, unless the stack is empty. Additionally, shifting from an empty buffer is also invalid. Furthermore, the root node must never be the target of a left-arc. Thus, instead of selecting the highest-scoring action, we select the highest scoring permitted transition.

Running the program like this should print CoNLL formatted parse trees for the sentences in the input.
```
python decoder.py data/model.h5 data/dev.conll
```

To evaluate the parser, run the program evaluate.py, which will compare your parser output to the target dependency structures and compute labeled and unlabeled attachment accuracy.

```
python decoder.py data/model.h5 data/test.conll
```

Labeled attachment score is the percentage of correct (parent, relation, child) predictions. Unlabeled attachment score is the percentage of correct (parent, child) predictions.
