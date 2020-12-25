from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras
from keras import backend

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words))) #create the buffer with all words
        state.stack.append(0)    
        while state.buffer:
            input = self.extractor.get_input_representation(words, pos, state)
            # For small amount of inputs that fit in one batch just call model(x)
            output_probs = self.model(input.reshape(1, input.shape[0]), training=False)
            probs_list = np.argsort(output_probs.numpy()[0])[::-1].tolist()
            for idx in probs_list:
                transition, label = self.output_labels[idx]
                if self.is_valid(transition, state.buffer, state.stack):
                    if transition == "shift":
                        state.shift()
                    elif transition == "left_arc":
                        state.left_arc(label)
                    else:
                        state.right_arc(label)
                    break
                    
        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        
    def is_valid(self, transition, buffer, stack):
        if transition == "shift":
            return len(buffer) != 0 if len(stack) == 0 else len(buffer) != 1 and len(buffer) != 0
        else:
            if transition == "left_arc":
                return len(stack) != 0 and stack[-1] != 0 and len(buffer) != 0
            return len(stack) != 0 and len(buffer) != 0


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])
    count = 0
    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
            count += 1
            if count == 3:
                break
        exit(0)
        
