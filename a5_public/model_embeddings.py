#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab.char2id['<pad>']
        # Define the character embedding size
        self.e_char_size = 50
        self.embed_size = embed_size  #The attribute name has to match the line 465 in nmt_model

        self.embeddings = nn.Embedding(len(vocab.char2id), self.e_char_size,
                                       padding_idx=pad_token_idx)
        self.cnn = CNN(input_channels=self.e_char_size,
                       output_channels=self.embed_size)
        self.highway = Highway(input_size=self.embed_size)
        self.dropout = nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        # Get the shape of the input
        sen_len, b, max_word = input.shape
        # 1. Embed the input with the character embedding
        embedded = self.embeddings(input)
            # Shape : sen_len, b, max_word, e_char

        # 2. Reshape the embedded tensor to the corresponding CNN input shape
        #    sen_len, b, max_word, e_char --> reshape
        #    (b*sen_len, m_word, e_char) --> permute
        #    (b*sen_len, e_char, m_word)
        embedded_reshape = embedded.reshape(b*sen_len, max_word,
                                            self.e_char_size).\
                                            permute(0,2,1)
        # 3. CNN (b*sen_len, e_word)
        conv = self.cnn(embedded_reshape)

        # 4. Highway & Dropout --> output_embedding
        #     Shape: (b*sen_len, e_word)
        highway = self.highway(conv)

        dropout = self.dropout(highway) # (b*sen_len, e_word)

        # Output tensor size: (sentence_length, batch_size, embed_size)
        word_embedding = dropout.reshape(sen_len, b, self.embed_size)

        return word_embedding
        ### END YOUR CODE

