#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        Dimension notation:
        b: batch
        l: Length
        H: Hidden size
        v: vocab size

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super().__init__()

        self.decoderCharEmb = nn.Embedding(num_embeddings=len(target_vocab.char2id),
                                           embedding_dim=char_embedding_size,
                                           padding_idx=target_vocab.char2id['<pad>']
                                           )
        self.charDecoder = nn.LSTM(input_size=char_embedding_size,
                                   hidden_size=hidden_size) # Default layer = 1
        self.char_output_projection = nn.Linear(
                                        in_features=hidden_size,
                                        out_features=len(target_vocab.char2id))
        self.target_vocab = target_vocab
        ### END YOUR CODE



    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        #1. Embed the char index to char_embedding: (l,b) --> (l,b, e_char)
        X = self.decoderCharEmb(input)
        #2. LSTM: (l,b, e_char) --> (l,b,h):(seq_len, batch, num_directions * hidden_size)
        output, dec_hidden = self.charDecoder(X,dec_hidden)
                #hidden: (num_layers * num_directions, batch, hidden_size) = (1,b,h)
        #3. Linear projection to the Scores: (l,b,v)
        scores = self.char_output_projection(output)

        return scores, dec_hidden
        ### END YOUR CODE


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        #1. Input and make prediction score using LSTM & Projection
        input_seq = char_sequence[:-1]  #size (l,b)
        scores, _ = self.forward(input_seq, dec_hidden) #size (l,b,h)
        scores = scores.permute(0,2,1) #(l,h,b) since h: number of classed should be in the middle

        #2. Remove the <START> token in the target sequence
        tar_seq = char_sequence[1:] #size (l,b)

        #3. Calculation the cross-entropy
        loss_ce = nn.CrossEntropyLoss(reduction='sum')
        loss = loss_ce(scores, tar_seq)

        return loss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        _, batch, h_size = initialStates[0].size()
        ## Extract the start and end token from the target vcab
        START, END = self.target_vocab.start_of_word, self.target_vocab.end_of_word
        ## Prepare empty target vovab with startiong token
        start_char_ids = [[START] * batch]
        ## List of chars to Tensor
        current_char_ids = torch.tensor(start_char_ids, device=device)

        current_states = initialStates

        output_word = [''] * batch


        for t in range(max_length):
            score, dec_hidden = self.forward(current_char_ids, current_states)  # Input needs to be tensor integer (l,b)

            # what if not sequeezing
            probability = torch.softmax(score.squeeze(0), dim=1) # Score(1,b,v) -> (b,v)
            current_char_ids = torch.argmax(probability, dim=1).unsqueeze(0)    #(1,b)
            for i, c in enumerate(current_char_ids.squeeze(0)):
                output_word[i] += self.target_vocab.id2char[int(c)]
        decodedWords = []
        for word in output_word:
            end_pos = word.find(self.target_vocab.id2char[END])
            decodedWords.append(word if end_pos == -1 else word[:end_pos])
        return decodedWords


    ### END YOUR CODE

