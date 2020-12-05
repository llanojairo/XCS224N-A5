#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

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

        ### YOUR CODE HERE for part 1f
        pad_token_idx = vocab.char2id['<pad>']
        self.embed_size = embed_size
        char_embed_size = 50

        self.char_embedding = nn.Embedding(len(vocab.char2id), 
            char_embed_size, 
            pad_token_idx)

        self.convNN = CNN(filters_num=self.embed_size)
        self.highway = Highway(size_embedding=self.embed_size)
        self.dropout = nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1f
        emb_word_list = []

        for padded_idx in input_tensor:
            embedding = self.char_embedding(padded_idx)
            reshaped_emb = torch.transpose(embedding, dim0=-1, dim1=-2)
            convl_out = self.convNN(reshaped_emb)
            highway = self.highway(convl_out)
            emb_word = self.dropout(highway)
            emb_word_list.append(emb_word)

        word_emb_f = torch.stack(emb_word_list)

        return word_emb_f
        ### END YOUR CODE
