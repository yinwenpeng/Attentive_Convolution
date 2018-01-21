import numpy
import theano
import random
import theano.tensor as T
#import theano.tensor.nlinalg
from theano.tensor.nnet import conv
from cis.deep.utils.theano import debug_print
from logistic_sgd import LogisticRegression
import numpy as np
from scipy.spatial.distance import cosine
from mlp import HiddenLayer
import cPickle

def create_AttentionMatrix_para(rng, n_in, n_out):

    W1_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)), dtype=theano.config.floatX)  # @UndefinedVariable
    W1 = theano.shared(value=W1_values, name='W1', borrow=True)
    W2_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)), dtype=theano.config.floatX)  # @UndefinedVariable
    W2 = theano.shared(value=W2_values, name='W2', borrow=True)

#     b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)  # @UndefinedVariable
    w_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (n_out+1)),
            high=numpy.sqrt(6. / (n_out+1)),
            size=(n_out,)), dtype=theano.config.floatX)  # @UndefinedVariable
    w = theano.shared(value=w_values, name='w', borrow=True)
    return W1,W2, w


def create_HiddenLayer_para(rng, n_in, n_out):

    W_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)), dtype=theano.config.floatX)  # @UndefinedVariable
    W = theano.shared(value=W_values, name='W', borrow=True)

    b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)  # @UndefinedVariable
    b = theano.shared(value=b_values, name='b', borrow=True)
    return W,b

def create_Bi_GRU_para(rng, word_dim, hidden_dim):
        # Initialize the network parameters
        U = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, word_dim))
        W = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        b = numpy.zeros((3, hidden_dim))
        # Theano: Created shared variables
        U = debug_print(theano.shared(name='U', value=U.astype(theano.config.floatX), borrow=True), 'U')
        W = debug_print(theano.shared(name='W', value=W.astype(theano.config.floatX), borrow=True), 'W')
        b = debug_print(theano.shared(name='b', value=b.astype(theano.config.floatX), borrow=True), 'b')

        Ub = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, word_dim))
        Wb = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        bb = numpy.zeros((3, hidden_dim))
        # Theano: Created shared variables
        Ub = debug_print(theano.shared(name='Ub', value=Ub.astype(theano.config.floatX), borrow=True), 'Ub')
        Wb = debug_print(theano.shared(name='Wb', value=Wb.astype(theano.config.floatX), borrow=True), 'Wb')
        bb = debug_print(theano.shared(name='bb', value=bb.astype(theano.config.floatX), borrow=True), 'bb')
        return U, W, b, Ub, Wb, bb

def create_GRU_para(rng, word_dim, hidden_dim):
        # Initialize the network parameters
#         U = numpy.random.uniform(-0.01, 0.01, (3, hidden_dim, word_dim))
        U=rng.normal(0.0, 0.01, (3, hidden_dim, word_dim))
#         W = numpy.random.uniform(-0.01, 0.01, (3, hidden_dim, hidden_dim))
        W=rng.normal(0.0, 0.01, (3, hidden_dim, hidden_dim))
        b = numpy.zeros((3, hidden_dim))
        # Theano: Created shared variables
        U = theano.shared(name='U', value=U.astype(theano.config.floatX), borrow=True)
        W =theano.shared(name='W', value=W.astype(theano.config.floatX), borrow=True)
        b = theano.shared(name='b', value=b.astype(theano.config.floatX), borrow=True)
        return U, W, b

def create_LSTM_para(rng, word_dim, hidden_dim):
    params={}
    #W play with input dimension
    W = rng.normal(0.0, 0.01, (word_dim, 4*hidden_dim))
    params['W'] = theano.shared(name='W', value=W.astype(theano.config.floatX), borrow=True)
    #U play with hidden states
    U = rng.normal(0.0, 0.01, (hidden_dim, 4*hidden_dim))
    params['U'] = theano.shared(name='U', value=U.astype(theano.config.floatX), borrow=True)
    b = numpy.zeros((4 * hidden_dim,))
    params['b'] = theano.shared(name='b', value=b.astype(theano.config.floatX), borrow=True)

    return params

def create_ensemble_para(rng, fan_in, fan_out):
#         W=rng.normal(0.0, 0.01, (fan_out,fan_in))
#
#         W =theano.shared(name='W', value=W.astype(theano.config.floatX), borrow=True)



        # initialize weights with random weights
        W_bound = numpy.sqrt(6. /(fan_in + fan_out))
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-0.01, high=0.01, size=(fan_out,fan_in)),
            dtype=theano.config.floatX),
                               borrow=True)

        return W
def create_ensemble_para_with_bounds(rng, fan_in, fan_out, lowerbound, upperbound):
        W = theano.shared(numpy.asarray(
            rng.uniform(low=lowerbound, high=upperbound, size=(fan_out,fan_in)),
            dtype=theano.config.floatX),
                               borrow=True)

        return W
def create_highw_para(rng, fan_in, fan_out):

        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(fan_out,fan_in)),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((fan_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return W, b

def create_conv_para(rng, filter_shape):
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-0.01, high=0.01, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return W, b
def create_conv_bias(rng, hidden_size):
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((hidden_size,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return b
def create_rnn_para(rng, dim):
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (2*dim + dim))
#         Whh = theano.shared(numpy.asarray(
#             rng.uniform(low=-W_bound, high=W_bound, size=(dim, dim)),
#             dtype=theano.config.floatX),
#                                borrow=True)
#         Wxh = theano.shared(numpy.asarray(
#             rng.uniform(low=-W_bound, high=W_bound, size=(dim, dim)),
#             dtype=theano.config.floatX),
#                                borrow=True)
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(2*dim, dim)),
            dtype=theano.config.floatX),
                               borrow=True)
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((dim,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return W, b

def ABCNN(left_T, right_T):
    dot_tensor3 = T.batched_dot(left_T.dimshuffle(0,2,1),right_T) #(batch, l_len, r_len)


    dot_matrix_for_right = T.nnet.softmax(T.max(dot_tensor3, axis=1)) #(batch, r_len)
    weighted_sum_r = T.batched_dot(dot_matrix_for_right.dimshuffle(0,'x',1), right_T.dimshuffle(0,2,1)).reshape((right_T.shape[0], right_T.shape[1])) #(batch,hidden, l_len)

    dot_matrix_for_left = T.nnet.softmax(T.max(dot_tensor3, axis=2))
    weighted_sum_l = T.batched_dot(dot_matrix_for_left.dimshuffle(0,'x',1), left_T.dimshuffle(0,2,1)).reshape((left_T.shape[0], left_T.shape[1])) #(batch,hidden, r_len)
    return weighted_sum_l,weighted_sum_r
def Conv_for_Self_Attend(input_tensor3, mask_matrix):
        #construct interaction matrix
        input_tensor3 = input_tensor3*mask_matrix.dimshuffle(0,'x',1)
        input_tensor3_r = input_tensor3
        mask_matrix_r = mask_matrix
        input_tensor3_r = input_tensor3_r*mask_matrix_r.dimshuffle(0,'x',1) #(batch, hidden, r_len)
        dot_tensor3 = T.batched_dot(input_tensor3.dimshuffle(0,2,1),input_tensor3_r) #(batch, l_len, r_len)

        dot_matrix_for_right = T.nnet.softmax(dot_tensor3.reshape((dot_tensor3.shape[0]*dot_tensor3.shape[1], dot_tensor3.shape[2])))
        dot_tensor3_for_right = dot_matrix_for_right.reshape((dot_tensor3.shape[0], dot_tensor3.shape[1], dot_tensor3.shape[2]))#(batch, l_len, r_len)
        weighted_sum_r = T.batched_dot(dot_tensor3_for_right, input_tensor3_r.dimshuffle(0,2,1)).dimshuffle(0,2,1)*mask_matrix.dimshuffle(0,'x',1) #(batch,hidden, l_len)

#         self.concat_output_tensor3 = T.concatenate([input_tensor3, weighted_sum_r], axis=1) #(batch, 2*hidden, len)
        sum_output_tensor3 = input_tensor3+weighted_sum_r
        return sum_output_tensor3

def tensor3_group_maxpool(tensor3, valid_left_vec, group_size):
    #tensor3 (batch, hidden, len)
    #valid_left_vec #batch
    #group_size 3
    def each_slice(matrix, left):
        #matrix (hidden , len)
        group_width = (matrix.shape[1]-left)/group_size
        if T.lt(group_width, 1):
            pool_vec_1 = (T.max(matrix[:,left:], axis=1)).dimshuffle(0,'x')  #(hidden, 1)
            pool_vec_2 = pool_vec_1
            pool_vec_3 = pool_vec_1
        else:
            pool_vec_1 = (T.max(matrix[:,left:left+group_width], axis=1)).dimshuffle(0,'x')  #(hidden, 1)
            pool_vec_2 = (T.max(matrix[:,left+group_width:left+2*group_width], axis=1)).dimshuffle(0,'x')
            pool_vec_3 = (T.max(matrix[:,left+2*group_width:], axis=1)).dimshuffle(0,'x')
        return T.concatenate([pool_vec_1,pool_vec_2,pool_vec_3], axis=1) #(hidden, 3)

    batch_return, _ = theano.scan(
        each_slice,
        sequences=[tensor3,valid_left_vec])  #(batch,hidden, 3)
    return batch_return

def fine_grained_softmax_tensor3(tensor3, left_vec):

    def process_matrix(matrix, left):
        submatrix = matrix[:,left:]
        sub_distr = T.nnet.softmax(submatrix)
        return T.concatenate([matrix[:,:left], sub_distr], axis=1)
    batch_return, _ = theano.scan(
        process_matrix,
        sequences=[tensor3,left_vec])  #(batch,hidden, len)

    return     batch_return

class Attentive_Conv_for_Pair(object):
    def __init__(self, rng, origin_input_tensor3,origin_input_tensor3_r,input_tensor3, input_tensor3_r, mask_matrix, mask_matrix_r,
                 filter_shape, filter_shape_context,image_shape, image_shape_r,W, b, W_context, b_context):
        '''
        Input:

        origin_input_tensor3: (batch_size, hidden_size, sen_length). tensor3 representation for sentence 1
        origin_input_tensor3_r: (batch_size, hidden_size, sen_length). tensor3 representation for sentence 2
        input_tensor3: (batch_size, hidden_size, sen_length). tensor3 representation for sentence 1. It can be the same with 'origin_input_tensor3' or be the output of gated convolution layer
        input_tensor3_r: (batch_size, hidden_size, sen_length). tensor3 representation for sentence 2. It can be the same with 'origin_input_tensor3_r' or be the output of gated convolution layer
        mask_matrix, mask_matrix_r: mask for the two sentences respectively. Each with (batch_size, sent_length), each row corresponding to one sentence
        filter_shape: standard filter shape for theano convolution function, in shape (hidden_size, 1, emb_size, filter_width)
        filter_shape_context: the filter shape to deal with the 'fake sentence' which contains attentive context. In shape (hidden_size, 1, emb_size, 1)
        image_shape, image_shape_r: standard tensor4 theano image_shape. (batch_size, 1, emb_size, sent_length)
        W, b:  parameters to deal with each sentence
        W_context, b_context: parameters to deal with 'fake attentive-context sentence' for each sentence


        Output:
        attentive_maxpool_vec_l: a vector to represent the sentence 1
        attentive_maxpool_vec_r: a vector to represent the sentence 2

        '''
        batch_size = origin_input_tensor3.shape[0]
        hidden_size = origin_input_tensor3.shape[1]
        l_len = origin_input_tensor3.shape[2]
        r_len = origin_input_tensor3_r.shape[2]
        #construct interaction matrix
        input_tensor3 = input_tensor3*mask_matrix.dimshuffle(0,'x',1)
        input_tensor3_r = input_tensor3_r*mask_matrix_r.dimshuffle(0,'x',1) #(batch, hidden, r_len)
        dot_tensor3 = T.batched_dot(input_tensor3.dimshuffle(0,2,1),input_tensor3_r) #(batch, l_len, r_len)

        l_max_cos = 1.0/(1.0+T.max(T.nnet.relu(dot_tensor3), axis=2))#1.0/T.exp(T.max(T.nnet.sigmoid(dot_tensor3), axis=2)) #(batch, l_len)
        r_max_cos = 1.0/(1.0+T.max(T.nnet.relu(dot_tensor3), axis=1))#1.0/T.exp(T.max(T.nnet.sigmoid(dot_tensor3), axis=1)) #(batch, r_len)

        dot_matrix_for_right = T.nnet.softmax(dot_tensor3.reshape((batch_size*l_len, r_len)))  #(batch*l_len, r_len)
        dot_tensor3_for_right = dot_matrix_for_right.reshape((batch_size, l_len, r_len))#(batch, l_len, r_len)

        weighted_sum_r = T.batched_dot(dot_tensor3_for_right, input_tensor3_r.dimshuffle(0,2,1)).dimshuffle(0,2,1)*mask_matrix.dimshuffle(0,'x',1) #(batch,hidden, l_len)

        dot_matrix_for_left = T.nnet.softmax(dot_tensor3.dimshuffle(0,2,1).reshape((batch_size*r_len, l_len))) #(batch*r_len, l_len)
        dot_tensor3_for_left = dot_matrix_for_left.reshape((batch_size, r_len, l_len))#(batch, r_len, l_len)

        weighted_sum_l = T.batched_dot(dot_tensor3_for_left, input_tensor3.dimshuffle(0,2,1)).dimshuffle(0,2,1)*mask_matrix_r.dimshuffle(0,'x',1) #(batch,hidden, r_len)

        #convolve left, weighted sum r
        biased_conv_model_l = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3*l_max_cos.dimshuffle(0,'x',1),
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape, W=W, b=b)
        biased_temp_conv_output_l = biased_conv_model_l.naked_conv_out
        self.conv_out_l = biased_conv_model_l.masked_conv_out
        self.maxpool_vec_l = biased_conv_model_l.maxpool_vec
        conv_model_l = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3,
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape, W=W, b=b)
        temp_conv_output_l = conv_model_l.naked_conv_out
        conv_model_weighted_r = Conv_with_Mask(rng, input_tensor3=weighted_sum_r,
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape_context, W=W_context, b=b_context) # note that b_context is not used
        temp_conv_output_weighted_r = conv_model_weighted_r.naked_conv_out
        '''
        combine
        '''
        mask_for_conv_output_l=T.repeat(mask_matrix.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_l=(1.0-mask_for_conv_output_l)*(mask_for_conv_output_l-10)

        self.biased_conv_attend_out_l = T.tanh(biased_temp_conv_output_l+ temp_conv_output_weighted_r+ b.dimshuffle('x', 0, 'x'))*mask_matrix.dimshuffle(0,'x',1)
        self.biased_attentive_sumpool_vec_l=T.sum(self.biased_conv_attend_out_l, axis=2)
        self.biased_attentive_meanpool_vec_l=self.biased_attentive_sumpool_vec_l/T.sum(mask_matrix,axis=1).dimshuffle(0,'x')
        masked_biased_conv_output_l=self.biased_conv_attend_out_l+mask_for_conv_output_l      #mutiple mask with the conv_out to set the features by UNK to zero
        self.biased_attentive_maxpool_vec_l=T.max(masked_biased_conv_output_l, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        self.conv_attend_out_l = T.tanh(temp_conv_output_l+ temp_conv_output_weighted_r+ b.dimshuffle('x', 0, 'x'))*mask_matrix.dimshuffle(0,'x',1)
        masked_conv_output_l=self.conv_attend_out_l+mask_for_conv_output_l      #mutiple mask with the conv_out to set the features by UNK to zero
        self.attentive_maxpool_vec_l=T.max(masked_conv_output_l, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        "convolve right, weighted sum l"
        biased_conv_model_r = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3_r*r_max_cos.dimshuffle(0,'x',1),
                 mask_matrix = mask_matrix_r,
                 image_shape=image_shape_r,
                 filter_shape=filter_shape, W=W, b=b)
        biased_temp_conv_output_r = biased_conv_model_r.naked_conv_out
        self.conv_out_r = biased_conv_model_r.masked_conv_out
        self.maxpool_vec_r = biased_conv_model_r.maxpool_vec
        conv_model_r = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3_r,
                 mask_matrix = mask_matrix_r,
                 image_shape=image_shape_r,
                 filter_shape=filter_shape, W=W, b=b)
        temp_conv_output_r = conv_model_r.naked_conv_out
        conv_model_weighted_l = Conv_with_Mask(rng, input_tensor3=weighted_sum_l,
                 mask_matrix = mask_matrix_r,
                 image_shape=image_shape_r,
                 filter_shape=filter_shape_context, W=W_context, b=b_context) # note that b_context is not used
        temp_conv_output_weighted_l = conv_model_weighted_l.naked_conv_out
        '''
        combine
        '''
        mask_for_conv_output_r=T.repeat(mask_matrix_r.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_r=(1.0-mask_for_conv_output_r)*(mask_for_conv_output_r-10)

        self.biased_conv_attend_out_r = T.tanh(biased_temp_conv_output_r+ temp_conv_output_weighted_l+ b.dimshuffle('x', 0, 'x'))*mask_matrix_r.dimshuffle(0,'x',1)
        self.biased_attentive_sumpool_vec_r=T.sum(self.biased_conv_attend_out_r, axis=2)
        self.biased_attentive_meanpool_vec_r=self.biased_attentive_sumpool_vec_r/T.sum(mask_matrix_r,axis=1).dimshuffle(0,'x')
        self.conv_attend_out_r = T.tanh(temp_conv_output_r+ temp_conv_output_weighted_l+ b.dimshuffle('x', 0, 'x'))*mask_matrix_r.dimshuffle(0,'x',1)
                # self.attentive_sumpool_vec_r=T.sum(self.conv_attend_out_r, axis=2)

        masked_biased_conv_output_r=self.biased_conv_attend_out_r+mask_for_conv_output_r      #mutiple mask with the conv_out to set the features by UNK to zero
        self.biased_attentive_maxpool_vec_r=T.max(masked_biased_conv_output_r, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        masked_conv_output_r=self.conv_attend_out_r+mask_for_conv_output_r      #mutiple mask with the conv_out to set the features by UNK to zero
        self.attentive_maxpool_vec_r=T.max(masked_conv_output_r, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

class Attentive_Conv_for_Pair_easy_version(object):
    def __init__(self, rng, input_tensor3, input_tensor3_r, mask_matrix, mask_matrix_r,
                 filter_shape, filter_shape_context,image_shape, image_shape_r,W, b, W_context, b_context):
        '''
        Input:

        input_tensor3: (batch_size, hidden_size, sen_length). tensor3 representation for sentence 1
        input_tensor3_r: (batch_size, hidden_size, sen_length). tensor3 representation for sentence 2
        mask_matrix, mask_matrix_r: mask for the two sentences respectively. Each with (batch_size, sent_length), each row corresponding to one sentence
        filter_shape: standard filter shape for theano convolution function, in shape (hidden_size, 1, emb_size, filter_width)
        filter_shape_context: the filter shape to deal with the 'fake sentence' which contains attentive context. In shape (hidden_size, 1, emb_size, 1)
        image_shape, image_shape_r: standard tensor4 theano image_shape. (batch_size, 1, emb_size, sent_length)
        W, b:  parameters to deal with each sentence
        W_context, b_context: parameters to deal with 'fake attentive-context sentence' for each sentence


        Output:
        attentive_maxpool_vec_l: a vector to represent the sentence 1
        attentive_maxpool_vec_r: a vector to represent the sentence 2

        '''
        origin_input_tensor3=input_tensor3
        origin_input_tensor3_r=input_tensor3_r
        batch_size = origin_input_tensor3.shape[0]
        hidden_size = origin_input_tensor3.shape[1]
        l_len = origin_input_tensor3.shape[2]
        r_len = origin_input_tensor3_r.shape[2]
        #construct interaction matrix
        input_tensor3 = input_tensor3*mask_matrix.dimshuffle(0,'x',1)
        input_tensor3_r = input_tensor3_r*mask_matrix_r.dimshuffle(0,'x',1) #(batch, hidden, r_len)
        dot_tensor3 = T.batched_dot(input_tensor3.dimshuffle(0,2,1),input_tensor3_r) #(batch, l_len, r_len)

        l_max_cos = 1.0/(1.0+T.max(T.nnet.relu(dot_tensor3), axis=2))#1.0/T.exp(T.max(T.nnet.sigmoid(dot_tensor3), axis=2)) #(batch, l_len)
        r_max_cos = 1.0/(1.0+T.max(T.nnet.relu(dot_tensor3), axis=1))#1.0/T.exp(T.max(T.nnet.sigmoid(dot_tensor3), axis=1)) #(batch, r_len)

        dot_matrix_for_right = T.nnet.softmax(dot_tensor3.reshape((batch_size*l_len, r_len)))  #(batch*l_len, r_len)
        dot_tensor3_for_right = dot_matrix_for_right.reshape((batch_size, l_len, r_len))#(batch, l_len, r_len)

        weighted_sum_r = T.batched_dot(dot_tensor3_for_right, input_tensor3_r.dimshuffle(0,2,1)).dimshuffle(0,2,1)*mask_matrix.dimshuffle(0,'x',1) #(batch,hidden, l_len)

        dot_matrix_for_left = T.nnet.softmax(dot_tensor3.dimshuffle(0,2,1).reshape((batch_size*r_len, l_len))) #(batch*r_len, l_len)
        dot_tensor3_for_left = dot_matrix_for_left.reshape((batch_size, r_len, l_len))#(batch, r_len, l_len)

        weighted_sum_l = T.batched_dot(dot_tensor3_for_left, input_tensor3.dimshuffle(0,2,1)).dimshuffle(0,2,1)*mask_matrix_r.dimshuffle(0,'x',1) #(batch,hidden, r_len)

        #convolve left, weighted sum r
        biased_conv_model_l = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3*l_max_cos.dimshuffle(0,'x',1),
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape, W=W, b=b)
        biased_temp_conv_output_l = biased_conv_model_l.naked_conv_out
        self.conv_out_l = biased_conv_model_l.masked_conv_out
        self.maxpool_vec_l = biased_conv_model_l.maxpool_vec
        conv_model_l = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3,
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape, W=W, b=b)
        temp_conv_output_l = conv_model_l.naked_conv_out
        conv_model_weighted_r = Conv_with_Mask(rng, input_tensor3=weighted_sum_r,
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape_context, W=W_context, b=b_context) # note that b_context is not used
        temp_conv_output_weighted_r = conv_model_weighted_r.naked_conv_out
        '''
        combine
        '''
        mask_for_conv_output_l=T.repeat(mask_matrix.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_l=(1.0-mask_for_conv_output_l)*(mask_for_conv_output_l-10)

        self.biased_conv_attend_out_l = T.tanh(biased_temp_conv_output_l+ temp_conv_output_weighted_r+ b.dimshuffle('x', 0, 'x'))*mask_matrix.dimshuffle(0,'x',1)
        self.biased_attentive_sumpool_vec_l=T.sum(self.biased_conv_attend_out_l, axis=2)
        self.biased_attentive_meanpool_vec_l=self.biased_attentive_sumpool_vec_l/T.sum(mask_matrix,axis=1).dimshuffle(0,'x')
        masked_biased_conv_output_l=self.biased_conv_attend_out_l+mask_for_conv_output_l      #mutiple mask with the conv_out to set the features by UNK to zero
        self.biased_attentive_maxpool_vec_l=T.max(masked_biased_conv_output_l, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        self.conv_attend_out_l = T.tanh(temp_conv_output_l+ temp_conv_output_weighted_r+ b.dimshuffle('x', 0, 'x'))*mask_matrix.dimshuffle(0,'x',1)
        masked_conv_output_l=self.conv_attend_out_l+mask_for_conv_output_l      #mutiple mask with the conv_out to set the features by UNK to zero
        self.attentive_maxpool_vec_l=T.max(masked_conv_output_l, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        "convolve right, weighted sum l"
        biased_conv_model_r = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3_r*r_max_cos.dimshuffle(0,'x',1),
                 mask_matrix = mask_matrix_r,
                 image_shape=image_shape_r,
                 filter_shape=filter_shape, W=W, b=b)
        biased_temp_conv_output_r = biased_conv_model_r.naked_conv_out
        self.conv_out_r = biased_conv_model_r.masked_conv_out
        self.maxpool_vec_r = biased_conv_model_r.maxpool_vec
        conv_model_r = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3_r,
                 mask_matrix = mask_matrix_r,
                 image_shape=image_shape_r,
                 filter_shape=filter_shape, W=W, b=b)
        temp_conv_output_r = conv_model_r.naked_conv_out
        conv_model_weighted_l = Conv_with_Mask(rng, input_tensor3=weighted_sum_l,
                 mask_matrix = mask_matrix_r,
                 image_shape=image_shape_r,
                 filter_shape=filter_shape_context, W=W_context, b=b_context) # note that b_context is not used
        temp_conv_output_weighted_l = conv_model_weighted_l.naked_conv_out
        '''
        combine
        '''
        mask_for_conv_output_r=T.repeat(mask_matrix_r.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_r=(1.0-mask_for_conv_output_r)*(mask_for_conv_output_r-10)

        self.biased_conv_attend_out_r = T.tanh(biased_temp_conv_output_r+ temp_conv_output_weighted_l+ b.dimshuffle('x', 0, 'x'))*mask_matrix_r.dimshuffle(0,'x',1)
        self.biased_attentive_sumpool_vec_r=T.sum(self.biased_conv_attend_out_r, axis=2)
        self.biased_attentive_meanpool_vec_r=self.biased_attentive_sumpool_vec_r/T.sum(mask_matrix_r,axis=1).dimshuffle(0,'x')
        self.conv_attend_out_r = T.tanh(temp_conv_output_r+ temp_conv_output_weighted_l+ b.dimshuffle('x', 0, 'x'))*mask_matrix_r.dimshuffle(0,'x',1)
                # self.attentive_sumpool_vec_r=T.sum(self.conv_attend_out_r, axis=2)

        masked_biased_conv_output_r=self.biased_conv_attend_out_r+mask_for_conv_output_r      #mutiple mask with the conv_out to set the features by UNK to zero
        self.biased_attentive_maxpool_vec_r=T.max(masked_biased_conv_output_r, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        masked_conv_output_r=self.conv_attend_out_r+mask_for_conv_output_r      #mutiple mask with the conv_out to set the features by UNK to zero
        self.attentive_maxpool_vec_r=T.max(masked_conv_output_r, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

class Conv_for_Pair_Multi_Perspective(object):
    """we define CNN by input tensor3 and output tensor3, like RNN, filter width must by 3,5,7..."""

    def __init__(self, rng, origin_input_tensor3, origin_input_tensor3_r, input_tensor3, input_tensor3_r, mask_matrix, mask_matrix_r,
                 filter_shape, filter_shape_context,image_shape, image_shape_r,W, b, W_context, b_context,
                 MP_W, psp_size):
        #MP_W: (emb_size, l), l meams perspectives
        MP_W=T.nnet.relu(MP_W) # first make sure each dim has weight 0-1
        #construct interaction matrix
        batch_size = input_tensor3.shape[0]
        hidden_size = filter_shape[0]
        l_len = input_tensor3.shape[2]
        r_len = input_tensor3_r.shape[2]

        input_tensor3 = input_tensor3*mask_matrix.dimshuffle(0,'x',1)
        input_tensor3_r = input_tensor3_r*mask_matrix_r.dimshuffle(0,'x',1)

        inter_tensor3 = T.extra_ops.repeat(input_tensor3, r_len, axis=2)*T.tile(input_tensor3_r, (1,1,l_len)) #(batch, hidden, l_len*r_len)
        inter_tensor3_psp = (inter_tensor3.dimshuffle(0,2,1).dot(MP_W)).dimshuffle(0,2,1)  #(batch, kerns,l_len*r_len )
        inter_tensor4_psp = inter_tensor3_psp.reshape((batch_size, psp_size, l_len, r_len)) #(batch, kerns, l_lrn, r_len)

        dot_tensor3 = inter_tensor4_psp.reshape((batch_size*psp_size,l_len,r_len)) #(batch*kerns, l_len, r_len)

        l_max_cos = 1.0/(1.0+T.max(T.nnet.relu(dot_tensor3), axis=2))##(batch*kerns, l_len)
        r_max_cos = 1.0/(1.0+T.max(T.nnet.relu(dot_tensor3), axis=1))##(batch*kerns, r_len)

        exp_inter_tensor4_psp = T.exp(inter_tensor4_psp)
        softmax_for_right = exp_inter_tensor4_psp/T.sum(exp_inter_tensor4_psp, axis=3).dimshuffle(0,1,2,'x') #(batch, kerns, l_lrn, r_len)
        softmax_for_left = exp_inter_tensor4_psp/T.sum(exp_inter_tensor4_psp, axis=2).dimshuffle(0,1,'x',2) #(batch, kerns, l_lrn, r_len)
        softmax_for_right_tensor3 = softmax_for_right.reshape((batch_size*psp_size,l_len,r_len))#(batch*kerns, l_lrn, r_len)
        softmax_for_left_tensor3 = softmax_for_left.reshape((batch_size*psp_size,l_len,r_len)).dimshuffle(0,2,1)#(batch*kerns, r_lrn, l_len)

        repeat_input_tensor3 = T.repeat(input_tensor3,psp_size, axis=0 ) #(batch*kerns, hidden, l_len)
        repeat_mask_matrix = T.repeat(mask_matrix,psp_size, axis=0 ) #(batch*kerns, l_len)
        repeat_input_tensor3_r = T.repeat(input_tensor3_r,psp_size, axis=0 ) #(batch*kerns, hidden, r_len)
        repeat_mask_matrix_r = T.repeat(mask_matrix_r,psp_size, axis=0 ) #(batch*kerns, r_len)

        weighted_sum_r = T.batched_dot(softmax_for_right_tensor3, repeat_input_tensor3_r.dimshuffle(0,2,1)).dimshuffle(0,2,1)*repeat_mask_matrix.dimshuffle(0,'x',1) #(batch*kerns,hidden, l_len)
        weighted_sum_l = T.batched_dot(softmax_for_left_tensor3, repeat_input_tensor3.dimshuffle(0,2,1)).dimshuffle(0,2,1)*repeat_mask_matrix_r.dimshuffle(0,'x',1) #(batch*kerns,hidden, r_len)


        repeat_origin_input_tensor3 = T.repeat(origin_input_tensor3, psp_size, axis=0 ) #(batch*kerns, hidden, l_len)
        biased_conv_model_l = Conv_with_Mask(rng, input_tensor3=repeat_origin_input_tensor3*l_max_cos.dimshuffle(0,'x',1),
                 mask_matrix = repeat_mask_matrix,
                 image_shape=(image_shape[0]*psp_size, image_shape[1],image_shape[2],image_shape[3]),
                 filter_shape=filter_shape, W=W, b=b)
        biased_temp_conv_output_l = biased_conv_model_l.naked_conv_out  #(batch*kerns, hidden, l_len)
#         self.conv_out_l = biased_conv_model_l.masked_conv_out
#         self.maxpool_vec_l = biased_conv_model_l.maxpool_vec
#         conv_model_l = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3,
#                  mask_matrix = mask_matrix,
#                  image_shape=image_shape,
#                  filter_shape=filter_shape, W=W, b=b)
#         temp_conv_output_l = conv_model_l.naked_conv_out
        conv_model_weighted_r = Conv_with_Mask(rng, input_tensor3=weighted_sum_r,
                 mask_matrix = repeat_mask_matrix,
                 image_shape=(image_shape[0]*psp_size, image_shape[1],image_shape[2],image_shape[3]),
                 filter_shape=filter_shape_context, W=W_context, b=b_context) # note that b_context is not used
        temp_conv_output_weighted_r = conv_model_weighted_r.naked_conv_out  #(batch*kerns, hidden, l_len)
        '''
        combine
        '''
        mask_for_conv_output_l=T.repeat(repeat_mask_matrix.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size*kerns, emb_size, maxSentLen)
        mask_for_conv_output_l=(1.0-mask_for_conv_output_l)*(mask_for_conv_output_l-10)

        self.biased_conv_attend_out_l = T.tanh(biased_temp_conv_output_l+ temp_conv_output_weighted_r+ b.dimshuffle('x', 0, 'x'))*repeat_mask_matrix.dimshuffle(0,'x',1)
        # self.attentive_sumpool_vec_l=T.sum(self.conv_attend_out_l, axis=2)
        masked_biased_conv_output_l=self.biased_conv_attend_out_l+mask_for_conv_output_l    ##(batch_size*kerns, emb_size, l_len)
        self.biased_attentive_maxpool_vec_l_psp=T.max(masked_biased_conv_output_l, axis=2) #(batch_size*kerns, hidden_size) # each sentence then have an embedding of length hidden_size
#         self.biased_attentive_maxpool_vec_l_psp=T.max(masked_biased_conv_output_l, axis=2).reshape((batch_size, psp_size*hidden_size)) #(batch_size*kerns, hidden_size) # each sentence then have an embedding of length hidden_size

#         self.conv_attend_out_l = T.tanh(temp_conv_output_l+ temp_conv_output_weighted_r+ b.dimshuffle('x', 0, 'x'))*mask_matrix.dimshuffle(0,'x',1)
#         masked_conv_output_l=self.conv_attend_out_l+mask_for_conv_output_l      #mutiple mask with the conv_out to set the features by UNK to zero
#         self.attentive_maxpool_vec_l=T.max(masked_conv_output_l, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size




        # self.group_max_pools_l = T.signal.pool.pool_2d(input=self.conv_out_l, ds=(1,10), ignore_border=True) #(batch, hidden, 5)
        # self.all_att_maxpool_vecs_l = T.concatenate([self.attentive_maxpool_vec_l.dimshuffle(0,1,'x'), self.group_max_pools_l], axis=2) #(batch, hidden, 6)


        repeat_origin_input_tensor3_r = T.repeat(origin_input_tensor3_r, psp_size, axis=0 ) #(batch*kerns, hidden, r_len)
        biased_conv_model_r = Conv_with_Mask(rng, input_tensor3=repeat_origin_input_tensor3_r*r_max_cos.dimshuffle(0,'x',1),
                 mask_matrix = repeat_mask_matrix_r,
                 image_shape=[image_shape_r[0]*psp_size,image_shape_r[1],image_shape_r[2],image_shape_r[3]],
                 filter_shape=filter_shape, W=W, b=b)
        biased_temp_conv_output_r = biased_conv_model_r.naked_conv_out
#         self.conv_out_r = biased_conv_model_r.masked_conv_out
#         self.maxpool_vec_r = biased_conv_model_r.maxpool_vec
#         conv_model_r = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3_r,
#                  mask_matrix = mask_matrix_r,
#                  image_shape=image_shape_r,
#                  filter_shape=filter_shape, W=W, b=b)
#         temp_conv_output_r = conv_model_r.naked_conv_out
        conv_model_weighted_l = Conv_with_Mask(rng, input_tensor3=weighted_sum_l,
                 mask_matrix = repeat_mask_matrix_r,
                 image_shape=[image_shape_r[0]*psp_size,image_shape_r[1],image_shape_r[2],image_shape_r[3]],
                 filter_shape=filter_shape_context, W=W_context, b=b_context) # note that b_context is not used
        temp_conv_output_weighted_l = conv_model_weighted_l.naked_conv_out
        '''
        combine
        '''
        mask_for_conv_output_r=T.repeat(repeat_mask_matrix_r.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_r=(1.0-mask_for_conv_output_r)*(mask_for_conv_output_r-10)

        self.biased_conv_attend_out_r = T.tanh(biased_temp_conv_output_r+ temp_conv_output_weighted_l+ b.dimshuffle('x', 0, 'x'))*repeat_mask_matrix_r.dimshuffle(0,'x',1)
#         self.conv_attend_out_r = T.tanh(temp_conv_output_r+ temp_conv_output_weighted_l+ b.dimshuffle('x', 0, 'x'))*mask_matrix_r.dimshuffle(0,'x',1)
                # self.attentive_sumpool_vec_r=T.sum(self.conv_attend_out_r, axis=2)

        masked_biased_conv_output_r=self.biased_conv_attend_out_r+mask_for_conv_output_r      #mutiple mask with the conv_out to set the features by UNK to zero
        self.biased_attentive_maxpool_vec_r_psp=T.max(masked_biased_conv_output_r, axis=2) #(batch_size*kerns, hidden_size) # each sentence then have an embedding of length hidden_size
#         self.biased_attentive_maxpool_vec_r_psp=T.max(masked_biased_conv_output_r, axis=2).reshape((batch_size, psp_size*hidden_size))
#         masked_conv_output_r=self.conv_attend_out_r+mask_for_conv_output_r      #mutiple mask with the conv_out to set the features by UNK to zero
#         self.attentive_maxpool_vec_r=T.max(masked_conv_output_r, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        # self.group_max_pools_r = T.signal.pool.pool_2d(input=self.conv_out_r, ds=(1,10), ignore_border=True) #(batch, hidden, 5)
        # self.all_att_maxpool_vecs_r = T.concatenate([self.attentive_maxpool_vec_r.dimshuffle(0,1,'x'), self.group_max_pools_r], axis=2) #(batch, hidden, 6)

def kmaxpooling_tensor3(tensor3, k):
    batch_size = tensor3.shape[0]
    hidden_size = tensor3.shape[1]
    length=tensor3.shape[2]
    matrix = tensor3.reshape((batch_size*hidden_size, length))
    idsort_matrix = T.argsort(matrix, axis=1)
    idsort_topk = idsort_matrix[:,-k:]
    restore_id_order_matrix = T.sort(idsort_topk, axis=1) # make y indices in acending lie

    ii = T.repeat(T.arange(batch_size*hidden_size), k)
    jj = restore_id_order_matrix.flatten()
    feature_vec  = matrix[ii, jj] # batch_size*hidden_size*k
    return feature_vec.reshape((batch_size, k*hidden_size))

class Conv_for_Pair_SoftAttend(object):
    """we define CNN by input tensor3 and output tensor3, like RNN, filter width must by 3,5,7..."""

    def __init__(self, rng, origin_input_tensor3, origin_input_tensor3_r, input_tensor3, input_tensor3_r, mask_matrix, mask_matrix_r,
                 filter_shape, filter_shape_context,image_shape, image_shape_r,W, b, W_context, b_context,
                 soft_att_W_big, soft_att_b_big,soft_att_W_small):
        #construct interaction matrix
        input_tensor3 = input_tensor3*mask_matrix.dimshuffle(0,'x',1)
        input_tensor3_r = input_tensor3_r*mask_matrix_r.dimshuffle(0,'x',1) #(batch, hidden, r_len)
#         dot_mask = T.batched_dot(mask_matrix.dimshuffle(0,1,'x'), mask_matrix_r.dimshuffle(0,'x',1)) #(batch, l_len, r_len)
#         dot_tensor3 = T.batched_dot(input_tensor3.dimshuffle(0,2,1),input_tensor3_r) #(batch, l_len, r_len)
#
#         self.cosine_tensor3 = dot_tensor3
        # self.cosine_tensor3 = dot_tensor3/(1e-8+T.batched_dot(T.sqrt(1e-8+T.sum(input_tensor3**2, axis=1)).dimshuffle(0,1,'x'), T.sqrt(1e-8+T.sum(input_tensor3_r**2, axis=1)).dimshuffle(0,'x', 1)))

#         sort_l= T.argsort(self.l_max_cos, axis=1)
#         self.l_topK_min_max_cos = self.l_max_cos[T.repeat(T.arange(input_tensor3.shape[0]), 10, axis=0), sort_l[:,:10].flatten()].reshape((input_tensor3.shape[0],10))
#         sort_r= T.argsort(self.r_max_cos, axis=1)
#         self.r_topK_min_max_cos = self.r_max_cos[T.repeat(T.arange(input_tensor3.shape[0]), 10, axis=0), sort_r[:,:10].flatten()].reshape((input_tensor3.shape[0],10))
        '''
        soft interaction matrix
        '''
        Conc_T = T.concatenate([T.extra_ops.repeat(input_tensor3, input_tensor3_r.shape[2], axis=2), T.tile(input_tensor3_r, (1,1,input_tensor3.shape[2]))], axis=1) #(batch, 2hidden, l_len*r_len)
        dot_tensor3 = T.tanh(Conc_T.dimshuffle(0,2,1).dot(soft_att_W_big)+soft_att_b_big.dimshuffle('x','x',0)).dot(soft_att_W_small).reshape((input_tensor3.shape[0], input_tensor3.shape[2], input_tensor3_r.shape[2]))#(batch, l_len, r_len)
        l_max_cos = T.max(dot_tensor3, axis=2) #(batch, l_len)
        r_max_cos = T.max(dot_tensor3, axis=1) #(batch, r_len)
#         l_max_cos = 1.0/(1.0+T.max(T.nnet.relu(dot_tensor3), axis=2))##(batch*kerns, l_len)
#         r_max_cos = 1.0/(1.0+T.max(T.nnet.relu(dot_tensor3), axis=1))##(batch*kerns, r_len)
#         dot_mask=T.cast((1.0-dot_mask)*(dot_mask-100000), 'float32')#(batch, l_len, r_len)
#         dot_tensor3 = dot_tensor3 + dot_mask

        dot_matrix_for_right = T.nnet.softmax(dot_tensor3.reshape((dot_tensor3.shape[0]*dot_tensor3.shape[1], dot_tensor3.shape[2])))
        dot_tensor3_for_right = dot_matrix_for_right.reshape((dot_tensor3.shape[0], dot_tensor3.shape[1], dot_tensor3.shape[2]))#(batch, l_len, r_len)
        weighted_sum_r = T.batched_dot(dot_tensor3_for_right, input_tensor3_r.dimshuffle(0,2,1)).dimshuffle(0,2,1)*mask_matrix.dimshuffle(0,'x',1) #(batch,hidden, l_len)

        dot_matrix_for_left = T.nnet.softmax(dot_tensor3.dimshuffle(0,2,1).reshape((dot_tensor3.shape[0]*dot_tensor3.shape[2], dot_tensor3.shape[1])))
        dot_tensor3_for_left = dot_matrix_for_left.reshape((dot_tensor3.shape[0], dot_tensor3.shape[2], dot_tensor3.shape[1]))#(batch, r_len, l_len)
        weighted_sum_l = T.batched_dot(dot_tensor3_for_left, input_tensor3.dimshuffle(0,2,1)).dimshuffle(0,2,1)*mask_matrix_r.dimshuffle(0,'x',1) #(batch,hidden, r_len)

        #convolve left, weighted sum r
        biased_conv_model_l = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3*l_max_cos.dimshuffle(0,'x',1),
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape, W=W, b=b)
        biased_temp_conv_output_l = biased_conv_model_l.naked_conv_out

        conv_model_l = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3,
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape, W=W, b=b)
        temp_conv_output_l = conv_model_l.naked_conv_out
        self.conv_out_l = conv_model_l.masked_conv_out
        self.maxpool_vec_l = conv_model_l.maxpool_vec
        self.kmaxpool_vec_l = kmaxpooling_tensor3(self.conv_out_l, 3)
        conv_model_weighted_r = Conv_with_Mask(rng, input_tensor3=weighted_sum_r,
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape_context, W=W_context, b=b_context) # note that b_context is not used
        temp_conv_output_weighted_r = conv_model_weighted_r.naked_conv_out
        '''
        combine
        '''
        mask_for_conv_output_l=T.repeat(mask_matrix.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_l=(1.0-mask_for_conv_output_l)*(mask_for_conv_output_l-10)

        self.biased_conv_attend_out_l = T.tanh(biased_temp_conv_output_l+ temp_conv_output_weighted_r+ b.dimshuffle('x', 0, 'x'))*mask_matrix.dimshuffle(0,'x',1)
        masked_biased_conv_output_l=self.biased_conv_attend_out_l+mask_for_conv_output_l      #mutiple mask with the conv_out to set the features by UNK to zero
        self.biased_attentive_maxpool_vec_l=T.max(masked_biased_conv_output_l, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size



        self.conv_attend_out_l = T.tanh(temp_conv_output_l+ temp_conv_output_weighted_r+ b.dimshuffle('x', 0, 'x'))*mask_matrix.dimshuffle(0,'x',1)
        self.attentive_sumpool_vec_l=T.sum(self.conv_attend_out_l, axis=2)

        masked_conv_output_l=self.conv_attend_out_l+mask_for_conv_output_l      #mutiple mask with the conv_out to set the features by UNK to zero
        self.attentive_maxpool_vec_l=T.max(masked_conv_output_l, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size
        self.attentive_kmaxpool_vec_l = kmaxpooling_tensor3(masked_conv_output_l, 3)


        #convolve right, weighted sum l
        biased_conv_model_r = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3_r*r_max_cos.dimshuffle(0,'x',1),
                 mask_matrix = mask_matrix_r,
                 image_shape=image_shape_r,
                 filter_shape=filter_shape, W=W, b=b)
        biased_temp_conv_output_r = biased_conv_model_r.naked_conv_out

        conv_model_r = Conv_with_Mask(rng, input_tensor3=origin_input_tensor3_r,
                 mask_matrix = mask_matrix_r,
                 image_shape=image_shape_r,
                 filter_shape=filter_shape, W=W, b=b)
        temp_conv_output_r = conv_model_r.naked_conv_out
        self.conv_out_r = conv_model_r.masked_conv_out
        self.maxpool_vec_r = conv_model_r.maxpool_vec
        self.kmaxpool_vec_r = kmaxpooling_tensor3(self.conv_out_r, 3)
        conv_model_weighted_l = Conv_with_Mask(rng, input_tensor3=weighted_sum_l,
                 mask_matrix = mask_matrix_r,
                 image_shape=image_shape_r,
                 filter_shape=filter_shape_context, W=W_context, b=b_context) # note that b_context is not used
        temp_conv_output_weighted_l = conv_model_weighted_l.naked_conv_out
        '''
        combine
        '''
        mask_for_conv_output_r=T.repeat(mask_matrix_r.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_r=(1.0-mask_for_conv_output_r)*(mask_for_conv_output_r-10)

        self.biased_conv_attend_out_r = T.tanh(biased_temp_conv_output_r+ temp_conv_output_weighted_l+ b.dimshuffle('x', 0, 'x'))*mask_matrix_r.dimshuffle(0,'x',1)
        masked_biased_conv_output_r=self.biased_conv_attend_out_r+mask_for_conv_output_r      #mutiple mask with the conv_out to set the features by UNK to zero
        self.biased_attentive_maxpool_vec_r=T.max(masked_biased_conv_output_r, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size


        self.conv_attend_out_r = T.tanh(temp_conv_output_r+ temp_conv_output_weighted_l+ b.dimshuffle('x', 0, 'x'))*mask_matrix_r.dimshuffle(0,'x',1)
        self.attentive_sumpool_vec_r=T.sum(self.conv_attend_out_r, axis=2)
        masked_conv_output_r=self.conv_attend_out_r+mask_for_conv_output_r      #mutiple mask with the conv_out to set the features by UNK to zero
        self.attentive_maxpool_vec_r=T.max(masked_conv_output_r, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size
        self.attentive_kmaxpool_vec_r = kmaxpooling_tensor3(masked_conv_output_r, 3)

class Conv_with_Mask_with_Gate(object):
    """we define CNN by input tensor3 and output tensor3, like RNN, filter width must by 3,5,7..."""

    def __init__(self, rng, input_tensor3, mask_matrix, filter_shape, image_shape, W, b, W_gate, b_gate):
        conv_layer = Conv_with_Mask(rng, input_tensor3=input_tensor3,
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape, W=W, b=b)

        gate_layer = Conv_with_Mask(rng, input_tensor3=input_tensor3,
                 mask_matrix = mask_matrix,
                 image_shape=image_shape,
                 filter_shape=filter_shape, W=W_gate, b=b_gate)
        self.output_tensor3 =   gate_layer.masked_conv_out_sigmoid*input_tensor3+(1.0-gate_layer.masked_conv_out_sigmoid)*conv_layer.masked_conv_out

        mask_for_conv_output=T.repeat(mask_matrix.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output=(1.0-mask_for_conv_output)*(mask_for_conv_output-10)
        masked_conv_output=self.output_tensor3+mask_for_conv_output      #mutiple mask with the conv_out to set the features by UNK to zero
        self.maxpool_vec=T.max(masked_conv_output, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size


class Conv_with_Mask(object):
    """we define CNN by input tensor3 and output tensor3, like RNN, filter width must by 3,5,7..."""

    def __init__(self, rng, input_tensor3, mask_matrix, filter_shape, image_shape, W, b):
        assert image_shape[1] == filter_shape[1]
        pad_size = filter_shape[3]/2
        remain_size = filter_shape[3]%2
        zero_pad_tensor4_1 = T.zeros((input_tensor3.shape[0], 1, input_tensor3.shape[1], pad_size), dtype=theano.config.floatX)+1e-8  # to get rid of nan in CNN gradient
        input = T.concatenate([zero_pad_tensor4_1,input_tensor3.dimshuffle(0,'x',1,2),
                    zero_pad_tensor4_1], axis=3)        #(batch_size, 1, emb_size, maxsenlen+width-1)

        self.input = input
        self.W = W
        self.b = b

        pad_images_shape=(image_shape[0], image_shape[1], image_shape[2], image_shape[3]+2*pad_size)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=pad_images_shape, border_mode='valid')    #here, we should pad enough zero padding for input
        #no tanh and bias
        # conv_with_bias = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        # conv_output_tensor3=conv_out.reshape((image_shape[0], filter_shape[0], image_shape[3])) #(batch, 1, kernerl, ishape[1]-filter_size1[1]+1)
        if remain_size==0:
            conv_out = conv_out[:,:,:,:-1]+conv_out[:,:,:,1:] #(batch, kerns, hidden, len)

        self.naked_conv_out=conv_out.reshape((image_shape[0], filter_shape[0], image_shape[3]))*mask_matrix.dimshuffle(0,'x',1) #(batch, hidden_size, len)


        #with tank and bias
        conv_with_bias = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        conv_output_tensor3=conv_with_bias.reshape((image_shape[0], filter_shape[0], image_shape[3])) #(batch, 1, kernerl, ishape[1]-filter_size1[1]+1)

        self.masked_conv_out=conv_output_tensor3*mask_matrix.dimshuffle(0,'x',1) #(batch, hidden_size, len)
        self.sumpool_vec = T.sum(self.masked_conv_out, axis=2) #(batch, hidden)
        conv_with_bias_sigmoid = T.nnet.sigmoid(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        conv_output_tensor3_sigmoid=conv_with_bias_sigmoid.reshape((image_shape[0], filter_shape[0], image_shape[3])) #(batch, 1, kernerl, ishape[1]-filter_size1[1]+1)

        self.masked_conv_out_sigmoid=conv_output_tensor3_sigmoid*mask_matrix.dimshuffle(0,'x',1) #(batch, hidden_size, len)


        mask_for_conv_output=T.repeat(mask_matrix.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output=(1.0-mask_for_conv_output)*(mask_for_conv_output-10)
        masked_conv_output=self.masked_conv_out+mask_for_conv_output      #mutiple mask with the conv_out to set the features by UNK to zero
        self.masked_conv_out_plus_mask = masked_conv_output
        self.maxpool_vec=T.max(self.masked_conv_out_plus_mask, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        self.params = [self.W, self.b]
class Conv_with_input_para(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, W, b, filter_type='valid'):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = W
        self.b = b

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode=filter_type)    #here, we should pad enough zero padding for input

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        conv_with_bias = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        wide_conv_out=conv_with_bias[:,:,filter_shape[2]-1,:].reshape((image_shape[0], 1, filter_shape[0], image_shape[3]+filter_shape[3]-1))
        narrow_conv_out=conv_with_bias.reshape((image_shape[0], 1, filter_shape[0], image_shape[3]-filter_shape[3]+1)) #(batch, 1, kernerl, ishape[1]-filter_size1[1]+1)

        self.narrow_conv_out=narrow_conv_out
        self.wide_conv_out=wide_conv_out
        #pad filter_size-1 zero embeddings at both sides
        left_padding = T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        right_padding = T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        self.output = T.concatenate([left_padding, narrow_conv_out, right_padding], axis=3)
        self.output_max_pooling_vec=T.max(narrow_conv_out.reshape((narrow_conv_out.shape[2], narrow_conv_out.shape[3])), axis=1)

        # store parameters of this layer
        self.params = [self.W, self.b]

class RNN_with_input_para(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, rnn_Whh, rnn_Wxh, rnn_b, dim):
        self.input = input.transpose(1,0) #iterate over first dim
        self.Whh = rnn_Whh
        self.Wxh=rnn_Wxh
        self.b = rnn_b
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(dim,
                                dtype=theano.config.floatX))
        def recurrence(x_t, h_tm1):
            w_t = T.nnet.sigmoid(T.dot(x_t, self.Wxh)
                                 + T.dot(h_tm1, self.Whh) + self.b)
            h_t=h_tm1*w_t+x_t*(1-w_t)
#             s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return h_t

        h, _ = theano.scan(fn=recurrence,
                                sequences=self.input,
                                outputs_info=self.h0,#[self.h0, None],
                                n_steps=self.input.shape[0])
        self.output=h.reshape((self.input.shape[0], self.input.shape[1])).transpose(1,0)


        # store parameters of this layer
        self.params = [self.Whh, self.Wxh, self.b]

def Matrix_Bit_Shift(input_matrix): # shit each column
    input_matrix=debug_print(input_matrix, 'input_matrix')

    def shift_at_t(t):
        shifted_matrix=debug_print(T.concatenate([input_matrix[:,t:], input_matrix[:,:t]], axis=1), 'shifted_matrix')
        return shifted_matrix

    tensor,_ = theano.scan(fn=shift_at_t,
                            sequences=T.arange(input_matrix.shape[1]),
                            n_steps=input_matrix.shape[1])

    return tensor

class Bi_GRU_Matrix_Input(object):
    def __init__(self, X, word_dim, hidden_dim, U, W, b, U_b, W_b, b_b, bptt_truncate):
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        def forward_prop_step(x_t, s_t1_prev):
            # GRU Layer 1
            z_t1 =T.nnet.sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
            return s_t1

        s, updates = theano.scan(
            forward_prop_step,
            sequences=X.transpose(1,0),
            truncate_gradient=self.bptt_truncate,
            outputs_info=dict(initial=T.zeros(self.hidden_dim)))

#         self.output_matrix=debug_print(s.transpose(), 'GRU_Matrix_Input.output_matrix')
#         self.output_vector_mean=T.mean(self.output_matrix, axis=1)
#         self.output_vector_max=T.max(self.output_matrix, axis=1)
#         self.output_vector_last=self.output_matrix[:,-1]
        #backward
        X_b=X[:,::-1]
        def backward_prop_step(x_t_b, s_t1_prev_b):
            # GRU Layer 1
            z_t1_b =T.nnet.sigmoid(U_b[0].dot(x_t_b) + W_b[0].dot(s_t1_prev_b) + b_b[0])
            r_t1_b = T.nnet.sigmoid(U_b[1].dot(x_t_b) + W_b[1].dot(s_t1_prev_b) + b_b[1])
            c_t1_b = T.tanh(U_b[2].dot(x_t_b) + W_b[2].dot(s_t1_prev_b * r_t1_b) + b_b[2])
            s_t1_b = (T.ones_like(z_t1_b) - z_t1_b) * c_t1_b + z_t1_b * s_t1_prev_b
            return s_t1_b

        s_b, updates_b = theano.scan(
            backward_prop_step,
            sequences=X_b.transpose(1,0),
            truncate_gradient=self.bptt_truncate,
            outputs_info=dict(initial=T.zeros(self.hidden_dim)))
        #dim: hidden_dim*2
#         output_matrix=T.concatenate([s.transpose(), s_b.transpose()[:,::-1]], axis=0)
        output_matrix=s.transpose()+s_b.transpose()[:,::-1]
        self.output_matrix=output_matrix+X # add input feature maps


        self.output_vector_mean=T.mean(self.output_matrix, axis=1)
        self.output_vector_max=T.max(self.output_matrix, axis=1)
        #dim: hidden_dim*4
        self.output_vector_last=T.concatenate([self.output_matrix[:,-1], self.output_matrix[:,0]], axis=0)

class Bi_GRU_Tensor3_Input(object):
    def __init__(self, T, lefts, rights, hidden_dim, U, W, b, Ub,Wb,bb):
        T=debug_print(T,'T')
        lefts=debug_print(lefts, 'lefts')
        rights=debug_print(rights, 'rights')
        def recurrence(matrix, left, right):
            sub_matrix=debug_print(matrix[:,left:-right], 'sub_matrix')
            GRU_layer=Bi_GRU_Matrix_Input(sub_matrix, sub_matrix.shape[0], hidden_dim,U,W,b, Ub,Wb,bb, -1)
            return GRU_layer.output_vector_mean
        new_M, updates = theano.scan(recurrence,
                                     sequences=[T, lefts, rights],
                                     outputs_info=None)
        self.output=debug_print(new_M.transpose(), 'Bi_GRU_Tensor3_Input.output')

class GRU_Matrix_Input(object):
    def __init__(self, X, word_dim, hidden_dim, U, W, b, bptt_truncate):
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        def forward_prop_step(x_t, s_t1_prev):
            # GRU Layer 1
            z_t1 =debug_print( T.nnet.sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + b[0]), 'z_t1')
            r_t1 = debug_print(T.nnet.sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + b[1]), 'r_t1')
            c_t1 = debug_print(T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + b[2]), 'c_t1')
            s_t1 = debug_print((T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev, 's_t1')
            return s_t1

        s, updates = theano.scan(
            forward_prop_step,
            sequences=X.transpose(1,0),
            truncate_gradient=self.bptt_truncate,
            outputs_info=dict(initial=T.zeros(self.hidden_dim)))

        self.output_matrix=debug_print(s.transpose(), 'GRU_Matrix_Input.output_matrix')
        self.output_vector_mean=T.mean(self.output_matrix, axis=1)
        self.output_vector_max=T.max(self.output_matrix, axis=1)
        self.output_vector_last=self.output_matrix[:,-1]

class GRU_Tensor3_Input(object):
    def __init__(self, T, lefts, rights, hidden_dim, U, W, b):
        T=debug_print(T,'T')
        lefts=debug_print(lefts, 'lefts')
        rights=debug_print(rights, 'rights')
        def recurrence(matrix, left, right):
            sub_matrix=debug_print(matrix[:,left:-right], 'sub_matrix')
            GRU_layer=GRU_Matrix_Input(sub_matrix, sub_matrix.shape[0], hidden_dim,U,W,b, -1)
            return GRU_layer.output_vector_mean
        new_M, updates = theano.scan(recurrence,
                                     sequences=[T, lefts, rights],
                                     outputs_info=None)
        self.output=debug_print(new_M.transpose(), 'GRU_Tensor3_Input.output')

def create_params_WbWAE(input_dim, output_dim):
    W = numpy.random.uniform(-numpy.sqrt(1./output_dim), numpy.sqrt(1./output_dim), (6, output_dim, input_dim))
    w = numpy.random.uniform(-numpy.sqrt(1./output_dim), numpy.sqrt(1./output_dim), (1,output_dim))

    W = theano.shared(name='W', value=W.astype(theano.config.floatX))
    w = theano.shared(name='w', value=w.astype(theano.config.floatX))

    return W, w

class Word_by_Word_Attention_EntailmentPaper(object):
    def __init__(self, l_hidden_M, r_hidden_M, W_y,W_h,W_r, w, W_t, W_p, W_x, r_dim):
        self.Y=l_hidden_M
        self.H=r_hidden_M
        self.attention_dim=r_dim
        self.r0 = theano.shared(name='r0', value=numpy.zeros(self.attention_dim, dtype=theano.config.floatX))
        def loop(h_t, r_t_1):
            M_t=T.tanh(W_y.dot(self.Y)+(W_h.dot(h_t)+W_r.dot(r_t_1)).dimshuffle(0,'x'))
            alpha_t=T.nnet.softmax(w.dot(M_t))
            r_t=self.Y.dot(alpha_t.reshape((self.Y.shape[1],1)))+T.tanh(W_t.dot(r_t_1))

            r_t=T.sum(M_t, axis=1)
            return r_t

        r, updates= theano.scan(loop,
                                sequences=self.H.transpose(),
                                outputs_info=self.r0
                                )

        H_star=T.tanh(W_p.dot(r[-1]+W_x.dot(self.H[:,-1])))
        self.output=H_star

class Bd_GRU_Batch_Tensor_Input_with_Mask(object):
    # Bidirectional GRU Layer.
    def __init__(self, X, Mask, hidden_dim, U, W, b, Ub, Wb, bb):
        fwd = GRU_Batch_Tensor_Input_with_Mask(X, Mask, hidden_dim, U, W, b)
        bwd = GRU_Batch_Tensor_Input_with_Mask(X[:,:,::-1], Mask[:,::-1], hidden_dim, Ub, Wb, bb)

        self.output_tensor_conc=T.concatenate([fwd.output_tensor, bwd.output_tensor[:,:,::-1]], axis=1) #(batch, 2*hidden, len)
        #for word level rep
        output_tensor=fwd.output_tensor+bwd.output_tensor[:,:,::-1]
        self.output_tensor=output_tensor

        #for final sentence rep
#         sent_output_tensor=fwd.output_tensor+bwd.output_tensor
#         self.output_tensor=output_tensor+X # add initialized emb
#         self.output_sent_rep=self.output_tensor[:,:,-1]
        self.output_sent_rep=fwd.output_tensor[:,:,-1]+bwd.output_tensor[:,:,-1]
        self.output_sent_rep_conc=T.concatenate([fwd.output_tensor[:,:,-1], bwd.output_tensor[:,:,-1]], axis=1)

class GRU_Batch_Tensor_Input_with_Mask(object):
    def __init__(self, X, Mask, hidden_dim, U, W, b):
        #now, X is (batch, emb_size, sentlength)
        #Mask is a matrix with (batch, sentlength), each row is binary vector for indicating this word is normal word or UNK
        self.hidden_dim = hidden_dim
#         self.bptt_truncate = bptt_truncate
        self.M=Mask.T

        new_tensor=X.dimshuffle(2,1,0)

        def forward_prop_step(x_t, mask, s_t1_prev):     #x_t is the embedding of current word, s_t1_prev is the generated hidden state of previous step
            # GRU Layer 1, the goal is to combine x_t and s_t1_prev to generate a hidden state "s_t1_m" for current word
            # z_t1, and r_t1 are called "gates" in GRU literatures, usually generated by "sigmoid" function
            z_t1 =T.nnet.sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + T.repeat(b[0].reshape((hidden_dim,1)), X.shape[0], axis=1))
            r_t1 = T.nnet.sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + T.repeat(b[1].reshape((hidden_dim,1)), X.shape[0], axis=1))
            # c layer is a temporary layer between layer x and output layer s
            c_t1 = T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + T.repeat(b[2].reshape((hidden_dim,1)), X.shape[0], axis=1))
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev   #the final hidden state s_t1 is weighted sum up of previous hidden state s_t1_prev and the temporary hidden state c_t1
            #mask here is used to choose the previous state s_t1_prev if current token is UNK, otherwise, choose the truely generated hidden state s_t1
            s_t1_m=s_t1*mask[None,:]+(1.0-mask[None,:])*s_t1_prev

            return s_t1_m

        '''
        theano.scan function works like for-loop in python, it processes slices of a list sequentially.
        you just need to define the function of each step, above we define the function "forward_prop_step" to combine x_t and s_t1_prev
        then for theano.scan, you need tell it the name of the step function ("forward_prop_step" here) and its parameters.
        parameter x_t is a slice of "new_tensor", mask is a slice of "self.M", s_t1_prev is initialized by "outputs_info" as a zero matrix (it is a matrix
        because this GRU deals with batch of sentences in the same time, so a row in the matrix is for one sentence, as a initialized hidden state)
        So, "s_t1_prev" is a zero vector in the begininig, then be updated by "s_t1_m". But theano scan will store all its updated vectors, returned as varaible "s" below,
        so "s" below is a list of vectors, namely the hidden states for all words sequentially.
        '''
        s, updates = theano.scan(
            forward_prop_step,
            sequences=[new_tensor, self.M],
            outputs_info=dict(initial=T.zeros((self.hidden_dim, X.shape[0]))))

#         self.output_matrix=debug_print(s.transpose(), 'GRU_Matrix_Input.output_matrix')
        self.output_tensor=s.dimshuffle(2,1,0)  #(batch, emb_size, sentlength) again

        self.output_sent_rep=self.output_tensor[:,:,-1] #it means we choose the last hidden state of gru as the sentence representation, this is a matrix, as it is for batch of sentences

class Bd_LSTM_Batch_Tensor_Input_with_Mask(object):
    # Bidirectional GRU Layer.
    def __init__(self, X, Mask, hidden_dim, fwd_params, bwd_params):
        fwd = LSTM_Batch_Tensor_Input_with_Mask(X, Mask, hidden_dim, fwd_params)
        bwd = LSTM_Batch_Tensor_Input_with_Mask(X[:,:,::-1], Mask[:,::-1], hidden_dim, bwd_params)

        self.output_tensor_conc=T.concatenate([fwd.output_tensor, bwd.output_tensor[:,:,::-1]], axis=1) #(batch, 2*hidden, len)
        self.output_sent_rep_conc=T.concatenate([fwd.output_sent_rep, bwd.output_sent_rep], axis=1) #(batch, 2*hidden)


class LSTM_Batch_Tensor_Input_with_Mask(object):
#     def __init__(self, X, Mask, hidden_dim, U, W, b):
    def __init__(self, X, Mask, hidden_size, tparams ):
        #X (batch, emb_size, senLen), Mask (batch, senLen)
        state_below=X.dimshuffle(2,0,1)
        mask=Mask.T
        # state_below, (senLen, batch_size, emb_size)
        nsteps = state_below.shape[0] #sentence length, as LSTM or GRU needs to know how many words/slices to deal with sequentially
        if state_below.ndim == 3:
            n_samples = state_below.shape[1] #batch_size
        else:
            n_samples = 1

        assert mask is not None

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(m_, x_, h_, c_): #mask, x_ current word embedding, h_ and c_ are hidden states in preceding step of two hidden layers
            preact = T.dot(h_, tparams['U'])
            preact += x_
            '''
            already remember that variables generated by "sigmoid" below are "gates", they are not hidden states, they are used to control how much information of
            some hidden states to be used for other steps, so "i", "f", "o" below are gates
            '''
            i = T.nnet.sigmoid(_slice(preact, 0, hidden_size))
            f = T.nnet.sigmoid(_slice(preact, 1, hidden_size))
            o = T.nnet.sigmoid(_slice(preact, 2, hidden_size))
            c = T.tanh(_slice(preact, 3, hidden_size))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * T.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

        state_below = (T.dot(state_below, tparams['W']) + tparams['b'])

        dim_proj = hidden_size

        '''
        pls understand this theano.scan by referring to the description of GRU's theano.scan
        '''

        rval, updates = theano.scan(_step,
                                    sequences=[mask, state_below],
                                    outputs_info=[T.alloc(numpy.asarray(0., dtype=theano.config.floatX),n_samples,dim_proj),
                                                  T.alloc(numpy.asarray(0., dtype=theano.config.floatX),n_samples,dim_proj)],
                                    n_steps=nsteps)
        self.output_tensor = rval[0].dimshuffle(1,2,0) #(batch, hidden_size, nsamples)
        self.output_sent_rep=self.output_tensor[:,:,-1] # (batch, hidden)

class GRU_Batch_Tensor_Input(object):
    def __init__(self, X, hidden_dim, U, W, b, bptt_truncate):
        #now, X is (batch, emb_size, sentlength)
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        new_tensor=debug_print(X.dimshuffle(2,1,0), 'new_tensor')

        def forward_prop_step(x_t, s_t1_prev):
            # GRU Layer 1
            z_t1 =debug_print( T.nnet.sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + T.repeat(b[0].reshape((hidden_dim,1)), X.shape[0], axis=1)), 'z_t1')  #maybe here has a bug, as b is vector while dot product is matrix
            r_t1 = debug_print(T.nnet.sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + T.repeat(b[1].reshape((hidden_dim,1)), X.shape[0], axis=1)), 'r_t1')
            c_t1 = debug_print(T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + T.repeat(b[2].reshape((hidden_dim,1)), X.shape[0], axis=1)), 'c_t1')
            s_t1 = debug_print((T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev, 's_t1')
            return s_t1

        s, updates = theano.scan(
            forward_prop_step,
            sequences=new_tensor,
            truncate_gradient=self.bptt_truncate,
            outputs_info=dict(initial=T.zeros((self.hidden_dim, X.shape[0]))))

#         self.output_matrix=debug_print(s.transpose(), 'GRU_Matrix_Input.output_matrix')
        self.output_tensor=debug_print(s.dimshuffle(2,1,0), 'self.output_tensor')

#         d0s, d2s=Dim_Align(self.output_tensor.shape[0])
#         d0s=debug_print(d0s, 'd0s')
#         d2s=debug_print(d2s, 'd2s')
#         self.output_matrix=debug_print(self.output_tensor[d0s,:,d2s].transpose(), 'self.output_matrix')  # before transpose, its (dim, hidden_size), each row is a hidden state

        d0s=Dim_Align_new(self.output_tensor.shape[0])
        self.output_matrix=self.output_tensor.transpose(0,2,1).reshape((self.output_tensor.shape[0]*self.output_tensor.shape[2], self.output_tensor.shape[1]))[d0s].transpose()
        self.dim=debug_print(self.output_tensor.shape[0]*(self.output_tensor.shape[0]+1)/2, 'self.dim')
        self.output_sent_rep=self.output_tensor[0,:,-1]
        self.output_sent_hiddenstates=self.output_tensor[0]
        self.ph_lengths=lenghs_phrases(self.output_tensor.shape[0])





def Dim_Align(x):
#     x = tt.lscalar()
    def series_sum(n):
        return n * (n + 1) / 2
    yz = T.zeros((series_sum(x),), dtype='int64')
#     yz = T.zeros((series_sum(x),), dtype='int32')#for gpu

    def step(x1, y1, y2):
        i = series_sum(x1)
        j = series_sum(x1 + 1)
        z1 = T.arange(x1 + 1)
        z2 = z1[::-1]
        y1 = T.set_subtensor(y1[i:j], z1)
        y2 = T.set_subtensor(y2[i:j], z2)
        return y1, y2

    (r1, r2), _ = theano.scan(step, sequences=[T.arange(x)], outputs_info=[yz, yz])
#     return theano.function([x], [y1[-1], y2[-1]])
    return r1[-1], r2[-1]

def Dim_Align_new(x):
    #there is a bug, when input x=1, namely the sentence has only one word
#     x = tt.lscalar()
    def series_sum(n):
        return n * (n + 1) / 2
    yz = T.zeros((series_sum(x),), dtype='int64')
#     yz = T.zeros((series_sum(x),), dtype='int32')#for gpu

    def step(x1, y1):
        i = series_sum(x1)
        j = series_sum(x1 + 1)
#         z1 = T.arange(x1 + 1)
        z1=T.arange(x1,x1*x+1,x-1)[::-1]
        y1 = T.set_subtensor(y1[i:j], z1)
        return y1

    r1, _ = theano.scan(step, sequences=[T.arange(x)], outputs_info=yz)
#     return theano.function([x], [y1[-1], y2[-1]])
    return r1[-1]

def lenghs_phrases(x):
#     x = tt.lscalar()
    def series_sum(n):
        return n * (n + 1) / 2
    yz = T.zeros((series_sum(x),), dtype='int64')
#     yz = T.zeros((series_sum(x),), dtype='int32')#for gpu

    def step(x1, y1):
        i = series_sum(x1)
        j = series_sum(x1 + 1)
#         z1 = T.arange(x1 + 1)
        z1=T.arange(1,x1+2)
        y1 = T.set_subtensor(y1[i:j], z1)
        return y1

    r1, _ = theano.scan(step, sequences=[T.arange(x)], outputs_info=yz) #[0,1,2,3]
#     return theano.function([x], [y1[-1], y2[-1]])
    return r1[-1]
class biRNN_with_input_para(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, rnn_W, rnn_b, rnn_W_r, rnn_b_r, dim):
        self.input = debug_print(input.transpose(1,0), 'self.input') #iterate over first dim
        self.rnn_W=rnn_W
        self.b = rnn_b

        self.Wr = rnn_W_r
        self.b_r = rnn_b_r
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(dim,
                                dtype=theano.config.floatX))
        self.h0_r = theano.shared(name='h0',
                                value=numpy.zeros(dim,
                                dtype=theano.config.floatX))
        def recurrence(x_t, h_tm1):
            concate=T.concatenate([x_t,h_tm1], axis=0)
#             w_t = T.nnet.sigmoid(T.dot(x_t, self.Wxh)
#                                  + T.dot(h_tm1, self.Whh) + self.b)
            w_t = T.nnet.sigmoid(T.dot(concate, self.rnn_W) + self.b)
            h_t=h_tm1*w_t+x_t*(1-w_t)
#             s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return h_t

        h, _ = theano.scan(fn=recurrence,
                                sequences=self.input,
                                outputs_info=self.h0,#[self.h0, None],
                                n_steps=self.input.shape[0])
        self.output_one=debug_print(h.reshape((self.input.shape[0], self.input.shape[1])).transpose(1,0), 'self.output_one')
        #reverse direction
        self.input_two=debug_print(input[:,::-1].transpose(1,0), 'self.input_two')
        def recurrence_r(x_t_r, h_tm1_r):
            concate=T.concatenate([x_t_r,h_tm1_r], axis=0)
#             w_t = T.nnet.sigmoid(T.dot(x_t, self.Wxh)
#                                  + T.dot(h_tm1, self.Whh) + self.b)
            w_t = T.nnet.sigmoid(T.dot(concate, self.Wr) + self.b_r)
#             h_t=h_tm1*w_t+x_t*(1-w_t)
# #             s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
#
#
#             w_t = T.nnet.sigmoid(T.dot(x_t_r, self.Wxh_r)
#                                  + T.dot(h_tm1_r, self.Whh_r) + self.b_r)
            h_t=h_tm1_r*w_t+x_t_r*(1-w_t)
#             s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return h_t
        h_r, _ = theano.scan(fn=recurrence_r,
                                sequences=self.input_two,
                                outputs_info=self.h0_r,#[self.h0, None],
                                n_steps=self.input_two.shape[0])
        self.output_two=debug_print(h_r.reshape((self.input_two.shape[0], self.input_two.shape[1])).transpose(1,0)[:,::-1], 'self.output_two')
        self.output=debug_print(self.output_one+self.output_two, 'self.output')
#         # store parameters of this layer
#         self.params = [self.Whh, self.Wxh, self.b]
class Conv_with_input_para_one_col_featuremap(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, W, b):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = W
        self.b = b

        input=debug_print(input, 'input_Conv_with_input_para_one_col_featuremap')
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='full')    #here, we should pad enough zero padding for input

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        conv_out=debug_print(conv_out, 'conv_out')
        conv_with_bias = debug_print(T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x')), 'conv_with_bias')
        posi=conv_with_bias.shape[2]/2
        conv_with_bias=conv_with_bias[:,:,posi:(posi+1),:]
        wide_conv_out=debug_print(conv_with_bias.reshape((image_shape[0], 1, filter_shape[0], image_shape[3]+filter_shape[3]-1)), 'wide_conv_out') #(batch, 1, kernerl, ishape[1]+filter_size1[1]-1)


        self.output_tensor = debug_print(wide_conv_out, 'self.output_tensor')
        self.output_matrix=debug_print(wide_conv_out.reshape((filter_shape[0], image_shape[3]+filter_shape[3]-1)), 'self.output_matrix')
        self.output_sent_rep_Dlevel=debug_print(T.max(self.output_matrix, axis=1), 'self.output_sent_rep_Dlevel')


        # store parameters of this layer
        self.params = [self.W, self.b]


class Conv(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='valid')    #here, we should pad enough zero padding for input

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        conv_with_bias = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        narrow_conv_out=conv_with_bias.reshape((image_shape[0], 1, filter_shape[0], image_shape[3]-filter_shape[3]+1)) #(batch, 1, kernerl, ishape[1]-filter_size1[1]+1)

        #pad filter_size-1 zero embeddings at both sides
        left_padding = 1e-20+T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        right_padding = 1e-20+T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        self.output = T.concatenate([left_padding, narrow_conv_out, right_padding], axis=3)


        # store parameters of this layer
        self.params = [self.W, self.b]

class Average_Pooling_for_Top(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_l, input_r, kern, left_l, right_l, left_r, right_r, length_l, length_r, dim, topk): # length_l, length_r: valid lengths after conv
#     layer3_DQ=Average_Pooling_for_Top(rng, input_l=layer2_DQ.output, input_r=layer2_Q.output_sent_rep_Dlevel, kern=nkerns[1],
#                      left_l=left_D, right_l=right_D, left_r=0, right_r=0,
#                       length_l=len_D+filter_sents[1]-1, length_r=1,
#                        dim=maxDocLength+filter_sents[1]-1, topk=3)


        fan_in = kern #kern numbers
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = kern
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
            dtype=theano.config.floatX),
                               borrow=True) #a weight matrix kern*kern

        input_r_matrix=debug_print(input_r,'input_r_matrix')

        input_l_matrix=debug_print(input_l.reshape((input_l.shape[2], input_l.shape[3])), 'origin_input_l_matrix')
        input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')



        simi_matrix=compute_simi_feature_matrix_with_column(input_l_matrix, input_r_matrix, length_l, 1, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, length_l)),'simi_question')

        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=input_l_matrix.transpose(1,0)[jj].reshape((topk, input_l_matrix.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))

        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))

        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)

        #with attention
#         output_D_doc_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'output_D_doc_level_rep') # is a column now
        output_D_doc_level_rep=debug_print(T.max(sub_matrix, axis=0), 'output_D_doc_level_rep') # is a column now
        self.output_D_doc_level_rep=output_D_doc_level_rep



        self.params = [self.W]



class Average_Pooling(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_D, input_r, kern, left_D, right_D,left_D_s, right_D_s, left_r, right_r, length_D_s, length_r, dim, doc_len, topk): # length_l, length_r: valid lengths after conv
#     layer1_DQ=Average_Pooling(rng, input_l=layer0_D_output, input_r=layer0_Q_output, kern=nkerns[0],
#                                       left_D=left_D, right_D=right_D,
#                      left_l=left_D_s, right_l=right_D_s, left_r=left_Q, right_r=right_Q,
#                       length_l=len_D_s+filter_words[1]-1, length_r=len_Q+filter_words[1]-1,
#                        dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)


        fan_in = kern #kern numbers
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = kern
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
            dtype=theano.config.floatX),
                               borrow=True) #a weight matrix kern*kern
        input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')
        input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')
        valid_D_s=[]
        for i in range(left_D, doc_len-right_D): # only consider valid sentences in doc
            input_l=input_D[i,:,:,:] # order-3 tensor
            left_l=left_D_s[i]
            right_l=right_D_s[i]
            length_l=length_D_s[i]


            input_l_matrix=debug_print(input_l.reshape((input_D.shape[2], input_D.shape[3])), 'origin_input_l_matrix')
            input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')



            simi_tensor=compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, self.W, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
            simi_question=debug_print(T.max(simi_tensor, axis=1).reshape((1, length_l)),'simi_question')

            neighborsArgSorted = T.argsort(simi_question, axis=1)
            kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
            kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
            jj = kNeighborsArgSorted.flatten()
            sub_matrix=input_l_matrix.transpose(1,0)[jj].reshape((topk, input_l_matrix.shape[0]))
            sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))

            sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
            #weights_answer=simi_answer/T.sum(simi_answer)
            #concate=T.concatenate([weights_question, weights_answer], axis=1)
            #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))

            sub_weights=T.repeat(sub_weights, kern, axis=1)
            #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)

            #with attention
            dot_l=debug_print(T.sum(sub_matrix*sub_weights, axis=0).transpose(1,0), 'dot_l') # is a column now
            valid_D_s.append(dot_l)
            #dot_r=debug_print(T.sum(input_r_matrix*weights_answer_matrix, axis=1),'dot_r')
            '''
            #without attention
            dot_l=debug_print(T.sum(input_l_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
            dot_r=debug_print(T.sum(input_r_matrix, axis=1),'dot_r')
            '''
            '''
            #with attention, then max pooling
            dot_l=debug_print(T.max(input_l_matrix*weights_question_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
            dot_r=debug_print(T.max(input_r_matrix*weights_answer_matrix, axis=1),'dot_r')
            '''
            #norm_l=debug_print(T.sqrt((dot_l**2).sum()),'norm_l')
            #norm_r=debug_print(T.sqrt((dot_r**2).sum()), 'norm_r')

            #self.output_vector_l=debug_print((dot_l/norm_l).reshape((1, kern)),'output_vector_l')
            #self.output_vector_r=debug_print((dot_r/norm_r).reshape((1, kern)), 'output_vector_r')
        valid_matrix=T.concatenate(valid_D_s, axis=1)
        left_padding = T.zeros((input_l_matrix.shape[0], left_D), dtype=theano.config.floatX)
        right_padding = T.zeros((input_l_matrix.shape[0], right_D), dtype=theano.config.floatX)
        matrix_padded = T.concatenate([left_padding, valid_matrix, right_padding], axis=1)
        self.output_D=matrix_padded
        self.output_D_valid_part=valid_matrix
        self.output_QA_sent_level_rep=T.mean(input_r_matrix, axis=1)

        #now, average pooling by comparing self.output_QA and self.output_D_valid_part
        simi_matrix=compute_simi_feature_matrix_with_column(self.output_D_valid_part, self.output_QA, doc_len-left_D-right_D, 1, doc_len) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, doc_len-left_D-right_D)),'simi_question')

        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=self.output_D_valid_part.transpose(1,0)[jj].reshape((topk, self.output_D_valid_part.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))

        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))

        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)

        #with attention
        output_D_sent_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0).transpose(1,0), 'output_D_sent_level_rep') # is a column now
        self.output_D_sent_level_rep=output_D_sent_level_rep



        self.params = [self.W]

class Average_Pooling_Scan(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_D, input_r, kern, left_D, right_D,left_D_s, right_D_s, left_r, right_r, length_D_s, length_r, dim, doc_len, topk): # length_l, length_r: valid lengths after conv
#     layer1_DQ=Average_Pooling(rng, input_l=layer0_D_output, input_r=layer0_Q_output, kern=nkerns[0],
#                                       left_D=left_D, right_D=right_D,
#                      left_l=left_D_s, right_l=right_D_s, left_r=left_Q, right_r=right_Q,
#                       length_l=len_D_s+filter_words[1]-1, length_r=len_Q+filter_words[1]-1,
#                        dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)


        fan_in = kern #kern numbers
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = kern
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
            dtype=theano.config.floatX),
                               borrow=True) #a weight matrix kern*kern

#         input_tensor_l=T.dtensor4("input_tensor_l")
#         input_tensor_r=T.dtensor4("input_tensor_r")
#         kern_scan=T.lscalar("kern_scan")
#         length_D_s_scan=T.lvector("length_D_s_scan")
#         left_D_s_scan=T.lvector("left_D_s_scan")
#         right_D_s_scan=T.lvector("right_D_s_scan")
#         length_r_scan=T.lscalar("length_r_scan")
#         left_r_scan=T.lscalar("left_r_scan")
#         right_r_scan=T.lscalar("right_r_scan")
#         dim_scan=T.lscalar("dim_scan")
#         topk_scan=T.lscalar("topk_scan")



        def sub_operation(input_l, length_l, left_l, right_l, input_r, kernn , length_r, left_r, right_r, dim, topk):
            input_l_matrix=debug_print(input_l.reshape((input_l.shape[1], input_l.shape[2])), 'origin_input_l_matrix')#input_l should be order3 tensor now
            input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')
#             input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')#input_r should be order4 tensor still
#             input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')
#
#
#             simi_tensor=compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, self.W, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
#             simi_question=debug_print(T.max(simi_tensor, axis=1).reshape((1, length_l)),'simi_question')
#
#             neighborsArgSorted = T.argsort(simi_question, axis=1)
#             kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
#             kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
#             jj = kNeighborsArgSorted.flatten()
#             sub_matrix=input_l_matrix.transpose(1,0)[jj].reshape((topk, input_l_matrix.shape[0]))
#             sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))
#
#             sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
#             sub_weights=T.repeat(sub_weights, kernn, axis=1)
#             dot_l=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'dot_l') # is a column now
#             dot_l=T.max(sub_matrix, axis=0)
            dot_l=debug_print(T.max(input_l_matrix, axis=1), 'dot_l') # max pooling
            return dot_l



#         results, updates = theano.scan(fn=sub_operation,
#                                        outputs_info=None,
#                                        sequences=[input_tensor_l, length_D_s_scan, left_D_s_scan, right_D_s_scan],
#                                        non_sequences=[input_tensor_r, kern_scan, length_r_scan, left_r_scan, right_r_scan, dim_scan, topk_scan])

        results, updates = theano.scan(fn=sub_operation,
                                       outputs_info=None,
                                       sequences=[input_D[left_D:doc_len-right_D], length_D_s[left_D: doc_len-right_D], left_D_s[left_D: doc_len-right_D], right_D_s[left_D: doc_len-right_D]],
                                       non_sequences=[input_r, kern, length_r, left_r, right_r, dim, topk])

#         scan_function = theano.function(inputs=[input_tensor_l, input_tensor_r, kern_scan, length_D_s_scan, left_D_s_scan, right_D_s_scan, length_r_scan, left_r_scan, right_r_scan, dim_scan, topk_scan],
#                                         outputs=results,
#                                         updates=updates)
#
#
#
#         sents=scan_function(input_D[left_D:doc_len-right_D], input_r, kern,
#                             length_D_s[left_D: doc_len-right_D], left_D_s[left_D: doc_len-right_D], right_D_s[left_D: doc_len-right_D],
#                             length_r,
#                             left_r,
#                             right_r,
#                             dim,
#                             topk)
        sents=results
        input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')
        input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')


        valid_matrix=debug_print(sents.transpose(1,0), 'valid_matrix')
        left_padding = T.zeros((input_D.shape[2], left_D), dtype=theano.config.floatX)
        right_padding = T.zeros((input_D.shape[2], right_D), dtype=theano.config.floatX)
        matrix_padded = T.concatenate([left_padding, valid_matrix, right_padding], axis=1)
        self.output_D=matrix_padded   #it shows the second conv for doc has input of all sentences
        self.output_D_valid_part=valid_matrix
        self.output_QA_sent_level_rep=T.max(input_r_matrix, axis=1)

        #now, average pooling by comparing self.output_QA and self.output_D_valid_part, choose one key sentence
        topk=1
        simi_matrix=debug_print(compute_simi_feature_matrix_with_column(self.output_D_valid_part, self.output_QA_sent_level_rep, doc_len-left_D-right_D, 1, doc_len), 'simi_matrix_matrix_with_column') #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, doc_len-left_D-right_D)),'simi_question')

        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=self.output_D_valid_part.transpose(1,0)[jj].reshape((topk, self.output_D_valid_part.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))

        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))

        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)

        #with attention
#         output_D_sent_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'output_D_sent_level_rep') # is a column now
        output_D_sent_level_rep=debug_print(T.max(sub_matrix, axis=0), 'output_D_sent_level_rep') # is a column now
        self.output_D_sent_level_rep=output_D_sent_level_rep



        self.params = [self.W]

class GRU_Average_Pooling_Scan(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_D, input_r, kern, left_D, right_D, dim, doc_len, topk): # length_l, length_r: valid lengths after conv
#     layer1_DQ=Average_Pooling(rng, input_l=layer0_D_output, input_r=layer0_Q_output, kern=nkerns[0],
#                                       left_D=left_D, right_D=right_D,
#                      left_l=left_D_s, right_l=right_D_s, left_r=left_Q, right_r=right_Q,
#                       length_l=len_D_s+filter_words[1]-1, length_r=len_Q+filter_words[1]-1,
#                        dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)


#         fan_in = kern #kern numbers
#         # each unit in the lower layer receives a gradient from:
#         # "num output feature maps * filter height * filter width" /
#         #   pooling size
#         fan_out = kern
#         # initialize weights with random weights
#         W_bound = numpy.sqrt(6. / (fan_in + fan_out))
#         self.W = theano.shared(numpy.asarray(
#             rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
#             dtype=theano.config.floatX),
#                                borrow=True) #a weight matrix kern*kern

#         input_tensor_l=T.dtensor4("input_tensor_l")
#         input_tensor_r=T.dtensor4("input_tensor_r")
#         kern_scan=T.lscalar("kern_scan")
#         length_D_s_scan=T.lvector("length_D_s_scan")
#         left_D_s_scan=T.lvector("left_D_s_scan")
#         right_D_s_scan=T.lvector("right_D_s_scan")
#         length_r_scan=T.lscalar("length_r_scan")
#         left_r_scan=T.lscalar("left_r_scan")
#         right_r_scan=T.lscalar("right_r_scan")
#         dim_scan=T.lscalar("dim_scan")
#         topk_scan=T.lscalar("topk_scan")



#         def sub_operation(input_l, length_l, left_l, right_l, input_r, kernn , length_r, left_r, right_r, dim, topk):
#             input_l_matrix=debug_print(input_l.reshape((input_l.shape[1], input_l.shape[2])), 'origin_input_l_matrix')#input_l should be order3 tensor now
#             input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')
# #             input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')#input_r should be order4 tensor still
# #             input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')
# #
# #
# #             simi_tensor=compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, self.W, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
# #             simi_question=debug_print(T.max(simi_tensor, axis=1).reshape((1, length_l)),'simi_question')
# #
# #             neighborsArgSorted = T.argsort(simi_question, axis=1)
# #             kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
# #             kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
# #             jj = kNeighborsArgSorted.flatten()
# #             sub_matrix=input_l_matrix.transpose(1,0)[jj].reshape((topk, input_l_matrix.shape[0]))
# #             sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))
# #
# #             sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
# #             sub_weights=T.repeat(sub_weights, kernn, axis=1)
# #             dot_l=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'dot_l') # is a column now
# #             dot_l=T.max(sub_matrix, axis=0)
#             dot_l=debug_print(T.max(input_l_matrix, axis=1), 'dot_l') # max pooling
#             return dot_l
#
#
#
# #         results, updates = theano.scan(fn=sub_operation,
# #                                        outputs_info=None,
# #                                        sequences=[input_tensor_l, length_D_s_scan, left_D_s_scan, right_D_s_scan],
# #                                        non_sequences=[input_tensor_r, kern_scan, length_r_scan, left_r_scan, right_r_scan, dim_scan, topk_scan])
#
#         results, updates = theano.scan(fn=sub_operation,
#                                        outputs_info=None,
#                                        sequences=[input_D[left_D:doc_len-right_D], length_D_s[left_D: doc_len-right_D], left_D_s[left_D: doc_len-right_D], right_D_s[left_D: doc_len-right_D]],
#                                        non_sequences=[input_r, kern, length_r, left_r, right_r, dim, topk])

#         scan_function = theano.function(inputs=[input_tensor_l, input_tensor_r, kern_scan, length_D_s_scan, left_D_s_scan, right_D_s_scan, length_r_scan, left_r_scan, right_r_scan, dim_scan, topk_scan],
#                                         outputs=results,
#                                         updates=updates)
#
#
#
#         sents=scan_function(input_D[left_D:doc_len-right_D], input_r, kern,
#                             length_D_s[left_D: doc_len-right_D], left_D_s[left_D: doc_len-right_D], right_D_s[left_D: doc_len-right_D],
#                             length_r,
#                             left_r,
#                             right_r,
#                             dim,
#                             topk)
#         sents=results
#         input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')
#         input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')


        valid_matrix=debug_print(input_D, 'valid_matrix')
        left_padding = T.zeros((input_D.shape[0], left_D), dtype=theano.config.floatX)
        right_padding = T.zeros((input_D.shape[0], right_D), dtype=theano.config.floatX)
        matrix_padded = T.concatenate([left_padding, valid_matrix, right_padding], axis=1)
        self.output_D=matrix_padded   #it shows the second conv for doc has input of all sentences
        self.output_D_valid_part=valid_matrix
        self.output_QA_sent_level_rep=input_r

        #now, average pooling by comparing self.output_QA and self.output_D_valid_part, choose one key sentence
        topk=1
        simi_matrix=debug_print(compute_simi_feature_matrix_with_column(self.output_D_valid_part, self.output_QA_sent_level_rep, doc_len-left_D-right_D, 1, doc_len), 'simi_matrix_matrix_with_column') #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, doc_len-left_D-right_D)),'simi_question')

        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=self.output_D_valid_part.transpose(1,0)[jj].reshape((topk, self.output_D_valid_part.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))

        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))

        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)

        #with attention
#         output_D_sent_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'output_D_sent_level_rep') # is a column now
        output_D_sent_level_rep=debug_print(T.max(sub_matrix, axis=0), 'output_D_sent_level_rep') # is a column now
        self.output_D_sent_level_rep=output_D_sent_level_rep



#         self.params = [self.W]

# def drop(input, p, rng):
#     """
#     :type input: numpy.array
#     :param input: layer or weight matrix on which dropout resp. dropconnect is applied
#
#     :type p: float or double between 0. and 1.
#     :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.
#
#     """
#     srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
#     mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
#     return input * mask

def dropit(srng, weight, drop):
    if T.gt(drop,0.0):#drop>0.0:
        # proportion of probability to retain
        retain_prob = 1 - drop
        mask = (srng.binomial(n=1, p=retain_prob, size=weight.shape, dtype='floatX'))/retain_prob #inverted dropout
        return T.cast(weight * mask, theano.config.floatX)
    else:
        return weight

def dont_dropit(weight, drop):
    # return (1 - drop)*T.cast(weight, theano.config.floatX)
    return weight  #do nothing in testing in inverted dropout

def dropout_layer(srng,weight, drop, train):
    result = theano.ifelse.ifelse(T.eq(train, 1),
                                    dropit(srng, weight, drop),
                                    dont_dropit(weight, drop))
    return result

class Average_Pooling_RNN(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_D, input_r, kern, left_D, right_D,doc_len, topk): # length_l, length_r: valid lengths after conv
#     layer1_DQ=Average_Pooling(rng, input_l=layer0_D_output, input_r=layer0_Q_output, kern=nkerns[0],
#                                       left_D=left_D, right_D=right_D,
#                      left_l=left_D_s, right_l=right_D_s, left_r=left_Q, right_r=right_Q,
#                       length_l=len_D_s+filter_words[1]-1, length_r=len_Q+filter_words[1]-1,
#                        dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)





        self.output_D_valid_part=input_D
        self.output_QA_sent_level_rep=input_r

        #now, average pooling by comparing self.output_QA and self.output_D_valid_part, choose one key sentence
        topk=1
        simi_matrix=debug_print(compute_simi_feature_matrix_with_column(self.output_D_valid_part, self.output_QA_sent_level_rep, doc_len-left_D-right_D, 1, doc_len), 'simi_matrix_matrix_with_column') #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, doc_len-left_D-right_D)),'simi_question')

        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=self.output_D_valid_part.transpose(1,0)[jj].reshape((topk, self.output_D_valid_part.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))

        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))

        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)

        #with attention
#         output_D_sent_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'output_D_sent_level_rep') # is a column now
        output_D_rep=debug_print(T.max(sub_matrix, axis=0), 'output_D_rep') # is a column now
        self.output_D_sent_level_rep=output_D_rep


def compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, para_matrix, dim):
    #matrix_r_after_translate=debug_print(T.dot(para_matrix, input_r_matrix), 'matrix_r_after_translate')
    matrix_r_after_translate=input_r_matrix

    input_l_tensor=input_l_matrix.dimshuffle('x',0,1)
    input_l_tensor=T.repeat(input_l_tensor, dim, axis=0)[:length_r,:,:]
    input_l_tensor=input_l_tensor.dimshuffle(2,1,0).dimshuffle(0,2,1)
    repeated_1=input_l_tensor.reshape((length_l*length_r, input_l_matrix.shape[0])).dimshuffle(1,0)

    input_r_tensor=matrix_r_after_translate.dimshuffle('x',0,1)
    input_r_tensor=T.repeat(input_r_tensor, dim, axis=0)[:length_l,:,:]
    input_r_tensor=input_r_tensor.dimshuffle(0,2,1)
    repeated_2=input_r_tensor.reshape((length_l*length_r, matrix_r_after_translate.shape[0])).dimshuffle(1,0)



    #cosine attention
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=0)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')

    multi=debug_print(repeated_1*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')

    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
    simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')


#     #euclid, effective for wikiQA
#     gap=debug_print(repeated_1-repeated_2, 'gap')
#     eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
#     simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((length_l, length_r)), 'simi_matrix')


    return simi_matrix#[:length_l, :length_r]

def compute_simi_feature_matrix_with_matrix(input_l_matrix, input_r_matrix, length_l, length_r, dim):
    #this function is the same with "compute_simi_feature_batch1_new", except that this has no input parameters
    matrix_r_after_translate=input_r_matrix

    input_l_tensor=input_l_matrix.dimshuffle('x',0,1)
    input_l_tensor=T.repeat(input_l_tensor, dim, axis=0)[:length_r,:,:]
    input_l_tensor=input_l_tensor.dimshuffle(2,1,0).dimshuffle(0,2,1)
    repeated_1=input_l_tensor.reshape((length_l*length_r, input_l_matrix.shape[0])).dimshuffle(1,0)

    input_r_tensor=matrix_r_after_translate.dimshuffle('x',0,1)
    input_r_tensor=T.repeat(input_r_tensor, dim, axis=0)[:length_l,:,:]
    input_r_tensor=input_r_tensor.dimshuffle(0,2,1)
    repeated_2=input_r_tensor.reshape((length_l*length_r, matrix_r_after_translate.shape[0])).dimshuffle(1,0)



    #cosine attention
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=0)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')

    multi=debug_print(repeated_1*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')

    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
    simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')


#     #euclid, effective for wikiQA
#     gap=debug_print(repeated_1-repeated_2, 'gap')
#     eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
#     simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((length_l, length_r)), 'simi_matrix')


    return simi_matrix#[:length_l, :length_r]

def compute_attention_feature_matrix_with_matrix(input_l_matrix, input_r_matrix, length_l, length_r, dim, W1, W2, w):
    #this function is the same with "compute_simi_feature_batch1_new", except that this has no input parameters
    matrix_r_after_translate=input_r_matrix

    input_l_tensor=input_l_matrix.dimshuffle('x',0,1)
    input_l_tensor=T.repeat(input_l_tensor, dim, axis=0)[:length_r,:,:]
    input_l_tensor=input_l_tensor.dimshuffle(2,1,0).dimshuffle(0,2,1)
    repeated_1=input_l_tensor.reshape((length_l*length_r, input_l_matrix.shape[0])).dimshuffle(1,0)

    input_r_tensor=matrix_r_after_translate.dimshuffle('x',0,1)
    input_r_tensor=T.repeat(input_r_tensor, dim, axis=0)[:length_l,:,:]
    input_r_tensor=input_r_tensor.dimshuffle(0,2,1)
    repeated_2=input_r_tensor.reshape((length_l*length_r, matrix_r_after_translate.shape[0])).dimshuffle(1,0)

    proj_1=W1.dot(repeated_1)
    proj_2=W2.dot(repeated_2)

    attentions=T.tanh(w.dot(proj_1+proj_2))
    attention_matrix=attentions.reshape((length_l, length_r))
#     #cosine attention
#     length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=0)),'length_1')
#     length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')
#
#     multi=debug_print(repeated_1*repeated_2, 'multi')
#     sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')
#
#     list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
#     simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')


#     #euclid, effective for wikiQA
#     gap=debug_print(repeated_1-repeated_2, 'gap')
#     eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
#     simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((length_l, length_r)), 'simi_matrix')


    return attention_matrix#[:length_l, :length_r]

def compute_simi_feature_matrix_with_column(input_l_matrix, column, length_l, length_r, dim):
    column=column.reshape((column.shape[0],1))
    repeated_2=T.repeat(column, dim, axis=1)[:,:length_l]



    #cosine attention
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(input_l_matrix), axis=0)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')

    multi=debug_print(input_l_matrix*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')

    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
    simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')


#     #euclid, effective for wikiQA
#     gap=debug_print(input_l_matrix-repeated_2, 'gap')
#     eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
#     simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((length_l, length_r)), 'simi_matrix')


    return simi_matrix#[:length_l, :length_r]

def cosine_row_wise_twoMatrix(M1, M2):
    #assume both (batch, hidden))
    dot=T.sum(M1*M2, axis=1) #(batch)
    norm1=T.sqrt(T.sum(M1**2, axis=1))
    norm2=T.sqrt(T.sum(M2**2, axis=1))
    return dot/(norm1*norm2)

def compute_acc(label_list, scores_list):
    #label_list contains 0/1, 500 as a minibatch, score_list contains score between -1 and 1, 500 as a minibatch
    if len(label_list)%500!=0 or len(scores_list)%500!=0:
        print 'len(label_list)%500: ', len(label_list)%500, ' len(scores_list)%500: ', len(scores_list)%500
        exit(0)
    if len(label_list)!=len(scores_list):
        print 'len(label_list)!=len(scores_list)', len(label_list), ' and ',len(scores_list)
        exit(0)
    correct_count=0
    total_examples=len(label_list)/500
    start_posi=range(total_examples)*500
    for i in start_posi:
        set_1=set()

        for scan in range(i, i+500):
            if label_list[scan]==1:
                set_1.add(scan)
        set_0=set(range(i, i+500))-set_1
        flag=True
        for zero_posi in set_0:
            for scan in set_1:
                if scores_list[zero_posi]> scores_list[scan]:
                    flag=False
        if flag==True:
            correct_count+=1

    return correct_count*1.0/total_examples
#def unify_eachone(tensor, left1, right1, left2, right2, dim, Np):

def Diversify_Reg(W):
    # (L, feature size), e.g., (output_size, input_size)
    loss=(T.nnet.sigmoid(W.dot(W.T)-T.eye(n=W.shape[0], m=W.shape[0], k=0, dtype=theano.config.floatX))**2).mean()
    return loss

def Determinant(W):
    prod=W.dot(W.T)
    loss=-T.log(T.nlinalg.Det()(prod))
    return loss

def normalize_matrix(M):
    norm=T.sqrt(1e-8+T.sum(T.sqr(M)))
    return M/norm

def normalize_matrix_row_wise(M):
    norm=T.sqrt(1e-8+T.sum(T.sqr(M), axis=1))
    return M/norm.dimshuffle(0,'x')
def normalize_matrix_col_wise(M):
    norm=T.sqrt(1e-8+T.sum(T.sqr(M), axis=0))
    return M/norm.dimshuffle('x',0)
def L2norm_paraList(params):
    sum=0.0
    for x in params:
        sum+=(x**2).sum()
    return sum

# def L2norm_paraList(paralist):
#     summ=0.0
#
#     for para in paralist:
#         summ+=(para** 2).mean()
#     return summ
def constant_param(value=0.0, shape=(0,)):
#     return theano.shared(lasagne.init.Constant(value).sample(shape), borrow=True)
    return theano.shared(numpy.full(shape, value, dtype=theano.config.floatX), borrow=True)


def normal_param(std=0.1, mean=0.0, shape=(0,)):
#     return theano.shared(lasagne.init.Normal(std, mean).sample(shape), borrow=True)
    U=numpy.random.normal(mean, std, shape)
    return theano.shared(name='U', value=U.astype(theano.config.floatX), borrow=True)
def cosine_simi(x, y):
    #this is better
    a = np.array(x)
    b = np.array(y)
    c = 1-cosine(a,b)
    return c

class Conv_then_GRU_then_Classify(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, concate_paragraph_input, Qs_emb, para_len_limit, q_len_limit, input_h_size_1, output_h_size, input_h_size_2, conv_width, batch_size, para_mask, q_mask, labels, layer_scalar):
        self.paragraph_para, self.paragraph_conv_output, self.paragraph_gru_output_tensor, self.paragraph_gru_output_reps=conv_then_gru(rng, concate_paragraph_input, output_h_size, input_h_size_1, conv_width, batch_size, para_len_limit, para_mask)

        self.q_para, self.q_conv_output, self.q_gru_output_tensor, self.q_gru_output_reps=conv_then_gru(rng, Qs_emb, output_h_size, input_h_size_2, conv_width, batch_size, q_len_limit, q_mask)

        LR_mask=para_mask[:,:-1]*para_mask[:,1:]
        self.classify_para, self.error, self.masked_dis=combine_for_LR(rng, output_h_size, self.paragraph_gru_output_tensor, self.q_gru_output_reps, LR_mask, batch_size, labels)
        self.masked_dis_inprediction=self.masked_dis*T.sqrt(layer_scalar)
        self.paras=self.paragraph_para+self.q_para+self.classify_para

def conv_then_gru(rng, input_tensor3, out_h_size, in_h_size, conv_width, batch_size, size_last_dim, mask):
    conv_input=input_tensor3.dimshuffle((0,'x', 1,2)) #(batch_size, 1, emb+3, maxparalen)
    conv_W, conv_b=create_conv_para(rng, filter_shape=(out_h_size, 1, in_h_size, conv_width))
    conv_para=[conv_W, conv_b]
    conv_model = Conv_with_input_para(rng, input=conv_input,
            image_shape=(batch_size, 1, in_h_size, size_last_dim),
            filter_shape=(out_h_size, 1, in_h_size, conv_width), W=conv_W, b=conv_b)
    conv_output=conv_model.narrow_conv_out #(batch, 1, hidden_size, maxparalen-1)

    U, W, b=create_GRU_para(rng, out_h_size, out_h_size)
    U_b, W_b, b_b=create_GRU_para(rng, out_h_size, out_h_size)
    gru_para=[U, W, b, U_b, W_b, b_b]
    gru_input=conv_output.reshape((conv_output.shape[0], conv_output.shape[2], conv_output.shape[3]))
    gru_mask=mask[:,:-1]*mask[:,1:]
    gru_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=gru_input, Mask=gru_mask, hidden_dim=out_h_size,U=U,W=W,b=b,Ub=U_b,Wb=W_b,bb=b_b)
    gru_output_tensor=gru_model.output_tensor #(batch, hidden_dim, para_len-1)
    gru_output_reps=gru_model.output_sent_rep_maxpooling.reshape((batch_size, 1, out_h_size)) #(batch, 2*out_size)

    overall_para= conv_para + gru_para
    return overall_para, conv_output, gru_output_tensor, gru_output_reps
def combine_for_LR(rng, hidden_size, para_reps, questions_reps, para_mask, batch_size, labels):
    #combine, then classify
    W_a1 = create_ensemble_para(rng, hidden_size, hidden_size)# init_weights((2*hidden_size, hidden_size))
    W_a2 = create_ensemble_para(rng, hidden_size, hidden_size)
    U_a = create_ensemble_para(rng, 2, hidden_size) # 3 extra features

    norm_W_a1=normalize_matrix(W_a1)
    norm_W_a2=normalize_matrix(W_a2)
    norm_U_a=normalize_matrix(U_a)

    LR_b = theano.shared(value=numpy.zeros((2,),
                                                 dtype=theano.config.floatX),  # @UndefinedVariable
                               name='LR_b', borrow=True)

    attention_paras=[W_a1, W_a2, U_a, LR_b]

    transformed_para_reps=T.tanh(T.dot(para_reps.transpose((0, 2,1)), norm_W_a2))
    transformed_q_reps=T.tanh(T.dot(questions_reps, norm_W_a1))
    #transformed_q_reps=T.repeat(transformed_q_reps, transformed_para_reps.shape[1], axis=1)

    add_both=0.5*(transformed_para_reps+transformed_q_reps)

    prior_att=add_both
    combined_size=hidden_size


    #prior_att=T.concatenate([transformed_para_reps, transformed_q_reps], axis=2)
    valid_indices=para_mask.flatten().nonzero()[0]

    layer3=LogisticRegression(rng, input=prior_att.reshape((batch_size*prior_att.shape[1], combined_size)), n_in=combined_size, n_out=2, W=norm_U_a, b=LR_b)
    #error =layer3.negative_log_likelihood(labels.flatten()[valid_indices])
    error = -T.mean(T.log(layer3.p_y_given_x)[valid_indices, labels.flatten()[valid_indices]])#[T.arange(y.shape[0]), y])

    distributions=layer3.p_y_given_x[:,-1].reshape((batch_size, para_mask.shape[1]))
    #distributions=layer3.y_pred.reshape((batch_size, para_mask.shape[1]))
    masked_dis=distributions*para_mask
    return  attention_paras, error, masked_dis

def cosine_matrix1_matrix2_rowwise(M1, M2):
    #assume both matrix are in shape (batch, hidden)
    dot_prod=T.sum(M1*M2, axis=1) #batch
    norm1=T.sqrt(1e-20+T.sum(M1**2,axis=1)) #batch
    norm2=T.sqrt(1e-20+T.sum(M2**2,axis=1)) #batch
    return dot_prod/(norm1*norm2+1e-20)
def cosine_tensors(tensor1, tensor2):
    #assume tensor in shape (batch, hidden, len))
    dot_prod=T.sum(tensor1*tensor2, axis=1) #(batch, len)
    norm_1=T.sqrt(1e-20+T.sum(tensor1**2, axis=1)) #(batch, len)
    norm_2=T.sqrt(1e-20+T.sum(tensor2**2, axis=1))
    return dot_prod/(1e-20+norm_1*norm_2)#(batch, len)

def BatchMatchMatrix_between_2tensors(tensor1, tensor2):
    #assume both are (batch, hidden ,para_len), (batch, hidden ,q_len)
    def example_in_batch(para_matrix, q_matrix):
        #assume both are (hidden, para_len),  (hidden, q_len)
        transpose_para_matrix=para_matrix.T  #(para_len, hidden)
        interaction_matrix=T.dot(transpose_para_matrix, q_matrix) #(para_len, q_len)
        return interaction_matrix
    batch_matrix,_ = theano.scan(fn=example_in_batch,
                                   outputs_info=None,
                                   sequences=[tensor1, tensor2])    #batch_q_reps (batch, hidden, para_len)
    return batch_matrix #(batch, para_len, q_len)

def load_model_from_file(file_path, params):
#     print 'loading model: ', file_path
    save_file = open(file_path)
#     save_file = open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/Best_Conv_Para_at_22')

    for para in params:
        para.set_value(cPickle.load(save_file), borrow=True)
    print 'model loaded successfully:', file_path
    save_file.close()
def store_model_to_file(file_path, best_params):
    save_file = open(file_path, 'wb')  # this will overwrite current contents
    for para in best_params:
        cPickle.dump(para.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    print 'para stored over...'
    save_file.close()

def add_HLs_2_Matrix(matrix, HL_1_para,HL_2_para,HL_3_para,HL_4_para):
#     HL_input = input4score
    HL_1_output = T.nnet.relu(T.dot(matrix, HL_1_para)) #(batch, p_len, hidden_size)
    HL_2_output = T.nnet.relu(T.dot(HL_1_output, HL_2_para))
    HL_3_output = T.nnet.relu(T.dot(HL_2_output+HL_1_output, HL_3_para))
    HL_4_output = T.nnet.relu(T.dot(HL_3_output+HL_2_output+HL_1_output, HL_4_para))
    return T.concatenate([matrix, HL_1_output, HL_2_output, HL_3_output, HL_4_output], axis=1)

def shuffle_big_list(lis):
    size=len(lis)
    times=10
    group_size = size/times
    sub = range(group_size)
    remain = size%len(sub)
    random.Random(20).shuffle(sub)
    newlis=[]
    ids= range(times)
    random.Random(20).shuffle(ids)
    for i in ids:
        newlis+=[x+i*group_size for x in sub]
    newlis+=lis[-remain:]
    return  newlis

def Gradient_Cost_Para_in_Group(cost_list,params_list,learning_rate):
    group_size = len(cost_list)
    updates=[]
    for i in range(group_size):
        updates_i = Gradient_Cost_Para(cost_list[i],params_list[i],learning_rate)
        updates+=updates_i
    return updates

def Gradient_Cost_Para(cost,params,learning_rate):
#     params = [embeddings]+NN_para+LR_para+HL_layer_1.params+HL_layer_2.params   # put all model parameters together
#     cost=loss#+L2_weight*L2_reg

    grads = T.grad(cost, params)    # create a list of gradients for all model parameters
    accumulator=[]
    for para_i in params:
        eps_p=np.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-8)))   #1e-8 is add to get rid of zero division
        updates.append((acc_i, acc))

    return updates

def Adam(cost, params, lr=0.001, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates
def elementwise_is_zero(mat):
    #if 0 to be 1.0, otherwise 0.0
    a = T.where( mat < 0, 1, mat)
    b = T.where( a > 0, 1, a)
    return  T.cast(1-b, 'int32')

def elementwise_is_two(mat):
    #if 0 to be 1.0, otherwise 0.0
    a = T.where( mat < 2, 1, mat)

    return  T.cast(a-1, 'int32')

def Two_Tensor2_Interact_ConvPool(tensor1, tensor2, mask1, mask2, W2score, conv_W, conv_b, filter_shape, images_shape, poolsize):
    #both tensor in form (batch, hidden, len)
    #pool size, e.g., (1,10)
    re_tensor1 = T.extra_ops.repeat(tensor1, tensor2.shape[2], axis=2) #(batch, hidden, l_len*r_len)
    re_tensor2 = T.tile(tensor2, (1,1,tensor1.shape[2])) #(batch, hidden, r_len*l_len)
    conc_tensor = T.concatenate([re_tensor1, re_tensor2], axis=1) #(batch, 2*hidden, len*len)
    conv_input = T.tanh(conc_tensor.dimshuffle(0,2,1).dot(W2score)).dimshuffle(0,2,1).reshape((tensor1.shape[0],W2score.shape[1], tensor1.shape[2], tensor2.shape[2])) #(batch, kern, l_len, r_len)

    mask = T.batched_dot(mask1.dimshuffle(0,1,'x'), mask2.dimshuffle(0,'x',1)).dimshuffle(0,'x',1,2) #(batch, 1, l_len, r_len)
    conv_input = conv_input*mask

#     conv_input = scores_tensor.dimshuffle(0,'x',1,2) #(batch, 1, l_len, r_len)
    raw_conv_out = conv.conv2d(input=conv_input, filters=conv_W,
                filter_shape=filter_shape, image_shape=images_shape, border_mode='valid')
    conv_out = T.tanh(raw_conv_out + conv_b.dimshuffle('x', 0, 'x', 'x'))
    pool_out = T.signal.pool.pool_2d(input=conv_out, ds=poolsize, ignore_border=True) #(batch, kern, height, width)

    output_matrix = pool_out.reshape((pool_out.shape[0], pool_out.shape[1]*pool_out.shape[2]*pool_out.shape[3]))
#     feature_size =
    return output_matrix

def Two_Tensor3_Conv_Mutually(rng, tensor1, tensor2, mask1, mask2, hidden_size, filter_size, l_len, r_len, b):
    #tensor3 (batch, hidden, len) to filter

    def one_pair(matrix1, matrix2, mask_1, mask_2):
        #mask is a vector
        s_tensor3_1 = matrix1.dimshuffle('x',0,1) #(1, hidden, l_len)
        s_tensor3_2 = matrix2.dimshuffle('x',0,1) #(1, hidden, r_len)
        matrix1_slices=[]
        for i in range(filter_size):
            matrix1_slices.append(matrix1[:,i:i+(matrix1.shape[1]-filter_size+1)].dimshuffle('x',0,1)) #(1, hidden_size, l_len-filter_size+1)
        s1_as_filter = T.concatenate(matrix1_slices, axis=0).dimshuffle(2,'x',1,0) #(l_len-filter+1, 1, emb, filter_size)

        matrix2_slices=[]
        for i in range(filter_size):
            matrix2_slices.append(matrix2[:,i:i+(matrix2.shape[1]-filter_size+1)].dimshuffle('x',0,1))
        s2_as_filter = T.concatenate(matrix2_slices, axis=0).dimshuffle(2,'x',1,0) #(r_len-filter+1, 1, emb, filter_size)

        conv_s1 = Conv_with_Mask(rng, input_tensor3=s_tensor3_1,
         mask_matrix = mask_1.dimshuffle('x',0),
         image_shape=(1, 1, hidden_size, l_len),
         filter_shape=(r_len-filter_size+1, 1, hidden_size, filter_size), W=s2_as_filter, b=b)

        conv_s2 = Conv_with_Mask(rng, input_tensor3=s_tensor3_2,
         mask_matrix = mask_2.dimshuffle('x',0),
         image_shape=(1, 1, hidden_size, r_len),
         filter_shape=(l_len-filter_size+1, 1, hidden_size, filter_size), W=s1_as_filter, b=b)

        return conv_s1.maxpool_vec, conv_s2.maxpool_vec  #(1, new_hidden_size)

    result_pair, updates = theano.scan(fn=one_pair,
                        sequences=[tensor1, tensor2,mask1, mask2])
    batch_l_maxpool = result_pair[0] #(batch, 1, r_len-filter_size+1)
    batch_r_maxpool = result_pair[1] #(batch, l_len-filter_size+1)
    return batch_l_maxpool.reshape((batch_l_maxpool.shape[0],batch_l_maxpool.shape[2])), batch_r_maxpool.reshape((batch_r_maxpool.shape[0],batch_r_maxpool.shape[2]))

def ACNN_entail_OneFilterWidth(rng,common_input_l,common_input_r,sents_mask_l,sents_mask_r,batch_size, emb_size,hidden_size, filter_size,maxSentLen,gate_filter_shape,
                               drop_conv_W_1_pre,conv_b_1_pre,drop_conv_W_1_gate,conv_b_1_gate,drop_conv_W_1,conv_b_1,drop_conv_W_1_context,conv_b_1_context,
                               drop_conv_W_1_pre_2,conv_b_1_pre_2,drop_conv_W_1_gate_2,conv_b_1_gate_2,drop_conv_W_1_2, conv_b_1_2,drop_conv_W_1_context_2,conv_b_1_context_2,
                               drop_conv_W_1_pre_3,conv_b_1_pre_3,drop_conv_W_1_gate_3,conv_b_1_gate_3,drop_conv_W_1_3, conv_b_1_3,drop_conv_W_1_context_3,conv_b_1_context_3,
                               drop_conv_W_1_pre_4,conv_b_1_pre_4,drop_conv_W_1_gate_4,conv_b_1_gate_4,drop_conv_W_1_4, conv_b_1_4,drop_conv_W_1_context_4,conv_b_1_context_4,
                               first_srng,drop_p,train_flag,labels):
    conv_layer_1_gate_l = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_l,
             mask_matrix = sents_mask_l,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=gate_filter_shape,
             W=drop_conv_W_1_pre, b=conv_b_1_pre,
             W_gate =drop_conv_W_1_gate, b_gate=conv_b_1_gate )
    conv_layer_1_gate_r = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_r,
             mask_matrix = sents_mask_r,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=gate_filter_shape,
             W=drop_conv_W_1_pre, b=conv_b_1_pre,
             W_gate =drop_conv_W_1_gate, b_gate=conv_b_1_gate )

    l_input_4_att = conv_layer_1_gate_l.output_tensor3#conv_layer_2_gate_l.masked_conv_out_sigmoid*conv_layer_2_pre_l.masked_conv_out+(1.0-conv_layer_2_gate_l.masked_conv_out_sigmoid)*common_input_l
    r_input_4_att = conv_layer_1_gate_r.output_tensor3#conv_layer_2_gate_r.masked_conv_out_sigmoid*conv_layer_2_pre_r.masked_conv_out+(1.0-conv_layer_2_gate_r.masked_conv_out_sigmoid)*common_input_r

    conv_layer_1 = Conv_for_Pair(rng,
            origin_input_tensor3=common_input_l,
            origin_input_tensor3_r = common_input_r,
            input_tensor3=l_input_4_att,
            input_tensor3_r = r_input_4_att,
             mask_matrix = sents_mask_l,
             mask_matrix_r = sents_mask_r,
             image_shape=(batch_size, 1, hidden_size[0], maxSentLen),
             image_shape_r = (batch_size, 1, hidden_size[0], maxSentLen),
             filter_shape=(hidden_size[1], 1, hidden_size[0], filter_size),
             filter_shape_context=(hidden_size[1], 1,hidden_size[0], 1),
             W=drop_conv_W_1, b=conv_b_1,
             W_context=drop_conv_W_1_context, b_context=conv_b_1_context)

#         attentive_sent_emb_l = conv_layer_1.attentive_maxpool_vec_l
#         attentive_sent_emb_r = conv_layer_1.attentive_maxpool_vec_r
    biased_attentive_sent_emb_l = conv_layer_1.biased_attentive_maxpool_vec_l
    biased_attentive_sent_emb_r = conv_layer_1.biased_attentive_maxpool_vec_r

#         cos = cosine_matrix1_matrix2_rowwise(attentive_sent_emb_l,attentive_sent_emb_r).dimshuffle(0,'x')
#         euc = 1.0/(1.0+T.sqrt(1e-20+T.sum((attentive_sent_emb_l-attentive_sent_emb_r)**2, axis=1))).dimshuffle(0,'x')
#         euc_sum = 1.0/(1.0+T.sqrt(1e-20+T.sum((conv_layer_1.attentive_sumpool_vec_l-conv_layer_1.attentive_sumpool_vec_r)**2, axis=1))).dimshuffle(0,'x')

    HL_layer_1_input = T.concatenate([biased_attentive_sent_emb_l,biased_attentive_sent_emb_r, biased_attentive_sent_emb_l*biased_attentive_sent_emb_r],axis=1)
    HL_layer_1_input_size = hidden_size[1]*3#+extra_size#+(maxSentLen*2+10*2)#+hidden_size[1]*3+1

    HL_layer_1_W, HL_layer_1_b = create_HiddenLayer_para(rng, HL_layer_1_input_size, hidden_size[0])
    HL_layer_1_params = [HL_layer_1_W, HL_layer_1_b]
    drop_HL_layer_1_W = dropout_layer(first_srng, HL_layer_1_W, drop_p, train_flag)
    HL_layer_1=HiddenLayer(rng, input=HL_layer_1_input, n_in=HL_layer_1_input_size, n_out=hidden_size[0], W=drop_HL_layer_1_W, b=HL_layer_1_b, activation=T.nnet.relu)

    HL_layer_2_W, HL_layer_2_b = create_HiddenLayer_para(rng, hidden_size[0], hidden_size[0])
    HL_layer_2_params = [HL_layer_2_W, HL_layer_2_b]
    drop_HL_layer_2_W = dropout_layer(first_srng, HL_layer_2_W, drop_p, train_flag)
    HL_layer_2=HiddenLayer(rng, input=HL_layer_1.output, n_in=hidden_size[0], n_out=hidden_size[0], W=drop_HL_layer_2_W, b=HL_layer_2_b, activation=T.nnet.relu)
    LR_input_size=HL_layer_1_input_size+2*hidden_size[0]
    U_a = create_ensemble_para(rng, 3, LR_input_size) # the weight matrix hidden_size*2
    drop_U_a = dropout_layer(first_srng, U_a, drop_p, train_flag)
    LR_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR_para=[U_a, LR_b]
    LR_input=T.tanh(T.concatenate([HL_layer_1_input, HL_layer_1.output, HL_layer_2.output],axis=1))
    layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=3, W=drop_U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    '''
    the second classifier
    '''
    conv_layer_2_gate_l = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_l,
             mask_matrix = sents_mask_l,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=gate_filter_shape,
             W=drop_conv_W_1_pre_2, b=conv_b_1_pre_2,
             W_gate =drop_conv_W_1_gate_2, b_gate=conv_b_1_gate_2 )
    conv_layer_2_gate_r = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_r,
             mask_matrix = sents_mask_r,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=gate_filter_shape,
             W=drop_conv_W_1_pre_2, b=conv_b_1_pre_2,
             W_gate =drop_conv_W_1_gate_2, b_gate=conv_b_1_gate_2 )

    l_input_4_att_2 = conv_layer_2_gate_l.output_tensor3#conv_layer_2_gate_l.masked_conv_out_sigmoid*conv_layer_2_pre_l.masked_conv_out+(1.0-conv_layer_2_gate_l.masked_conv_out_sigmoid)*common_input_l
    r_input_4_att_2 = conv_layer_2_gate_r.output_tensor3#conv_layer_2_gate_r.masked_conv_out_sigmoid*conv_layer_2_pre_r.masked_conv_out+(1.0-conv_layer_2_gate_r.masked_conv_out_sigmoid)*common_input_r

    conv_layer_2 = Conv_for_Pair(rng,
            origin_input_tensor3=common_input_l,
            origin_input_tensor3_r = common_input_r,
            input_tensor3=l_input_4_att_2,
            input_tensor3_r = r_input_4_att_2,
             mask_matrix = sents_mask_l,
             mask_matrix_r = sents_mask_r,
             image_shape=(batch_size, 1, hidden_size[0], maxSentLen),
             image_shape_r = (batch_size, 1, hidden_size[0], maxSentLen),
             filter_shape=(hidden_size[1], 1, hidden_size[0], filter_size),
             filter_shape_context=(hidden_size[1], 1,hidden_size[0], 1),
             W=drop_conv_W_1_2, b=conv_b_1_2,
             W_context=drop_conv_W_1_context_2, b_context=conv_b_1_context_2)
    biased_attentive_sent_emb_l_2 = conv_layer_2.attentive_maxpool_vec_l
    biased_attentive_sent_emb_r_2 = conv_layer_2.attentive_maxpool_vec_r

    HL_layer_3_input = T.concatenate([biased_attentive_sent_emb_l_2,biased_attentive_sent_emb_r_2, biased_attentive_sent_emb_l_2*biased_attentive_sent_emb_r_2],axis=1)
    HL_layer_3_input_size = hidden_size[1]*3#+extra_size#+(maxSentLen*2+10*2)#+hidden_size[1]*3+1

    HL_layer_3_W, HL_layer_3_b = create_HiddenLayer_para(rng, HL_layer_3_input_size, hidden_size[0])
    HL_layer_3_params = [HL_layer_3_W, HL_layer_3_b]
    drop_HL_layer_3_W = dropout_layer(first_srng, HL_layer_3_W, drop_p, train_flag)
    HL_layer_3=HiddenLayer(rng, input=HL_layer_3_input, n_in=HL_layer_3_input_size, n_out=hidden_size[0], W=drop_HL_layer_3_W, b=HL_layer_3_b, activation=T.nnet.relu)

    HL_layer_4_W, HL_layer_4_b = create_HiddenLayer_para(rng, hidden_size[0], hidden_size[0])
    HL_layer_4_params = [HL_layer_4_W, HL_layer_4_b]
    drop_HL_layer_4_W = dropout_layer(first_srng, HL_layer_4_W, drop_p, train_flag)
    HL_layer_4=HiddenLayer(rng, input=HL_layer_3.output, n_in=hidden_size[0], n_out=hidden_size[0], W=drop_HL_layer_4_W, b=HL_layer_4_b, activation=T.nnet.relu)
    LR2_input_size=HL_layer_3_input_size+2*hidden_size[0]
    U2_a = create_ensemble_para(rng, 3, LR2_input_size) # the weight matrix hidden_size*2
    drop_U2_a = dropout_layer(first_srng, U2_a, drop_p, train_flag)
    LR2_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR2_para=[U2_a, LR2_b]
    LR2_input=T.tanh(T.concatenate([HL_layer_3_input, HL_layer_3.output, HL_layer_4.output],axis=1))
    layer_LR2=LogisticRegression(rng, input=LR2_input, n_in=LR2_input_size, n_out=3, W=drop_U2_a, b=LR2_b) #basically it is a multiplication between weight matrix and input feature vector

    '''
    the third classifier
    '''
    conv_layer_3_gate_l = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_l,
             mask_matrix = sents_mask_l,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=gate_filter_shape,
             W=drop_conv_W_1_pre_3, b=conv_b_1_pre_3,
             W_gate =drop_conv_W_1_gate_3, b_gate=conv_b_1_gate_3 )
    conv_layer_3_gate_r = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_r,
             mask_matrix = sents_mask_r,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=gate_filter_shape,
             W=drop_conv_W_1_pre_3, b=conv_b_1_pre_3,
             W_gate =drop_conv_W_1_gate_3, b_gate=conv_b_1_gate_3)

    l_input_4_att_3 = conv_layer_3_gate_l.output_tensor3#conv_layer_2_gate_l.masked_conv_out_sigmoid*conv_layer_2_pre_l.masked_conv_out+(1.0-conv_layer_2_gate_l.masked_conv_out_sigmoid)*common_input_l
    r_input_4_att_3 = conv_layer_3_gate_r.output_tensor3#conv_layer_2_gate_r.masked_conv_out_sigmoid*conv_layer_2_pre_r.masked_conv_out+(1.0-conv_layer_2_gate_r.masked_conv_out_sigmoid)*common_input_r

    conv_layer_3 = Conv_for_Pair(rng,
            origin_input_tensor3=common_input_l,
            origin_input_tensor3_r = common_input_r,
            input_tensor3=l_input_4_att_3,
            input_tensor3_r = r_input_4_att_3,
             mask_matrix = sents_mask_l,
             mask_matrix_r = sents_mask_r,
             image_shape=(batch_size, 1, hidden_size[0], maxSentLen),
             image_shape_r = (batch_size, 1, hidden_size[0], maxSentLen),
             filter_shape=(hidden_size[1], 1, hidden_size[0], filter_size),
             filter_shape_context=(hidden_size[1], 1,hidden_size[0], 1),
             W=drop_conv_W_1_3, b=conv_b_1_3,
             W_context=drop_conv_W_1_context_3, b_context=conv_b_1_context_3)
    biased_attentive_sent_emb_l_3 = conv_layer_3.biased_attentive_maxpool_vec_l
    biased_attentive_sent_emb_r_3 = conv_layer_3.biased_attentive_maxpool_vec_r

    HL_layer_5_input = T.concatenate([biased_attentive_sent_emb_l_3,biased_attentive_sent_emb_r_3, biased_attentive_sent_emb_l_3*biased_attentive_sent_emb_r_3],axis=1)
    HL_layer_5_input_size = hidden_size[1]*3#+extra_size#+(maxSentLen*2+10*2)#+hidden_size[1]*3+1

    HL_layer_5_W, HL_layer_5_b = create_HiddenLayer_para(rng, HL_layer_5_input_size, hidden_size[0])
    HL_layer_5_params = [HL_layer_5_W, HL_layer_5_b]
    drop_HL_layer_5_W = dropout_layer(first_srng, HL_layer_5_W, drop_p, train_flag)
    HL_layer_5=HiddenLayer(rng, input=HL_layer_5_input, n_in=HL_layer_5_input_size, n_out=hidden_size[0], W=drop_HL_layer_5_W, b=HL_layer_5_b, activation=T.nnet.relu)

    HL_layer_6_W, HL_layer_6_b = create_HiddenLayer_para(rng, hidden_size[0], hidden_size[0])
    HL_layer_6_params = [HL_layer_6_W, HL_layer_6_b]
    drop_HL_layer_6_W = dropout_layer(first_srng, HL_layer_6_W, drop_p, train_flag)
    HL_layer_6=HiddenLayer(rng, input=HL_layer_5.output, n_in=hidden_size[0], n_out=hidden_size[0], W=drop_HL_layer_6_W, b=HL_layer_6_b, activation=T.nnet.relu)
    LR3_input_size=HL_layer_5_input_size+2*hidden_size[0]
    U3_a = create_ensemble_para(rng, 3, LR3_input_size) # the weight matrix hidden_size*2
    drop_U3_a = dropout_layer(first_srng, U3_a, drop_p, train_flag)
    LR3_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR3_para=[U3_a, LR3_b]
    LR3_input=T.tanh(T.concatenate([HL_layer_5_input, HL_layer_5.output, HL_layer_6.output],axis=1))
    layer_LR3=LogisticRegression(rng, input=LR3_input, n_in=LR3_input_size, n_out=3, W=drop_U3_a, b=LR3_b) #basically it is a multiplication between weight matrix and input feature vector

    '''
    the fourth classifier
    '''
    conv_layer_4_gate_l = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_l,
             mask_matrix = sents_mask_l,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=gate_filter_shape,
             W=drop_conv_W_1_pre_4, b=conv_b_1_pre_4,
             W_gate =drop_conv_W_1_gate_4, b_gate=conv_b_1_gate_4 )
    conv_layer_4_gate_r = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_r,
             mask_matrix = sents_mask_r,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=gate_filter_shape,
             W=drop_conv_W_1_pre_4, b=conv_b_1_pre_4,
             W_gate =drop_conv_W_1_gate_4, b_gate=conv_b_1_gate_4)

    l_input_4_att_4 = conv_layer_4_gate_l.output_tensor3#conv_layer_2_gate_l.masked_conv_out_sigmoid*conv_layer_2_pre_l.masked_conv_out+(1.0-conv_layer_2_gate_l.masked_conv_out_sigmoid)*common_input_l
    r_input_4_att_4 = conv_layer_4_gate_r.output_tensor3#conv_layer_2_gate_r.masked_conv_out_sigmoid*conv_layer_2_pre_r.masked_conv_out+(1.0-conv_layer_2_gate_r.masked_conv_out_sigmoid)*common_input_r

    conv_layer_4 = Conv_for_Pair(rng,
            origin_input_tensor3=common_input_l,
            origin_input_tensor3_r = common_input_r,
            input_tensor3=l_input_4_att_4,
            input_tensor3_r = r_input_4_att_4,
             mask_matrix = sents_mask_l,
             mask_matrix_r = sents_mask_r,
             image_shape=(batch_size, 1, hidden_size[0], maxSentLen),
             image_shape_r = (batch_size, 1, hidden_size[0], maxSentLen),
             filter_shape=(hidden_size[1], 1, hidden_size[0], filter_size),
             filter_shape_context=(hidden_size[1], 1,hidden_size[0], 1),
             W=drop_conv_W_1_4, b=conv_b_1_4,
             W_context=drop_conv_W_1_context_4, b_context=conv_b_1_context_4)
    biased_attentive_sent_emb_l_4 = conv_layer_4.attentive_maxpool_vec_l
    biased_attentive_sent_emb_r_4 = conv_layer_4.attentive_maxpool_vec_r

    HL_layer_7_input = T.concatenate([biased_attentive_sent_emb_l_4,biased_attentive_sent_emb_r_4, biased_attentive_sent_emb_l_4*biased_attentive_sent_emb_r_4],axis=1)
    HL_layer_7_input_size = hidden_size[1]*3#+extra_size#+(maxSentLen*2+10*2)#+hidden_size[1]*3+1

    HL_layer_7_W, HL_layer_7_b = create_HiddenLayer_para(rng, HL_layer_7_input_size, hidden_size[0])
    HL_layer_7_params = [HL_layer_7_W, HL_layer_7_b]
    drop_HL_layer_7_W = dropout_layer(first_srng, HL_layer_7_W, drop_p, train_flag)
    HL_layer_7=HiddenLayer(rng, input=HL_layer_7_input, n_in=HL_layer_7_input_size, n_out=hidden_size[0], W=drop_HL_layer_7_W, b=HL_layer_7_b, activation=T.nnet.relu)

    HL_layer_8_W, HL_layer_8_b = create_HiddenLayer_para(rng, hidden_size[0], hidden_size[0])
    HL_layer_8_params = [HL_layer_8_W, HL_layer_8_b]
    drop_HL_layer_8_W = dropout_layer(first_srng, HL_layer_8_W, drop_p, train_flag)
    HL_layer_8=HiddenLayer(rng, input=HL_layer_7.output, n_in=hidden_size[0], n_out=hidden_size[0], W=drop_HL_layer_8_W, b=HL_layer_8_b, activation=T.nnet.relu)
    LR4_input_size=HL_layer_7_input_size+2*hidden_size[0]
    U4_a = create_ensemble_para(rng, 3, LR4_input_size) # the weight matrix hidden_size*2
    drop_U4_a = dropout_layer(first_srng, U4_a, drop_p, train_flag)
    LR4_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR4_para=[U4_a, LR4_b]
    LR4_input=T.tanh(T.concatenate([HL_layer_7_input, HL_layer_7.output, HL_layer_8.output],axis=1))
    layer_LR4=LogisticRegression(rng, input=LR4_input, n_in=LR4_input_size, n_out=3, W=drop_U4_a, b=LR4_b) #basically it is a multiplication between weight matrix and input feature vector

    loss_0=(layer_LR.negative_log_likelihood(labels)+layer_LR2.negative_log_likelihood(labels)+layer_LR3.negative_log_likelihood(labels)+layer_LR4.negative_log_likelihood(labels))/4.0  #for classification task, we usually used negative log likelihood as loss, the lower the better.
    para_0 = LR_para+HL_layer_1_params+HL_layer_2_params+LR2_para+HL_layer_3_params+HL_layer_4_params +LR3_para+HL_layer_5_params+HL_layer_6_params +LR4_para+HL_layer_7_params+HL_layer_8_params


#         loss = loss_0+loss_1+loss_2
    batch_distr = layer_LR.p_y_given_x+layer_LR2.p_y_given_x+layer_LR3.p_y_given_x+layer_LR4.p_y_given_x#T.sum((layer_LR.p_y_given_x).reshape((batch_size, multi_psp_size,3)), axis=1)  #(batch, 3)
    batch_error = (layer_LR.errors(labels)+layer_LR2.errors(labels)+layer_LR3.errors(labels)+layer_LR4.errors(labels))/4.0

    return loss_0, para_0, batch_distr,batch_error

def maskMatrix_to_posiMatrix(mask):
    def vec(vector):
        summ = T.cast(T.sum(vector), 'int32')
        new_vec = T.concatenate([T.zeros((vector.shape[0]-summ,)), (T.arange(summ)+1)*1.0/summ], axis=0)
        return new_vec

    results, _ = theano.scan(fn=vec,sequences=mask)
    return T.cast(results, 'float32')

def one_classifier_in_one_copy(rng, common_input_l,common_input_r,sents_mask_l,sents_mask_r,batch_size, emb_size,maxSentLen,gate_filter_shape,hidden_size,filter_size,
                               first_srng, drop_p,train_flag,labels,
                               drop_conv_W_1_pre,conv_b_1_pre,drop_conv_W_1_gate,conv_b_1_gate,
                               drop_conv_W_1,conv_b_1,drop_conv_W_1_context,conv_b_1_context,
                               biased):
    conv_layer_1_gate_l = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_l,
             mask_matrix = sents_mask_l,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=gate_filter_shape,
             W=drop_conv_W_1_pre, b=conv_b_1_pre,
             W_gate =drop_conv_W_1_gate, b_gate=conv_b_1_gate )
    conv_layer_1_gate_r = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_r,
             mask_matrix = sents_mask_r,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=gate_filter_shape,
             W=drop_conv_W_1_pre, b=conv_b_1_pre,
             W_gate =drop_conv_W_1_gate, b_gate=conv_b_1_gate )

    drop_l_input_4_att = dropout_layer(first_srng, conv_layer_1_gate_l.output_tensor3, drop_p, train_flag)#conv_layer_2_gate_l.masked_conv_out_sigmoid*conv_layer_2_pre_l.masked_conv_out+(1.0-conv_layer_2_gate_l.masked_conv_out_sigmoid)*common_input_l
    drop_r_input_4_att = dropout_layer(first_srng,conv_layer_1_gate_r.output_tensor3, drop_p, train_flag)#conv_layer_2_gate_r.masked_conv_out_sigmoid*conv_layer_2_pre_r.masked_conv_out+(1.0-conv_layer_2_gate_r.masked_conv_out_sigmoid)*common_input_r

    conv_layer_1 = Conv_for_Pair(rng,
            origin_input_tensor3=common_input_l,
            origin_input_tensor3_r = common_input_r,
            input_tensor3=drop_l_input_4_att,
            input_tensor3_r = drop_r_input_4_att,
             mask_matrix = sents_mask_l,
             mask_matrix_r = sents_mask_r,
             image_shape=(batch_size, 1, hidden_size[0], maxSentLen),
             image_shape_r = (batch_size, 1, hidden_size[0], maxSentLen),
             filter_shape=(hidden_size[1], 1, hidden_size[0], filter_size[0]),
             filter_shape_context=(hidden_size[1], 1,hidden_size[0], 1),
             W=drop_conv_W_1, b=conv_b_1,
             W_context=drop_conv_W_1_context, b_context=conv_b_1_context)

#         attentive_sent_emb_l = conv_layer_1.attentive_maxpool_vec_l
#         attentive_sent_emb_r = conv_layer_1.attentive_maxpool_vec_r
    if biased:
        biased_attentive_sent_emb_l = conv_layer_1.biased_attentive_maxpool_vec_l
        biased_attentive_sent_emb_r = conv_layer_1.biased_attentive_maxpool_vec_r
    else:
        biased_attentive_sent_emb_l = conv_layer_1.attentive_maxpool_vec_l
        biased_attentive_sent_emb_r = conv_layer_1.attentive_maxpool_vec_r

#         cos = cosine_matrix1_matrix2_rowwise(attentive_sent_emb_l,attentive_sent_emb_r).dimshuffle(0,'x')
#         euc = 1.0/(1.0+T.sqrt(1e-20+T.sum((attentive_sent_emb_l-attentive_sent_emb_r)**2, axis=1))).dimshuffle(0,'x')
#         euc_sum = 1.0/(1.0+T.sqrt(1e-20+T.sum((conv_layer_1.attentive_sumpool_vec_l-conv_layer_1.attentive_sumpool_vec_r)**2, axis=1))).dimshuffle(0,'x')



    HL_layer_1_input = dropout_layer(first_srng, T.concatenate([biased_attentive_sent_emb_l,biased_attentive_sent_emb_r, biased_attentive_sent_emb_l*biased_attentive_sent_emb_r],axis=1), drop_p, train_flag)
    HL_layer_1_input_size = hidden_size[1]*3#+extra_size#+(maxSentLen*2+10*2)#+hidden_size[1]*3+1

    HL_layer_1_W, HL_layer_1_b = create_HiddenLayer_para(rng, HL_layer_1_input_size, hidden_size[0])
    HL_layer_1_params = [HL_layer_1_W, HL_layer_1_b]
    HL_layer_1=HiddenLayer(rng, input=HL_layer_1_input, n_in=HL_layer_1_input_size, n_out=hidden_size[0], W=HL_layer_1_W, b=HL_layer_1_b, activation=T.nnet.relu)
    HL_layer_1_output = dropout_layer(first_srng, HL_layer_1.output, drop_p, train_flag)

    HL_layer_2_W, HL_layer_2_b = create_HiddenLayer_para(rng, hidden_size[0], hidden_size[0])
    HL_layer_2_params = [HL_layer_2_W, HL_layer_2_b]
    HL_layer_2=HiddenLayer(rng, input=HL_layer_1_output, n_in=hidden_size[0], n_out=hidden_size[0], W=HL_layer_2_W, b=HL_layer_2_b, activation=T.nnet.relu)
    HL_layer_2_output = dropout_layer(first_srng, HL_layer_2.output, drop_p, train_flag)


    LR_input=  T.tanh(T.concatenate([HL_layer_1_input,HL_layer_1_output ,HL_layer_2_output], axis=1) )       #T.concatenate([T.tanh(HL_layer_1_input), HL_layer_1_output, HL_layer_2_output],axis=1)
    LR_input_size=HL_layer_1_input_size+2*hidden_size[0]
    U_a = create_ensemble_para(rng, 3, LR_input_size) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR_para=[U_a, LR_b]

    layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=3, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector

    loss = layer_LR.negative_log_likelihood(labels)
    distr = layer_LR.p_y_given_x
    params = LR_para+HL_layer_1_params+HL_layer_2_params
    return loss, distr, params
