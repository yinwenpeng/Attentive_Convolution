# -*- coding: utf-8 -*-
import numpy
import codecs
import string
import nltk
from nltk.stem import SnowballStemmer
from itertools import izip
import re
import string
from difflib import SequenceMatcher
from numpy import linalg as LA
_digits = re.compile('\d')

def denoise_oov(word_str, wordset, trichar2wordlist):
    if word_str in wordset or len(word_str)<=6:
        return word_str
    else:
        word_len = len(word_str)
        tri_char_list = [ word_str[i:i+3]  for i in range(word_len-2)]
        tri_char_size = word_len-2
        accu_pre_set=set([])
        for i in range(tri_char_size):
            #give the i-th tri-char
            if i ==0:
                pre_set = set(trichar2wordlist.setdefault(tri_char_list[1],[]))
            else:
                pre_set = set(trichar2wordlist.setdefault(tri_char_list[0],[]))
            for j in range(tri_char_size):
                if j !=i:
                    pre_set = pre_set&set(trichar2wordlist.setdefault(tri_char_list[j],[]))
            accu_pre_set=accu_pre_set | pre_set
        if len(accu_pre_set)>0:
            #find correct ones
            correct_list = list(accu_pre_set)
            cand2simi={}
            top_simi=0.0
            top_cand=''
            if len(correct_list) ==1:
                top_cand = correct_list[0]
            else:
                for cand in correct_list:
                    simi = SequenceMatcher(None, cand.lower(), word_str).ratio()
                    if simi == 1.0:
                        top_cand = cand
                        top_simi = simi
                        break
                    if simi>top_simi:
                        top_cand = cand
                        top_simi = simi
                if top_simi < 0.9:
                    top_cand = 'UNK'

            print 'OOV word: ', word_str, ' --> ',top_cand, ' from:', correct_list
            return top_cand
        else:
            return 'UNK'  #can not find by remove single tri-char





def transfer_wordlist_2_idlist_with_maxlen_denoiseOOV(token_list, vocab_map, maxlen, wordset, trichar2wordlist):
    '''
    From such as ['i', 'love', 'Munich'] to idlist [23, 129, 34], if maxlen is 5, then pad two zero in the left side, becoming [0, 0, 23, 129, 34]
    '''
    idlist=[]
    for word in token_list:
        position = word.find('-')
        if position<0:
            word = denoise_oov(word, wordset, trichar2wordlist)
            id=vocab_map.get(word)
            if id is None: # if word was not in the vocabulary
                id=len(vocab_map)+1  # id of true words starts from 1, leaving 0 to "pad id"
                vocab_map[word]=id
            idlist.append(id)
        else:
            subwords = word.split('-')
            for subword in subwords:
                subword = denoise_oov(subword, wordset, trichar2wordlist)
                id=vocab_map.get(subword)
                if id is None: # if word was not in the vocabulary
                    id=len(vocab_map)+1  # id of true words starts from 1, leaving 0 to "pad id"
                    vocab_map[subword]=id
                idlist.append(id)

    mask_list=[1.0]*len(idlist) # mask is used to indicate each word is a true word or a pad word
    pad_size=maxlen-len(idlist)
    if pad_size>0:
        idlist=[0]*pad_size+idlist
        mask_list=[0.0]*pad_size+mask_list
    else: # if actual sentence len is longer than the maxlen, truncate
        idlist=idlist[:maxlen]
        mask_list=mask_list[:maxlen]
    return idlist, mask_list



def transfer_wordlist_2_idlist_with_maxlen(token_list, vocab_map, maxlen):
    '''
    From such as ['i', 'love', 'Munich'] to idlist [23, 129, 34], if maxlen is 5, then pad two zero in the left side, becoming [0, 0, 23, 129, 34]
    '''
    idlist=[]
    for word in token_list:
        position = word.find('-')
        if position<0:
            if word not in string.punctuation:
                word =  word.translate(None, string.punctuation)
            id=vocab_map.get(word)
            if id is None: # if word was not in the vocabulary
                id=len(vocab_map)+1  # id of true words starts from 1, leaving 0 to "pad id"
                vocab_map[word]=id
            idlist.append(id)
        else:
            subwords = word.split('-')
            for subword in subwords:
                if subword not in string.punctuation:
                    subword =  subword.translate(None, string.punctuation)
                id=vocab_map.get(subword)
                if id is None: # if word was not in the vocabulary
                    id=len(vocab_map)+1  # id of true words starts from 1, leaving 0 to "pad id"
                    vocab_map[subword]=id
                idlist.append(id)

    mask_list=[1.0]*len(idlist) # mask is used to indicate each word is a true word or a pad word
    pad_size=maxlen-len(idlist)
    if pad_size>0:
        idlist=[0]*pad_size+idlist
        mask_list=[0.0]*pad_size+mask_list
    else: # if actual sentence len is longer than the maxlen, truncate
        idlist=idlist[:maxlen]
        mask_list=mask_list[:maxlen]
    return idlist, mask_list

def transfer_wordlist_2_idlist_with_maxlen_return_wordlist(token_list, vocab_map, maxlen):
    subword_tokenlist=[]
    for token in token_list:
        position = token.find('-')
        if position<0:
            subword_tokenlist.append(token)
        else:
            subwords = token.split('-')
            for subword in subwords:
                subword_tokenlist.append(subword)
    token_list =   subword_tokenlist
    pad_size = maxlen - len(token_list)
    if pad_size > 0:
        token_list=['uuuuuu']*pad_size+token_list
    else:
        token_list = token_list[:maxlen]
    idlist=[]
    mask_list=[]
    if pad_size > 0:
        idlist+=[0]*pad_size
        mask_list+=[0.0]*pad_size
        valid_token_list = token_list[pad_size:]
    else:
        valid_token_list = token_list
    for word in valid_token_list:
        id=vocab_map.get(word)
        if id is None: # if word was not in the vocabulary
            id=len(vocab_map)+1  # id of true words starts from 1, leaving 0 to "pad id"
            vocab_map[word]=id
        idlist.append(id)
        mask_list.append(1.0)
    return idlist, mask_list, token_list

def load_yelp_dataset(maxlen=100, minlen=4):
    root="/mounts/data/proj/wenpeng/Dataset/yelp/"
    files=['yelp_train_500k', 'yelp_valid_2000', 'yelp_test_2000']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences=[]
    all_masks=[]
    all_labels=[]
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'
        max_len=0
        sents=[]
        sents_masks=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            parts=line.strip().lower().split() #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts) > minlen: # we only consider some sentences that are not too short, controlled by minlen
                label=int(parts[0])-1  # keep label be 0 or 1
                sentence_wordlist=parts[1:]
                if len(sentence_wordlist)> max_len:
                    max_len = len(sentence_wordlist)

                labels.append(label)
                sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
                sents.append(sent_idlist)
                sents_masks.append(sent_masklist)
        all_sentences.append(sents)
        all_masks.append(sents_masks)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels), ' max_len: ', max_len
    print 'dataset loaded over, totally ', len(word2id), 'words'
    return all_sentences, all_masks, all_labels, word2id

def load_sentiment_dataset(maxlen=40, minlen=4):
    root="/mounts/data/proj/wenpeng/Dataset/StanfordSentiment/stanfordSentimentTreebank/5classes/"
    files=['1train.txt', '1dev.txt', '1test.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences=[]
    all_masks=[]
    all_labels=[]
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents=[]
        sents_masks=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            parts=line.strip().lower().split() #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts) > minlen: # we only consider some sentences that are not too short, controlled by minlen
                label=int(parts[0])-1  # keep label be 0 or 1
                sentence_wordlist=parts[1:]
                if sentence_wordlist[-1]=='.':
                    sentence_wordlist=sentence_wordlist[:-1]

                labels.append(label)
                sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
                sents.append(sent_idlist)
                sents_masks.append(sent_masklist)
        all_sentences.append(sents)
        all_masks.append(sents_masks)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels)
    print 'dataset loaded over, totally ', len(word2id), 'words'
    return all_sentences, all_masks, all_labels, word2id

def load_word2vec():
    word2vec = {}

    print "==> loading 300d word2vec"
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
    # f=open('/save/wenpeng/datasets/word2vec_words_300d.txt', 'r')#glove.6B.300d.txt, fasttext.en.vec, word2vec_words_300d.txt, glove.840B.300d.txt
    f=open('/mounts/data/proj/wenpeng/Dataset/word2vec_words_300d.txt', 'r')

    for line in f:
        l = line.split()
        word2vec[l[0]] = map(float, l[1:])

    print "==> word2vec is loaded"

    return word2vec

def load_glove():
    word2vec = {}

    print "==> loading 300d glove"
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
    f=open('/save/wenpeng/datasets/Dataset/glove.6B.300d.txt', 'r')#glove.6B.300d.txt, word2vec_words_300d.txt, glove.840B.300d.txt
    wordlist=[]
    emblist=[]
    for line in f:
        l = line.split()
        wordlist.append(l[0])
        emblist.append(map(float, l[1:]))
        # word2vec[l[0]] = map(float, l[1:])
    emb_array=numpy.asarray(emblist)
    norm_emb_array = LA.norm(emb_array, axis=0)

    emb_array = emb_array/norm_emb_array[None,:]
    word2vec = {}
    for i in range(len(wordlist)):
        word2vec[wordlist[i]] = list(emb_array[i])

    print "==> word2vec is loaded"

    return word2vec

def load_word2vec_return_word2vec_trichar2wordlist():
    word2vec = {}
    trichar2wordlist = {}

    print "==> loading 300d word2vec"
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
    f=open('/save/wenpeng/datasets/word2vec_words_300d.txt', 'r')#glove.6B.300d.txt, word2vec_words_300d.txt, glove.840B.300d.txt
    for line in f:
        l = line.split()
        word = l[0]
        word2vec[word] = map(float, l[1:])
        # if word do not conain pub:
        for i in range(len(word)-2):
            new_tri_char = word[i:i+3]
            # exist_word_list = trichar2wordlist.get(new_tri_char)
            # if exist_word_list is None:
            #     exist_word_list=[]
            # exist_word_list.append(word)


            trichar2wordlist.setdefault(new_tri_char, [] ).append(word)
            # trichar2wordlist[new_tri_char] = exist_word_list

    print "==> word2vec is loaded"

    return word2vec, trichar2wordlist
def load_word2vec_file(filename):
    emb_dic = '/mounts/data/proj/wenpeng/Dataset/'
    word2vec = {}

    print "==> loading 300d word2vec"
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
    f=open(emb_dic+filename, 'r')#glove.6B.300d.txt, word2vec_words_300d.txt, glove.6B.50d.txt
    for line in f:
        l = line.split()
        word2vec[l[0]] = map(float, l[1:])

    print "==> word2vec is loaded"

    return word2vec
def load_word2vec_to_init(rand_values, ivocab, word2vec):
    fail=0
    for id, word in ivocab.iteritems():
        emb=word2vec.get(word)
        if emb is not None:
            rand_values[id]=numpy.array(emb)
        else:
            # print word
            fail+=1
    print '==> use word2vec initialization over...fail ', fail
    return rand_values

def load_SNLI_dataset(maxlen=40):
    '''
    load raw SNLI data, truncate sentences into "maxlen". For example, with "maxlen=5", sentence "it is interesting" will
    generate two vectors: id_list [0, 0, 1, 2, 3] and mask_list [0, 0, 1, 1, 1] in which "1" denotes valid word, "0" otherwise.
    '''
    root="/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/"#   "/save/wenpeng/datasets/StanfordEntailment/"
    files=['train.txt', 'dev.txt', 'test.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences_l=[]
    all_masks_l=[]
    all_sentences_r=[]
    all_masks_r=[]
    all_labels=[]
    max_sen_len=0
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents_l=[]
        sents_masks_l=[]
        sents_r=[]
        sents_masks_r=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)==3:

                label=int(parts[0])  # keep label be 0 or 1
                sentence_wordlist_l=parts[1].strip().lower().split()
                # if sentence_wordlist_l[-1]=='.':
                #     sentence_wordlist_l=sentence_wordlist_l[:-1]
                sentence_wordlist_r=parts[2].strip().lower().split()
                # if sentence_wordlist_r[-1]=='.':
                #     sentence_wordlist_r=sentence_wordlist_r[:-1]
                l_len=len(sentence_wordlist_l)
                r_len = len(sentence_wordlist_r)
                if l_len > max_sen_len:
                    max_sen_len=l_len
                if r_len > max_sen_len:
                    max_sen_len=r_len

                sent_idlist_l, sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_l, word2id, maxlen)
                sent_idlist_r, sent_masklist_r=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_r, word2id, maxlen)

                sents_l.append(sent_idlist_l)
                sents_masks_l.append(sent_masklist_l)
                sents_r.append(sent_idlist_r)
                sents_masks_r.append(sent_masklist_r)
                labels.append(label)


        all_sentences_l.append(sents_l)
        all_sentences_r.append(sents_r)
        all_masks_l.append(sents_masks_l)
        all_masks_r.append(sents_masks_r)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels), 'pairs'
    print 'dataset loaded over, totally ', len(word2id), 'words, max sen len:',   max_sen_len
    return all_sentences_l, all_masks_l, all_sentences_r, all_masks_r,all_labels, word2id

def load_SNLI_dataset_with_vocab(maxlen=40, wordset=None, trichar2wordlist=None):
    root="/save/wenpeng/datasets/StanfordEntailment/"
    files=['train.txt', 'dev.txt', 'test.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences_l=[]
    all_masks_l=[]
    all_sentences_r=[]
    all_masks_r=[]
    all_labels=[]
    max_sen_len=0
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents_l=[]
        sents_masks_l=[]
        sents_r=[]
        sents_masks_r=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)==3:

                label=int(parts[0])  # keep label be 0 or 1
                sentence_wordlist_l=parts[1].strip().lower().split()
                sentence_wordlist_r=parts[2].strip().lower().split()
                l_len=len(sentence_wordlist_l)
                r_len = len(sentence_wordlist_r)
                if l_len > max_sen_len:
                    max_sen_len=l_len
                if r_len > max_sen_len:
                    max_sen_len=r_len

                sent_idlist_l, sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen_denoiseOOV(sentence_wordlist_l, word2id, maxlen, wordset, trichar2wordlist)
                sent_idlist_r, sent_masklist_r=transfer_wordlist_2_idlist_with_maxlen_denoiseOOV(sentence_wordlist_r, word2id, maxlen, wordset, trichar2wordlist)

                sents_l.append(sent_idlist_l)
                sents_masks_l.append(sent_masklist_l)
                sents_r.append(sent_idlist_r)
                sents_masks_r.append(sent_masklist_r)
                labels.append(label)


        all_sentences_l.append(sents_l)
        all_sentences_r.append(sents_r)
        all_masks_l.append(sents_masks_l)
        all_masks_r.append(sents_masks_r)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels), 'pairs'
    print 'dataset loaded over, totally ', len(word2id), 'words, max sen len:',   max_sen_len
    return all_sentences_l, all_masks_l, all_sentences_r, all_masks_r,all_labels, word2id

def extra_two_wordlist_SNLI(wordlist1, wordlist2):
    return [0.0,0.0,0.0,0.0]
#     set1=set(wordlist1)
#     word_overalp = set1 & set(wordlist2)
#     overlap_size = len(word_overalp)*1.0
#     len2 = len(wordlist2)
#     feature_1 = overlap_size/len(wordlist1)
#     feature_2 = overlap_size/len2
#
#     feature_3=len2*1.0/len(wordlist1)
#
#     last_match = len2
#     for index, word in enumerate(wordlist2[::-1]):
#         if word in set1:
#             last_match=index
#             break
#     feature_4 = 1.0-last_match*1.0/len2
#     return [feature_1,feature_2, feature_3, feature_4]#, 1.0/(len(wordlist1)+1.0),1.0/(len(wordlist2)+1.0)]

def load_SNLI_dataset_with_extra_with_test(maxlen=40):
    root="/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/"
#     files=['train.norm.to.word2vec.vocab.txt', 'dev.norm.to.word2vec.vocab.txt', 'test.norm.to.word2vec.vocab.txt']
    files=['train.txt', 'dev.txt', 'test.txt']
#     files=['train_removed_overlap.txt', 'dev_removed_overlap.txt', 'test_removed_overlap.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences_l=[]
    all_masks_l=[]
    all_sentences_r=[]
    all_masks_r=[]
    all_labels=[]
    all_extra=[]
    test_rows = []
    max_sen_len=0
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents_l=[]
        sents_masks_l=[]
        sents_r=[]
        sents_masks_r=[]
        extra=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            if i == 2: #test file
                test_rows.append(line.strip())
            parts=line.strip().lower().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)==3:

                label=int(parts[0])  # keep label be 0 or 1
                sentence_wordlist_l=parts[1].strip().lower().split()
                if sentence_wordlist_l[-1]=='.':
                    sentence_wordlist_l=sentence_wordlist_l[:-1]

#                 sub_sentence_wordlist_l=sentence_wordlist_l[:(len(sentence_wordlist_l)/2)] # only a half length

                sentence_wordlist_r=parts[2].strip().lower().split()
                if sentence_wordlist_r[-1]=='.':
                    sentence_wordlist_r=sentence_wordlist_r[:-1]
                l_len=len(sentence_wordlist_l)
                r_len = len(sentence_wordlist_r)
                if l_len > max_sen_len:
                    max_sen_len=l_len
                if r_len > max_sen_len:
                    max_sen_len=r_len

                sent_idlist_l, sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_l, word2id, maxlen)
                sent_idlist_r, sent_masklist_r=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_r, word2id, maxlen)

                half_sentence_wordlist_l = sentence_wordlist_l[:(len(sentence_wordlist_l)/2)]
                half_sent_idlist_l, half_sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(half_sentence_wordlist_l, word2id, maxlen)

#                 sub_sent_idlist_l, sub_sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(sub_sentence_wordlist_l, word2id, maxlen)
                extra_instance = extra_two_wordlist_SNLI(sentence_wordlist_l, sentence_wordlist_r)

                sents_l.append(sent_idlist_l)
                sents_masks_l.append(sent_masklist_l)
                sents_r.append(sent_idlist_r)
                sents_masks_r.append(sent_masklist_r)
                extra.append(extra_instance)
                labels.append(label)

#                 if i==0:#train file
#                     if label == 1: #contr
#                         sents_l.append(sent_idlist_r)
#                         sents_masks_l.append(sent_masklist_r)
#                         sents_r.append(sent_idlist_l)
#                         sents_masks_r.append(sent_masklist_l)
#                         labels.append(label) #contral
#                     elif label==2: # entail
#                         sents_l.append(sent_idlist_r)
#                         sents_masks_l.append(sent_masklist_r)
#                         sents_r.append(sent_idlist_l)
#                         sents_masks_r.append(sent_masklist_l)
#                         labels.append(0) #neutral
                        #aug entail
#                         sents_l.append(half_sent_idlist_l)
#                         sents_masks_l.append(half_sent_masklist_l)
#                         sents_r.append(sent_idlist_l)
#                         sents_masks_r.append(sent_masklist_l)
#                         labels.append(0)#neutral



        all_sentences_l.append(sents_l)
        all_sentences_r.append(sents_r)
        all_masks_l.append(sents_masks_l)
        all_masks_r.append(sents_masks_r)
        all_extra.append(extra)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels), 'pairs'
    print 'dataset loaded over, totally ', len(word2id), 'words, max sen len:',   max_sen_len
    return all_sentences_l, all_masks_l, all_sentences_r, all_masks_r,all_extra, all_labels, word2id,test_rows


def load_SNLI_dataset_with_extra(maxlen=40):
    root="/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/"
#     files=['train.norm.to.word2vec.vocab.txt', 'dev.norm.to.word2vec.vocab.txt', 'test.norm.to.word2vec.vocab.txt']
    files=['train.txt', 'dev.txt', 'test.txt']
#     files=['train_removed_overlap.txt', 'dev_removed_overlap.txt', 'test_removed_overlap.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences_l=[]
    all_masks_l=[]
    all_sentences_r=[]
    all_masks_r=[]
    all_labels=[]
    all_extra=[]
    max_sen_len=0
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents_l=[]
        sents_masks_l=[]
        sents_r=[]
        sents_masks_r=[]
        extra=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            parts=line.strip().lower().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)==3:

                label=int(parts[0])  # keep label be 0 or 1
                sentence_wordlist_l=parts[1].strip().lower().split()
                if sentence_wordlist_l[-1]=='.':
                    sentence_wordlist_l=sentence_wordlist_l[:-1]

#                 sub_sentence_wordlist_l=sentence_wordlist_l[:(len(sentence_wordlist_l)/2)] # only a half length

                sentence_wordlist_r=parts[2].strip().lower().split()
                if sentence_wordlist_r[-1]=='.':
                    sentence_wordlist_r=sentence_wordlist_r[:-1]
                l_len=len(sentence_wordlist_l)
                r_len = len(sentence_wordlist_r)
                if l_len > max_sen_len:
                    max_sen_len=l_len
                if r_len > max_sen_len:
                    max_sen_len=r_len

                sent_idlist_l, sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_l, word2id, maxlen)
                sent_idlist_r, sent_masklist_r=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_r, word2id, maxlen)

                half_sentence_wordlist_l = sentence_wordlist_l[:(len(sentence_wordlist_l)/2)]
                half_sent_idlist_l, half_sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(half_sentence_wordlist_l, word2id, maxlen)

#                 sub_sent_idlist_l, sub_sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(sub_sentence_wordlist_l, word2id, maxlen)
                extra_instance = extra_two_wordlist_SNLI(sentence_wordlist_l, sentence_wordlist_r)

                sents_l.append(sent_idlist_l)
                sents_masks_l.append(sent_masklist_l)
                sents_r.append(sent_idlist_r)
                sents_masks_r.append(sent_masklist_r)
                extra.append(extra_instance)
                labels.append(label)

#                 if i==0:#train file
#                     if label == 1: #contr
#                         sents_l.append(sent_idlist_r)
#                         sents_masks_l.append(sent_masklist_r)
#                         sents_r.append(sent_idlist_l)
#                         sents_masks_r.append(sent_masklist_l)
#                         labels.append(label) #contral
#                     elif label==2: # entail
#                         sents_l.append(sent_idlist_r)
#                         sents_masks_l.append(sent_masklist_r)
#                         sents_r.append(sent_idlist_l)
#                         sents_masks_r.append(sent_masklist_l)
#                         labels.append(0) #neutral
                        #aug entail
#                         sents_l.append(half_sent_idlist_l)
#                         sents_masks_l.append(half_sent_masklist_l)
#                         sents_r.append(sent_idlist_l)
#                         sents_masks_r.append(sent_masklist_l)
#                         labels.append(0)#neutral



        all_sentences_l.append(sents_l)
        all_sentences_r.append(sents_r)
        all_masks_l.append(sents_masks_l)
        all_masks_r.append(sents_masks_r)
        all_extra.append(extra)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels), 'pairs'
    print 'dataset loaded over, totally ', len(word2id), 'words, max sen len:',   max_sen_len
    return all_sentences_l, all_masks_l, all_sentences_r, all_masks_r,all_extra, all_labels, word2id
def load_SNLI_dataset_with_Nonoverlap(maxlen=40):
    root="/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/"
    files=['train.txt', 'dev.txt', 'test.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences_l=[]
    all_masks_l=[]
    all_sentences_r=[]
    all_masks_r=[]

    all_nonoverlap_sentences_l=[]
    all_nonoverlap_masks_l=[]
    all_nonoverlap_sentences_r=[]
    all_nonoverlap_masks_r=[]

    all_labels=[]

    max_sen_len=0
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents_l=[]
        sents_masks_l=[]
        sents_r=[]
        sents_masks_r=[]
        nonoverlap_sents_l=[]
        nonoverlap_sents_masks_l=[]
        nonoverlap_sents_r=[]
        nonoverlap_sents_masks_r=[]
        labels=[]

        readfile=open(root+files[i], 'r')
        for line in readfile:
            parts=line.strip().lower().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)==3:

                label=int(parts[0])  # keep label be 0 or 1
                sentence_wordlist_l=parts[1].strip().split()
                sentence_wordlist_r=parts[2].strip().split()
                word_overalp = set(sentence_wordlist_l) & set(sentence_wordlist_r)
                nonoverlap_sentence_wordlist_l = [x for x in sentence_wordlist_l if x not in word_overalp]
                nonoverlap_sentence_wordlist_r = [x for x in sentence_wordlist_r if x not in word_overalp]
                if len(nonoverlap_sentence_wordlist_l)==0 or len(nonoverlap_sentence_wordlist_r)==0:
                    continue

                l_len=len(sentence_wordlist_l)
                r_len = len(sentence_wordlist_r)
                if l_len > max_sen_len:
                    max_sen_len=l_len
                if r_len > max_sen_len:
                    max_sen_len=r_len
                labels.append(label)
                sent_idlist_l, sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_l, word2id, maxlen)
                sent_idlist_r, sent_masklist_r=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_r, word2id, maxlen)
                nonoverlap_sent_idlist_l, nonoverlap_sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(nonoverlap_sentence_wordlist_l, word2id, maxlen)
                nonoverlap_sent_idlist_r, nonoverlap_sent_masklist_r=transfer_wordlist_2_idlist_with_maxlen(nonoverlap_sentence_wordlist_r, word2id, maxlen)

                sents_l.append(sent_idlist_l)
                sents_masks_l.append(sent_masklist_l)
                sents_r.append(sent_idlist_r)
                sents_masks_r.append(sent_masklist_r)

                nonoverlap_sents_l.append(nonoverlap_sent_idlist_l)
                nonoverlap_sents_masks_l.append(nonoverlap_sent_masklist_l)
                nonoverlap_sents_r.append(nonoverlap_sent_idlist_r)
                nonoverlap_sents_masks_r.append(nonoverlap_sent_masklist_r)

        all_sentences_l.append(sents_l)
        all_sentences_r.append(sents_r)
        all_masks_l.append(sents_masks_l)
        all_masks_r.append(sents_masks_r)
        all_nonoverlap_sentences_l.append(nonoverlap_sents_l)
        all_nonoverlap_masks_l.append(nonoverlap_sents_masks_l)
        all_nonoverlap_sentences_r.append(nonoverlap_sents_r)
        all_nonoverlap_masks_r.append(nonoverlap_sents_masks_r)
        all_labels.append(labels)

        print '\t\t\t size:', len(labels), 'pairs'
    print 'dataset loaded over, totally ', len(word2id), 'words, max sen len:',   max_sen_len
    return all_sentences_l, all_masks_l, all_sentences_r, all_masks_r, all_nonoverlap_sentences_l,all_nonoverlap_masks_l,all_nonoverlap_sentences_r  ,all_nonoverlap_masks_r,               all_labels, word2id
def wordList_to_charIdList(word_list, char_len, char2id):
#     sent_len = len(word_list)
#     pad_size = word_size_limit - sent_len
#     if pad_size > 0:
#         word_list = ['u'*char_len]*pad_size + word_list
#     else:
#         word_list = word_list[:word_size_limit]
    char_idlist=[]
    mask=[]
    for word in word_list:
        sub_char_idlist=[]
        word_len = len(word)
        for char in word:
            char_id = char2id.get(char)
            if char_id is None:
                char_id = len(char2id)+1
                char2id[char]=char_id
            sub_char_idlist.append(char_id)
        char_pad_size = char_len - len(sub_char_idlist)
        if char_pad_size > 0:
            sub_char_idlist = [0]*char_pad_size + sub_char_idlist
            sub_char_mask = [0.0]*char_pad_size + [1.0]*word_len
        else:
            sub_char_idlist=sub_char_idlist[:char_len]
            sub_char_mask = [1.0]*char_len
        char_idlist+=sub_char_idlist
        mask+=sub_char_mask
    return char_idlist, mask
def load_SNLI_dataset_char(maxlen=40, char_len=15):
    root="/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/"
    files=['train.txt', 'dev.txt', 'test.txt']
    word2id={}  # store vocabulary, each word map to a id
    char2id={}
    all_sentences_l=[]
    all_masks_l=[]
    all_sentences_r=[]
    all_masks_r=[]

    all_char_sentences_l=[]
    all_char_masks_l=[]
    all_char_sentences_r=[]
    all_char_masks_r=[]

    all_labels=[]
    max_sen_len=0
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents_l=[]
        sents_masks_l=[]
        sents_r=[]
        sents_masks_r=[]
        sents_char_l=[]
        sents_char_masks_l=[]
        sents_char_r=[]
        sents_char_masks_r=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            parts=line.strip().lower().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)==3:

                label=int(parts[0])  # keep label be 0 or 1
                sentence_wordlist_l=parts[1].strip().split()
                sentence_wordlist_r=parts[2].strip().split()
                l_len=len(sentence_wordlist_l)
                r_len = len(sentence_wordlist_r)
                if l_len > max_sen_len:
                    max_sen_len=l_len
                if r_len > max_sen_len:
                    max_sen_len=r_len
                labels.append(label)
                sent_idlist_l, sent_masklist_l, trunc_l=transfer_wordlist_2_idlist_with_maxlen_return_wordlist(sentence_wordlist_l, word2id, maxlen)
                sent_idlist_r, sent_masklist_r, trunc_r=transfer_wordlist_2_idlist_with_maxlen_return_wordlist(sentence_wordlist_r, word2id, maxlen)
                l_char_idlist, l_char_mask = wordList_to_charIdList(trunc_l, char_len, char2id)
                r_char_idlist, r_char_mask = wordList_to_charIdList(trunc_r, char_len, char2id)
                sents_l.append(sent_idlist_l)
                sents_masks_l.append(sent_masklist_l)
                sents_r.append(sent_idlist_r)
                sents_masks_r.append(sent_masklist_r)

                sents_char_l.append(l_char_idlist)
                sents_char_masks_l.append(l_char_mask)
                sents_char_r.append(r_char_idlist)
                sents_char_masks_r.append(r_char_mask)

        all_sentences_l.append(sents_l)
        all_sentences_r.append(sents_r)
        all_masks_l.append(sents_masks_l)
        all_masks_r.append(sents_masks_r)

        all_char_sentences_l.append(sents_char_l)
        all_char_masks_l.append(sents_char_masks_l)
        all_char_sentences_r.append(sents_char_r)
        all_char_masks_r.append(sents_char_masks_r)

        all_labels.append(labels)
        print '\t\t\t size:', len(labels), 'pairs'
    print 'dataset loaded over, totally ', len(word2id), 'words, max sen len:',   max_sen_len
    return all_sentences_l, all_masks_l, all_sentences_r, all_masks_r, all_char_sentences_l,all_char_masks_l,all_char_sentences_r,all_char_masks_r,all_labels, word2id,char2id

def load_guu_data_4_CompTransE(maxPathLen=20):
    rootPath='/mounts/data/proj/wenpeng/Dataset/FB_socher/path/'
    files=['train_ent_recovered.txt', 'dev_ent_recovered.txt', 'test_ent_recovered.txt']
#     rootPath='/mounts/data/proj/wenpeng/Dataset/FB_socher/length_1/'
#     files=['/mounts/data/proj/wenpeng/Dataset/FB_socher/length_1/train.txt', '/mounts/data/proj/wenpeng/Dataset/FB_socher/path/train_ent_recovered.txt', '/mounts/data/proj/wenpeng/Dataset/FB_socher/path/test_ent_recovered.txt']
    relation_str2id={}
    relation_id2wordlist={}
    rel_id2inid={}
    ent_str2id={}
    tuple2tailset={}
    rel2tailset={}
    ent2relset={}
    ent2relset_maxSetSize=0

    train_paths_store=[]
    train_ents_store=[]
    train_masks_store=[]


    dev_paths_store=[]
    dev_ents_store=[]
    dev_masks_store=[]

    test_paths_store=[]
    test_ents_store=[]
    test_masks_store=[]

    max_path_len=0
    for file_id, fil in enumerate(files):

            filename=rootPath+fil
            print 'loading', filename, '...'
            readfile=open(filename, 'r')
            line_co=0
            for line in readfile:

                parts=line.strip().split('\t')
                ent_list=[]
                rel_list=[]
                for i in range(len(parts)):
                    if i%2==0:
                        ent_list.append(parts[i])
                    else:
                        rel_list.append(parts[i].replace('**', '_'))
                if len(ent_list)!=len(rel_list)+1:
                    print 'len(ent_list)!=len(rel_list)+1:', len(ent_list),len(rel_list)
                    print 'line:', line
                    exit(0)
                ent_path=keylist_2_valuelist(ent_list, ent_str2id, 0)
                one_path=[]
                for potential_relation in rel_list:

                    rel_id=relation_str2id.get(potential_relation)
                    if rel_id is None:
                        rel_id=len(relation_str2id)+1
                        relation_str2id[potential_relation]=rel_id
                    wordlist=potential_relation.split('_')
#                                 wordIdList=strs2ids(potential_relation.split(), word2id)
                    relation_id2wordlist[rel_id]=wordlist
                    one_path.append(rel_id)
                    if rel_id not in rel_id2inid and potential_relation[0]=='_':
                        inID=relation_str2id.get(potential_relation[1:])
                        if inID is not None:
                            rel_id2inid[rel_id]=inID
                add_tuple2tailset(ent_path, one_path, tuple2tailset)
                add_rel2tailset(ent_path, one_path, rel2tailset)
                ent2relset_maxSetSize=add_ent2relset(ent_path, one_path, ent2relset, ent2relset_maxSetSize)

                #pad
                valid_size=len(one_path)
                if valid_size > max_path_len:
                    max_path_len=valid_size
                pad_size=maxPathLen-valid_size
                if pad_size > 0:
                    one_path=[0]*pad_size+one_path
                    # ent_path=ent_path[:pad_size]+ent_path
                    ent_path=ent_path[:1]*(pad_size+1)+ent_path[1:]
                    one_mask=[0.0]*pad_size+[1.0]*valid_size
                else:
                    one_path=one_path[-maxPathLen:]  # select the last max_len relations
                    ent_path=ent_path[:1]+ent_path[-maxPathLen:]
                    one_mask=[1.0]*maxPathLen

                if file_id < 1: #train
                    if len(ent_path)!=maxPathLen+1 or len(one_path) != maxPathLen:
                        print 'len(ent_path)!=5:',len(ent_path), len(one_path)
                        print 'line:', line
                        exit(0)
                    train_paths_store.append(one_path)
                    train_ents_store.append(ent_path)
                    train_masks_store.append(one_mask)
                elif file_id ==1:
                    dev_paths_store.append(one_path)
                    dev_ents_store.append(ent_path)
                    dev_masks_store.append(one_mask)
                else:
                    test_paths_store.append(one_path)
                    test_ents_store.append(ent_path)
                    test_masks_store.append(one_mask)

                # line_co+=1
                # if line_co==10000:#==0:
                #     #  print line_co
                #     break

            readfile.close()
            print '\t\t\t\tload over, overall ',    len(train_paths_store), ' train,', len(dev_paths_store), ' dev,', len(test_paths_store), ' test,', 'tuple2tailset size:', len(tuple2tailset),', max path len:', max_path_len, 'max ent2relsetSize:', ent2relset_maxSetSize

    return ((train_paths_store, train_masks_store, train_ents_store),
            (dev_paths_store, dev_masks_store, dev_ents_store),
            (test_paths_store, test_masks_store, test_ents_store)) , relation_id2wordlist,ent_str2id, relation_str2id, tuple2tailset, rel2tailset, ent2relset, ent2relset_maxSetSize, rel_id2inid
def keylist_2_valuelist(keylist, dic, start_index=0):
    value_list=[]
    for key in keylist:
        value=dic.get(key)
        if value is None:
            value=len(dic)+start_index
            dic[key]=value
        value_list.append(value)
    return value_list

def add_tuple2tailset(ent_path, one_path, tuple2tailset):
    size=len(one_path)
    if len(ent_path)!=size+1:
        print 'len(ent_path)!=len(one_path)+1:', len(ent_path),size
        exit(0)
    for i in range(size):
        tuple=(ent_path[i], one_path[i])
        tail=ent_path[i+1]
        tailset=tuple2tailset.get(tuple)
        if tailset is None:
            tailset=set()
        if tail not in tailset:
            tailset.add(tail)
            tuple2tailset[tuple]=tailset
def add_rel2tailset(ent_path, one_path, rel2tailset):
    size=len(one_path)
    if len(ent_path)!=size+1:
        print 'len(ent_path)!=len(one_path)+1:', len(ent_path),size
        exit(0)
    for i in range(size):
#         tuple=(ent_path[i], one_path[i])
        tail=ent_path[i+1]
        rel=one_path[i]
        tailset=rel2tailset.get(rel)
        if tailset is None:
            tailset=set()
        if tail not in tailset:
            tailset.add(tail)
            rel2tailset[rel]=tailset
def add_ent2relset(ent_path, one_path, ent2relset, maxSetSize):
    size=len(one_path)
    if len(ent_path)!=size+1:
        print 'len(ent_path)!=len(one_path)+1:', len(ent_path),size
        exit(0)
    for i in range(size):
        ent_id=ent_path[i+1]
        rel_id=one_path[i]
        relset=ent2relset.get(ent_id)
        if relset is None:
            relset=set()
        if rel_id not in relset:
            relset.add(rel_id)
            if len(relset) > maxSetSize:
                maxSetSize=len(relset)
            ent2relset[ent_id]=relset
    return maxSetSize

def sent_parse_relclassify(raw_sent):
    ent1_left=raw_sent.find('<e1>')
    ent1_right = raw_sent.find('</e1>')
    ent2_left=raw_sent.find('<e2>')
    ent2_right = raw_sent.find('</e2>')
    if ent1_left==-1 or ent1_right ==-1 or ent2_left==-1 or ent2_right ==-1:
        print 'ent1_left==-1 or ent1_right ==-1 or ent2_left==-1 or ent2_right ==-1:', raw_sent
        exit(0)
    else:
        ent1_str=raw_sent[ent1_left+4:ent1_right]
        ent2_str=raw_sent[ent2_left+4:ent2_right]
        left_context=raw_sent[:ent1_left].strip()
        mid_context = raw_sent[ent1_right+5:ent2_left].strip()
        right_context = raw_sent[ent2_right+5:].strip()
        if left_context =='':
            left_context='<PAD>'
        if mid_context =='':
            mid_context ='<PAD>'
        if right_context =='':
            right_context ='<PAD>'
        return left_context, ent1_str+' '+mid_context+' '+ent2_str, right_context

def load_heike_rel_dataset(maxlen=20):
    root="/mounts/data/proj/wenpeng/Dataset/rel_classify_heike/"
    files=['SemEval2010_task8_train.txt', 'SemEval2010_task8_test_withLabels.txt']
    word2id={}  # store vocabulary, each word map to a id
    left_sents=[]
    left_masks=[]
    mid_sents=[]
    mid_masks=[]
    right_sents=[]
    right_masks=[]
    all_labels=[]
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        left_s=[]
        left_m=[]
        mid_s=[]
        mid_m=[]
        right_s=[]
        right_m=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            split_point=line.strip().find(':')
            if split_point == -1:
                continue
            else:
                label=int(line.strip()[:split_point].strip())
                raw_sent=line.strip()[split_point+1:].strip()
                left_context, mid_context, right_context = sent_parse_relclassify(raw_sent)




                left_idlist, left_masklist=transfer_wordlist_2_idlist_with_maxlen(left_context.split(), word2id, maxlen)
                mid_idlist, mid_masklist=transfer_wordlist_2_idlist_with_maxlen(mid_context.split(), word2id, maxlen)
                right_idlist, right_masklist=transfer_wordlist_2_idlist_with_maxlen(right_context.split(), word2id, maxlen)

                left_s.append(left_idlist)
                left_m.append(left_masklist)
                mid_s.append(mid_idlist)
                mid_m.append(mid_masklist)
                right_s.append(right_idlist)
                right_m.append(right_masklist)
                labels.append(label)
        left_sents.append(left_s)
        left_masks.append(left_m)
        mid_sents.append(mid_s)
        mid_masks.append(mid_m)
        right_sents.append(right_s)
        right_masks.append(right_m)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels)
    print 'dataset loaded over, totally ', len(word2id), 'words'
    return     left_sents,left_masks,mid_sents,mid_masks,right_sents,right_masks,all_labels, word2id

def process_one_block_wikiQA(block, word2id, maxlen):
    Q=''
    AP=[]
    AN=[]
    for (Q_i, A_i, label_i) in block:
        if Q!='' and Q_i!=Q:
            print 'Q!='' and Q_i!=Q:', Q,Q_i
            exit(0)
        Q=Q_i
        if label_i =='1':
            AP.append(A_i)
        else:
            AN.append(A_i)
#     if len(AP)>1:
#         print 'more than one positive answers:', block
#         exit(0)
#     if len(AP)==0:
#         AP.append(Q)

    Q_id_list=[]
    Q_mask_list=[]
    AP_id_list=[]
    AP_mask_list=[]
    AN_id_list=[]
    AN_mask_list=[]
    Q_id, Q_mask = transfer_wordlist_2_idlist_with_maxlen(Q.strip().split(), word2id, maxlen)
    for ap in AP:
        for an in AN:
            ap_idlist, ap_masklist=transfer_wordlist_2_idlist_with_maxlen(ap.strip().split(), word2id, maxlen)
            an_idlist, an_masklist=transfer_wordlist_2_idlist_with_maxlen(an.strip().split(), word2id, maxlen)
            Q_id_list.append(Q_id)
            Q_mask_list.append(Q_mask)
            AP_id_list.append(ap_idlist)
            AP_mask_list.append(ap_masklist)
            AN_id_list.append(an_idlist)
            AN_mask_list.append(an_masklist)

    return     Q_id_list,Q_mask_list,AP_id_list,AP_mask_list,AN_id_list,AN_mask_list


def load_wikiQA_train(filename, word2id, maxlen=20):

    Q_ids=[]
    Q_masks=[]
    AP_ids=[]
    AP_masks=[]
    AN_ids=[]
    AN_masks=[]

    readfile=open(filename, 'r')
    old_Q=''
    block_store=[]
    for line in readfile:
        parts=line.strip().split('\t')
        Q=parts[0]
        A=parts[1]
        label=parts[2]
        if Q != old_Q:
            if len(block_store)>0: #start a new block
                Q_id_list, Q_mask_list, AP_id_list, AP_mask_list, AN_id_list, AN_mask_list = process_one_block_wikiQA(block_store, word2id, maxlen)

                Q_ids+=Q_id_list
                Q_masks+=Q_mask_list
                AP_ids+=AP_id_list
                AP_masks+=AP_mask_list
                AN_ids+=AN_id_list
                AN_masks+=AN_mask_list

                block_store=[]
#             block_store.append((Q,A,label))
            old_Q=Q

        block_store.append((Q,A,label))
    readfile.close()
    print 'load training data over, totally size:', len(Q_ids)
    return     Q_ids,Q_masks,AP_ids,AP_masks,AN_ids,AN_masks, word2id

def load_wikiQA_devOrTest(filename, word2id, maxlen=20):
    Q_ids=[]
    Q_masks=[]
    AP_ids=[]
    AP_masks=[]

    readfile=open(filename, 'r')
    for line in readfile:
        parts=line.strip().split('\t')
        Q=parts[0]
        A=parts[1]
#         label=parts[2]
        Q_idlist, Q_masklist=transfer_wordlist_2_idlist_with_maxlen(Q.strip().split(), word2id, maxlen)
        A_idlist, A_masklist=transfer_wordlist_2_idlist_with_maxlen(A.strip().split(), word2id, maxlen)

        Q_ids.append(Q_idlist)
        Q_masks.append(Q_masklist)
        AP_ids.append(A_idlist)
        AP_masks.append(A_masklist)

    readfile.close()
    print 'load test or dev data over, totally size:', len(Q_ids)
    return     Q_ids,Q_masks,AP_ids,AP_masks,word2id

def compute_map_mrr(filename, probs):
    #file
    testread=open(filename, 'r')
    separate=[]
    labels=[]
    pre_q=' '
    line_no=0
    for line in testread:
        parts=line.strip().split('\t')
        if len(parts)>=3:
            if parts[0]!=pre_q:
                separate.append(line_no)
            labels.append(int(parts[2]))
            pre_q=parts[0]
            line_no+=1
    testread.close()
    separate.append(line_no)#the end of file
    #compute MAP, MRR
    question_no=len(separate)-1
    all_map=0.0
    all_mrr=0.0
    all_corr_answer=0
    all_acc=0.0
    for i in range(question_no):
        sub_labels=labels[separate[i]:separate[i+1]]
        sub_probs=probs[separate[i]:separate[i+1]]
        sub_dict = [(prob, label) for prob, label in izip(sub_probs, sub_labels)] # a list of tuple
        #sorted_probs=sorted(sub_probs, reverse = True)
        sorted_tuples=sorted(sub_dict,key=lambda tup: tup[0], reverse = True)
        map=0.0
        find=False
        corr_no=0

        #MAP
        for index, (prob,label) in enumerate(sorted_tuples):
            if label==1:
                corr_no+=1 # the no of correct answers
                all_corr_answer+=1
                map+=1.0*corr_no/(index+1)
                find=True
                if index ==0:
                    all_acc+=1
        #MRR
        for index, (prob,label) in enumerate(sorted_tuples):
            if label==1:
                all_mrr+=1.0/(index+1)
                break # only consider the first correct answer

        if corr_no ==0:
            map=0.0
        else:
            map=map/corr_no
        all_map+=map
    MAP=all_map/question_no
    MRR=all_mrr/question_no
    ACC=all_acc/question_no


    return MAP, MRR, ACC

def retrieve_top1_sent(filename, probs, topN):
    #file
    testread=open(filename, 'r')
    separate=[]
    labels=[]
    sents=[]
    questions=[]
    answers=[]
    q_ids=[]
    pre_q=' '
    line_no=0
    for line in testread:
        parts=line.strip().split('\t')
        if len(parts)>=3:
            if parts[0]!=pre_q:
                separate.append(line_no)
            questions.append(parts[0])
            sents.append(parts[1])
            labels.append(int(parts[2]))
            if len(parts)==4:
                answers.append(parts[3])
            else:
                answers.append(' ')
            pre_q=parts[0]
            line_no+=1
        else:
            q_ids.append(parts[0])


    testread.close()
    writefile = open('/mounts/data/proj/wenpeng/Dataset/SQuAD/dev-TwoStageRanking-SpanLevel-20170802.txt', 'w')
    separate.append(line_no)#the end of file
    #compute MAP, MRR
    question_no=len(separate)-1
    if question_no!=len(q_ids):
        print 'question_no!=len(q_ids):', question_no, len(q_ids)
        exit(0)
    all_acc=0.0
    for i in range(question_no):
        q_id = q_ids[i]
        sub_labels=labels[separate[i]:separate[i+1]]
        sub_probs=probs[separate[i]:separate[i+1]]
        sub_questions = questions[separate[i]:separate[i+1]]
        sub_sents = sents[separate[i]:separate[i+1]]
        sents_ids = range(separate[i+1] - separate[i])
        sub_ans = [x for x in answers[separate[i]:separate[i+1]] if x !=' ']
        sub_dict = [(prob, label, sent, sent_id) for prob, label, sent, sent_id in izip(sub_probs, sub_labels, sub_sents, sents_ids)] # a list of tuple
        sorted_tuples=sorted(sub_dict,key=lambda tup: tup[0], reverse = True)

        #MAP
        top1_sent=''
        top1_label=0
        top1_sent_id=0
        in_top2_flag=False
        for index, (prob,label, sent, sent_id) in enumerate(sorted_tuples):
            if index==0:
                if label ==1:
#                     all_acc+=1
#                     top1_label = label
                    in_top2_flag = True
                top1_sent += ' '+sent
                top1_sent_id = sent_id
            elif index ==1:
                if sent_id+1 == top1_sent_id:
                    top1_sent= sent+' '+top1_sent
                    if label ==1:
                        in_top2_flag=True
                elif top1_sent_id+1 == sent_id:
                    top1_sent +=' '+sent
                    if label ==1:
                        in_top2_flag=True
                else:
                    break


            else:
                break
        if len(set(sub_questions))!=1:
            print 'len(set(sub_questions))!=1:'
            print sub_questions
            exit(0)
        if in_top2_flag:
            all_acc+=1
            top1_label = 1
        writefile.write(q_id+'\t'+sub_questions[0]+'\t'+top1_sent.strip()+'\t'+str(top1_label)+'\t'+' || '.join(sub_ans)+'\n')

    ACC=all_acc/question_no
    print '\t\t\t\tretrieve top 1 dev sent over, p@1 indeed: ', ACC
    writefile.close()

def load_POS_dataset(maxlen=40):

    files=['/mounts/Users/student/wenpeng/FLORS/datasets/train-wsj-02-21',
           '/mounts/Users/student/wenpeng/FLORS/datasets/Google_Task/target/wsj/gweb-wsj-dev',
           '/mounts/Users/student/wenpeng/FLORS/datasets/Google_Task/target/wsj/gweb-wsj-test']
    word2id={}  # store vocabulary, each word map to a id
    pos2id={}
    all_sentences=[]
    all_masks=[]
    all_labels=[]
    for i in range(len(files)):
        print 'loading file:', files[i], '...'

        sents=[]
        sents_masks=[]
        labels=[]
        readfile=open(files[i], 'r')
        sentence_wordlist=[]
        sentence_poslist =[]
        for line in readfile:
            if len(line.strip()) > 0:
                parts=line.strip().split('\t') #word, pos
                if len(parts)!=2:
                    print 'len(parts)!=2:', line
                    exit(0)
                sentence_wordlist.append(parts[0])
                sentence_poslist.append(parts[1])
#                 sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
#                 if len(parts) > minlen: # we only consider some sentences that are not too short, controlled by minlen
#                     label=int(parts[0])-1  # keep label be 0 or 1
#                     sentence_wordlist=parts[1:]
#
#                     labels.append(label)
#                     sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
#                     sents.append(sent_idlist)
#                     sents_masks.append(sent_masklist)
            else:#store current sentence
                sent_idlist, sent_masklist1=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
                pos_idlist, sent_masklist2=transfer_wordlist_2_idlist_with_maxlen(sentence_poslist, pos2id, maxlen)
                if       sent_masklist1 !=sent_masklist2:
                    print        'sent_masklist1 !=sent_masklist2:', sent_masklist1,sent_masklist2
                    exit(0)
                sents.append(sent_idlist)
                sents_masks.append(sent_masklist1)
                labels.append(pos_idlist)
                sentence_wordlist=[]
                sentence_poslist =[]
        all_sentences.append(sents)
        all_masks.append(sents_masks)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels)
    print 'dataset loaded over, totally ', len(word2id), 'words'
    return all_sentences, all_masks, all_labels, word2id, pos2id

def load_duyu_marco_dataset(maxlen_q=15, maxlen_s=40):
    root="/mounts/data/proj/wenpeng/Dataset/SQuAD/"
    files=['train-TwoStageRanking.txt', 'dev-TwoStageRanking.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences_l=[]
    all_masks_l=[]
    all_sentences_r=[]
    all_masks_r=[]
    all_labels=[]
    max_sen_len_q=0
    max_sen_len_s=0
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents_l=[]
        sents_masks_l=[]
        sents_r=[]
        sents_masks_r=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            parts=line.strip().lower().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)>=3:

                label=int(parts[2])  # keep label be 0 or 1
                sentence_wordlist_l=parts[0].strip().split()
                sentence_wordlist_r=parts[1].strip().split()
                l_len=len(sentence_wordlist_l)
                r_len = len(sentence_wordlist_r)
                if l_len > max_sen_len_q:
                    max_sen_len_q=l_len
                if r_len > max_sen_len_s:
                    max_sen_len_s=r_len
                labels.append(label)
                sent_idlist_l, sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_l, word2id, maxlen_q)
                sent_idlist_r, sent_masklist_r=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_r, word2id, maxlen_s)
                sents_l.append(sent_idlist_l)
                sents_masks_l.append(sent_masklist_l)
                sents_r.append(sent_idlist_r)
                sents_masks_r.append(sent_masklist_r)
        all_sentences_l.append(sents_l)
        all_sentences_r.append(sents_r)
        all_masks_l.append(sents_masks_l)
        all_masks_r.append(sents_masks_r)
        all_labels.append(labels)
        labels_array = numpy.asarray(labels)
        print '\t\t\t size:', len(labels), 'pairs, posi rato: ', numpy.sum(labels_array)*1.0/len(labels)
    print 'dataset loaded over, totally ', len(word2id), 'words, max sen len:',   max_sen_len_q, max_sen_len_s
    return all_sentences_l, all_masks_l, all_sentences_r, all_masks_r,all_labels, word2id


# def extra_two_wordlist_for_truncateQuestion(wordlist1, wordlist2, stopwords,stemmer):
#     wordlist1 = [x for x in wordlist1 if x not in string.punctuation]
#     wordlist2 = [x for x in wordlist2 if x not in string.punctuation]
#
#     wordlist1_nostop = [x for x in wordlist1 if x not in stopwords]
#     wordlist2_nostop = [x for x in wordlist2 if x not in stopwords]
#
#     wordlist1_stem = [stemmer.stem(x) for x in wordlist1_nostop]
#     wordlist2_stem = [stemmer.stem(x) for x in wordlist2_nostop]
#
#
#
#
#     if len(wordlist1_nostop)==0 or len(wordlist2_nostop) ==0:
#         return [0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0]
#     else:
#         word_overalp = set(wordlist1_nostop) & set(wordlist2_nostop)
#         word_overalp_stem = set(wordlist1_stem) & set(wordlist2_stem)
#         feature_1 = len(word_overalp)*1.0/len(set(wordlist1))
#         feature_2 = len(word_overalp)*1.0/len(set(wordlist2))
#         feature_3 = len(word_overalp_stem)*1.0/len(set(wordlist1))
#         feature_4 = len(word_overalp_stem)*1.0/len(set(wordlist2))
#
#         feature_5 = len(word_overalp)*1.0/len(set(wordlist1_nostop))
#         feature_6 = len(word_overalp)*1.0/len(set(wordlist2_nostop))
#         feature_7 = len(word_overalp_stem)*1.0/len(set(wordlist1_nostop))
#         feature_8 = len(word_overalp_stem)*1.0/len(set(wordlist2_nostop))
#         return [feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8]#, 1.0/(len(wordlist1)+1.0),1.0/(len(wordlist2)+1.0)]



def contains_digits(d):
    return bool(_digits.search(d))

def extra_two_wordlist_lowercase(wordlist1, wordlist2, stopwords,stemmer):
    wordlist1 = [x for x in wordlist1 if x not in string.punctuation]
    wordlist2 = [x for x in wordlist2 if x not in string.punctuation]

    wordlist1_nostop = [x for x in wordlist1 if x not in stopwords]
    wordlist2_nostop = [x for x in wordlist2 if x not in stopwords]

    wordlist1_stem = [stemmer.stem(x) for x in wordlist1_nostop]
    wordlist2_stem = [stemmer.stem(x) for x in wordlist2_nostop]

#     extra_vec = extra_two_wordlist_for_truncateQuestion(wordlist1[:2]+wordlist1[-3:-1], wordlist2, stopwords,stemmer)
    epslon=1.0
    months=set(['january','february','march','april','may','june','july','august','september','october','november','december'])
    digit_features=[0.0, 0.0, 0.0] #two features: year, month, value
    for word in wordlist2:
        if word.isdigit():
            if len(word)==4:
                digit_features[0]=epslon  #year
            else:
                digit_features[2]=epslon
        elif contains_digits(word):
            digit_features[2]=epslon
    if len(set(wordlist2) & months)>0:
        digit_features[1]=epslon




    qtypes=['what', 'when', 'where','which','how long', 'how many', 'how much', 'who', 'whose', 'why','how','year']    # size 12
    qtype_vec=[0.0]*len(qtypes) #what, when, which, how many, how much, who, year
    q_sent=' '.join(wordlist1)
    q_word_set=set(wordlist1)
    for i, type_word in enumerate(qtypes):
        if type_word in q_word_set or q_sent.find(type_word)>=0:
            qtype_vec[i]=epslon



    if len(wordlist1_nostop)==0 or len(wordlist2_nostop) ==0:
        return [0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0]+digit_features+qtype_vec
    else:
        word_overalp = set(wordlist1_nostop) & set(wordlist2_nostop)
        word_overalp_stem = set(wordlist1_stem) & set(wordlist2_stem)
        feature_1 = len(word_overalp)*1.0/len(set(wordlist1))
        feature_2 = len(word_overalp)*1.0/len(set(wordlist2))
        feature_3 = len(word_overalp_stem)*1.0/len(set(wordlist1))
        feature_4 = len(word_overalp_stem)*1.0/len(set(wordlist2))

        feature_5 = len(word_overalp)*1.0/len(set(wordlist1_nostop))
        feature_6 = len(word_overalp)*1.0/len(set(wordlist2_nostop))
        feature_7 = len(word_overalp_stem)*1.0/len(set(wordlist1_nostop))
        feature_8 = len(word_overalp_stem)*1.0/len(set(wordlist2_nostop))
        return [feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8]+digit_features+qtype_vec#, 1.0/(len(wordlist1)+1.0),1.0/(len(wordlist2)+1.0)]


def extra_two_wordlist(wordlist1, wordlist2, stopwords,stemmer):
    wordlist1_lower = [x.lower() for x in wordlist1]
    wordlist2_lower = [x.lower() for x in wordlist2]

    wordlist1 = [x for x in wordlist1 if x not in string.punctuation]
    wordlist2 = [x for x in wordlist2 if x not in string.punctuation]

    wordlist1_nostop = [x for x in wordlist1 if x not in stopwords]
    wordlist2_nostop = [x for x in wordlist2 if x not in stopwords]

    wordlist1_stem = [stemmer.stem(x) for x in wordlist1_nostop]
    wordlist2_stem = [stemmer.stem(x) for x in wordlist2_nostop]


    lower_vec = extra_two_wordlist_lowercase(wordlist1_lower, wordlist2_lower,stopwords,stemmer)
#     extra_vec = extra_two_wordlist_for_truncateQuestion(wordlist1[:2]+wordlist1[-3:-1], wordlist2, stopwords,stemmer)
    epslon=1.0
    months=set(['January','February','March','April','May','June','July','August','September','October','November','December'])  #12
    digit_features=[0.0, 0.0, 0.0] #two features: year, month, digit
    for word in wordlist2:
        if word.isdigit():
            if len(word)==4:
                digit_features[0]=epslon  #year
            else:
                digit_features[2]=epslon
        elif contains_digits(word):
            digit_features[2]=epslon
    if len(set(wordlist2) & months)>0:
        digit_features[1]=epslon




    qtypes=['What', 'When', 'Where','Which','How long', 'How many', 'How much', 'Who', 'Whose', 'Why','How','year']    # size 12
    qtype_vec=[0.0]*len(qtypes) #what, when, which, how many, how much, who, year
    q_sent=' '.join(wordlist1)
    q_word_set=set(wordlist1)
    for i, type_word in enumerate(qtypes):
        if type_word in q_word_set or q_sent.find(type_word)>=0:
            qtype_vec[i]=epslon



    if len(wordlist1_nostop)==0 or len(wordlist2_nostop) ==0:
        return [0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0]+digit_features+qtype_vec+lower_vec
    else:
        word_overalp = set(wordlist1_nostop) & set(wordlist2_nostop)
        word_overalp_stem = set(wordlist1_stem) & set(wordlist2_stem)
        feature_1 = len(word_overalp)*1.0/len(set(wordlist1))
        feature_2 = len(word_overalp)*1.0/len(set(wordlist2))
        feature_3 = len(word_overalp_stem)*1.0/len(set(wordlist1))
        feature_4 = len(word_overalp_stem)*1.0/len(set(wordlist2))

        feature_5 = len(word_overalp)*1.0/len(set(wordlist1_nostop))
        feature_6 = len(word_overalp)*1.0/len(set(wordlist2_nostop))
        feature_7 = len(word_overalp_stem)*1.0/len(set(wordlist1_nostop))
        feature_8 = len(word_overalp_stem)*1.0/len(set(wordlist2_nostop))
        return [feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8]+digit_features+qtype_vec+lower_vec#, 1.0/(len(wordlist1)+1.0),1.0/(len(wordlist2)+1.0)]


def load_squad_TwoStageRanking_dataset(maxlen_q=15, maxlen_s=40):
#     from nltk.stem import SnowballStemmer
    stopwords = []
    stopfile=open('stopwords.txt', 'r')
    for line in stopfile:
        stopwords.append(line.strip())
    stopfile.close()
    stopwords=set(stopwords)

    stemmer = SnowballStemmer('english')
    root="/mounts/data/proj/wenpeng/Dataset/SQuAD/"
    files=['train-TwoStageRanking.txt', 'dev-TwoStageRanking.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences_l=[]
    all_masks_l=[]
    all_sentences_r=[]
    all_masks_r=[]
    all_labels=[]
    all_extra = []
    max_sen_len_q=0
    max_sen_len_s=0
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents_l=[]
        sents_masks_l=[]
        sents_r=[]
        sents_masks_r=[]
        extra=[]
        labels=[]
        readfile=codecs.open(root+files[i], 'r', "utf-8")
        co_line=0
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)>=3:

                label=int(parts[2])  # keep label be 0 or 1
                # sentence_wordlist_l=[i for i in parts[0].strip().split() if i not in string.punctuation]
                # sentence_wordlist_r=[i for i in parts[1].strip().split() if i not in string.punctuation]

                sentence_wordlist_l=parts[0].strip().split()#questions
                sentence_wordlist_r=parts[1].strip().split()
                extra.append(extra_two_wordlist(sentence_wordlist_l, sentence_wordlist_r,stopwords, stemmer))
                l_len=len(sentence_wordlist_l)
                r_len = len(sentence_wordlist_r)
                if l_len > max_sen_len_q:
                    max_sen_len_q=l_len
                if r_len > max_sen_len_s:
                    max_sen_len_s=r_len
                labels.append(label)
                sent_idlist_l, sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_l, word2id, maxlen_q)
                sent_idlist_r, sent_masklist_r=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_r, word2id, maxlen_s)
                sents_l.append(sent_idlist_l)
                sents_masks_l.append(sent_masklist_l)
                sents_r.append(sent_idlist_r)
                sents_masks_r.append(sent_masklist_r)
            co_line+=1
#             if co_line%1000==0:
#                 print co_line, '...'
        all_sentences_l.append(sents_l)
        all_sentences_r.append(sents_r)
        all_masks_l.append(sents_masks_l)
        all_masks_r.append(sents_masks_r)
        all_labels.append(labels)
        all_extra.append(extra)
        labels_array = numpy.asarray(labels)
        print '\t\t\t size:', len(labels), 'pairs, posi rato: ', numpy.sum(labels_array)*1.0/len(labels)
        readfile.close()
    print 'dataset loaded over, totally ', len(word2id), 'words, max sen len:',   max_sen_len_q, max_sen_len_s
    return all_sentences_l, all_masks_l, all_sentences_r, all_masks_r,all_labels, all_extra, word2id

def create_squad_question_classify_wh_word():
    import random
    root="/mounts/data/proj/wenpeng/Dataset/SQuAD/"
    files=['train-TwoStageRanking.txt', 'dev-TwoStageRanking.txt']
    writefiles=['train-question_classify_wh_word.txt', 'dev-question_classify_wh_word.txt']
    wh_word_set = set(['What', 'Where', 'When', 'Who', 'Which', 'Whom', 'what', 'where', 'when', 'who', 'which', 'whom'])
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'
        writefile=codecs.open(root+writefiles[i], 'w', "utf-8")


        readfile=codecs.open(root+files[i], 'r', "utf-8")
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)>=3:
                question = parts[0]
                question_wordlist = question.split()
                sentence = parts[1]
                label=int(parts[2])  # keep label be 0 or 1
                overlap_wh = wh_word_set & set(question.split())
                if label == 1 and len(overlap_wh)>0:
                    #write pos
                    ans = parts[3]
#                     if question = 'What sits on top of the Main Building at Notre Dame ?':
#                         print sentence
                    start = len(sentence[:sentence.find(ans)].split())
                    end = start + len(ans.split())-1
                    writefile.write(question+'\t'+sentence+'\t'+str(start)+'\t'+str(end)+'\t'+str(label)+'\t'+ans+'\n')
                    #write neg
                    new_question = []
                    for word in question_wordlist:
                        if word in wh_word_set:
                            replace_wh_word = random.sample(wh_word_set-set([word]), 1)[0]
                            new_question.append(replace_wh_word)
                        else:
                            new_question.append(word)
                    writefile.write(' '.join(new_question)+'\t'+sentence+'\t'+str(start)+'\t'+str(end)+'\t'+str(0)+'\t'+ans+'\n')


        readfile.close()
        writefile.close()
    print 'over'
def load_squad_question_classify_wh_word(maxlen=40, maxlen_q=40):
    root="/mounts/data/proj/wenpeng/Dataset/SQuAD/"
    files=['train-question_classify_wh_word.txt', 'dev-question_classify_wh_word.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences=[]
    all_masks=[]
    all_qs=[]
    all_qs_masks=[]
    boundaries=[]
    all_labels=[]
    max_right_b = 0
    dev_content = []
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents=[]
        sents_masks=[]
        q=[]
        q_mask=[]
        boundary=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            if i ==1:
                dev_content.append(line.strip())
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            question_wordlist = parts[0].split()
            sentence_wordlist = parts[1].split()

            start = int(parts[2])
            end = int(parts[3])
            if end > max_right_b:
                max_right_b = end
            label=int(parts[4])  # keep label be 0 or 1

            labels.append(label)
            sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
            q_idlist, q_masklist=transfer_wordlist_2_idlist_with_maxlen(question_wordlist, word2id, maxlen_q)

            sents.append(sent_idlist)
            sents_masks.append(sent_masklist)
            q.append(q_idlist)
            q_mask.append(q_masklist)
            boundary.append([start, end])

        all_sentences.append(sents)
        all_masks.append(sents_masks)
        all_qs.append(q)
        all_qs_masks.append(q_mask)
        boundaries.append(boundary)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels), ' max_right_b:', max_right_b
    print 'dataset loaded over, totally ', len(word2id), 'words'
    return all_sentences, all_masks, all_qs, all_qs_masks, boundaries, all_labels, word2id,dev_content
def load_duyu_dataset(strr='marco', maxlen_q=15, maxlen_s=40):
    #used for ranking loss
    root="/mounts/Users/student/wenpeng/Duyu/to-wenpeng-"+strr+"/"
    files=['train.txt', 'dev.txt', 'test.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences_l=[]
    all_masks_l=[]
    all_sentences_r=[]
    all_masks_r=[]
    all_labels=[]
    max_sen_len_q=0
    max_sen_len_s=0
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents_l=[]
        sents_masks_l=[]
        sents_r=[]
        sents_masks_r=[]
        labels=[]
        readfile=open(root+files[i], 'r')
        for line in readfile:
            parts=line.strip().lower().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)==3:

                label=int(parts[2])  # keep label be 0 or 1
                sentence_wordlist_l=parts[0].strip().lower().split()
                sentence_wordlist_r=parts[1].strip().lower().split()
                l_len=len(sentence_wordlist_l)
                r_len = len(sentence_wordlist_r)
                if l_len > max_sen_len_q:
                    max_sen_len_q=l_len
                if r_len > max_sen_len_s:
                    max_sen_len_s=r_len
                labels.append(label)
                sent_idlist_l, sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_l, word2id, maxlen_q)
                sent_idlist_r, sent_masklist_r=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist_r, word2id, maxlen_s)
                sents_l.append(sent_idlist_l)
                sents_masks_l.append(sent_masklist_l)
                sents_r.append(sent_idlist_r)
                sents_masks_r.append(sent_masklist_r)
        all_sentences_l.append(sents_l)
        all_sentences_r.append(sents_r)
        all_masks_l.append(sents_masks_l)
        all_masks_r.append(sents_masks_r)
        all_labels.append(labels)
        labels_array = numpy.asarray(labels)
        print '\t\t\t size:', len(labels), 'pairs, posi rato: ', numpy.sum(labels_array)*1.0/len(labels)
    print 'dataset loaded over, totally ', len(word2id), 'words, max sen len:',   max_sen_len_q, max_sen_len_s
    return all_sentences_l, all_masks_l, all_sentences_r, all_masks_r,all_labels, word2id

def extend_word2vec_lowercase(word2vec):
    vocab_set = set(word2vec.keys())
    size=0
    for word in vocab_set:
        if word.islower() == False: # has upper
            word_lower = word.lower()
            if word_lower not in vocab_set:
                word2vec[word_lower]=word2vec.get(word)
                size+=1
    print 'extend word2vec over, size: ', size
    return word2vec
