import logging
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk.data
from gensim.models import word2vec
import json


def doc_towordlist(doc, RemoveStopWords = False):
    # Convert all letter to lower case
    doc = doc.lower()
    # Remove non-letter and split it
    tokenizer = RegexpTokenizer(r'\w+')
    wordlist = tokenizer.tokenize(doc)
    # Remove the stop word if necessary
    if RemoveStopWords == True:
        stopw = stopwords.words('english')
        tmpwords = wordlist[:]
        for item in tmpwords:
            if item.lower() in stopw:
                wordlist.remove(item)
    return wordlist


def trainword2vec(doclist):
    # Get a list of list of word
    sentences = []
    for item in doclist:
        sentences += [doc_towordlist(item['question'])]
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # Initialize the parameters
    num_features = 300
    min_word_count = 10
    num_workers = 4
    context = 10
    downsampling = 1e-3
    print('Training...')
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
    model.init_sims(replace=True)
    model_name = "question__word_features"
    print('Saving...')
    model.save(model_name)
    print('Done')
    # print(model.most_similar("queen"))
    return model


if __name__ == '__main__': 
    # Read data from .csv files
    print('Reading Trainset for word2vec')
    f = open('F:\浙大\科研\SRTP\dataset\OpenEnded_mscoco_train2014_questions.json', 'r')
    jsontext = json.loads(f.readline())
    qlist = jsontext['questions']
    # Word Vectors loading
    try:
        print('Model loading')
        model = word2vec.Word2Vec.load("question__word_features")
        print('Done')
    except:
        print('preModel not found, now get a new trainning model')
        model = trainword2vec(qlist)
    
    # Add vectors
    word_vector = model.syn0
    word_index = model.index2word
    img_id = []
    ques_id = []
    ques_vector = []
    i = 0
    for item in qlist:
        tmpques = doc_towordlist(item['question'])
        tmp_vector = np.zeros(word_vector.shape[1])
        for word in tmpques:
            if word in word_index:
                tmp_vector += word_vector[word_index.index(word)]
        tmp_imgid = np.zeros(word_vector.shape[1]).astype(np.int32) + item['image_id']
        tmp_quesid = np.zeros(word_vector.shape[1]).astype(np.int32) + item['question_id']
        # 300 rows per question
        ques_vector += tmp_vector.tolist()
        img_id += tmp_imgid.tolist()
        ques_id += tmp_quesid.tolist() 
        i += 1
        print('Question', i, 'processed')
    print('Write into csv files...')
    output = pd.DataFrame(data={"img_id":img_id, "ques_id":ques_id, "vectors":ques_vector})
    output.to_csv( "question_vector.csv", index=False, quoting=3 )
