import kerasvgg as kvgg
from sklearn.externals import joblib
from spacy.en import English
from keras.models import model_from_json 
from keras.optimizers import SGD
import cv2, numpy as np
from utils import grouper, selectFrequentAnswers
import spacy
import os

def get_ques_tensor(questions, nlp, timesteps):
    nb_samples = len(questions)
    word_vec_dim = nlp(questions[0])[0].vector.shape[0]
    questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
    for i in xrange(len(questions)):
        tokens = nlp(questions[i])
        for j in xrange(len(tokens)):
            if j<timesteps:
                questions_tensor[i,j,:] = tokens[j].vector
    return questions_tensor

#TODO: Find a way to return a list  of top five elements from the list.

def load():
    print ("Loading English")
    nlp = spacy.load('en', vectors = 'en_glove_cc_300_1m_vectors')
    
    
    print ("Loading encoder")
    encoder = joblib.load(os.path.abspath('../models/encoder.pkl'))
    print (len(list(encoder.classes_)))
    
    print ("Loading model")
    model = model_from_json(open(os.path.abspath('../models/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3.json')).read())
    print ("Loading Weights")
    model.load_weights(os.path.abspath('../models/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_epoch_070.hdf5'))
    print ("Compiling Model")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop')
    print ('Loaded')
    return nlp, encoder, model    

def image_matrix(nlp, encoder, model, path):
    print("Loading Image")     
    im = cv2.resize(cv2.imread(path), (224, 224))
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    print("Generating Image vectors")
    img_model = kvgg.VGG_16('../models/vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    img_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    img_matrix = img_model.predict(im)
    return img_matrix


##Below code heavily derived from Avi Singh's repo
def predict(nlp, encoder, model, question, img_matrix):
    
        timesteps = len(nlp(question))
        print("Getting question tensor")
        ques_tensor = get_ques_tensor([question], nlp, timesteps)
        concat_matrix = [ques_tensor, img_matrix]
        print ("Predicting")
        Y_predicted = model.predict_classes(concat_matrix, verbose = 0)
        #print (Y_predicted)
        #print( encoder.inverse_transform(Y_predicted))
    return encoder.inverse_transform(Y_predicted)
    


if __name__ == '__main__':
    predict(path, ques)
