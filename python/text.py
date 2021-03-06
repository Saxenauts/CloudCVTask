import json 
from spacy.en import English
import operator 
import progressbar 
from sklearn import preprocessing
from sklearn.externals import joblib
from utils import grouper, selectFrequentAnswers

def getModalAnswer(answers):
	candidates = {}
	for i in xrange(10):
		candidates[answers[i]['answer']] = 1

	for i in xrange(10):
		candidates[answers[i]['answer']] += 1

	return max(candidates.iteritems(), key=operator.itemgetter(1))[0]


"""
def convert():
    print("Loading English()")
    
    nlp = English()
    
    print ('Loaded English()')
    #Given the sources
    ann_file = "../data/mscoco_train2014_annotations.json"
    ques_file = "../data/OpenEnded_mscoco_train2014_questions.json"
    
    #Add files 
    ques_txt = open("../data/ques_txt.txt", 'w')
    ques_id_txt = open("../data/ques_id_txt.txt", 'w')
    ques_length_file = open("../ques_len_txt.txt", 'w')
    
    ans_txt = open("../data/ans_txt.txt", 'w')
    img_id = open("../data/img_id_txt.txt", 'w')
    
    questions = json.load(open(ques_file, 'r'))
    ques = questions['questions']
    
    answers = json.load(open(ann_file, 'r'))
    ans = answers['annotations']
    print("Loaded all files")
    
    #TODO: Why is encoding important? 
    pbar = progressbar.ProgressBar()
    print ("Converting to text files")
    
     
    for i, q in pbar(zip(xrange(len(ques)), ques)):
        ques_txt.write((q['question'] + '\n').encode('utf8'))
        ques_length_file.write((str(len(nlp(q['question'])))+'\n').encode('utf8'))
        ques_id_txt.write((str(q['question_id']) + '\n').encode('utf8'))
        
        img_id.write((str(q['image_id']) + '\n').encode('utf8'))
        #TODO: Understand the meaning of this and rewrite the function
        ans_txt.write(getModalAnswer(ans[i]['answers']).encode('utf8'))
        ans_txt.write('\n'.encode('utf8'))
        
    print 'Translation done'

convert()
"""

def save_pickle():
    
    answers = open('../data/ans_txt.txt', 'r').read().decode('utf8').splitlines()
    print(len(answers))
    questions = open("../data/ques_txt.txt", 'r').read().decode('utf8').splitlines()
    images = open("../data/img_id_txt.txt", 'r').read().decode('utf8').splitlines()
    maxAnswers = 1000
    questions, answers, images = selectFrequentAnswers(questions,answers,images, maxAnswers)
    
    encoder = preprocessing.LabelEncoder()
    encoder.fit(answers)
    print ("Number of classes: " + str(len(list(encoder.classes_))))
    joblib.dump(encoder, '../data/encoder.pkl')
    return "DONE"
    
save_pickle()

