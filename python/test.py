import predict as pdict
from os.path import join
import os



img_path = "../static/zebra.jpg"
print(os.path.abspath(img_path))

ques = "What is the this animal?"

pdict.predict(os.path.abspath(img_path), ques)

print ("Successful")
