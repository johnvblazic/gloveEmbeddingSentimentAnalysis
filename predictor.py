import numpy as np
import glob
import collections
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

t0 = time()

embedSize = 300
#embedding = Word2Vec(size=embedSize, window=5, min_count=5, workers=4)
embedding = KeyedVectors.load_word2vec_format("WordVectors/w2v.42B.300d.txt")


training_xs = []
training_ys = []
for file_name in glob.glob("txt_sentoken/pos/*"):
    with open(file_name) as train_file:
        for line in train_file.readlines():
            sLine = line.rstrip()
            doc_vec = [0 for x in range(embedSize)]
            for word in sLine:
                if word in embedding.vocab:
                    doc_vec = doc_vec + embedding[word]
        #print(sLine)
        #print(doc_vec)
        training_xs.append(doc_vec)
        training_ys.append('positive')

for file_name in glob.glob("txt_sentoken/neg/*"):
    with open(file_name) as train_file:
        for line in train_file.readlines():
            sLine = line.rstrip()
            doc_vec = [0 for x in range(embedSize)]
            for word in sLine:
                if word in embedding.vocab:
                    doc_vec = doc_vec + embedding[word]
        #print(sLine)
        #print(doc_vec)
        training_xs.append(doc_vec)
        training_ys.append('negative')

#training_ys = np.ravel(data[:,[1]])
#training_xs = rows[:,[0]]

model5 = SVC()
model1 = KNeighborsClassifier(5)
model3 = RandomForestClassifier()

model2 = DecisionTreeClassifier(max_depth=5)
model6 = LogisticRegression()
model4 =AdaBoostClassifier()

voter = VotingClassifier(estimators=[('dec-tree',model2),('ada',model4),('lr',model6),('knn',model1),('rfc',model3),('svc',model5)])


voter.fit(training_xs,training_ys)

#scores = cross_val_score(model, training_xs, training_ys, cv=10)
prections = cross_val_predict(voter, training_xs, training_ys, cv=10)
score = accuracy_score(training_ys,prections)
print(score)


print('Total time: ',time() - t0)

#load test set
test_xs = []
test_ys = []
for file_name in glob.glob("txt_sentoken/test/pos/*"):
    with open(file_name) as test_file:
        for line in test_file.readlines():
            sLine = line.rstrip()
            doc_vec = [0 for x in range(embedSize)]
            for word in sLine:
                if word in embedding.vocab:
                    doc_vec = doc_vec + embedding[word]
        #print(sLine)
        #print(doc_vec)
        test_xs.append(doc_vec)
        test_ys.append('positive')

for file_name in glob.glob("txt_sentoken/test/neg/*"):
    with open(file_name) as test_file:
        for line in test_file.readlines():
            sLine = line.rstrip()
            doc_vec = [0 for x in range(embedSize)]
            for word in sLine:
                if word in embedding.vocab:
                    doc_vec = doc_vec + embedding[word]
        #print(sLine)
        #print(doc_vec)
        test_xs.append(doc_vec)
        test_ys.append('negative')

#predict on test set and print
prections = voter.predict(test_xs)
score = accuracy_score(test_ys,prections)
print(score)



test_report = classification_report(
    y_true=le.transform(gold_polarity),
    y_pred=predicted_polarity,
    labels=le.transform(le.classes_),
    target_names=le.classes_)
print(test_report)

confusion_matrix = confusion_matrix(
	y_true=le.transform(gold_polarity),
    y_pred=predicted_polarity,
    labels=le.transform(le.classes_))

print(confusion_matrix)
