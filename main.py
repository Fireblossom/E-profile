import models.naive_bayes

file_obj = open('prprprprprpr', 'r')
model = models.naive_bayes.NB_classifier(5)
model.train(file_obj)
# Prototype with ugly coding ...
