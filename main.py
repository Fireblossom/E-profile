import models.naive_bayes
import eval

#file_obj = open('prprprprprpr', 'r')
#model = models.naive_bayes.NB_classifier(5)
#model.train(file_obj)
# Prototype with ugly coding ...

eval_obj = eval.Eval('predict.csv', 'gold.csv')
print(eval_obj.get_score())
