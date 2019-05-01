import models.naive_bayes
import eval

file_obj = open('gold.csv', 'r')
model = models.naive_bayes.NB_classifier(8)
model.train(file_obj)
# Prototype with ugly coding ...
print(model.predict('happy'))

eval_obj = eval.Eval('predict.csv', 'gold.csv')
print(eval_obj.get_score())
