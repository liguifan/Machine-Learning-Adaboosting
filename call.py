from Combo import *
from Orange import *




data=Orange.data.Table("/Users/liguifan/Desktop/orange_test/connect-4/connect-4")
learner = Orange.regression.tree.TreeLearner(max_depth=3, measure="gini", m_pruning=2)
classifier = learner(data)

print data[1],data[1].getclass() 
learner = Orange.classification.bayes.NaiveLearner()

for d in data[:100]:
    c = classifier(d)
    print "%10s -> %s" % (classifier(d), d.getclass())

combo=ComboLearner(Orange.classification.tree.TreeLearner(max_depth=2, measure="gini", m_pruning=2),maxT=100)

adaboost=combo.__call__(data)


for x in data[0:100]:
	temp1=adaboost.__call__(x)
	print temp1
