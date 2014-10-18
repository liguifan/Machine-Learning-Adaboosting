from AdaboostMM_f import *
from Orange import *

train_size=16000
test_size=20000-train_size
Y_train=[]
training_error=[]
test_error=[]
adaMM=[]
train_result=[]


data=Orange.data.Table("/Users/liguifan/Desktop/orange_test/dataset2")
'''
learner = Orange.regression.tree.TreeLearner()
classifier = learner(data)
'''

for instance in data[0:train_size]:
	Y_train.append(instance.getclass())





T_experiment=5
T_range=range(1,T_experiment+1)
print 'Adaboost learning'
for i in T_range:
	combo=ComboLearner(Orange.classification.tree.TreeLearner(max_depth=4, measure="gini", m_pruning=2),maxT=i)
	adaboost=combo.__call__(data)
	adaMM.append(adaboost)

print 'length of adaMM is ',len(adaMM)
train_file=open('/Users/liguifan/Desktop/orange_test/train_error_result','w')
test_file=open('/Users/liguifan/Desktop/orange_test/test_error_result','w')

errorRate=0

for i in range(T_experiment):
	for instance in data[0:train_size]:
		iClass=instance.getclass()
		iLabel=adaMM[i].__call__(instance)
		if iClass != iLabel : errorRate = errorRate + 1.
	training_error_temp=errorRate/float(16000)
	train_file.write(str(training_error_temp)+'\n')
	errorRate=0

train_file.close()


for i in range(T_experiment):
	for instance in data[train_size:]:
		iClass=instance.getclass()
		iLabel=adaMM[i].__call__(instance)
		if iClass != iLabel : errorRate = errorRate + 1.
	testing_error_temp=errorRate/float(4000)
	test_file.write(str(testing_error_temp)+'\n')
	errorRate=0



test_file.close()
