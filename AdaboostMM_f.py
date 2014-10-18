import Orange
import Orange.core as orange
import math
import copy
import random

class ProgressBar:
	'''
	Progress bar
	'''
	def __init__ (self, valmax, maxbar, title):
		if valmax == 0:  valmax = 1
		if maxbar > 200: maxbar = 200
		self.valmax = valmax
		self.maxbar = maxbar
		self.title  = title

	def update(self, val):
		import sys
		# process
		perc  = round((float(val) / float(self.valmax)) * 100)
		scale = 100.0 / float(self.maxbar)
		bar   = int(perc / scale)

		# render
		out = '\r %20s [%s%s%s] %3d %%' % (self.title, '=' * bar, '>' , ' ' * (self.maxbar - bar), perc)
		sys.stdout.write(out)
		sys.stdout.flush()

#
# learner = the weak learning algorithm used during the training phase
# maxT = number of iterations (default value is 10)
class ComboLearner(orange.Learner):
	def __new__(cls, learner, instances=None, weight_id=None, **kwargs):
		self = orange.Learner.__new__(cls, **kwargs)
		if instances is not None:
			self.__init__(self, learner, **kwargs)
			return self.__call__(instances, weight_id)
		else:
			return self

	def __init__(self, learner = Orange.classification.tree.TreeLearner(max_depth=4, measure="gini", m_pruning=2), maxT = 10, verbose = True, alphaMethod=1):
		self._T=maxT # number of iterations
		self._h=[] # array containing the weak classifiers
		self._alpha = [] # array containing the coefficients alpha_t
		self._dist = [] # array containing the distribution over the training examples : dist[i] = -cost[i][y_i]
		self._learner = learner
		self._verbose = verbose
		self._method = alphaMethod # 1 : alpha is a function of the edge delta, 2 : alpha is a function of the training cost (see Mukherjee et al. 2011)

	def __call__(self, sample, orig_weight = 0):
		# show the progression of the algorithm  as a progress bar
		bar = ProgressBar(int(self._T),100,"CoMBo")
		# initialization phase
		self.initial(sample)

		currentStep=0
		while currentStep < int(self._T):
			if self._verbose :
				print "Iteration "+str(currentStep)
				print "  Learning ..."

			h = self._learner(self._sample, self._weight) # train the weak classifier
			#h here is acturally a tree
			print 'self weight is ',self._weight
			if self._verbose :
				print "  Computing alpha ..."
			perf = self.computeErrorAlpha(h) # computes the value of alpha
			error = perf[0]
			alpha = perf[1]
			if alpha <= 0. : # sif alpha is negative then the learnt classifier does not verify the weak learning condition
				if self._verbose :
					print "alpha <= 0 : Exit"
				self._sample.removeMetaAttribute(self._weight)
				return ComboClassifier(classifiers = self._h, name="CoMBo", class_var=self._sample.domain.class_var)

			self._h.append((h, alpha)) # adds the weak classifier to the ensemble of weak classifiers
			if self._verbose :
				print "  Updating Costs ..."
			perf1 = self.updateCosts(alpha, h) # updates cost matrix and distribution
			if self._verbose :
				print "Current classifier's error = "+str(error)+"; alpha = "+str(alpha)
				print "---> train = "+str(perf1[0])+"; weighted_train = "+str(perf1[1])
			currentStep = currentStep + 1
			if not self._verbose :
				bar.update(currentStep)
		print self._sample[-1]
		self._sample.removeMetaAttribute(self._weight)
		print self._sample[-1]
		return ComboClassifier(classifiers = self._h, name="CoMBo", class_var=self._sample.domain.class_var)

	# initializes the various variables
	def initial(self, sample):
		self._sample=sample # training sample

		self._f = []
		self._cost = [] # cost matrix
		self._numclass = 0 # number of classes
		self._classes = [] # the classes

		self._h=[] # array containing the weak classifiers
		self._alpha = [] # array containing the coefficients alpha_t

		# compute the number of classes
		for i in self._sample:
			cl = i.getclass()
			if cl not in self._classes : self._classes.append(cl)
		self._numclass = len(self._classes)

		# contains the number of examples per class
		self._perclassnum = [0 for i in range(self._numclass)]

		# creating the distribution as a meta attribute in Orange
		self._weight = Orange.feature.Descriptor.new_meta_id()
		self._sample.addMetaAttribute(self._weight, float(1./len(self._sample)))
		#print self._sample[1:5],float(1./len(self._sample))
		# initialization of f (see AdaBoost.MM paper) and computation of the number of exemples for each class
		z = len(self._sample)

		for i in self._sample:
			iClass = self._classes.index(i.getclass())

			# init of f
			self._f.append([0 for c in self._classes])

			# init of perclassnum
			self._perclassnum[iClass] = self._perclassnum[iClass] + 1.


		# initialization of the cost matrix and of the distribution
		iIndex = 0
		for i in self._sample:
			iClass = self._classes.index(i.getclass())

			# init of the cost matrix
			self._cost.append([1./(self._perclassnum[iClass]) for c in self._classes])
			self._cost[iIndex][iClass] = - (self._numclass - 1.)/(self._perclassnum[iClass])

			# init of the distribution
			i.setweight(self._weight, 1./(self._perclassnum[iClass]*self._numclass))

			iIndex = iIndex + 1


	# computes the coefficient alpha using the AdaBoost.MM formula
	# returns the error rate of the weak classifier
	def computeErrorAlpha(self, classifier):
		errorRate = 0.
		alpha = 0.

		if self._method == 1:
			cCost = 0. # classification cost
			tCost = 0. # total cost
		else :
			# second method
			ccCost = 0. # correctly classified cost
			mcCost = 0. # missclassified cost

		iIndex = 0 # the index of example i

		for i in self._sample :
			iClass = self._classes.index(i.getclass())
			iLabel = self._classes.index(classifier(i))
			if iClass != iLabel : errorRate = errorRate + 1.

			if self._method == 1 :
				cCost = cCost + self._cost[iIndex][iLabel]
				tCost = tCost - self._cost[iIndex][iClass]
				
			else :
				# second method
				if iLabel==iClass :
					ccCost = ccCost - self._cost[iIndex][iClass]
				else :
					mcCost = mcCost + self._cost[iIndex][iLabel]

			iIndex = iIndex + 1

		if self._method == 1 :
			delta = -cCost/tCost
			if delta == 1 :
				return [1.,-1.]

			alpha = 0.5*math.log((1.+delta)/(1.-delta))
		else :
			# second method
			if mcCost == 0 :
				return [1.,-1.]

			alpha = 0.5*math.log(ccCost/mcCost)

		errorRate = errorRate/len(self._sample)

		return [errorRate,alpha]

	# updates the cost matrix, the f function and the distribution
	# h is the weak classifier and alpha its coefficient
	# returns the weighted and unweighted errors of the classifier
	def updateCosts(self, alpha, h):

		# updates the cost matrix
		iIndex = 0 # the example's index
		dist = 0. # = sum_{i}(-c[i][y_i])
		for i in self._sample :
			iClass = self._classes.index(i.getclass())
			iLabel = self._classes.index(h(i))

			# update f
			self._f[iIndex][iLabel] = self._f[iIndex][iLabel] + alpha

			# update the cost matrix
			costclass = 0.

			for k in range(self._numclass):
				if iClass != k :
					self._cost[iIndex][k] = math.exp(self._f[iIndex][k]-self._f[iIndex][iClass])
					costclass = costclass + self._cost[iIndex][k]
			self._cost[iIndex][iClass] = - costclass

			dist = dist - self._cost[iIndex][iClass]
			print 'DISTribution is ',iIndex,dist
			iIndex = iIndex + 1

		# updated the distribution
		iIndex = 0
		error = 0.
		errorp = 0. # weighted error
		errorcl = [0 for i in range(self._numclass)] # per class errors
		for i in self._sample :
			iClass = self._classes.index(i.getclass())
			#print 'before ',i
			#print self._weight
			i.setweight(self._weight, - self._cost[iIndex][iClass]/dist)
			#print 'after ',i
			#print 'i weight is ', i.getweight(self._weight)
			#print 'please look ',self._cost[iIndex][iClass]
			#print 'weight is ', - self._cost[iIndex][iClass] / dist
			# compute the predicted class for example i
			maxf = max(self._f[iIndex])
			imaxf = self._f[iIndex].index(maxf)

			# update the empirical error, the weighted error and the per class errors
			if imaxf != iClass :
				error = error + 1.
				errorp = errorp + i.getweight(self._weight)
				errorcl[iClass] = errorcl[iClass] + 1.

			iIndex = iIndex + 1

		for i in range(self._numclass):
			errorcl[i] = errorcl[i]/self._perclassnum[i]

		if self._verbose :
			print "error per class",errorcl

		error = error/len(self._sample)

		return [error, errorp]

ComboLearner = Orange.utils.deprecated_members({"examples":"instances", "classVar":"class_var", "weightId":"weigth_id", "origWeight":"orig_weight"})(ComboLearner)


class ComboClassifier(orange.Classifier):
	def __init__(self, classifiers, name, class_var, **kwds):
		self.classifiers = classifiers
		self.name = name
		self.class_var = class_var
		self.__dict__.update(kwds)

	def __call__(self, instance, result_type = orange.GetValue, size=-1):
		votes = Orange.statistics.distribution.Discrete(self.class_var)
		if size>0: # use only the size first classifiers in the prediction
			lclass = len(self.classifiers)
			for iSize in range(size):
				if iSize >= lclass : break
				c,e = self.classifiers[iSize]
				votes[int(c(instance))] += e
		else :
			for c, e in self.classifiers:
				votes[int(c(instance))] += e
		index = Orange.utils.selection.select_best_index(votes)
		value = Orange.data.Value(self.class_var, index)
		if result_type == orange.GetValue:
			return value
		sv = sum(votes)
		for i in range(len(votes)):
			votes[i] = votes[i]/sv
		if result_type == orange.GetProbabilities:
			return votes
		elif result_type == orange.GetBoth:
			return (value, votes)
		else:
			return value

	def __reduce__(self):
		return type(self), (self.classifiers, self.name, self.class_var), dict(self.__dict__)
ComboClassifier = Orange.utils.deprecated_members({"classVar":"class_var", "resultType":"result_type"})(ComboClassifier)

########################################################
