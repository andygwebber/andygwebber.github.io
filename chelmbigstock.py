#!/usr/bin/env python3

"""
Code to use stock history to see if future fluctuations can be predicted

@Author: Andy Webber
Created: March 1, 2014
"""
# A python script to learn about stock picking
import sys

from operator import itemgetter

import numpy as np
from random import random
from sklearn import linear_model
import timeit
from scipy.stats import anderson

import dateutl
from Stock import Stock
from LearningData import LearningData

from sklearn import preprocessing

"""std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test) """

def form_data(stocks, init_param):
    """ This function constructs the training, testing and cross validation
        objects for the stock market analysis """
    
    rs = stocks[1].rsi
    ts = stocks[1].tsi
    a = 1
    
        
    for date in init_param.train_dates:
        try:
            training_data
        except NameError:
            training_data = LearningData()
            training_data.construct(stocks, date, init_param.future_day, init_param.features)
        else:
            training_data.append(stocks, date, init_param.future_day, init_param.features)
            
    for date in init_param.test_dates:
        try:
            test_data
        except NameError:
            test_data = LearningData()
            test_data.construct(stocks, date, init_param.future_day, init_param.features)
        else:
            test_data.append(stocks, date, init_param.future_day, init_param.features)
    
    #reference_date = dateutl.days_since_1900('1991-01-01')
    #test_data.construct(stocks,[reference_date, day_history, init_param.future_day])
    
    return training_data, test_data

def output(training_data, cv_data):
    " This function outputs the data in csv form so it can be examined in Matlab"
    
    f = open('training_x.txt','w')
    for i in range(0,training_data.m):
        x_str = ','.join(str(x) for x in training_data.X[i])
        print(x_str)
        f.write(x_str + '\n')
    f.close
    
    f = open('training_y.txt','w')
    y_str = ','.join(str(y) for y in training_data.y)
    f.write(y_str)
    f.close
    
    f = open('cv_x.txt','w')
    for i in range(0,cv_data.m):
        x_str = ','.join(str(x) for x in cv_data.X[i])
        print(x_str)
        f.write(x_str + '\n')
    f.close
    
    f = open('cv_y.txt','w')
    y_str = ','.join(str(y) for y in cv_data.y)
    f.write(y_str)
    f.close
    
def logistic_reg(training_data):
    """ This function does the actual training. It takes in training data
        and cross validation data and returns the model and optimal 
        regularization parameter """
    
    """ Setting guesses for minimum and maximum values of regularization parameter then
        find the value of parameter that minimizes error on cross validation data. If
        local minimum is found the return this model. If not, extend minimum or maximum 
        appropriately and repeat """
    from sklearn.linear_model import LogisticRegression
    C_min = 1.0e-5
    C_max = 1.0e5
    regularization_flag = 1 # To set 1 until local minimum is found
    regularization_param = 0
    
#    while regularization_flag != 0:
#        regularization_param, regularization_flag = set_reg_param(training_data, cv_data, alpha_min, alpha_max)
#        if regularization_flag == -1:
#            """ The local minimum is at point less than alpha_min """
#            alpha_min = alpha_min * 0.3
#        if regularization_flag == 1:
#            """ The local minimum is at point greater then alpha_max """
#            alpha_max = alpha_max * 3
            
    lr = LogisticRegression (C=C_max, random_state=0)
    lr.fit(training_data.X, training_data.y)
    return lr, C_max

def set_reg_param(training_data, cv_data, alpha_min, alpha_max):
    """ This function does a linear regression with regularization for training_data
        then tests prediction for training_data and cv_data over a range of regularization
        parameters. If a local minimum is found it returns the parameter and a 0 to indicate
        it is complete. If minimum it below alpha_min it returns -1 for flag. If it is above
        alpha_max, it returns 1 for flag. """
        
    f = open('alpha.txt', 'w')
    
    alph = alpha_min
    min_alpha = alpha_min # This is the value of alpha in our range that gives minimum for cv data
    alpha_largest = alpha_min # Learning is not generally done at alpha_min, this tracks larget alpha
    while alph < alpha_max:
        """ Learn for this parameter """
        clf = linear_model.Ridge (alpha=alph, fit_intercept=False)
        clf.fit(training_data.X, training_data.y)
        
        """ Get prediction for this parameter """
        predict_data = clf.predict(training_data.X)
        predict_cv = clf.predict(cv_data.X)
        
        """ Caculate the differences from actual data for training and cv data"""
        diff_training = (1.0/training_data.m) * np.linalg.norm(predict_data - training_data.y)
        diff_cv = (1.0/cv_data.m) * np.linalg.norm(predict_cv - cv_data.y)
        
        """ Write out the values for plotting. Do appropriate work to determine min_val_alpha """
        f.write(str(alph) +  " " + str(diff_training) + " " + str(diff_cv) +  "\n")
        if alph == alpha_min:
            min_diff = diff_cv # Just setting default value for first value of alph 
            min_alpha = alpha_min
        if diff_cv < min_diff:
            """ We have a new minimum so value and alph must be recored """
            min_diff = diff_cv
            min_alpha = alph
        alpha_largest = alph # Keep track of largest alpha used
        alph = alph * 1.5 # increment alph
    f.close()
            
    """ Loop is now complete. If min_value_alpha is not alpha_min or alpha_max, return flag of 0
            else return -1 or 1 so min or max can be adjusted and loop completed again """
    if abs(min_alpha - alpha_min) < alpha_min/10.0:
        flag = -1 # Local minimum is less than alpha_min so return -1 
    elif abs(min_alpha - alpha_largest) < alpha_min/10.0:
        flag = 1 # Local minimum is greater than alpha_max so return 1 
    else:
        flag = 0 # Local minimum is in range so return 0 
        
    return min_alpha, flag
    
def examine(stocks, init_param, C_in, gamma_in, verbose):
    """ This plot takes in the stocks and features. It plots a ROC curve
        returns the Area under the curve"""
    from sklearn.svm import SVC
    from sklearn import metrics
    import matplotlib.pyplot as plt
#    import pandas as pd
    
    training_data, test_data = form_data(stocks, init_param)
    std_scale = preprocessing.StandardScaler().fit(training_data.X)
    training_data.X = std_scale.transform(training_data.X)
    test_data.X = std_scale.transform(test_data.X)
    
    svm = SVC(kernel='rbf', random_state=0, gamma = gamma_in, C=C_in, probability=True)
    svm.fit(training_data.X, training_data.y)
    preds = svm.predict_proba(test_data.X)[:,1]
    fpr, tpr, _ = metrics.roc_curve(test_data.y, preds)

#    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    roc_auc = metrics.auc(fpr,tpr)
    
    if verbose:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    
    return roc_auc
    
def choose_features(stocks, init_param, C, gamma):
    """ This function chooses the feature from the available_features array
        that when added to chosen_features give maximium area under curve. 
        It returns chosen features and available_features arrays with 
        the best feature added to the former and removed from the latter.
        It also appends the best aoc onto the aoc array"""
        
    chosen_features = []
    available_features = init_param.features[:]
    """The code is written to edit init_param.features but make a copy to 
       restore things after the loop"""
    init_param_features = init_param.features[:]
    aoc = []
    
    while (len(available_features) > 5):
        best_aoc = 0
        for feature in available_features:
            input_features = chosen_features[:]
            input_features.append(feature)
            init_param.features = input_features
            feature_aoc = examine(stocks, init_param, C, gamma, False)
            if feature_aoc > best_aoc:
                best_aoc = feature_aoc
                best_feature = feature
            
        chosen_features.append(best_feature)
        available_features.remove(best_feature)
        aoc.append(best_aoc)
    
    """ Restore init_param.features """
    init_param.features = init_param_features[:]
    return chosen_features, available_features, aoc
  
class Particle:
    pass
           
def parab(C,gamma):
    """ This is a simple parabolic function """
    return 5.678 - ((1.234 - C)**2 + (3.456 - gamma)**2)
 
def pso(stocks, init_param):
    iter_max = 10000
    pop_size = 40
    dimensions = 2
    c1 = 3
    c2 = 3
    err_crit = 0.000001
    #initialize the particles
    particles = []
    for i in range(pop_size):
        p = Particle()
        p.params = np.array([random() for i in range(dimensions)])
        p.fitness = -float('inf')
        p.v = 0.0
        particles.append(p)
    
    # let the first particle be the global best
    gbest = particles[0]
    gbest_hist = np.ones(3)
    i = 0
    while i < iter_max :
        print("doing i = ",i)
        for p in particles:
    #        fitness,err = f6(p.params)
            p.params[0] = max(p.params[0], 0.01)
            p.params[1] = max(p.params[1], 0.01)
            C = p.params[0]
            gamma = p.params[1]
            fitness = parab(C,gamma)
            verbose = False
            fitness = examine(stocks, init_param, C, gamma, verbose)
            if fitness > p.fitness:
                p.fitness = fitness
                p.best = p.params
    
            if fitness > gbest.fitness:
                gbest = p
            v = p.v + c1 * random() * (p.best - p.params) \
                    + c2 * random() * (gbest.params - p.params)
            p.params = p.params + v
        
        gbest_hist[i%3] = gbest.fitness    
        i  += 1
        if np.std(gbest_hist) < err_crit:
            break
        #progress bar. '.' = 10%
        if i % (iter_max/10) == 0:
            print ('.')
    return gbest.fitness, gbest.params[0], gbest.params[1], i    

def recalc_stocks(stocks, feature, args):
    """ This method goes through each stock in stocks and recalculates
        the given feature with args. For example the default time on 
        rsi is 14 days. If I wanted to recalculate each rsi based on 21 days,
        feature would be 'rsi' and args would be '21' """
        
    for stock in stocks:
        expression = 'stock.' + feature + '_calc(' + args + ')'
        exec(expression)
        
    return
        

def execute(init_param): 
    """ execute is the function where each run is done. main sets parameters then calls execute"""
    
    
    from sklearn.svm import SVC
    import matplotlib.pyplot as plt
    start = timeit.timeit()
#    fitness, param0, param1, i = pso()
    stocks = Stock.read_stocks('../data/stocks_read.txt', init_param.max_stocks)
    init_param.features = ['rsi','uo','stoch']
    C = 1.0
    gamma = 0.2
    feature_aoc = examine(stocks, init_param, C, gamma, True)
    
    
    training_data, test_data = form_data(stocks, init_param)
    std_scale = preprocessing.StandardScaler().fit(training_data.X)
    training_data.X = std_scale.transform(training_data.X)
    test_data.X = std_scale.transform(test_data.X)
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from sklearn.metrics import accuracy_score
    tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=1, random_state=0)
    tree.fit(training_data.X, training_data.y)
    y_pred = tree.predict_proba(test_data.X)[:,1]
    fpr, tpr, thresholds = roc_curve(y_true=test_data.y, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr)
    plt.show()
    print('aoc of clf is \n', roc_auc)
    
    from sklearn.ensemble import AdaBoostClassifier
    ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators = 100,
                         learning_rate = 0.1,
                         random_state = 0)
    ada = ada.fit(training_data.X, training_data.y)
    y_train_pred = ada.predict(training_data.X)
    y_test_pred = ada.predict(test_data.X)
    ada_train = accuracy_score(training_data.y, y_train_pred)
    ada_test = accuracy_score(test_data.y, y_test_pred)
    fpr, tpr, thresholds = roc_curve(y_true=test_data.y, y_score=y_test_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr)
    plt.show()
    print('Decision tree train/test accuracies %.3f/%.3f'
        % (ada_train, ada_test))
    print('aoc of ada is \n', roc_auc)
    
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(training_data.X, training_data.y)
    y_train_pred = qda.predict(training_data.X)
    y_test_pred = qda.predict(test_data.X)
    qda_train = accuracy_score(training_data.y, y_train_pred)
    qda_test = accuracy_score(test_data.y, y_test_pred)

    fpr, tpr, thresholds = roc_curve(y_true=test_data.y, y_score=y_test_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr)
    plt.show()
    print('Quadratic Discriminant train/test accuracies %.3f/%.3f'
        % (qda_train, qda_test))
    print('aoc of qda is \n', roc_auc)
    
    
    recalc_stocks(stocks, 'rsi', '20')
 #   stocks = 1
    
    """ Chose the best feature """
#    chosen_features = []
#    available_features = init_param.features
    C = 1.6
    gamma = 0.23
    
    f = open('200_stocks_results.txt','w')
    
    print("doing 5")
    C = 1
    gamma = 0.2
    init_param.future_day = 5
    chosen_features, available_features, aoc = choose_features(stocks, init_param, C, gamma)
    init_param.features = chosen_features[0:2]
    fitness, C, gamma, i = pso(stocks, init_param)
    
    f.write('Results for 5 days out \n')
    f.write('fitness = ' + str(fitness) + '\t' + 'C = ' + str(C) + '\t' + 'gamma = ' + str(gamma) + '\n')
    for i in range(0,len(aoc)):
        f.write(chosen_features[i] + '\t' + str(aoc[i]) + '\n')
    f.write('\n')
        
    
    print("doing 10")
    C = 1
    gamma = 0.2
    init_param.future_day = 10
    init_param.features = ['rsi','tsi','ppo','adx','dip14','dim14','cci','cmo','mfi','natr','roc','stoch','uo']
    chosen_features, available_features, aoc = choose_features(stocks, init_param, C, gamma)
    init_param.features = chosen_features[0:2]
    fitness, C, gamma, i = pso(stocks, init_param)
    
    f.write('Results for 10 days out \n')
    f.write('fitness = ' + str(fitness) + '\t' + 'C = ' + str(C) + '\t' + 'gamma = ' + str(gamma) + '\n')
    for i in range(0,len(aoc)):
        f.write(chosen_features[i] + '\t' + str(aoc[i]) + '\n')
    f.write('\n')
    
    print("doing 15")
    C = 1
    gamma = 0.2
    init_param.future_day = 15
    init_param.features = ['rsi','tsi','ppo','adx','dip14','dim14','cci','cmo','mfi','natr','roc','stoch','uo']
    chosen_features, available_features, aoc = choose_features(stocks, init_param, C, gamma)
    init_param.features = chosen_features[0:2]
    fitness, C, gamma, i = pso(stocks, init_param)
    
    f.write('Results for 15 days out \n')
    f.write('fitness = ' + str(fitness) + '\t' + 'C = ' + str(C) + '\t' + 'gamma = ' + str(gamma) + '\n')
    for i in range(0,len(aoc)):
        f.write(chosen_features[i] + '\t' + str(aoc[i]) + '\n')
    f.write('\n')
    
    print("doing 20")
    C = 1
    gamma = 0.2
    init_param.future_day = 20
    init_param.features = ['rsi','tsi','ppo','adx','dip14','dim14','cci','cmo','mfi','natr','roc','stoch','uo']
    chosen_features, available_features, aoc = choose_features(stocks, init_param, C, gamma)
    init_param.features = chosen_features[0:2]
    fitness, C, gamma, i = pso(stocks, init_param)
    
    f.write('Results for 20 days out \n')
    f.write('fitness = ' + str(fitness) + '\t' + 'C = ' + str(C) + '\t' + 'gamma = ' + str(gamma) + '\n')
    for i in range(0,len(aoc)):
        f.write(chosen_features[i] + '\t' + str(aoc[i]) + '\n')
    f.write('\n')
    
    print("doing 25")
    C = 1
    gamma = 0.2
    init_param.future_day = 25
    init_param.features = ['rsi','tsi','ppo','adx','dip14','dim14','cci','cmo','mfi','natr','roc','stoch','uo']
    chosen_features, available_features, aoc = choose_features(stocks, init_param, C, gamma)
    init_param.features = chosen_features[0:2]
    fitness, C, gamma, i = pso(stocks, init_param)
    
    f.write('Results for 25 days out \n')
    f.write('fitness = ' + str(fitness) + '\t' + 'C = ' + str(C) + '\t' + 'gamma = ' + str(gamma) + '\n')
    for i in range(0,len(aoc)):
        f.write(chosen_features[i] + '\t' + str(aoc[i]) + '\n')
    f.write('\n')
    
    print("doing 30")
    C = 1
    gamma = 0.2
    init_param.future_day = 30
    init_param.features = ['rsi','tsi','ppo','adx','dip14','dim14','cci','cmo','mfi','natr','roc','stoch','uo']
    chosen_features, available_features, aoc = choose_features(stocks, init_param, C, gamma)
    init_param.features = chosen_features[0:2]
    fitness, C, gamma, i = pso(stocks, init_param)
    
    f.write('Results for 30 days out \n')
    f.write('fitness = ' + str(fitness) + '\t' + 'C = ' + str(C) + '\t' + 'gamma = ' + str(gamma) + '\n')
    for i in range(0,len(aoc)):
        f.write(chosen_features[i] + '\t' + str(aoc[i]) + '\n')
    f.write('\n')
    
    print("doing 35")
    C = 1
    gamma = 0.2
    init_param.future_day = 35
    init_param.features = ['rsi','tsi','ppo','adx','dip14','dim14','cci','cmo','mfi','natr','roc','stoch','uo']
    chosen_features, available_features, aoc = choose_features(stocks, init_param, C, gamma)
    init_param.features = chosen_features[0:2]
    fitness, C, gamma, i = pso(stocks, init_param)
    
    f.write('Results for 35 days out \n')
    f.write('fitness = ' + str(fitness) + '\t' + 'C = ' + str(C) + '\t' + 'gamma = ' + str(gamma) + '\n')
    for i in range(0,len(aoc)):
        f.write(chosen_features[i] + '\t' + str(aoc[i]) + '\n')
    f.write('\n')
    f.close()
    
    init_param.features = ['rsi','tsi','ppo','adx','dip14','dim14','cci','cmo','mfi','natr','roc','stoch','uo']
    chosen_features, available_features, aoc = choose_features(stocks, init_param, C, gamma)
    init_param.features = chosen_features[0:2]
    fitness, C, gamma, i = pso(stocks, init_param)

    init_param.features = ['rsi','tsi']
    verbose = True
    examine(stocks, init_param, verbose, C, gamma)
                         
    training_data, test_data = form_data(stocks, init_param)
    std_scale = preprocessing.StandardScaler().fit(training_data.X)
    training_data.X = std_scale.transform(training_data.X)
    test_data.X = std_scale.transform(test_data.X)
    end1 = timeit.timeit()
    print("form_data took ", (end1-start))
    print("training_data has ",len(training_data.y)," elements")
    print("test_data has ",len(test_data.y)," elements")
    
    if init_param.output:
        output(training_data, cv_data)
    
    #clf, regularization_parameter = learn(training_data, cv_data)
    """  lr, C = logistic_reg(training_data)
    test_predict = lr.predict(test_data.X)
    errors = np.count_nonzero(test_predict - test_data.y)
    accuracy = 1.0 - (errors/len(test_predict))
    print("accuracy is ",accuracy)
    end2 = timeit.timeit()
    print("regression took ",(end2-end1))"""
    train_errors, test_errors, C_arr = [], [], []
    train_accuracy, test_accuracy = [],[]
    C_i = 0.01
    while C_i < 10:
        svm = SVC(kernel='rbf', random_state=0, gamma = 0.2, C=C_i)
        svm.fit(training_data.X, training_data.y)
        train_errors.append(np.count_nonzero(svm.predict(training_data.X)-training_data.y))
        accuracy = 1.0 - np.count_nonzero(svm.predict(training_data.X)-training_data.y)/len(training_data.y)
        train_accuracy.append(accuracy)
        test_errors.append(np.count_nonzero(svm.predict(test_data.X)-test_data.y))
        accuracy = 1.0 - np.count_nonzero(svm.predict(test_data.X)-test_data.y)/len(test_data.y)
        test_accuracy.append(accuracy)
        C_arr.append(C_i)
        C_i = C_i *1.1
        
    plt.plot(C_arr, train_accuracy,c='r')
    plt.plot(C_arr, test_accuracy,c='b')
    plt.xscale('log')
    plt.show()
    
    yy = np.asarray(training_data.y)
    XX = np.asarray(training_data.X)
    XX0 = XX[yy == 0]
    XX1 = XX[yy == 1]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(XX0[:,0], XX0[:,12],c='red')
    ax1.scatter(XX1[:,0], XX1[:,12],c='blue')
    plt.show()
        
    
#    init_param2 = init_param
#    init_param2.reference_dates = [dateutl.days_since_1900('2000-01-01')]
#    init_param2.test_dates = [dateutl.days_since_1900('2010-01-01')]
#    training_data2, test_data2 = form_data(init_param2)
#    lr, C = logistic_reg(training_data2)
#    test_predict2 = lr.predict(test_data2.X)
#    errors = np.count_nonzero(test_predict2 - test_data2.y)
#    accuracy = 1.0 - (errors/len(test_predict))
    print("accuracy is ",accuracy)
    
    print("run finished with accuracy", accuracy)
    
class InitialParameters(object):
    """ This class defines an object of parameters used to run the code. It
        is set in main and the parameters are passed to execute """
    
    def __init__(self):
        """ The object is defined with default values that can then be changed in main()"""
        
        #self.max_stocks = 100
        self.max_stocks = 200
        """ cv_factor determines what portion of stocks to put in cross validation set and what portion
            to leave in training set. cv_factor = 2 means every other stock goes into cross validation
            set. cv_factor = 3 means every third stock goes into cross validation set """
        self.cv_factor = 2 
        """ future_day is how many training days in the future we train for. Setting future_day = 25
            means we are measuring how the stock does 25 days out """
        self.future_day = 25
        """ The train_dates are the dates for training and cross validation"""
        self.train_dates = []
        first_train_date = dateutl.days_since_1900('2001-01-01')
        num_train_dates = 10
        train_date_increment = 60
        self.train_dates.append(first_train_date)
        for iday in range(1,num_train_dates):
            last_train_date = self.train_dates[iday-1]
            self.train_dates.append(last_train_date + train_date_increment)
        """self.train_dates[1] -= 1 """
        
        """ test_dates are the dates we are using for testing """
        self.test_dates = []
        first_test_date = dateutl.days_since_1900('2010-01-01')
        num_test_dates = 10
        test_date_increment = 60
        self.test_dates.append(first_test_date) 
        for iday in range(1,num_test_dates):
            last_test_date = self.test_dates[iday-1]
            self.test_dates.append(last_test_date + test_date_increment)
        """self.test_dates[1] -= 1
        self.test_dates[3] += 1
        self.test_dates[4] += 3
        self.test_dates[5] += 4
        self.test_dates.append(dateutl.days_since_1900('2010-01-01'))
        self.test_dates.append(dateutl.days_since_1900('2010-03-01'))
        self.test_dates.append(dateutl.days_since_1900('2010-05-01'))
        self.test_dates.append(dateutl.days_since_1900('2010-07-01'))
        self.test_dates.append(dateutl.days_since_1900('2010-09-01'))
        self.test_dates.append(dateutl.days_since_1900('2010-11-01'))"""
        """train_history_days and train_increment set how many historical days we use to
           train and the increment used. Setting train_history_days = 21 and train_increment = 5
           means we are using the values at days days 5, 10, 15 and 20 days before the reference day
           as input features """
        self.train_days = 21
        self.train_increment = 5
        self.features = ['rsi','tsi','ppo','adx','dip14','dim14','cci', \
                         'cmo','mfi','natr','roc','stoch','uo']
        """ output is just a boolean about calling the output function to write out 
            appropriate X and y matricies. The default is False meaning do not write out
            matricies """
        self.output = False
    
def main(argv):
    
    init_param = InitialParameters()
    #init_param.reference_dates.append(dateutl.days_since_1900('1981-01-01'))
    #init_param.reference_dates.append(dateutl.days_since_1900('2001-01-01'))
    execute(init_param)


if __name__ == "__main__":
    main(sys.argv)
