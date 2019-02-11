import numpy
import pandas
import time
import keras
import sklearn.datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import random
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import warnings
warnings.filterwarnings("ignore")
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
startTime = int(time.time())
print ([1.0 / (10**i) for i in range(7)])

# load dataset
dataframe = pandas.read_csv("pima.csv", header=0,encoding = "ISO-8859-1").dropna()
dataset = dataframe.values
# split into input (X) and output (Y) variables

#RusLoading
#X = dataset[:,5:7].astype(float)
#Y = dataset[:,29]

#csgoloading
#X = dataset[:,2:7].astype(float)
#Y = dataset[:,8]


#lolloading
Xfull = dataset[:,0:8].astype(float)
Yfull = dataset[:,8]


#USLoading
#Xfull = dataset[:,9:14].astype(float)
#Yfull = dataset[:,8]


#forestloading
#Xfull = dataset[:,0:4].astype(float)
#Yfull = dataset[:,4]

#dataset = sklearn.datasets.load_breast_cancer()
#Xfull = dataset.data.astype(float)
#Yfull = dataset.target


X,Xval,Y,Yval = train_test_split(Xfull,Yfull)
print (len(X),len(Xval))
print (X[0],Y[0])

#Xval = X[cutoff:]
#X = X[:cutoff]

#Yval = Y[cutoff:]
#Y = Y[:cutoff]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

encoder2 = LabelEncoder()
encoder.fit(Yval)
encoded_Yval = encoder.transform(Yval)


# larger model
def create_larger(optimizer = 'Adamax',learn_rate = 1e-4):
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(60, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(60, kernel_initializer='normal', activation='relu'))

    #model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(50, kernel_initializer='normal', activation='relu'))

    #model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(15, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(60, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(120, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(240, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal', activation='softmax'))
    # Compile model
    opt = None
    if(optimizer == "Adam"):
            opt = keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    if(optimizer == "SGD"):
            opt = keras.optimizers.SGD(lr=learn_rate, momentum=0.0, decay=0.0, nesterov=False)
    if(optimizer == "RMSprop"):
            opt = keras.optimizers.RMSprop(lr=learn_rate, rho=0.9, epsilon=None, decay=0.0)
    if(optimizer == "Adamax"):
            opt = keras.optimizers.Adamax(lr= learn_rate)
    #adadelta = keras.optimizers.Adadelta(lr= learn_rate)
    print("using" ,opt," with learn rate",learn_rate)
    model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
    return model

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=numpy.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    print("in func")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose = 0)
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)
    print(train_sizes)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# smaller model
def create_smaller(learn_rate = .0001):
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=8, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    opt = keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    print ("model created with learn rate",learn_rate)
    return model


def runSonarNN(params):
  ####NOTE: Scikit Doesn't like Kerasclassier paramaters being passed in so i've just hardcoded the ones grid search found####
    mod = KerasClassifier(build_fn=create_smaller, epochs=1200, batch_size=5, verbose=0,learn_rate=.0001)
    history = mod.fit(X, encoded_Y, validation_split=.2)
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    #print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    title = "NN curve"
    x = plot_learning_curve(mod, title, X, encoded_Y, cv=3, n_jobs=1)
    x.show()
    return mod.score(Xval,encoded_Yval)



def GridNN():
    epochs = [100,200,400,1000,1200]
    batch_sizes = [5,25,100]
    param_grid = dict(batch_size=batch_sizes,epochs = epochs)
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=epochs, batch_size=batch_sizes, verbose=0)))
    pipeline = Pipeline(estimators)
    print(pipeline.get_params().keys())
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=KerasClassifier(build_fn=create_smaller, epochs=epochs, batch_size=batch_sizes, verbose=0), param_grid=param_grid)#, n_jobs=-1)
    grid_result = grid.fit(X,encoded_Y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_params_
#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=0, batch_size=50, verbose=2)))
#pipeline = Pipeline(estimators)
#kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
#results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
#print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#print (results)
#print(estimators)
#print(estimators[1])

def runNN(params):

    estimator = KerasClassifier(build_fn=create_smaller,epochs = params["epochs"], batch_size=params["batch_size"], verbose=0)
    #optimizer = ['SGD', 'RMSprop', 'Adam', 'Adamax']
    #epoch = [3000,5000]
    #learn_rate = [0.01 / (10**i) for i in range(6)]
    #param_grid = dict(learn_rate=learn_rate,epochs = epoch)
    #print(param_grid)
    #grid = GridSearchCV(estimator=estimator, param_grid=param_grid)#, n_jobs=-1)
    #grid_result = grid.fit(X,encoded_Y,validation_split=.33)
    # summarize results
    #print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #means = grid_result.cv_results_['mean_test_score']
    #stds = grid_result.cv_results_['std_test_score']
    #params = grid_result.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #    print("%f (%f) with: %r" % (mean, stdev, param))
    history = estimator.fit(X, encoded_Y, validation_split=0.2)
    results = cross_val_score(estimator, X, encoded_Y,cv=3)
    print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    #print(int(time.time()) - startTime)
    #print(history.history.keys())
    # summarize history for accuracy
    #plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper right')
    #plt.show()
    # summarize history for loss
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper right')
    #plt.show()
    #print(estimator.score(Xval,encoded_Yval),"nn test")
    #print(estimator.score(X, encoded_Y),"nn train")
    title = "NN curve"
    #cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    print("Right before curve")
    x = plot_learning_curve(estimator, title, X, encoded_Y, cv=3, n_jobs=1)
    x.show()
    print("Training Time:",int(time.time()) - startTime,"\n")
    print (results)

    return estimator.score(Xval,encoded_Yval)
    #return results.mean()*100,results.mean()*100






#startTime = time.time()
#pclf = svm.SVC(gamma=.00001,probability=True,kernel='poly',C=1)
#pclf.fit(X, encoded_Y)
#print(pclf.score(Xval,encoded_Yval),"poly svm test")
#print(pclf.score(X, encoded_Y),"poly svm train")
#print("Training Time:",int(time.time()) - startTime,"\n")

def runSVM(params):
    startTime = time.time()
    #clf = svm.SVC(kernel="sigmoid",probability=True,verbose=False,C=1)
    clf = svm.SVC(kernel=params['kernel'],C=params['C'],gamma=params['gamma'])


    clf.fit(X, encoded_Y)

    print(clf.score(Xval,encoded_Yval),"svm test")
    print(clf.score(X, encoded_Y),"svm train")
    validscore = clf.score(Xval,encoded_Yval)
    testscore = numpy.mean(cross_val_score(clf, X, encoded_Y, cv=5))
    title = "Svm curve"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    x = plot_learning_curve(clf, title, X, encoded_Y, cv=cv, n_jobs=1)
    x.show()
    print("Training Time:",int(time.time()) - startTime,"\n")
    return validscore


def GridSVM():
    tuned_parameters = [{'kernel': ['rbf','linear',"sigmoid"], 'gamma': [1e-3, 1e-4],
                         'C': [.1,1, 10,100]}]

    scores = ['precision']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=3,
                           scoring='%s_macro' % score)
        clf.fit(X, encoded_Y)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = encoded_Yval, clf.predict(Xval)
        print(classification_report(y_true, y_pred))
        print()

    print(clf.score(Xval,encoded_Yval),"svm test")
    print(clf.score(X, encoded_Y),"svm train")
    print("Training Time:",int(time.time()) - startTime,"\n")
    return clf.best_params_



def runKNN(params):
    startTime = time.time()
    knnTests = []
    knnsTrains = []
    param = params["n_neighbors"]
    neigh = KNeighborsClassifier(n_neighbors=param)
    neigh.fit(X, encoded_Y)
    print(neigh.score(Xval,encoded_Yval),"knn test",param)
    print(neigh.score(X, encoded_Y),"knn train")
    knnTests.append(neigh.score(Xval,encoded_Yval))
    knnsTrains.append(neigh.score(X,encoded_Y))
    validscore = neigh.score(Xval,encoded_Yval)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    title = "KNN Curve"
    x = plot_learning_curve(neigh, title, X, encoded_Y, cv=cv, n_jobs=1)
    x.show()


    #plt.plot(knnTests)
    #plt.plot(knnsTrains)
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('# of neighbors')
    #plt.legend(['train', 'test'], loc='upper right')
    #plt.show()
    return validscore

def runDT(params):

    startTime = time.time()
    dt = DecisionTreeClassifier(max_depth=params['max_depth'],max_features=params["max_features"])
    dt.fit(X,encoded_Y)
    validscore = dt.score(Xval,encoded_Yval)
    testscore = numpy.mean(cross_val_score(dt, X, encoded_Y, cv=5))
    title = "Dt curve"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    x = plot_learning_curve(dt, title, X, encoded_Y, cv=cv, n_jobs=1)
    x.show()
    print (validscore,testscore)
    print("Training Time:",int(time.time()) - startTime,"\n")
    return validscore

def GridDT():
    #dt = DecisionTreeClassifier()
    parameter_grid = {'max_depth': [i for i in range(1,50)],
                  'max_features': [i for i in range(1,8)]}
    return GridSearch(DecisionTreeClassifier(),parameter_grid)

def GridSearch(m,tuned_parameters):

    scores = ['precision']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(m, tuned_parameters, cv=3,
                           scoring='%s_macro' % score)
        clf.fit(X, encoded_Y)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #    print("%0.3f (+/-%0.03f) for %r"
        #          % (mean, std * 2, params))
        print()
        return clf.best_params_


def runBoost(params):
    startTime = time.time()
    #dt = DecisionTreeClassifier(max_depth=params["max_depth"],max_features=params["max_features"])
    boost = AdaBoostClassifier(base_estimator = params["base_estimator"],n_estimators = params["n_estimators"], learning_rate = params["learning_rate"])
    boost.fit(X,encoded_Y)
    print(boost.score(Xval,encoded_Yval),"boost test")
    print(boost.score(X, encoded_Y),"boost train")
    print("Training Time:",int(time.time()) - startTime,"\n")
    validscore = boost.score(Xval,encoded_Yval)
    testscore = numpy.mean(cross_val_score(boost, X, encoded_Y, cv=5))
    title = "Boosting curve"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    x = plot_learning_curve(boost, title, X, encoded_Y, cv=cv, n_jobs=1)
    x.show()
    print (validscore,testscore)
    print("Training Time:",int(time.time()) - startTime,"\n")
    return validscore

def GridBoost():
    dts= []
    for j in range(10,51,5):
        for p in range(1,8):
            dts.append(DecisionTreeClassifier(max_depth = j,max_features = p))
    parameter_grid = {'base_estimator': dts,
                  'n_estimators': [10,50,100,500,1000],
                  'learning_rate': [.1,.01,.001]}
    return GridSearch(AdaBoostClassifier(),parameter_grid)


def GridKNN():
    knnTests = []
    knnsTrains = []
    for j in range(1,100):
        neigh = KNeighborsClassifier(n_neighbors=j)
        neigh.fit(X, encoded_Y)
        #print(neigh.score(Xval,encoded_Yval),"knn test",j)
        #print(neigh.score(X, encoded_Y),"knn train")
        knnTests.append(neigh.score(Xval,encoded_Yval))
        knnsTrains.append(neigh.score(X,encoded_Y))
    plt.plot(knnsTrains)
    plt.plot(knnTests)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('# of neighbors')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    parameter_grid = {'n_neighbors': [j for j in range(1,100)]}
    return GridSearch(neigh,parameter_grid)
    #return 1 + knnTests.index(max(knnTests))




X,Xval,Y,Yval = train_test_split(Xfull,Yfull,shuffle=True)
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
#dummy_y = np_utils.to_categorical(encoded_Y)

encoder2 = LabelEncoder()
encoder.fit(Yval)
encoded_Yval = encoder.transform(Yval)
NNtestAccs = []
SVMtestAccs = []
KNNtestAccs = []
DTtestAccs = []
BoosttestAccs = []
testLens = []

totStartTime = time.time()

nnParams = GridNN()
svmParam = GridSVM()
knnParam = GridKNN()
dtParam = GridDT()
boostParam = GridBoost()

nnAcc = runSonarNN(nnParams)
svmAcc =runSVM(svmParam)
knnAcc = runKNN(knnParam)
dtAcc = runDT(dtParam)
boostAcc = runBoost(boostParam)

print (knnParam)
print([nnAcc,svmAcc,knnAcc,dtAcc,boostAcc])

plt.bar(["nn","svm","knn","dt","boost"],[nnAcc,svmAcc,knnAcc,dtAcc,boostAcc])
plt.ylabel('Accuracy')
plt.title('Accuracy By Model')
plt.show()
