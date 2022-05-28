import csv
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, make_scorer
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from  sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# Read training data
train_domains = []
y_train = []
with open("train.txt", 'r') as f:
    for line in f:
        l = line.split(',')
        train_domains.append(l[0])
        y_train.append(l[1][:-1])

# Read test data
y_test = []
test_domains = []
with open("test.txt", 'r') as f:
    for line in f:
        l = line.split(',')
        test_domains.append(l[0])

# Create a directed graph
G = nx.read_edgelist('edgelist.txt', delimiter=' ', create_using=nx.DiGraph())

print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())


# Create the training matrix. Each row corresponds to a web host.
# Use the following 3 features for each web host:
# (1) out-degree of node
# (2) in-degree of node
# (3) average degree of neighborhood of node
x_train = np.zeros((len(train_domains), 3))
avg_neig_deg = nx.average_neighbor_degree(G, nodes=train_domains)
for i in range(len(train_domains)):
    x_train[i,0] = G.in_degree(train_domains[i])
    x_train[i,1] = G.out_degree(train_domains[i])
    x_train[i,2] = avg_neig_deg[train_domains[i]]

# Create the test matrix. Use the same 3 features as above
x_test = np.zeros((len(test_domains), 3))
avg_neig_deg = nx.average_neighbor_degree(G, nodes=test_domains)
for i in range(len(test_domains)):
    x_test[i,0] = G.in_degree(test_domains[i])
    x_test[i,1] = G.out_degree(test_domains[i])
    x_test[i,2] = avg_neig_deg[test_domains[i]]

print("Train matrix dimensionality: ", x_train.shape)
print("Test matrix dimensionality: ", x_test.shape)

para = dict()
para['solver'] = ['newton-cg', 'lbfgs','sag','saga']
para['C'] = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

logreg = LogisticRegression()


# Use logistic regression to classify the webpages of the test set
clf = GridSearchCV(logreg, param_grid = para, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1))
model =clf.fit(x_train, y_train)

y_pred = model.predict_proba(x_test)
print('Best Score: %s' % model.best_score_)
print('Best Hyperparameters: %s' % model.best_params_)

alt =svm.SVC(probability=True)
alt_param= dict()
alt_param['C']= [0.1,1,10,100,1000]
alt_param['gamma']=[1,0.1,0.001,0.0001]
alt_param['kernel']=['rbf']
clf = GridSearchCV(alt, param_grid=alt_param, cv=5)
alt_model = clf.fit(x_train,y_train)

ny_pred = alt_model.predict_proba(x_test)
print('Best Score: %s' % alt_model.best_score_)
print('Best Hyperparameters: %s' % alt_model.best_params_)




# Write predictions to a file
with open('sample_submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = []
    for i in range(10):
        lst.append('class_'+str(i))
    lst.insert(0, "domain_name")
    writer.writerow(lst)
    for i,test_domain in enumerate(test_domains):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_domain)
        writer.writerow(lst)

