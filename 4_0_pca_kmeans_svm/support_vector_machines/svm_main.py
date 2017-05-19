import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import preprocessing

def visualize_2D(features, labels, clf_a, clf_b, clf_c):
    h = 0.02
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    for i, clf in enumerate((clf_a, clf_b, clf_c)):
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        if clf.probability == False:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            Z = Z[:,1].reshape(xx.shape)
        plt.subplot(2, 2, i + 1)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(features[:, 0], features[:, 1], c=labels, cmap=plt.cm.coolwarm)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 2')
        plt.title('SVMs')
    plt.show()
    
# Load features and labels from file.
features = np.loadtxt("features_svm.txt")
n_samples, n_features = np.shape(features)
labels = np.loadtxt("labels_svm.txt")
features_evaluation = np.loadtxt("features_evaluation.txt")
visualization = True

# Check if scale normalizes the data! Normalize your input feature data (the evaluation data is already normalized).
features = preprocessing.scale(features)

# Train 3 different classifiers as specified in the exercise sheet (exchange the value for None).
classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(features,labels)
classifier_g_perfect = svm.SVC(kernel='rbf', C=1000000)
classifier_g_perfect.fit(features,labels)
classifier_g_slack = svm.SVC(kernel='rbf', C=1)
classifier_g_slack.fit(features,labels)

# Train 3 different classifiers only on first 2 dimensions for visualization (exchange the value for None).
if visualization:
    classifier_linear_viz = svm.SVC(kernel='linear').fit(features[:,0:2],labels)
    classifier_g_perfect_viz = svm.SVC(kernel='rbf', C=1000000).fit(features[:,0:2],labels)
    classifier_g_slack_viz = svm.SVC(kernel='rbf', C=1).fit(features[:,0:2],labels)
    visualize_2D(features, labels, classifier_linear_viz, classifier_g_perfect_viz, classifier_g_slack_viz)

# Classify evaluation data and store classifications to file (exchange the value for None).
Y_linear = classifier_linear.predict(features_evaluation)
Y_g_perfect = classifier_g_perfect.predict(features_evaluation)
Y_g_slack = classifier_g_perfect.predict(features_evaluation)

# Save probability results to file.
np.savetxt('results_svm_Y_linear.txt', Y_linear)
np.savetxt('results_svm_Y_g_perfect.txt', Y_g_perfect)
np.savetxt('results_svm_Y_g_slack.txt', Y_g_slack)

# Train 3 different classifiers as specified in the exercise sheet (exchange the value for None).
classifier_linear = svm.SVC(kernel='linear', probability=True)
classifier_linear.fit(features,labels)
classifier_g_perfect = svm.SVC(kernel='rbf', C=1000000, probability=True)
classifier_g_perfect.fit(features,labels)
classifier_g_slack = svm.SVC(kernel='rbf', C=1, probability=True)
classifier_g_slack.fit(features,labels)

# Train 3 different classifiers only on first 2 dimensions for visualization (exchange the value for None).
if visualization:
    classifier_linear_viz = svm.SVC(kernel='linear',probability=True).fit(features[:,0:2],labels)
    classifier_g_perfect_viz = svm.SVC(kernel='rbf', C=100000, probability=True).fit(features[:,0:2],labels)
    classifier_g_slack_viz = svm.SVC(kernel='rbf', C=1,probability=True).fit(features[:,0:2],labels)
    visualize_2D(features, labels, classifier_linear_viz, classifier_g_perfect_viz, classifier_g_slack_viz)

# Classify newly loaded features and store classification probabilities to file (exchange the value for None).
P_linear = classifier_linear.predict_proba(features_evaluation)
P_g_perfect = classifier_g_perfect.predict_proba(features_evaluation)
P_g_slack = classifier_g_slack.predict_proba(features_evaluation)

# Save probability results to file. 
# P_linear, P_g_perfect and P_g_slack are of the form N x 2 dimensions,
#  with number of features N and classification probabilities for the two classes. 
np.savetxt('results_svm_P_linear.txt', P_linear)
np.savetxt('results_svm_P_g_perfect.txt', P_g_perfect)
np.savetxt('results_svm_P_g_slack.txt', P_g_slack)