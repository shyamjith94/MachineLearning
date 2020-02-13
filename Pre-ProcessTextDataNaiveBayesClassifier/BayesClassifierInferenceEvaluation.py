import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# constants
SPAM_PROB_FILE = '/home/shyam/GitRespository/MachineLearningUdumy/Pre-ProcessTextDataNaiveBayesClassifier/SampleData' \
                 '/SpamData/SpamData/03_Testing/prob_spam.txt'

NON_SPAM_PROB_FILE = '/home/shyam/GitRespository/MachineLearningUdumy/Pre-ProcessTextDataNaiveBayesClassifier' \
                     '/SampleData/SpamData/SpamData/03_Testing/prob_nonspam.txt'

TOKEN_ALL_PROB_FILE = '/home/shyam/GitRespository/MachineLearningUdumy/Pre-ProcessTextDataNaiveBayesClassifier' \
                      '/SampleData/SpamData/SpamData/03_Testing/pro_all.txt'

TEST_FEATURE_MATRIX = '/home/shyam/GitRespository/MachineLearningUdumy/Pre-ProcessTextDataNaiveBayesClassifier' \
                      '/SampleData/SpamData/SpamData/03_Testing/test-features.txt'

TEST_TARGET_FILE = '/home/shyam/GitRespository/MachineLearningUdumy/Pre-ProcessTextDataNaiveBayesClassifier' \
                   '/SampleData/SpamData/SpamData/03_Testing/test-target.txt'
VOCAB_SIZE = 2500

x_test = np.loadtxt(TEST_FEATURE_MATRIX, delimiter=' ')
y_test = np.loadtxt(TEST_TARGET_FILE, delimiter=' ')
prob_token_spam = np.loadtxt(SPAM_PROB_FILE, delimiter=' ')
prob_token_non_spam = np.loadtxt(NON_SPAM_PROB_FILE, delimiter=' ')
prob_token_all = np.loadtxt(TOKEN_ALL_PROB_FILE, delimiter=' ')

# join conditional probability
# multiplying all conditional probability


# calculate join probability
# dot product help to multiply array x_text dot product
spam_dot_product = x_test.dot(prob_token_spam).shape
print(f'spam shape of dot product is- {spam_dot_product[0]}')
PROB_SPAM = 0.3116
joint_log_spam = x_test.dot(np.log(prob_token_spam) - np.log(prob_token_all)) + np.log(PROB_SPAM)
print('joint probability sample\n', joint_log_spam)
# join probability email non spam
non_spam_dot_product = x_test.dot(prob_token_non_spam).shape
print(f'non spam shape of dot product is- {non_spam_dot_product[0]}')
joint_log_non_spam = x_test.dot(np.log(prob_token_non_spam) - np.log(prob_token_all)) + np.log(1 - PROB_SPAM)
print('joint probability non spam sample\n', joint_log_non_spam)
# making predictions
# checking higher join probability
# p(spam|x) > p(non_spam|x)
# p(spam|x) < p(non_spam|x)
prediction = joint_log_spam > joint_log_non_spam
print('sample for prediction\n', prediction[:5])
# matrix evaluation
# accurate
correct_doc = (y_test == prediction).sum()
print(f'correctly classified- {correct_doc}')
wrong_doc = (y_test.shape[0] - correct_doc)
print(f'wrongly classified- {wrong_doc}')
accurate_fraction = correct_doc / len(x_test)
wrong_fraction = wrong_doc / len(x_test)
print(f'fraction accurate dock- {accurate_fraction} and wrong is {wrong_fraction}')

# visualization the result
plt.subplot(1, 2, 1)
y_axis_label = 'p(x|spam)'
x_axis_label = 'p(x|non_spam'
# plot line
line_data = np.linspace(start=-1400, stop=1, num=1000)
plt.figure(figsize=(11, 7))
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.xlim([-14000, 1])
plt.ylim([-14000, 1])
plt.scatter(joint_log_non_spam, joint_log_spam, color='navy')
plt.plot(line_data, line_data, color='orange')
plt.subplot(1, 2, 2)
plt.xlim([-100, 1])
plt.ylim([-100, 1])
plt.scatter(joint_log_non_spam, joint_log_spam, color='navy')
plt.plot(line_data, line_data, color='orange')

plt.show()

sns.set_style('whitegrid')
labels = 'actual category'
plt.xlim([-2000, 1])
plt.ylim([-2000, 1])
summary_df = pd.DataFrame({y_axis_label: joint_log_spam, x_axis_label: joint_log_non_spam, labels: y_test})
sns.lmplot(x=x_axis_label, y=y_axis_label, data=summary_df, fit_reg=False, scatter_kws={'alpha': 0.5, 's': 25},
           hue=labels, markers=['o', 'x'], palette='hls', legend=False)
plt.plot(line_data, line_data, color='orange')
plt.legend(('decision boundary', 'non_spam', 'spam'), loc='lower right', fontsize=14)
plt.show()

# false positive and false negative
np.unique(prediction, return_counts=True)
true_positive = (y_test == 1) & (prediction == 1)
true_positive.sum()
false_positive = (y_test == 0) & (prediction == 1)
false_positive.sum()
false_negative = (y_test == 1) & (prediction == 0)
false_negative.sum()

# recall score
recall_score = true_positive.sum() / (true_positive.sum() + false_negative.sum())
print(f'recall score is- {recall_score}')

# precision score
precision_score = true_positive.sum() / (true_positive.sum() + false_positive.sum())
print(f'precision score is- {precision_score}')

# f-care
f_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
print(f'f-score score is- {f_score}')
