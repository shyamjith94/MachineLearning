import numpy as np
import pandas as pd

TEST_DATA = '/home/shyam/GitRespository/MachineLearningUdumy/ ' \
            'Pre-ProcessTextDataNaiveBayesClassifier/SampleData/SpamDatanext/SpamData/02_Training/test-data.txt'
TRAIN_DATA = '/home/shyam/GitRespository/MachineLearningUdumy/ ' \
             'Pre-ProcessTextDataNaiveBayesClassifier/SampleData/SpamDatanext/SpamData/02_Training/train-data.txt'

SPAM_PROB_FILE = '/home/shyam/GitRespository/MachineLearningUdumy/ ' \
                 'Pre-ProcessTextDataNaiveBayesClassifier/SampleData/SpamDatanext/SpamData/03_Testing/prob_spam.txt'
NON_SPAM_PROB_FILE = '/home/shyam/GitRespository/MachineLearningUdumy/ ' \
                     'Pre-ProcessTextDataNaiveBayesClassifier/SampleData/SpamDatanext/SpamData/03_Testing' \
                     '/prob_nonspam.txt'
TOKEN_ALL_PROB_FILE = '/home/shyam/GitRespository/MachineLearningUdumy/ ' \
                      'Pre-ProcessTextDataNaiveBayesClassifier/SampleData/SpamDatanext/SpamData/03_Testing/pro_all.txt'

TEST_FEATURE_MATRIX = '/home/shyam/GitRespository/MachineLearningUdumy/ ' \
                      'Pre-ProcessTextDataNaiveBayesClassifier/SampleData/SpamDatanext/SpamData/03_Testing/test' \
                      '-features.txt'

TEST_TARGET_FILE = '/home/shyam/GitRespository/MachineLearningUdumy/ ' \
                   'Pre-ProcessTextDataNaiveBayesClassifier/SampleData/SpamDatanext/SpamData/03_Testing/test-target.txt'

VOCAB_SIZE = 100


# reduce run time
# VOCAB_SIZE = 2500


def make_full_matrix(sparse_matrix, nr_words, doc_idx=0, word_idx=1,
                     cat_idx=2, freq_idx=3):
    """
    full matrix from a sparse matrix. return Data frame.
    :param sparse_matrix: numpy array
    :param nr_words: size of vocab total number of token
    :param doc_idx: position of of the doc id in sparse matrix default 1st column
    :param word_idx: position of the word id  in sparse matrix default 2nd column
    :param cat_idx: position of the label (spam is one and non spam is 0) default 3rd column
    :param freq_idx: position occurrence of word in sparse matrix default 4th column
    :return: DataFrame
    """
    columns_names = ['DOC_ID'] + ['CATEGORY'] + list(range(0, VOCAB_SIZE))
    doc_id_names = np.unique(sparse_matrix[:, 0])
    full_train_data = pd.DataFrame(index=doc_id_names, columns=columns_names)
    full_train_data.fillna(value=0, inplace=True)
    for i in range(0, len(sparse_matrix)):
        doc_nr = sparse_matrix[i][doc_idx]
        word_id = sparse_matrix[i][word_idx]
        label = sparse_matrix[i][cat_idx]
        occurrence = sparse_matrix[i][freq_idx]
        full_train_data.at[doc_nr, 'DOC_ID'] = doc_nr
        full_train_data.at[doc_nr, 'CATEGORY'] = label
        full_train_data.at[doc_nr, word_id] = occurrence
    full_train_data.set_index('DOC_ID', inplace=True)
    print(full_train_data.head())
    return full_train_data


def loading_reading_file():
    """read and load .txt file to numpy array"""

    sparse_train_data = np.loadtxt(TRAIN_DATA, delimiter=' ', dtype=int)
    sparse_test_data = np.loadtxt(TEST_DATA, delimiter=' ', dtype=int)
    print('number of lines in test data', sparse_test_data.shape[0])
    print('number of lines in train data', sparse_train_data.shape[0])
    print('number of email training files', np.unique(sparse_train_data[:, 0]).size)
    print('number of email testing files', np.unique(sparse_test_data[:, 0]).size)
    # create full matrix data frame
    full_train_data = make_full_matrix(sparse_train_data, VOCAB_SIZE)
    # Train the naive model
    # calculate the probability
    prob_spam = full_train_data.CATEGORY.sum() / full_train_data.size
    print(f'probability of spam {prob_spam}')
    full_train_features = full_train_data.loc[:, full_train_data.columns != 'CATEGORY']
    # each email length
    email_length = full_train_features.sum(axis=1)
    # sum of word count all email
    total_word_count = email_length.sum()
    print(f'total word count {total_word_count}')
    # number of tokens in spam and non spam email
    spam_length = email_length[full_train_data.CATEGORY == 1]
    spam_word_count = spam_length.sum()
    print(f'spam word count{spam_word_count}')
    non_spam_length = email_length[full_train_data.CATEGORY == 0]
    non_spam_word_cont = non_spam_length.sum()
    print(f'non spam word count{non_spam_word_cont}')
    # valid subset data frame correct
    print(f'check subset are proper{spam_length.shape[0] - non_spam_length.shape[0]}')
    print(f'average number of words in spam email{spam_word_count / spam_length.shape[0]}')
    print(f'average number of words in spam email{non_spam_word_cont / non_spam_length.shape[0]}')

    # summing the tokens
    train_spam_tokens = full_train_features.loc[full_train_data.CATEGORY == 1]
    summed_spam_tokens = train_spam_tokens.sum(axis=0) + 1
    print(f'summed spam tokens {summed_spam_tokens}')

    train_non_spam_tokens = full_train_features.loc[full_train_data.CATEGORY == 0]
    summed_non_spam_tokens = train_non_spam_tokens.sum(axis=0) + 1
    print(f'summed non spam tokens {summed_non_spam_tokens}')

    # calculate probability token occur at then given email spam
    # p(token|spam)
    prob_token_spam = summed_spam_tokens / (spam_word_count + VOCAB_SIZE)
    print('probability of spam\n', prob_token_spam)
    print(f'total probability of spam email{prob_token_spam.sum()}')
    # calculate probability token occur at then given email non spam
    # p(token|non spam)
    prob_token_non_spam = summed_non_spam_tokens / (non_spam_word_cont + VOCAB_SIZE)
    print('probability of spam\n', prob_token_non_spam)
    print(f'total probability of non spam email{prob_token_non_spam.sum()}')

    # probability of token occur
    # p(token)
    prob_token_all = full_train_features.sum(axis=0) / total_word_count
    print(f'probability of token all email{prob_token_all.sum()}')
    # creating txt file of probability
    np.savetxt(SPAM_PROB_FILE, prob_token_spam)
    np.savetxt(NON_SPAM_PROB_FILE, prob_token_non_spam)
    np.savetxt(TOKEN_ALL_PROB_FILE, prob_token_all)

    # saving text file to
    full_test_data = make_full_matrix(sparse_test_data, VOCAB_SIZE)
    x_test = full_test_data.loc[:, full_train_data.columns != 'CATEGORY']
    y_test = full_test_data.CATEGORY
    np.savetxt(TEST_TARGET_FILE, y_test)
    np.savetxt(TEST_FEATURE_MATRIX, x_test)


loading_reading_file()
