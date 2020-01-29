import pandas as pd
import numpy as np
from os import walk
from os.path import join
import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer, word_tokenize

from bs4 import BeautifulSoup

from wordcloud import WordCloud
from PIL import Image

from sklearn.model_selection import train_test_split

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', None)

# ntlk download language pack need to run only once

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('gutenberg')
# nltk.download('shakespeare')

FILE_PATH = '/home/shyam/GitRespository/MachineLearningUdumy/ ' \
            'Pre-ProcessTextDataNaiveBayesClassifier/SampleData/SpamData/SpamData/01_Processing/practice_email.txt'

ROOT = '/home/shyam/GitRespository/MachineLearningUdumy/ ' \
       'Pre-ProcessTextDataNaiveBayesClassifier/SampleData/SpamData/SpamData/01_Processing/spam_assassin_corpus/'
SPAM_1_PATH = ROOT + 'spam_1'
SPAM_2_PATH = ROOT + 'spam_2'
EASY_NON_SPAM_1 = ROOT + 'easy_ham_1'
EASY_NON_SPAM_2 = ROOT + 'easy_ham_2'
DATA_JSON_FILE = '/home/shyam/GitRespository/MachineLearningUdumy/ Pre-ProcessTextDataNaiveBayesClassifier/' \
                 'SampleData/SpamData/SpamData/01_Processing/email-text-data.json'
WHALE_FILE = 'SampleData/SpamData/SpamData/01_Processing/wordcloud_resources/whale-icon.png'
NON_SPAM_THUMPS_UP = 'SampleData/SpamData/SpamData/01_Processing/wordcloud_resources/thumbs-up.png'
SPAM_THUMPS_DOWN = 'SampleData/SpamData/SpamData/01_Processing/wordcloud_resources/thumbs-down.png'
WORD_ID_FILE = 'SampleData/SpamData/SpamData/01_Processing/words-by-id.csv'
# VOCAB_SIZE = 2500
VOCAB_SIZE = 100
CHECK_VOCAB = '/home/shyam/GitRespository/MachineLearningUdumy/ Pre-ProcessTextDataNaiveBayesClassifier/SampleData/' \
              'SpamData/SpamData/01_Processing/word-by-id.csv'

TRAINING_DATA_FILE = 'SampleData/SpamData/SpamData/02_Training/train_data.txt'
TESTING_DATA_FILE = 'SampleData/SpamData/SpamData/02_Training/test_data.txt'


def word_cloud():
    # creating New image file using pillow
    # testing words could with novel spakepear
    novel_corpus = nltk.corpus.gutenberg.words('melville-moby_dick.txt')
    word_list = [''.join(words) for words in novel_corpus]
    word_string = ' '.join(word_list)

    icon = Image.open(WHALE_FILE)
    image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
    image_mask.paste(icon, box=icon)

    rgb_array = np.array(image_mask)
    word_could = WordCloud(mask=rgb_array, background_color='white')
    word_could.generate(str(word_string))
    plt.imshow(word_could, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def read_test_email():
    stream = open(FILE_PATH, encoding='latin-1')
    is_body = False
    lines = []
    for line in stream:
        print(line)
        if is_body:
            lines.append(line)
        elif line == '\n':
            is_body = True
    email_body = '\n'.join(lines)
    stream.close()
    return email_body


def tokenizer_remove_punctuation():
    """remove punctuation, stop words and tokenize """
    # example
    msg = 'All work no play make jack a dull boy, to be or not to be. ??? no body expect to the Spanish ' \
          'Inquisition'
    tokenize_words = word_tokenize(msg.lower())
    streamer = SnowballStemmer('english')
    stop_words = stopwords.words('english')
    stop_words = set(stop_words)
    filtered_words = []
    for words in tokenize_words:
        if (words not in stop_words) and (words.isalpha()):
            streamer_word = streamer.stem(words)
            filtered_words.append(streamer_word)
    print(filtered_words)


def clean_emails(message, streamer=PorterStemmer()):
    """remove punctuation, stop words and tokenize """
    filtered_words = []
    stop_words = set(stopwords.words('english'))
    message_words = word_tokenize(message.lower())
    for words in message_words:
        if (words not in stop_words) and (words.isalpha()):
            filtered_words.append(streamer.stem(words))
    return filtered_words


def clean_mail_html_tags(message, streamer=PorterStemmer()):
    """remove punctuation, stop words and tokenize """
    filtered_words = []
    # remove html tags
    soup = BeautifulSoup(message, 'html.parser')
    cleaned_text = soup.get_text()
    # tokenize words
    stop_words = set(stopwords.words('english'))
    message_words = word_tokenize(cleaned_text.lower())
    for words in message_words:
        if (words not in stop_words) and (words.isalpha()):
            filtered_words.append(streamer.stem(words))
    return filtered_words


def word_is_part_vocabulary(word):
    """Checking If Words In Vocab"""
    vocab = pd.read_csv(CHECK_VOCAB)
    return word in set(vocab.VOCAB_WORD)


def find_longest_email(email_data):
    return email_data.MESSAGE.str.len()


def make_sparse_matrix_test(test_df, indexed_words, test_label):
    """
    df: A Data Frame With Words in The Columns With A Document Id As an Index (X_TEST, Y_TRAIN)
    indexed_words: Index Of Words Order by Word Id
    labels: Category as Series (Y_TRAIN, Y_TEST)
    :return: Sparse Matrix as data Frame test data
    """
    nr_rows = test_df.shape[0]
    nr_cols = test_df.shape[1]
    word_set = set(indexed_words)
    dict_list = []
    for i in range(nr_rows):
        for j in range(nr_cols):
            word = test_df.iat[i, j]
            if word in word_set:
                doc_id = test_df.index[i]
                word_id = indexed_words.get_loc(word)
                category = test_label.at[doc_id]
                item = {'LABEL': category, 'DOC_ID': doc_id, 'OCCURRENCE': 1, 'WORD_ID': word_id}
                dict_list.append(item)
    return pd.DataFrame(dict_list)


def make_sparse_matrix_train(train_df, indexed_words, train_label):
    """
    :return: Sparse Matrix as data Frame for train data
    """
    nr_rows = train_df.shape[0]
    nr_cols = train_df.shape[1]
    word_set = set(indexed_words)
    dict_list = []
    for i in range(nr_rows):
        for j in range(nr_cols):
            word = train_df.iat[i, j]
            if word in word_set:
                doc_id = train_df.index[i]
                word_id = indexed_words.get_loc(word)
                category = train_label.at[doc_id]
                item = {'LABEL': category, 'DOC_ID': doc_id, 'OCCURRENCE': 1, 'WORD_ID': word_id}
                dict_list.append(item)
    return pd.DataFrame(dict_list)


class EmailBodyExtraction:
    """Read all email using Generator"""

    def __init__(self, path, category):
        self.path = path
        self.category = category

    def email_body_generator(self):
        for root, dir_names, file_names in walk(self.path):
            for file_name in file_names:
                file_path = join(root, file_name)
                stream = open(file_path, encoding='latin-1')
                is_body = False
                lines = []
                for line in stream:
                    if is_body:
                        lines.append(line)
                    elif line == '\n':
                        is_body = True
                email_body = '\n'.join(lines)
                yield file_name, email_body

    def df_from_directory(self):
        rows = []
        rows_name = []
        for file_name, email_body in self.email_body_generator():
            rows.append({'MESSAGE': email_body, 'CATEGORY': self.category})
            rows_name.append(file_name)
        return pd.DataFrame(data=rows, index=rows_name)


class EmailAnalysis:
    def __init__(self):
        self.df_all_email = pd.DataFrame()
        self.non_spam_string = ''
        self.spam_string = ''
        self.words_data_frame = None

    def clean_data(self):
        print(self.df_all_email)
        # checking Null values
        print('if any null columns')
        print(self.df_all_email['MESSAGE'].isnull().values.any())
        # check if there is a empty email (string length zero)
        print('length of each string')
        print(self.df_all_email.MESSAGE.str.len().head())
        print('if any string have length zero')
        print((self.df_all_email.MESSAGE.str.len() == 0).any())
        print('how any string have length zero')
        print((self.df_all_email.MESSAGE.str.len() == 0).sum())
        # locate empty email in data frame
        print('index of empty email')
        empty_index = self.df_all_email[self.df_all_email.MESSAGE.str.len() == 0]
        print(empty_index)
        print('fetch location of empty string')
        self.df_all_email.reset_index(inplace=True)
        print(self.df_all_email)
        location = self.df_all_email.loc[self.df_all_email['index'] == 'cmds']
        print(location)
        print(self.df_all_email.iloc[1711])
        print(self.df_all_email.iloc[4263])
        print(self.df_all_email.iloc[5355])
        # remove file from data frame
        self.df_all_email.drop(index=[1711, 4263, 5355], inplace=True)
        print('after drop empty check any exist')
        empty_index = self.df_all_email[self.df_all_email.MESSAGE.str.len() == 0]
        print(empty_index)
        # set document id to index track email
        self.df_all_email.set_index('index', inplace=True)
        self.df_all_email['FILENAME'] = self.df_all_email.index
        document_id = range(0, len(self.df_all_email.index))
        self.df_all_email['DOC_ID'] = document_id
        self.df_all_email.set_index('DOC_ID', inplace=True)
        self.df_all_email = self.df_all_email.iloc[0:100]
        print(self.df_all_email)

    def visualization(self):
        print('count of sam and non spam')
        print(self.df_all_email.CATEGORY.value_counts())
        amount_of_spam = self.df_all_email.CATEGORY.value_counts()[1]
        amount_of_non_spam = self.df_all_email.CATEGORY.value_counts()[0]
        category_names = ['Spam', 'Legit Mail']
        size_data = [amount_of_spam, amount_of_non_spam]
        print(size_data)
        plt.pie(size_data, labels=category_names, startangle=90, autopct='%1.2f%%', explode=[0, 0.1])
        plt.show()

        # donut chart
        plt.pie(size_data, labels=category_names, startangle=90, autopct='%1.2f%%')
        center_circle = plt.Circle((0, 0), radius=0.6, fc='white')
        plt.gca().add_artist(center_circle)
        plt.show()

        # for try donut chart with multiple category
        category_names = ['Spam', 'Legit Mail', 'Update', 'Promotion']
        size_data = [25, 43, 19, 22]
        gape = [0.05, 0.05, 0.05, 0.05]
        plt.pie(size_data, labels=category_names, startangle=90, autopct='%1.2f%%', explode=gape)
        center_circle = plt.Circle((0, 0), radius=0.6, fc='white')
        plt.gca().add_artist(center_circle)
        plt.show()

    def spam_non_spam_word_cloud(self):
        icon_non_spam = Image.open(NON_SPAM_THUMPS_UP)
        icon_spam = Image.open(SPAM_THUMPS_DOWN)
        non_spam_image_mask = Image.new(mode='RGB', size=icon_non_spam.size, color=(255, 255, 255))
        non_spam_image_mask.paste(icon_non_spam, box=icon_non_spam)

        spam_image_mask = Image.new(mode='RGB', size=icon_spam.size, color=(255, 255, 255))
        spam_image_mask.paste(icon_spam)
        rgb_spam_array = np.array(spam_image_mask)
        rgb_non_spam_array = np.array(non_spam_image_mask)
        spam_word_cloud = WordCloud(mask=rgb_spam_array, background_color='white', colormap='gist_heat')
        spam_word_cloud.generate(str(self.spam_string))
        non_spam_word_cloud = WordCloud(mask=rgb_non_spam_array, background_color='white', colormap='gist_heat')
        non_spam_word_cloud.generate(str(self.non_spam_string))
        plt.imshow(non_spam_word_cloud)
        plt.axis('off')
        plt.show()
        plt.imshow(spam_word_cloud)
        plt.axis('off')
        plt.show()

    def vocabulary_dictionary(self):
        # gathering index of spam and non message
        # cleaning html tags
        main_list = self.df_all_email.MESSAGE
        main_list = main_list.apply(clean_mail_html_tags)
        print('generate flat list vocabulary')
        flat_list_non_spam = [item for sub_list in main_list for item in sub_list]
        # making to list to pd Series
        normal_words = pd.Series(flat_list_non_spam)
        print('count of words in non spam email\n', normal_words.value_counts())
        normal_words = normal_words.value_counts()
        print(normal_words)
        normal_words = normal_words.iloc[0: VOCAB_SIZE]
        # creating Data frame in 2500 row
        # creating index id
        words_id = list(range(0, VOCAB_SIZE))
        vocab = pd.DataFrame({'VOCAB_WORD': normal_words.index.values}, index=words_id)
        vocab.index.name = 'WORDS_ID'
        # once done then reading from file
        # vocab.to_csv(DATA_JSON_FILE, index_label=vocab.index.name)
        series = pd.Series(main_list)
        # data frame have created for each word in column
        self.words_data_frame = pd.DataFrame.from_records(series.to_list())

    def train_test_data_frame(self):
        x_train, x_test, y_train, y_test = train_test_split(self.words_data_frame, self.df_all_email.CATEGORY,
                                                            test_size=0.3, random_state=42)
        print('no of train sample\t', x_train.shape[0])
        print('fraction of training set\t', x_train.shape[0] / self.words_data_frame.shape[0])
        # creating spark matrix
        # print(x_train.head())
        print(self.words_data_frame)
        vocab = pd.read_csv(CHECK_VOCAB)
        # line using reduce time to run script need to remove final stage
        vocab = vocab.iloc[0:9]
        word_index = pd.Index(vocab.VOCAB_WORD)
        # calling function to create sparse matrix
        sparse_matrix_train = make_sparse_matrix_train(train_df=x_train, indexed_words=word_index, train_label=y_train)
        # combine occurrence sparse matrix using group by method
        sparse_matrix_train = sparse_matrix_train.groupby(['DOC_ID', 'WORD_ID', 'LABEL']) \
            .sum() \
            .reset_index()
        # saving text file using numpy
        np.savetxt(TRAINING_DATA_FILE, sparse_matrix_train, fmt='%d')
        print(sparse_matrix_train)
        # calling function to create sparse matrix test data
        sparse_matrix_train_test = make_sparse_matrix_test(test_df=x_test, indexed_words=word_index, test_label=y_test)
        # combine occurrence sparse matrix using group by method
        sparse_matrix_train_test = sparse_matrix_train_test.groupby(['DOC_ID', 'WORD_ID', 'LABEL']) \
            .sum() \
            .reset_index()
        # saving text file using numpy
        np.savetxt(TESTING_DATA_FILE, sparse_matrix_train_test, fmt='%d')
        print(sparse_matrix_train_test)

    def cleaning_message(self):
        """remove punctuation, stop words and tokenize """
        # gathering index of spam and non message
        doc_ids_spam = self.df_all_email[self.df_all_email.CATEGORY == 0].index
        doc_ids_non_spam = self.df_all_email[self.df_all_email.CATEGORY == 1].index
        main_list = self.df_all_email.MESSAGE
        main_list = main_list.apply(clean_emails)
        non_span_word_list = main_list.loc[doc_ids_non_spam]
        span_word_list = main_list.loc[doc_ids_spam]
        print('generate flat list')
        flat_list_non_spam = [item for sub_list in non_span_word_list for item in sub_list]
        flat_list_spam = [item for sub_list in span_word_list for item in sub_list]
        # making to list to pd Series
        normal_words = pd.Series(flat_list_non_spam)
        un_normal_words = pd.Series(flat_list_spam)
        print('count of words in non spam email\n', normal_words.value_counts())
        print('count of words in spam email\n', un_normal_words.value_counts())
        # creating words cloud for spam and non message
        self.non_spam_string = ' '.join(normal_words)
        self.spam_string = ' '.join(un_normal_words)

    def __call__(self, *args, **kwargs):
        spam_email = EmailBodyExtraction(path=SPAM_1_PATH, category=1)
        df_spam_email = spam_email.df_from_directory()
        spam_email = EmailBodyExtraction(path=SPAM_2_PATH, category=1)
        df_spam_email = df_spam_email.append(spam_email.df_from_directory())
        spam_email = EmailBodyExtraction(path=EASY_NON_SPAM_1, category=0)
        df_non_spam_email = spam_email.df_from_directory()
        spam_email = EmailBodyExtraction(path=EASY_NON_SPAM_2, category=0)
        df_non_spam_email = df_non_spam_email.append(spam_email.df_from_directory())
        self.df_all_email = pd.concat([df_non_spam_email, df_spam_email])
        self.clean_data()
        # visualize email using pie diagram
        # self.visualization()

        # call cleaning method
        # self.cleaning_message()

        # words cloud thumps up and down graphical representationdf_all_email
        # self.spam_non_spam_word_cloud()

        # creating vocabulary dictionary and clean all html tags
        self.vocabulary_dictionary()

        # word is part of vocabulary
        # print(self.word_is_part_vocabulary(word='check'))

        # find email with most number of words

        # find length of mail
        # longest_mail = find_longest_email(self.df_all_email)
        # print('mail length\n', longest_mail)
        # print('max length mail\n', longest_mail.max())
        # print('min length mail\n', longest_mail.min())

        # train and test data
        self.train_test_data_frame()


email_analysis = EmailAnalysis()
email_analysis()
# word_cloud()
