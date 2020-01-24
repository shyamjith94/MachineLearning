import pandas as pd
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
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', None)

nltk.download('punkt')
nltk.download('stopwords')
FILE_PATH = '/home/shyam/GitRespository/MachineLearningUdumy/ ' \
            'Pre-ProcessTextDataNaiveBayesClassifier/SampleData/SpamData/SpamData/01_Processing/practice_email.txt'

ROOT = '/home/shyam/GitRespository/MachineLearningUdumy/ ' \
       'Pre-ProcessTextDataNaiveBayesClassifier/SampleData/SpamData/SpamData/01_Processing/spam_assassin_corpus/'
SPAM_1_PATH = ROOT + 'spam_1'
SPAM_2_PATH = ROOT + 'spam_2'
EASY_NON_SPAM_1 = ROOT + 'easy_ham_1'
EASY_NON_SPAM_2 = ROOT + 'easy_ham_2'


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
    def word_cloud(self):
        pass
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
        # gathering index of spam and non message
        doc_ids_spam = self.df_all_email[self.df_all_email.CATEGORY == 0].index
        doc_ids_non_spam = self.df_all_email[self.df_all_email.CATEGORY == 1].index
        main_list = self.df_all_email.MESSAGE
        main_list = main_list.apply(clean_emails)
        non_span_word_list = main_list.loc[doc_ids_non_spam]
        span_word_list = main_list.loc[doc_ids_spam]
        flat_list_non_spam = [item for sub_list in non_span_word_list for item in sub_list]
        flat_list_spam = [item for sub_list in span_word_list for item in sub_list]
        # making to list to pd Series
        normal_words = pd.Series(flat_list_non_spam)
        un_normal_words = pd.Series(flat_list_spam)
        print('count of words in non spam email\n', normal_words.value_counts())
        print('count of words in spam email\n', un_normal_words.value_counts())
        print(df_spam_email.MESSAGE)
        word_coud_test = WordCloud().generate(df_spam_email.MESSAGE[0])
        plt.imshow(word_coud_test)
        plt.axis('off')
        plt.show()
email_analysis = EmailAnalysis()
email_analysis()
