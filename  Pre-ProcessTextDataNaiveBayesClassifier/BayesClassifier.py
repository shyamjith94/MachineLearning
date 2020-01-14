import pandas as pd
from os import walk
from os.path import join
import matplotlib.pyplot as plt

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', None)

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
        plt.pie(size_data, labels=category_names, startangle=90, autopct='%1.2f%%')
        plt.show()

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
        self.visualization()


email_analysis = EmailAnalysis()
email_analysis()
