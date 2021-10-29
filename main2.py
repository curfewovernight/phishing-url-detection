import re
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def extract_feature_train(url, label):
    # set 1 if length of url greater than 54
    l_url = len(url)
    if l_url > 54:
        length_of_url = 1
    else:
        length_of_url = 0

    # set 1 if url has http/s string
    if ("http://" in url) or ("https://" in url):
        http_has = 1
    else:
        http_has = 0

    # set 1 if url has suspicious character
    if ("@" in url) or ("//" in url):
        suspicious_char = 1
    else:
        suspicious_char = 0

    # if has prefix or suffix ("-")
    if "-" in url:
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    # set 1 if no. of dots between 1 - 5
    if "." in url:
        dot_count = len(url.split('.')) - 1
        if dot_count > 5:
            dots = 0
        else:
            dots = 1
    else:
        dots = 0

    # set 1 if no. of slash between 1 - 5
    if "/" in url:
        slash_count = len(url.split('/')) - 1
        if slash_count > 5:
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    # url has phishing terms
    if (("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or (
            "banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0

    # set 1 if length of subdomain < 5
    it = url.index("//") + 2
    if "." in url:
        j = url.index(".")
        sd_char_len = j - it
    else:
        sd_char_len = 3

    if sd_char_len > 5:
        sub_domain = 0
    else:
        sub_domain = 1

    # url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", url):
        ip_contain = 1
    else:
        ip_contain = 0

    # label - phishing or legit
    lbl = label

    return lbl, length_of_url, http_has, suspicious_char, prefix_suffix, dots, slash, phis_term, sub_domain, ip_contain


def extract_feature_test(url, label):
    # set 1 if length of url greater than 54
    l_url = len(url)
    if l_url > 54:
        length_of_url = 1
    else:
        length_of_url = 0

    # set 1 if url has http/s string
    if ("http://" in url) or ("https://" in url):
        http_has = 1
    else:
        http_has = 0

    # set 1 if url has suspicious character
    if ("@" in url) or ("//" in url):
        suspicious_char = 1
    else:
        suspicious_char = 0

    # if has prefix or suffix ("-")
    if "-" in url:
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    # set 1 if no. of dots between 1 - 5
    if "." in url:
        dot_count = len(url.split('.')) - 1
        if dot_count > 5:
            dots = 0
        else:
            dots = 1
    else:
        dots = 0

    # set 1 if no. of slash between 1 - 5
    if "/" in url:
        slash_count = len(url.split('/')) - 1
        if slash_count > 5:
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    # url has phishing terms
    if (("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or (
            "banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0

    # set 1 if length of subdomain < 5
    it = url.index("//") + 2
    j = url.index(".")
    sd_char_len = j - it

    if sd_char_len > 5:
        sub_domain = 0
    else:
        sub_domain = 1

    # url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", url):
        ip_contain = 1
    else:
        ip_contain = 0

    # label - phishing or legit
    lbl = label

    return lbl, length_of_url, http_has, suspicious_char, prefix_suffix, dots, slash, phis_term, sub_domain, ip_contain


def import_train_data():
    balance_data = pd.read_csv('Extracted_Training_Features.csv', sep=',', header=1, usecols=range(1, 11), encoding='utf-8')
    print(type(balance_data))

    # print dataset shape
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)

    # print dataset observations
    print("Dataset: \n", balance_data.head())
    return balance_data


def import_test_data():
    balance_data = pd.read_csv('Extracted_Testing_Features.csv', sep=',', header=1, usecols=range(1, 11), encoding='utf-8')

    # print dataset shape
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)

    # print dataset observations
    print("Dataset: \n", balance_data.head())
    return balance_data


def split_dataset(balance_data):

    # separate label (y/n) from features
    # X has features
    # Y has label (y/n)
    X = balance_data.values[:, 1:10]
    Y = balance_data.values[:, 0]

    # splitting the dataset into train and test
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

    return X, Y


def extract_feature_usertest(url):
    # set 1 if length of url greater than 54
    l_url = len(url)
    if l_url > 54:
        length_of_url = 1
    else:
        length_of_url = 0

    # set 1 if url has http/s string
    if ("http://" in url) or ("https://" in url):
        http_has = 1
    else:
        http_has = 0

    # set 1 if url has suspicious character
    if ("@" in url) or ("//" in url):
        suspicious_char = 1
    else:
        suspicious_char = 0

    # if has prefix or suffix ("-")
    if "-" in url:
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    # set 1 if no. of dots between 1 - 5
    if "." in url:
        dot_count = len(url.split('.')) - 1
        if dot_count > 5:
            dots = 0
        else:
            dots = 1
    else:
        dots = 0

    # set 1 if no. of slash between 1 - 5
    if "/" in url:
        slash_count = len(url.split('/')) - 1
        if slash_count > 5:
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    # url has phishing terms
    if (("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or (
            "banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0

    # set 1 if length of subdomain < 5
    it = url.index("//") + 2
    j = url.index(".")
    sd_char_len = j - it

    if sd_char_len > 5:
        sub_domain = 0
    else:
        sub_domain = 1

    # url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", url):
        ip_contain = 1
    else:
        ip_contain = 0

    return length_of_url, http_has, suspicious_char, prefix_suffix, dots, slash, phis_term, sub_domain, ip_contain


def cal_accuracy(y_test, y_pred):

    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred) * 100, "%\n")

    return accuracy_score(y_test, y_pred) * 100


def main():
    # training datatset file
    training_excel_file = 'training.xlsx'
    df = pd.DataFrame(pd.read_excel(training_excel_file))

    # testing dataset file
    testing_excel_file = 'test1.xlsx'
    df1 = pd.DataFrame(pd.read_excel(testing_excel_file))

    training_url_list = []
    training_label_list = []

    testing_url_list = []
    testing_label_list = []

    for url_items in df['url']:
        training_url_list.append(url_items)

    for label_items in df['phishing']:
        training_label_list.append(label_items)

    for url_items1 in df1['url']:
        testing_url_list.append(url_items1)

    for label_items1 in df1['result']:
        testing_label_list.append(label_items1)

    # features list
    extract_training_feature_list = []
    extract_testing_feature_list = []

    for url_items1, label_items1 in zip(training_url_list, training_label_list):
        url = url_items1
        label = label_items1
        extract_training_feature_list.append(extract_feature_train(url, label))

    for url_items1, label_items1 in zip(testing_url_list, testing_label_list):
        url = url_items1
        label = label_items1
        extract_testing_feature_list.append(extract_feature_test(url, label))

    df_train = pd.DataFrame(extract_training_feature_list,
                            columns=['label', 'length_of_url', 'http_has', 'suspicious_char', 'prefix_suffix', 'dots',
                                     'slash', 'phis_term', 'sub_domain', 'ip_contain'])

    df_train.to_csv('Extracted_Training_Features.csv', sep=',', encoding='utf-8')

    df_test = pd.DataFrame(extract_testing_feature_list,
                           columns=['label', 'length_of_url', 'http_has', 'suspicious_char', 'prefix_suffix', 'dots',
                                    'slash', 'phis_term', 'sub_domain', 'ip_contain'])

    df_test.to_csv('Extracted_Testing_Features.csv', sep=',', encoding='utf-8')

    train_data = import_train_data()
    test_data = import_test_data()

    X, Y = split_dataset(train_data)
    X1, Y1 = split_dataset(test_data)

    # RF Report
    print("RF Rep")
    rfc = RandomForestClassifier()
    rfc.fit(X, Y)
    y_pred1 = rfc.predict(X1)
    acc1 = cal_accuracy(Y1, y_pred1)

    # svm Report
    print("SVM Rep")
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X, Y)
    y_pred2 = clf.predict(X1)
    acc2 = cal_accuracy(Y1, y_pred2)

    # DT Report
    print("DT Rep")
    dct = DecisionTreeClassifier()
    dct.fit(X, Y)
    y_pred3 = dct.predict(X1)
    acc3 = cal_accuracy(Y1, y_pred3)

    # GNB Report
    print("Gaussian Naive Bayes Rep")
    gnb = GaussianNB()
    gnb.fit(X, Y)
    y_pred4 = gnb.predict(X1)
    acc4 = cal_accuracy(Y1, y_pred4)

    # KNeighborsClassifier Report
    print("k-nearest neighbors Rep")
    neigh = KNeighborsClassifier(2)
    neigh.fit(X, Y)
    y_pred5 = neigh.predict(X1)
    acc5 = cal_accuracy(Y1, y_pred5)

    # AdaBoost Report
    print("AdaBoost Rep")
    ada = AdaBoostClassifier()
    ada.fit(X, Y)
    y_pred6 = ada.predict(X1)
    acc6 = cal_accuracy(Y1, y_pred6)

    # input url from user
    g_url = input("\n\nEnter url: ").strip()

    # RF
    def rfc_pred():
        try:
            rfc_url = g_url
            e = np.array([extract_feature_usertest(rfc_url)])
            userpredict1 = rfc.predict(e.reshape(1, -1))
            print("\n---------------------------Random Forest---------------------------")
            if (userpredict1[0] == 'no'):
                print('Legitimate')
            else:
                print('Phising')

        except Exception as e:
            print('Error: ', str(e))
            print("Check if http protocol is specified!")

    rfc_pred()

    # SVM
    def svm_pred():
        try:
            url2 = g_url
            e2 = np.array([extract_feature_usertest(url2)])
            userpredict2 = clf.predict(e2.reshape(1, -1))
            print("\n---------------------------SVM---------------------------")
            if (userpredict2[0] == 'no'):
                print('Legitimate')
            else:
                print('Phising')

        except Exception as e:
            print('Error: ', str(e))
            print("Check if http protocol specified!")

    svm_pred()

    # unified function
    def comm_pred(c_clf, c_name):
        try:
            url = g_url
            e = np.array([extract_feature_usertest(url)])
            userpredict = c_clf.predict(e.reshape(1, -1))
            print("\n---------------------------" + c_name + "---------------------------")
            if (userpredict[0] == 'no'):
                print('Legitimate')
            else:
                print('Phising')

        except Exception as e:
            print('Error: ', str(e))
            print("Check if http protocol specified!")

    comm_pred(dct, "DT")
    comm_pred(gnb, "Gaussian")
    comm_pred(neigh, "k-nearest")
    comm_pred(ada, "AdaBoost")


if __name__ == "__main__":
    main()
