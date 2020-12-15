# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Sven Bartz
# Created Date: 2020-10-01 18:54:00
# =============================================================================
"""This module has been built to identify User Entrepreneurial Characteristics
   in Crowdfunding Campaigns (e.g. kickstarter.com)"""
# =============================================================================
# requires the following modules:
#   pandas, scikit-learn, matplotlib, spacy, nltk, xlrd, openpyxl

# optional: pycontractions
#           (which outperforms module "contractions" by over 10% accuracy
#            but requires JavaVM-JDK8 an Visual C++ 14.0 on the operating system)
# ===============================
import re
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import spacy
from spacy.cli import download
spacy.cli.download('en_core_web_sm')
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


"""=========================== PREPROCESSING =============================="""

def expand_contractions(text):
    
    if contraction_model_selected == 'BASIC':
        contractions_df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                   'contractions.csv'))
        for i in range(len(contractions_df)):
            text = text.replace(contractions_df['contracted'].iloc[i],
                                contractions_df['expanded'].iloc[i])
        return text

    if contraction_model_selected == 'ADVANCED':
        from pycontractions import Contractions
        contractions_model = Contractions(api_key='glove-twitter-200')
        text = list(contractions_model.expand_texts([text]))[0]
        return text
    
    
def capital_letters_to_lowercase(text):
    text = text.lower()
    return text


def convert_accented_characters(text):
    import unicodedata
    text = unicodedata.normalize('NFKD',
                                 text).encode('ascii',
                                              'ignore').decode('utf-8',
                                                               'ignore')
    return text


def remove_special_characters(text):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


def remove_numbers(text):
    pattern = r'\d+'
    text = re.sub(pattern, '', text)
    return text


def remove_stopwords(text):
    from collections import Counter
    stopwords_df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                            'stopwords.csv'))
    stopwords_list = stopwords_df['stopwords'].values.tolist()
    stopwords = stopwords_list
    stopwords_dict = Counter(stopwords)
    text = ' '.join([word for word in text.split()
                     if word not in stopwords_dict])
    return text


def apply_lemmatizing(text):
    nlp = spacy.load('en_core_web_sm')
    text = nlp(text)
    text = ' '.join([token.lemma_
                     if token.lemma_ != '-PRON-'
                     else token.text for token in text])
    return text


def apply_stemming(text):
    stemmer = PorterStemmer()
    token_words = word_tokenize(text)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(stemmer.stem(word))
        stem_sentence.append(' ')
    return ''.join(stem_sentence)


# ========================= Main Preprocess Function ========================= 

def text_preprocessing(received_text, func1, func2, func3, func4,
                       func5, func6, func7, func8):
    processed_text = received_text
    if func1:
        processed_text = expand_contractions(processed_text)
    if func2:
        processed_text = capital_letters_to_lowercase(processed_text)
    if func3:
        processed_text = convert_accented_characters(processed_text)
    if func4:
        processed_text = remove_special_characters(processed_text)
    if func5:
        processed_text = remove_numbers(processed_text)
    if func6:
        processed_text = remove_stopwords(processed_text)
    if func7:
        processed_text = apply_lemmatizing(processed_text)
    if func8:
        processed_text = apply_stemming(processed_text)
    return processed_text


# ====================== User Dialog - Preprocess Setup ======================

preprocess_functions = ['Expand Contractions', 'Capital Letters to Lowercase',
                        'Remove Accented Characters',
                        'Remove Special Characters', 'Remove Numbers',
                        'Remove Stopwords', 'Apply Stemming',
                        'Apply Lemmatizing']
preprocess_default = [True, True, True, True, True, True, False, False]
preprocess_answers = preprocess_default


classifiers = ['Complement Naive Bayes', 'Multinomial Naive Bayes',
               'Linear Model with Stochastic Gradient Descent Learning',
               'Linear Support Vector Classifier']
classifier_default = ('Complement Naive Bayes')
selected_classifier = classifier_default


contraction_model = ['BASIC', 'ADVANCED']
contraction_model_description = ('    (runs a simple compare and replace function)', ' (runs module "pycontractions" which outperforms the BASIC function' + '\n' +
                                 '                     by over 10% but it requires JavaVirtualMachines-OpenJDK8' + '\n'
                                 '                     and Visual C++ 14.0, it takes about 4 minutes for a text of 200 sentence)')
contraction_model_default = ('BASIC')
contraction_model_selected = contraction_model_default

cleaned_text = ()
restart_question_low = ('y')

print('\n\n' + '________________ User Entrepreneurial Characteristic Identifier ________________' + '\n\n' +
      'This program is written to identify user entrepreneurial characteristics in crowdfunding campaign descriptions.' + '\n' +
      'Based on a handmade training classification set, containing over 17039 training elements, a machine learning classifier (average accuracy 70%)' + '\n' +
      'assigns  each sentence of a campaign description to one out of 10 characteristics whereas the characteristic are:' + '\n' +
      '      - Characteristic 1:  Address Target Group             (user entrepreneurial-major)' + '\n' +
      '           because User Entrepreneur has extensive experience with a product, he knows which feature is needed and can therefor very explicitly address the target group' + '\n' +
      '      - Characteristic 2:  Product History                 (user entrepreneurial-major)' + '\n' +
      '           because User Entrepreneur has extensive experience with a product, he has a history to talk about' + '\n' +
      '      - Characteristic 3:  Usage of Patents                (user entrepreneurial-minor)' + '\n' +
      '           because User Entrepreneur tends to save his intellectual property' + '\n' +
      '      - Characteristic 4:  Existing Prototypes             (user entrepreneurial-minor)' + '\n' +
      '           because User Entrepreneur is already using it by his own before commercializing' + '\n' +
      '      - Characteristic 5:  Community, Feedback             (user entrepreneurial-major)' + '\n' +
      '           because User Entrepreneur was supported by others to bring the innovation to life' + '\n' +
      '      - Characteristic 6:  Future Vision                   (general)' + '\n' +
      '      - Characteristic 7:  Direct Appeal                   (general)' + '\n' +
      '      - Characteristic 8:  Kickstarter related             (general)' + '\n' +
      '      - Characteristic 9:  R&D and Science                 (general)' + '\n' +
      '      - Characteristic 10: General Description             (general)' + '\n\n' +
      'Based on literature(please see paper xy) the first five characteristics are typical for user entrepreneurs.' + '\n' +
      'After selecting text preprocesses and choosing the classifier the predicted results where graphically compared' + '\n' +
      'to a former analysis of the characteristic distribution of 50 designated Lead User campaigns.' + '\n' +
      'Finally the user can choose whether to print out the cleaned text, the classified sentences or to examine another text.' + '\n\n' +
      'Due to privacy and data protection reasons this program does not crawl any website but works with text which is pasted by the user.' + '\n' +
      'This has the advantage that it is robust against any kind of possible website-structure changes.' + '\n' +
      'Furthermore it is applicable to a wide variety of English written texts, unrestricted from any platforms.')

while restart_question_low == 'y':
    while True:
        query1 = input('Please paste the text to be examined and push enter:'+'\n\n' + '--->' + '\n\n') or 'missing user input'

        while True:
            if query1 == 'missing user input':
                print('Please paste text and push enter:'+'\n\n'+'--->')
                query1 = input('') or 'missing user input'
            else:
                break

        if query1 != 'missing user input':
            original_text = query1
            print('\n\n\n\n' + '_______________________________ Settings _______________________________________' + '\n\n' +
                  'Text Preprocessing: ' + '\n')

            for i in range(len(preprocess_functions)):
                print('  [{}]  {}{}'.format('x' if preprocess_answers[i] is True else ' ',
                                            preprocess_functions[i],
                                            ' (BASIC)' if i == 0 and contraction_model_selected == 'BASIC' else
                                            ' (ADVANCED)' if i == 0 and contraction_model_selected == 'ADVANCED' else ''))

            print('\n'+'Classifier: '+'\n')

            for i in range(len(classifiers)):
                print('  [{}]  {}'.format('x' if selected_classifier == classifiers[i] else ' ', classifiers[i]))

            while True:                                                        # loop till user enters valid answer
                query2 = input('Do you want to change Text Preprocessing settings?' +
                               '\n' + 'Please answer with [y]es or [n]o: ') or 'empty'
                query2_low = query2[0].lower()
                if query2_low in ['y', 'n']:
                    break

            if query2_low == 'y':
                for i in range(8):
                    while True:
                        query3 = input('  Should the function ' + preprocess_functions[i] + ' be applied?' + '\n' + '  Please answer with [y]es or [n]o: ') or 'empty'
                        query3_low = query3[0].lower()
                        if query3_low in ['y', 'n']:
                            break

                    if query3_low == 'y':
                        preprocess_answers[i] = True
                    if query3_low == 'n':
                        preprocess_answers[i] = False

                    if preprocess_functions[i] == preprocess_functions[0] and query3_low == 'y':
                        print('')
                        while True:
                            for i in range(len(contraction_model)):
                                print('     [{}] = {}{}'.format(i+1, contraction_model[i], contraction_model_description[i]))

                            query4 = input('     Please choose [1] or [2]: ') or 'empty'
                            query4_low = query4[0].lower()
                            if query4_low == '1':
                                contraction_model_selected = 'BASIC'
                            if query4_low == '2':
                                contraction_model_selected = 'ADVANCED'
                            if query4_low in ['1', '2']:
                                break
            if contraction_model_selected == 'ADVANCED':
                print('\n' + 'This process step may take up to 4 minutes due to selecting ADVANCED mode in Expand Contractions!')


# =========================== Apply Preprocessing ============================

        delimited_text = original_text.replace('\n', ' yyy ').\
            replace('\r', ' yyy ').replace('.', ' yyy ').replace('!', ' yyy ').\
            replace('?', ' yyy ').replace('   ', ' ').replace('  ', ' ')
        cleaned_text = text_preprocessing(delimited_text, preprocess_answers[0],
                                          preprocess_answers[1], preprocess_answers[2],
                                          preprocess_answers[3], preprocess_answers[4],
                                          preprocess_answers[5], preprocess_answers[6],
                                          preprocess_answers[7])

        if cleaned_text == '':
            print('After appling preprocess execution, no text is left over')
        else:
            break

# ============= Preparation of Cleaned Text for Classification ===============

    text_to_list = re.split('yyy', cleaned_text)
    cleaned_list = []
    for i in range(len(text_to_list)):
        a = text_to_list[i].lstrip(' ').rstrip(' ')
        cleaned_list.append(a)
    while '' in cleaned_list:
        cleaned_list.remove('')
    data_predict = cleaned_list


    """===================== CLASSIFICATION ================================="""

    characteristic = ['C1-UE', 'C2-UE', 'C3-UE', 'C4-UE', 'C5-UE',
                      'C6   ', 'C7   ', 'C8   ', 'C9   ', 'C10  ']
    X_train = ()
    X_test = ()
    
    pd.set_option('max_colwidth', 150)
    data_train = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                          'classifier_training_data.csv'))


# ============================= Count Vectorizer =============================

    count_vect = CountVectorizer(lowercase=None)

    X_train_count_vect = count_vect.fit_transform(data_train['content'])
    X_predict_count_vect = count_vect.transform(data_predict)

    X_train = X_train_count_vect
    X_test = X_predict_count_vect


# ======================== TF(Term Frequency) & IDF(Inverse Document Frequency)

    tfidf_transformer = TfidfTransformer()

    X_train_tfidf = tfidf_transformer.fit_transform(X_train_count_vect)
    X_test_tfidf = tfidf_transformer.transform(X_predict_count_vect)


# ==================== User Dialog - Classification Setup ====================

    while True:                                                                # loop till user enters valid answer
        query5 = input('\n'+'Do you want to use TF-IDF?' + '\n' +
                       'Please answer with [y]es or [n]o: ') or 'empty'
        query5_low = query5[0].lower()
        if query5_low in ['y', 'n']:
            break
    if query5_low == 'y':                                                      # interrogation of applied classifier
        X_train = X_train_tfidf
        X_test = X_test_tfidf
        
    while True:
        query5 = input('\n'+'Do you want to change the Classifier?' + '\n' +
                       'Please answer with [y]es or [n]o: ') or 'empty'
        query5_low = query5[0].lower()
        if query5_low in ['y', 'n']:
            break
    if query5_low == 'y':
        while True:
            print('\n\n'+'Available Classifiers:' + '\n')

            for i in range(len(classifiers)):
                print('  [{}]  {}'.format(i+1, classifiers[i]))
            query6 = input('\n'+'  Please choose [1], [2], [3], [4]: ') or 'empty'
            query6_low = query6[0].lower()
            if query6_low in ['1', '2', '3', '4']:
                break

        if query6_low == '1':
            selected_classifier = classifiers[0]
        if query6_low == '2':
            selected_classifier = classifiers[1]
        if query6_low == '3':
            selected_classifier = classifiers[2]
        if query6_low == '4':
            selected_classifier = classifiers[3]


# ============================ Setup Information =============================

    print('\n\n\n\n\n\n\n' + '____________________ Training- & Predict-Data Information ______________________' +
          '\n\n' + 'The Training-Data contains ' + str(X_train_count_vect.shape[0]) +
          ' training sentences/elements with a total of ' + str(X_train_count_vect.shape[1]) + ' unique words. ' + '\n'
          'The text to be examined contains ' + str(X_predict_count_vect.shape[0]) + ' sentences/elements.' + '\n')

    if not any(preprocess_answers) is True:                                    # show which text preprocess is applied
        print('Applied Text Preprocessing function/s:' + '\n\n' +
              '       - No Text Preprocessing was applied')
    if not any(preprocess_answers) is False:
        print('Applied Text Preprocessing function/s:' + '\n')
    for i in range(8):
        if preprocess_answers[i] is True:
            print('       - '+preprocess_functions[i])

    print('\n' + 'Applied Classifier:' + '\n\n' +
          '       - '+selected_classifier)


# ==================== Complement Naive Bayes Classifier ==================== was choosen as the best classifier according to results of analysis in thesis

    def CNB(train_count_vect, train_characteristic, predict_count_vect):
        from sklearn.naive_bayes import ComplementNB
        CNB_clf = ComplementNB(alpha=1.947826087,
                               fit_prior=True,
                               norm=True) 

        CNB_clf.fit(train_count_vect, train_characteristic)
        clf_prediction_CNB = CNB_clf.predict(predict_count_vect)
        return clf_prediction_CNB


# ==================== Multinomial Naive Bayes Classifier ==================== second best classifier according to results of analysis in thesis

    def MNB(train_count_vect, train_characteristic, predict_count_vect):
        from sklearn.naive_bayes import MultinomialNB
        MNB_clf = MultinomialNB()

        MNB_clf.fit(train_count_vect, train_characteristic)
        clf_prediction_MNB = MNB_clf.predict(predict_count_vect)
        return clf_prediction_MNB


# ===================== Linear classifiers With SGD Training ================= third best classifier according to results of analysis in thesis

    def SGD(train_count_vect, train_characteristic, predict_count_vect):
        from sklearn.linear_model import SGDClassifier
        SVM_clf = SGDClassifier()

        train_count_vect_dense = train_count_vect.todense()
        predict_count_vect_dense = predict_count_vect.todense()

        SVM_clf.fit(train_count_vect_dense, train_characteristic)
        clf_prediction_SVM = SVM_clf.predict(predict_count_vect_dense)
        return clf_prediction_SVM


# ===================== Linear Support Vector Classifier ===================== fourth best classifier according to results of analysis in thesis

    def LinearSVC(train_count_vect, train_characteristic, predict_count_vect):
        from sklearn.svm import LinearSVC
        LSVC_clf = LinearSVC(dual=True, max_iter=2000)

        LSVC_clf.fit(train_count_vect, train_characteristic)
        clf_prediction_LSVC = LSVC_clf.predict(predict_count_vect)
        return clf_prediction_LSVC


    """=========================== Results / Analysis ======================"""

    sentences_count = len(data_predict)
    if selected_classifier == 'Complement Naive Bayes':
        clf_prediction = CNB(X_train, data_train['characteristic'], X_test)
    if selected_classifier == 'Multinomial Naive Bayes':
        clf_prediction = MNB(X_train, data_train['characteristic'], X_test)
    if selected_classifier == 'Linear Model with Stochastic Gradient Descent Learning':
        clf_prediction = SGD(X_train, data_train['characteristic'], X_test)
    if selected_classifier == 'Linear Support Vector Classifier':
        clf_prediction = LinearSVC(X_train, data_train['characteristic'], X_test)


# ====================== Characteristic Occurrence Count ======================

    predicted_characteristic = [None]*len(characteristic)
    for i in range(len(characteristic)):
        predicted_characteristic[i] = np.count_nonzero(clf_prediction == 'C{}'.format(i+1))


# =========================== Absolute Occurrence =============================

    predicted_characteristic_absolute = [None]*len(characteristic)
    for i in range(len(characteristic)):
        predicted_characteristic_absolute[i] = (predicted_characteristic[i] / sentences_count) * 100


# ============= UE Characteristic Distribution, inserted manually ============

    UE_characteristic_distribution = [8, 22, 1, 2, 5, 4, 5, 3, 5, 39]

    print('\n\n\n\n\n\n\n' + '______________________________ Results / Analysis ______________________________' + '\n\n' +
          'The examined text contained {} sentence/s, classified as following:'.format(sentences_count) + '\n')

    characteristic_keywords = ['Address Target Group', ' Product History     ',
                                'Usage of Patents/IP ', 'Existing Prototypes ',
                                'Community, Feedback ', 'Future Vision       ',
                                'Direct Appeal       ', 'Kickstarter related ',
                                'R&D and Science     ', 'General Description ']

    for i in range(len(characteristic)):
        print('{}  ({}):   {} sentence/s which is   {:.2f}%     (average UE: {}%)'.format(characteristic[i],
                                                                                          characteristic_keywords[i],
                                                                                          predicted_characteristic[i],
                                                                                          predicted_characteristic_absolute[i],
                                                                                          UE_characteristic_distribution[i]))


# ================================== Plot ====================================

    predicted_characteristic_distribution = [None]*len(characteristic)
    for i in range(len(characteristic)):
        predicted_characteristic_distribution[i] = round(predicted_characteristic_absolute[i], 2)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ind = np.arange(10)
    width = 0.4


# ================================== Bars ====================================

    rects1 = ax.bar(ind, UE_characteristic_distribution, width,
                    color='darkblue')
    rects2 = ax.bar(ind+width, predicted_characteristic_distribution, width,
                    color='darkorange')


# ============================ Axes and Labels ===============================
    
    characteristic_keywords_clean = [x.strip(' ') for x in characteristic_keywords]
    ax.set_xlim(-width, len(ind))
    ax.set_ylim(0, 110)
    ax.set_ylabel('Absolute Share', fontsize=14)
    ax.set_title('Comparing: User Entrepreneurial- vs. Predicted Characteristic', fontsize=16)
    xTickMarks = [characteristic_keywords_clean[i] for i in range(10)]
    yTickMarks =['', '20%', '40%', '60%', '80%', '100%']
    ax.set_xticks(ind+width/2)
    xtickNames = ax.set_xticklabels(xTickMarks)
    ytickNames = ax.set_yticklabels(yTickMarks)
    plt.setp(xtickNames, rotation=75, fontsize=14, va='top', ha='right')
    plt.setp(ytickNames, fontsize=14)


# ================================== Bars ====================================

    ax.legend((rects1[0], rects2[0]), ('User Entrepreneurial Average', 'Examined Text'), fontsize=14)


# =========================== Values over Bars ===============================

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                    '%d' % int(height)+'%',
                    ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)   
    print('\n\n\n'+'-> please see the plot <-'+'\n\n')
    plt.show()


# ============== characteristics exceeding UE-characteristics =================

    exceed = bool
    for i in range(len(characteristic)-5):
        if predicted_characteristic_absolute[i] > UE_characteristic_distribution[i]:
            exceed = True
    if exceed is False:
        print('\n' + 'The examined text exhibits no pronounced User Entrepreneurial Characteristic.')
    if exceed is True:
        print('\n' + 'The examined text exhibits a pronounced User Entrepreneurial Characteristic in:' + '\n')
        for i in range(len(characteristic)-5):
            if predicted_characteristic_absolute[i] > UE_characteristic_distribution[i]:
                print('  {} - {}'.format(characteristic[i], characteristic_keywords[i]))


# ================================= Output ===================================

    print('\n\n\n\n\n\n\n'+'__________________________________ Printouts ___________________________________')

    prediction_list = clf_prediction.tolist()
    results = pd.DataFrame({'predicted characteristic': prediction_list,
                            'content': cleaned_list})

    while True:
        query7 = input('Do you want to print out the cleaned text?' +
                       '\n' + 'Please answer with [y]es or [n]o: ') or 'empty'
        query7_low = query7[0].lower()
        if query7_low in ['y', 'n']:
            break
    if query7_low == 'y':
        print('\n' + '--------------------------------- Cleaned Text ---------------------------------' + '\n\n' +
              '. '.join(cleaned_list)+'\n\n' +
              '------------------------------- Cleaned Text End -------------------------------')

    while True:
        print('\n\n\n\n\n\n' + 'Do you want to print out the classification results for each sentence:' + '\n\n' +
              '    [n]  no print out' + '\n\n' +
              '    [1]  csv in console' + '\n' +
              '    [2]  xlsx in same python file directory' + '\n' +
              '    [3]  do 2 and 3' + '   ') or 'empty'

        query8 = input('Please choose [n] or [1], [2], [3]: ')
        query8_low = query8[0].lower()
        if query8_low in ['n', '1', '2', '3']:
            break

    if query8_low == '1':
        print('\n' + '--------------------------- Predicted Characteristic ---------------------------' + '\n\n' +
              results.to_csv(index=False) + '\n' +
              '------------------------- Predicted Characteristic End -------------------------' + '\n\n')

    if query8_low == '2':
        from datetime import datetime
        dateTimeObj = datetime.now()
        date = dateTimeObj.strftime('%Y-%b-%d_%H-%M-%S')
        results = results.to_excel(os.path.join(os.path.dirname(__file__), '{}_predicted_characteristic.xlsx'.format(date)), index=False, header=True)
        print('\n\n' + 'File was saved in: '+os.path.join(os.path.dirname(__file__), '{}_predicted_characteristic.xlsx'.format(date)+'\n\n'))

    if query8_low == '3':
        print('\n' + '--------------------------- Predicted Characteristic ---------------------------' + '\n\n' +
              results.to_csv(index=False) + '\n' +
              '------------------------- Predicted Characteristic End -------------------------')
        from datetime import datetime
        dateTimeObj = datetime.now()
        date = dateTimeObj.strftime('%Y-%b-%d_%H-%M-%S')
        results = results.to_excel(os.path.join(os.path.dirname(__file__), '{}_predicted_characteristic.xlsx'.format(date)), index=False, header=True)
        print('\n\n' + 'File was saved in: '+os.path.join(os.path.dirname(__file__), '{}_predicted_characteristic.xlsx'.format(date))+'\n\n')

    restart_question_low = ('')
    while restart_question_low not in ['y', 'n']:
        restart_question = input('\n\n\n' + 'Do you want to examine another text?'+'\n' + 'Please answer with [y]es or [n]o: ') or 'empty'
        restart_question_low = restart_question[0].lower()

print('\n\n\n\n\n\n'+'=============================== End of Program =================================')
