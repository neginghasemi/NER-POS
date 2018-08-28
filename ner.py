import nltk
from nltk import classify
from collections import Counter


real = []
predicted = []


def exact_match():
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    boundary = 0
    matched = False
    both = False
    for n in range(0, len(real)):
        if real[n] == 'O':
            if matched:
                tp += 1
                matched = False
                both = False
            elif not matched and boundary > 0 and both:
                fp += 1
                fn += 1
                both = False
            elif not matched and boundary > 0 and not both:
                fn += 1

            if predicted[n] == real[n]:
                tn += 1
            else:
                fp += 1
            boundary = 0
        else:
            if boundary == 0 and predicted[n] == real[n]:
                matched = True
                both = True
            elif boundary > 0 and predicted[n] == real[n] and matched:
                matched = True
                both = True
            else:
                matched = False
            boundary += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Exact Matching Precision: ", precision)
    print("Exact Matching Recall: ", recall)
    print("TP, TN, FP, FN: ", tp, tn, fp, fn)


def boundary_match():
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    boundary = 0
    last_label = 'N'
    matched = False
    both = False
    for n in range(0, len(real)):
        if real[n] == 'O':
            if matched:
                tp += 1
                matched = False
                both = False
            elif not matched and boundary > 0 and both:
                fp += 1
                fn += 1
                both = False
            elif not matched and boundary > 0 and not both:
                fn += 1

            if predicted[n] == real[n]:
                tn += 1
            else:
                fp += 1
            boundary = 0
        else:
            if boundary == 0 and predicted[n] != 'O':
                matched = True
                both = True
                last_label = predicted[n]
            elif boundary > 0 and predicted[n] == last_label and matched:
                both = True
                matched = True
            else:
                matched = False
            boundary += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Boundary Matching Precision: ", precision)
    print("Boundary Matching Recall: ", recall)
    print("TP, TN, FP, FN: ", tp, tn, fp, fn)


def type_match():
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    boundary = 0
    matched = False
    for n in range(0, len(real)):
        if real[n] == 'O':
            if matched:
                tp += 1
                matched = False
            elif not matched and boundary > 0:
                fn += 1
            if predicted[n] == real[n]:
                tn += 1
            else:
                fp += 1
            boundary = 0
        else:
            if boundary == 0 and predicted[n] == real[n]:
                matched = True
            elif boundary > 0 and predicted[n] == real[n]:
                matched = True
            elif predicted[n] != real[n] and not matched:
                matched = False
            boundary += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Type Matching Precision: ", precision)
    print("Type Matching Recall: ", recall)
    print("TP, TN, FP, FN: ", tp, tn, fp, fn)


if __name__ == '__main__':

    with open('./dataset/NERtr.txt', 'r') as fileTrain:
        train_file = fileTrain.read().split('\n')
        fileTrain.close()

    with open('./dataset/NERte.txt', 'r') as fileTest:
        test_file = fileTest.read().split('\n')
        fileTest.close()

    current_sentence = []
    train = []
    for i in range(0, len(train_file) - 2):
        current_line = train_file[i]
        word_ne = current_line.split('\t')
        if word_ne[0] == '':
            pos_tags = nltk.pos_tag(list(zip(*current_sentence))[0])
            for cw in range(0, len(pos_tags)):
                current_word = current_sentence[cw][0]
                features = dict()
                capital = 0
                if current_word[0].isupper():
                    capital = 1

                new_entry = {"word": current_word, "pos": pos_tags[cw], "isCapital": capital, "lowercased_word": current_word.lower()}
                features.update(new_entry)
                train.append((features, current_sentence[cw][1]))
            current_sentence = []
        else:
            current_sentence.append((word_ne[0], word_ne[1]))

    current_sentence = []
    test = []
    for i in range(0, len(test_file)):
        current_line = test_file[i]
        word_ne = current_line.split('\t')
        if word_ne[0] == '':
            pos_tags = nltk.pos_tag(list(zip(*current_sentence))[0])
            for cw in range(0, len(pos_tags)):
                current_word = current_sentence[cw][0]
                features = dict()
                capital = 0
                if current_word[0].isupper():
                    capital = 1

                new_entry = {"word": current_word, "pos": pos_tags[cw], "isCapital": capital, "lowercased_word": current_word.lower()}
                features.update(new_entry)
                test.append((features, current_sentence[cw][1]))
            current_sentence = []
        else:
            current_sentence.append((word_ne[0], word_ne[1]))

    classifier = nltk.MaxentClassifier.train(train, max_iter=10)
    print("Accuracy: ", classify.accuracy(classifier, test))

    real = list(zip(*test))[1]
    predicted = classifier.classify_many(list(zip(*test))[0])

    exact_match()
    boundary_match()
    type_match()
