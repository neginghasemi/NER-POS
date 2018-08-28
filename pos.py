from nltk.tag import hmm
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.probability import LidstoneProbDist, MLEProbDist, SimpleGoodTuringProbDist, KneserNeyProbDist, WittenBellProbDist

if __name__ == '__main__':

    with open('./dataset/POStrutf.txt', 'r') as fileTrain:
        train_file = fileTrain.read().split('\n')
        fileTrain.close()

    with open('./dataset/POSteutf.txt', 'r') as fileTest:
        test_file = fileTest.read().split('\n')
        fileTest.close()

    current_sentence = []
    words = set()
    tags = set()
    train = []
    for i in range(0, len(train_file)-1):
        current_line = train_file[i]
        word_tag = current_line.split('\t\t')
        words.add(word_tag[0])
        tags.add(word_tag[1])
        current_sentence.append((word_tag[0], word_tag[1]))
        if word_tag[0] == '.':
            train.append(current_sentence)
            current_sentence = []

    current_sentence = []
    test = []
    for i in range(0, len(test_file)-1):
        current_line = test_file[i]
        word_tag = current_line.split('\t\t')
        words.add(word_tag[0])
        tags.add(word_tag[1])
        current_sentence.append((word_tag[0], word_tag[1]))
        if word_tag[0] == '.':
            test.append(current_sentence)
            current_sentence = []

    tags = list(tags)
    words = list(words)
    trainer = hmm.HiddenMarkovModelTrainer(tags, words)
    # tagger = trainer.train_supervised(train, estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins))
    # tagger = trainer.train_supervised(train, estimator=lambda fd, bins: MLEProbDist(fd))
    tagger = trainer.train_supervised(train, estimator=lambda fd, bins: SimpleGoodTuringProbDist(fd, bins))
    # tagger = trainer.train_supervised(train, estimator=lambda fd, bins: WittenBellProbDist(fd, bins))
    # tagger = trainer.train_supervised(train, estimator=lambda fd, bins: KneserNeyProbDist(fd, bins))

    print("here")
    predicted = []
    real = []
    for i in range(0, len(test)-1):
        current = list(zip(*test[i]))
        tagged = tagger.tag(list(current[0]))
        current_tags = list(list(zip(*tagged))[1])
        predicted += current_tags
        real += list(current[1])

    print(tags)
    confusion = confusion_matrix(predicted, real, labels=tags)
    row_sums = confusion.sum(axis=1)
    average_confusion_matrix = confusion.astype('float') / row_sums[:, np.newaxis]
    nans = np.isnan(average_confusion_matrix)
    average_confusion_matrix[nans] = 0
    np.savetxt("Confusion.csv", confusion, fmt='%i', delimiter=",")
    np.savetxt("Average_Confusion.csv", average_confusion_matrix, fmt='%10.3f', delimiter=",")
    print(accuracy_score(predicted, real))

    with open('in.txt', 'r') as fileInput:
        input_file = fileInput.read().split('\n')
        fileInput.close()

    output_file = open('out.txt', 'w')
    current_sentence = []
    input_test = []
    for i in range(0, len(input_file)):
        sentences = input_file[i].split('. ')
        for j in range(0, len(sentences)-1):
            current_sentence = sentences[j]
            tag_input = tagger.tag(current_sentence.split())
            for t in tag_input:
                output_file.write(t[0] + "\t" + t[1] + "\n")
    output_file.close()
