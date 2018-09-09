# NER-POS
 Implementation of a system for Named-Entity Recognition, POS Tagging.

## POS
 Implementation of a system for POS Tagging.
 
 At first, we should split our train and test data into sentences. Then each sentences words to tuples (word, pos). 
 
 In training step, I used Hidden Markov Model from nltk. We can use test tuples to evaluate our tagger.

## NER
 Implementation of a system for Named-Entity Recognition.
 
 For each word, I extracted four features. Then I used a MaxentClassifier from nltk. 
 
 Features I used:
  - word: The target word.
  - pos: The pos tag of target word.
  - isCapital: 1 if the target word starts with capital letter. otherwise 0.
  - lowercased_word: The target word in lowercase.
  
 Evaluation:
  - Exact Match: detected entity has correct type and boundary
  - Boundry Match: detected entity has correct type but wrong boundary
  - Type Match: detected entity has correct boundary but wrong type

## Uses
 - [Numpy](http://www.numpy.org/) version 1.14.5
 - [Sklearn](http://scikit-learn.org/stable/)
 - [Nltk](https://www.nltk.org) version 3.3

## Run
 - `pos.py` will implement and evaluate a hmm based pos tagger
 - `ner.py` will implement and evaluate a maxEnt based named-entity recognition

## POS Output
 - Average Confusion Matrix as `Average_Confusion.csv` file
 - Test Accurecy

## NER Output
 - Precision
 - Recall
