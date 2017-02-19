# Text-Classification

In this project i have implemented text classification Naïve Bayes algorithm for 20-NewsGroups dataset.
Overall accuracy, F1 score and Vocalbulary size is displayed when the code is executed.

The following categories are used from 20news­bydate dataset : comp.graphics,
rec.autos, and talk.politics.guns has been used

he following preprocessing steps have been done :
- Special characters and numeric characters have been removed from text document using regular expression
- Word lowercasing is done
- Words with string length less than three is removed
- Stopwords are removed from the text document using NLTK library
- Multinomial Naive bayes algorithm is used for text document classification
