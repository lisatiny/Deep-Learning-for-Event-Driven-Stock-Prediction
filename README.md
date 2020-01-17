# Deep-Learning-for-Event-Driven-Stock-Prediction

I refered the paper named ["Deep Learning for Event Driven Stock Prediction"](https://pdfs.semanticscholar.org/fa8c/f4efda3b31c4fc43293d83e21ac4ce1d2d8f.pdf)

The authors used event extraction technique name ReVerb [Fader et al., 2011] and ZPar [Zhang and Clark, 2011]. But I used Standford Open Information Extraction Package named ["openie"](https://nlp.stanford.edu/software/openie.html) for simplification.

openie used to return expression which has multiple words(i.e idioms) such as "is related to". If we convert those expressions into one single index then, there can be "out of vocabulary" issue. So I splitted it ['is','related','to'],put them to embedding matrix and applied average operation. i.e) embedded_matrix.mean(axis=1). This can be replaced and developed by using phase embedding such as phase2vec.
