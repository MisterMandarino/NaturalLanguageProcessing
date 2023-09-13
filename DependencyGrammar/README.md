**This file is not mandatory**
But if you want, here your can add your comments or anything that you want to share with us
regarding the exercise.

Lab Exercise

    -Parse 100 last sentences from dependency treebank using spacy and stanza
        -are the depedency tags of spacy the same of stanza?
    -Evaluate against the ground truth the parses using DependencyEvaluator
        -print LAS and UAS for each parser

(BUT! To evaluate the parsers, the sentences parsed by spacy and stanza have to be DependencyGraph objects. To do this , you have to covert the output of the spacy/stanza to ConLL formant, from this format extract the columns following the Malt-Tab format and finally convert the resulting string into a DependecyGraph. Lucky for you there is a library that gets the job done. You have to install the library spacy_conll and use and adapt to your needs the code that you can find below.)