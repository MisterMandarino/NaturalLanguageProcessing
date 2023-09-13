**This file is not mandatory**
But if you want, here your can add your comments or anything that you want to share with us
regarding the exercise.

Lab Exercise: Language Model Evaluation

Write your own implementation of the Stupid backoff algorithm. Train it and compare the perplexity with the one provided by NLKT. The dataset that you have to use is the Shakespeare Macbeth.

Stupid Backoff algorithm (use ⍺=0.4): https://aclanthology.org/D07-1090.pdf

NLTK (StupidBackoff): https://www.nltk.org/api/nltk.lm.html

Suggestion: adapt the compute_ppl function to compute the perplexity of your model. The PPL has to be computed on the whole corpus and not at sentence level.
