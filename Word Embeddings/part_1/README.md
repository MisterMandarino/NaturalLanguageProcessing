**This file is not mandatory**
But if you want, here your can add your comments or anything that you want to share with us
regarding the exercise.

Exercise 1 (2 points)

Modify the baseline LM_RNN (the idea is to add a set of improvements and see how these affect the performance). Furthremore, you have to play with the hyperparameters to minimise the PPL and thus print the results achieved with the best configuration. Here are the links to the state-of-the-art papers which uses vanilla RNN paper1, paper2.

    -Replace RNN with LSTM (output the PPL)
    -Add two dropout layers: (output the PPL)
        -one on embeddings,
        -one on the output
    -Replace SGD with AdamW (output the PPL)

Exercise 2 (4 points)

Add the following regularizations described in this paper:

    -Weight Tying
    -Variational Dropout
    -Non-monotonically Triggered AvSGD