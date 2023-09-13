# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    
    SUB_TRAIN = True   ## SET TO TRUE FOR TRAIN THE SUBJECTIVITY MODEL
    POL_TRAIN = False   ## SET TO TRUE FOR TRAIN THE POLARITY MODEL

    ## Instantiate the BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ## Get the data for the subjectivity task
    sub_dataset = get_subjectivity_data()

    ## Get the maximum length of the sentences in the subjectivity dataset
    max_len = get_tokenizer_max_length(dataset=sub_dataset, tokenizer=tokenizer)

    ## Instantiate the language mapper
    lang = Lang(subjectivity.words(), cutoff=0)

    ## Using the Binary-CrossEntropy Loss
    subjectivity_loss = nn.BCELoss()
    accuracy, sub_model = subjectivity_task(n_splits=10,
                                dataset=sub_dataset, 
                                tokenizer=tokenizer,
                                max_len=max_len, 
                                lr=0.001, 
                                n_epochs=5, 
                                criterion=subjectivity_loss,
                                train=SUB_TRAIN)
    print('Mean Accuracy (Subjectivity Task): ', accuracy)

    ## Get the data for the polarity task
    pol_dataset = get_polarity_data()

    ## Remove the objective sentences from the polarity dataset
    clean_pol_dataset = remove_objective_sents(pol_dataset, tokenizer, sub_model)

    polarity_loss = nn.BCELoss()

    ## Polarity task using the original dataset
    accuracy = polarity_task(n_splits=10,
                             dataset=pol_dataset,
                             tokenizer=tokenizer,
                             max_len=512,
                             lr=0.00001,
                             n_epochs=2,
                             criterion=polarity_loss,
                             train=POL_TRAIN,
                             raw_data=True)
    print('Mean Accuracy: (Polarity Task)', accuracy)

    # polarity task using the dataset with the removal of the objective sentences
    accuracy = polarity_task(n_splits=10,
                             dataset=clean_pol_dataset,
                             tokenizer=tokenizer,
                             max_len=512,
                             lr=0.00001,
                             n_epochs=2,
                             criterion=polarity_loss,
                             train=POL_TRAIN,
                             raw_data=False)
    print('Mean Accuracy: (Subjectivity + Polarity Task )', accuracy)
