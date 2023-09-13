# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":

    train=False   ## SET TO TRUE FOR TRAIN THE MODEL

    # Load data
    train_raw, dev_raw, test_raw = get_data()

    # Compute the vocabulary 
    corpus = train_raw + dev_raw + test_raw
    words = sum([x['utterance'].split() for x in train_raw], [])  # number of words
    slots = get_slots_n(corpus) # number of unique slots
    intents = get_intents_n(corpus)  # number of unique intents
    lang = Lang(words, intents, slots, cutoff=0)

    # Instantiate the Bert Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = get_max_token_len(corpus, tokenizer)

    # Create the datasets
    train_dataset = IntentDataset(train_raw, lang, tokenizer, max_len)
    dev_dataset = IntentDataset(dev_raw, lang, tokenizer, max_len)
    test_dataset = IntentDataset(test_raw, lang, tokenizer, max_len)

    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    # criterion for the loss computation
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    if train:
        # Initialize the hyperparameters
        runs = 5  # number of runs
        lr = 0.0001 # learning rate

        slot_f1s, intent_acc = [], []
        best_score = 0
        for x in range(0, runs):
            # Instantiate the model
            model = IntentBERT(out_slot, out_int).to(device)

            # Instantiate the optimizer
            optimizer = optim.Adam(model.parameters(), lr=lr)

            n_epochs = 10 # number of epochs each run
            patience = 3 # patience for early stopping
            best_f1 = 0
            for x in tqdm(range(1,n_epochs)):
                loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)

                if x % 5 == 0:
                    results_dev, _, _ = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                    f1 = results_dev['total']['f']

                    if f1 > best_f1:
                        best_f1 = f1
                        patience = 3
                    else:
                        patience -= 1
                    if patience <= 0: # Early stopping with patience
                        break # Not nice but it keeps the code clean
            results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
            print('intent acc: ', intent_test['accuracy'], '\tslots f1: ', results_test['total']['f'])
            intent_acc.append(intent_test['accuracy'])
            slot_f1s.append(results_test['total']['f'])
            if (intent_test['accuracy'] + results_test['total']['f']) > best_score:
                best_score = intent_test['accuracy'] + results_test['total']['f']
                torch.save(model.state_dict(), __file__ + "\\..\\bin\\best_bert.pt")
        slot_f1s = np.asarray(slot_f1s)
        intent_acc = np.asarray(intent_acc)
        print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
        print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(intent_acc.std(), 3))
    else:
        # Instantiate the model
        model = IntentBERT(out_slot, out_int).to(device)

        # load the model's parameters
        model_params = torch.load(__file__ + "\\..\\bin\\best_bert.pt")
        model.load_state_dict(model_params)

        # evaluate the model
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        print('Slot F1: ', results_test['total']['f'])
        print('Intent Accuracy:', intent_test['accuracy'])
