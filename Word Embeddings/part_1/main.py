# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *

if __name__ == "__main__":

    train=False   ## SET TO TRUE FOR TRAIN ALL THE THREE MODELS WITH GRID SEARCH FINE-TUNING
    
    # Dataloader instantiation
    train_loader, dev_loader, test_loader, lang = get_dataloaders()

    #initialize the hyperparameters
    hid_size = 200 # hidden size
    emb_size = 300 # embedding size
    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # gpu device if possible
    vocab_len = len(lang.word2id) # vocabulary length

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    print('RNN Model')

    #instantiate the baseline 
    model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)

    if train: 
        #modify the hyperparameters with a grid search approach     
        print('fine-tuning Baseline')
        momentums = [0.9, 0.95, 0.99]
        nesterov_momentum = [False, True]
        weight_decays = [0.001, 0.0001]
        lr = 0.1 #increasing the learning rate as shown in the paper 'Recurrent neural network based language model'

        best_ppl = float('inf')
        best_momentum = 0
        best_weight_decay = 0
        use_nesterov = False
        for momentum in momentums:
            for wd in weight_decays:
                for nm in nesterov_momentum:

                    model.apply(init_weights) # re-initialize the model weights to make a fair comparison

                    print('-------------------------------------')
                    print(f'Param search:\tUse Nesterov: [{nm}]\t Momentum: [{momentum}]\t Weight decay: [{wd}]')
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=nm)

                    # Evaluate on little epochs due to time resources limitations
                    ppl = evaluate_model(model, optimizer, train_loader, test_loader, dev_loader, criterion_eval, criterion_train, clip, n_epochs=20, patience=3, device=device)

                    # save the best combination
                    if ppl < best_ppl:
                        best_ppl = ppl
                        best_momentum = momentum
                        best_weight_decay = wd
                        use_nesterov = nm
        print(f'Best Parameters:\tUse Nesterov: [{use_nesterov}]\t Momentum: [{best_momentum}]\t Weight decay: [{best_weight_decay}]')
        print(f'Best PPL: {best_ppl}')

        #evaluate the fine-tuned baseline
        model.apply(init_weights)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=best_momentum, weight_decay=best_weight_decay, nesterov=use_nesterov)
        evaluate_model(model, optimizer, train_loader, test_loader, dev_loader, criterion_eval, criterion_train, clip, n_epochs=100, patience=3, device=device)
        torch.save(model.state_dict(), __file__ + "\\..\\bin\\rnn.pt")
    else:
        model_params = torch.load(__file__ + "\\..\\bin\\rnn.pt")
        model.load_state_dict(model_params)
        test_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print('Test ppl: ', test_ppl)

    
    print('LSTM model')
    # Replace the RNN layer with LSTM layer
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)

    if train:
        model.apply(init_weights) #initialize the model weights
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=best_momentum, weight_decay=best_weight_decay, nesterov=use_nesterov)
        evaluate_model(model, optimizer, train_loader, test_loader, dev_loader, criterion_eval, criterion_train, clip, n_epochs=100, patience=3, device=device)
        torch.save(model.state_dict(), __file__ + "\\..\\bin\\lstm.pt")
    else:
        model_params = torch.load(__file__ + "\\..\\bin\\lstm.pt")
        model.load_state_dict(model_params)
        test_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print('Test ppl: ', test_ppl)

    print('LSTM model (with dropout layers)')
    # Add Dropout layers to the LSTM model
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], with_dropout=True, emb_dropout=0.1, out_dropout=0.1).to(device)
    if train:
        model.apply(init_weights) #initialize the model weights
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=best_momentum, weight_decay=best_weight_decay, nesterov=use_nesterov)
        evaluate_model(model, optimizer, train_loader, test_loader, dev_loader, criterion_eval, criterion_train, clip, n_epochs=100, patience=3, device=device)
        torch.save(model.state_dict(), __file__ + "\\..\\bin\\lstm_dropout.pt")
    else:
        model_params = torch.load(__file__ + "\\..\\bin\\lstm_dropout.pt")
        model.load_state_dict(model_params)
        test_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print('Test ppl: ', test_ppl)
    
    print('LSTM model (with AdamW optimizer)')
    # Replace the SGD optimizer with AdamW
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], with_dropout=True, emb_dropout=0.1, out_dropout=0.1).to(device)

    if train:
        print('fine-tuning AdamW optimizer')
        learning_rates = [0.1, 0.01, 0.001]
        weight_decays = [0.01, 0.001, 0.0001]

        best_ppl = float('inf')
        best_lr = 0
        best_wd = 0
        for lr in learning_rates:
            for wd in weight_decays:

                model.apply(init_weights) # re-initialize the model weights to make a fair comparison

                print('-------------------------------------')
                print(f'Param search:\tLearning rate: [{lr}]\t Weight decay: [{wd}]')
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

                # Evaluate on little epochs due to time resources limitations
                ppl = evaluate_model(model, optimizer, train_loader, test_loader, dev_loader, criterion_eval, criterion_train, clip, n_epochs=20, patience=3, device=device)

                # save the best combination
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_lr = lr
                    best_wd = wd

        print(f'Best Parameters:\t Learning rate: [{best_lr}]\t Weight decay: [{best_wd}]')
        print(f'Best PPL: {best_ppl}')

        model.apply(init_weights) #initialize the model weights
        optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
        evaluate_model(model, optimizer, train_loader, test_loader, dev_loader, criterion_eval, criterion_train, clip, n_epochs=100, patience=3, device=device)
        torch.save(model.state_dict(), __file__ + "\\..\\bin\\lstm_adamw.pt")
    else:
        model_params = torch.load(__file__ + "\\..\\bin\\lstm_adamw.pt")
        model.load_state_dict(model_params)
        test_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print('Test ppl: ', test_ppl)

