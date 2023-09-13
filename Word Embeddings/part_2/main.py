# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *

if __name__ == "__main__":
    
    train=False  ## CHANGE THIS TO RE-TRAIN THE MODEL

    # Dataloader instantiation
    train_loader, dev_loader, test_loader, lang = get_dataloaders()

    #initialize the hyperparameters
    hid_size = 200 # hidden size
    emb_size = 300 # embedding size
    clip = 5 # Clip the gradient
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # gpu device if possible
    vocab_len = len(lang.word2id) # vocabulary length

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    # using the optimizer hyperparameters found using grid search in the part 1
    lr = 0.1 # learning rate
    wd = 0.0001 # weight decay
    momentum=0.99 # momentum
    use_nesterov=True

    # Weight Tying optimization
    print('LSTM Weight Tying optimization')
    model = LM_LSTM_Optimized(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], variational_dropout=False, weight_tying=True).to(device)

    if train:
        model.apply(init_weights) #initialize the model weights
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=use_nesterov)
        evaluate_model(model, optimizer, train_loader, test_loader, dev_loader, criterion_eval, criterion_train, clip, n_epochs=100, patience=100, device=device, nt_avsgd=False, n=2)
        torch.save(model.state_dict(), __file__ + "\\..\\bin\\lstm_wt.pt")
    else:
        model_params = torch.load(__file__ + "\\..\\bin\\lstm_wt.pt")
        model.load_state_dict(model_params)
        test_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print('Test ppl: ', test_ppl)

    # Variational Dropout optimization
    print('LSTM Variational Dropout optimization')
    model = LM_LSTM_Optimized(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], variational_dropout=True, weight_tying=True).to(device)

    if train:
        model.apply(init_weights) #initialize the model weights
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=use_nesterov)
        evaluate_model(model, optimizer, train_loader, test_loader, dev_loader, criterion_eval, criterion_train, clip, n_epochs=100, patience=100, device=device, nt_avsgd=False, n=2)
        torch.save(model.state_dict(), __file__ + "\\..\\bin\\lstm_vd.pt")
    else:
        model_params = torch.load(__file__ + "\\..\\bin\\lstm_vd.pt")
        model.load_state_dict(model_params)
        test_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print('Test ppl: ', test_ppl)
    
    # Non-monotonically Triggered AvSGD Optimization
    print('LSTM Non-monotonically Triggered AvSGD Optimization')
    model = LM_LSTM_Optimized(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], variational_dropout=True, weight_tying=True).to(device)

    if train:
        model.apply(init_weights) #initialize the model weights
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=use_nesterov)
        evaluate_model(model, optimizer, train_loader, test_loader, dev_loader, criterion_eval, criterion_train, clip, n_epochs=100, patience=5, device=device, nt_avsgd=True, n=2)
        torch.save(model.state_dict(), __file__ + "\\..\\bin\\lstm_optim.pt")
    else:
        model_params = torch.load(__file__ + "\\..\\bin\\lstm_optim.pt")
        model.load_state_dict(model_params)
        test_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print('Test ppl: ', test_ppl)
