# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from utils import *
from model import *

def train_loop(data, optimizer, criterion, model):
    model.train()

    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['ids'], sample['mask']) # Get model's predictions
        gt = sample['label'][:, 0:1] # get the ground truth
        loss = criterion(output, gt) # compute the loss
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, model):
    model.eval()

    ref_intents = []
    hyp_intents = []

    with torch.no_grad():
        for sample in data:
            output = model(sample['ids'], sample['mask'])
            pred = torch.round(output.cpu()) # compute the binary prediction
            gt = sample['label'][:, 0:1].cpu() # get the ground truth

            # classification evaluation
            ref_intents.extend(gt.numpy())
            hyp_intents.extend(pred.numpy())

    # Return the accuracy score
    results = accuracy_score(ref_intents, hyp_intents)

    return results

def subjectivity_task(n_splits, dataset, tokenizer, max_len, lr, n_epochs, criterion, train):
    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    samples = [sample for sample, label in dataset]
    labels = [label for sample, label in dataset]

    scores_clf = []
    best_accuracy = 0
    for i, (train_index, test_index) in enumerate(skf.split(samples, labels)):

        print('split ', i + 1)

        x_train, x_test = [dataset[indx] for indx in train_index], [dataset[indx] for indx in test_index]

        train_data = Subjectivity_Dataset(x_train, tokenizer, max_len)
        test_data = Subjectivity_Dataset(x_test, tokenizer, max_len)

        train_loader = DataLoader(dataset=train_data, batch_size=SUB_TRAIN_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=SUB_TEST_BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

        model = Subjectivity_BERT(dropout_prob=0.1).to(device)

        if train:
            optimizer = optim.AdamW(model.parameters(), lr=lr)

            ## TRAIN
            for x in tqdm(range(1,n_epochs)):
                loss = train_loop(train_loader, optimizer, criterion, model)
                print('mean loss: ', np.mean(loss))

            ## EVAL
            accuracy = eval_loop(test_loader, model)
            scores_clf.append(accuracy)
            print(f'Accuracy (split {i+1}): {accuracy}')

            ## Save the best model
            if accuracy > best_accuracy:
                torch.save(model.state_dict(), __file__ + "\\..\\bin\\subjectivity_model.pt")
                best_accuracy = accuracy
        else:
            ## Load the best model
            model_params = torch.load(__file__ + "\\..\\bin\\subjectivity_model.pt")
            model.load_state_dict(model_params)

            ## EVAL
            accuracy = eval_loop(test_loader, model)
            scores_clf.append(accuracy)
            print(f'Accuracy (split {i+1}): {accuracy}')

    return round(sum(scores_clf)/len(scores_clf), 3), model

def polarity_task(n_splits, dataset, tokenizer, max_len, lr, n_epochs, criterion, train, raw_data):

    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    samples = [sample for sample, label in dataset]
    labels = [label for sample, label in dataset]

    scores_clf = []
    best_accuracy = 0
    for i, (train_index, test_index) in enumerate(skf.split(samples, labels)):

        print('split ', i + 1)

        x_train, x_test = [dataset[indx] for indx in train_index], [dataset[indx] for indx in test_index]

        train_data = Polarity_Dataset(x_train, tokenizer, max_len)
        test_data = Polarity_Dataset(x_test, tokenizer, max_len)

        train_loader = DataLoader(dataset=train_data, batch_size=POL_TRAIN_BATCH_SIZE, collate_fn=polarity_collate_fn, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=POL_TEST_BATCH_SIZE, collate_fn=polarity_collate_fn, shuffle=False)

        model = Polarity_BERT(dropout_prob=0.1).to(device)

        if train:
            optimizer = optim.AdamW(model.parameters(), lr=lr)

            ## TRAIN
            for x in tqdm(range(1,n_epochs)):
                loss = train_loop(train_loader, optimizer, criterion, model)
                print('mean loss: ', np.mean(loss))

            ## EVAL
            accuracy = eval_loop(test_loader, model)
            scores_clf.append(accuracy)
            print(f'Accuracy (split {i+1}): {accuracy}')

            ## Save the best model
            if accuracy > best_accuracy:
                if raw_data:
                    torch.save(model.state_dict(), __file__ + "\\..\\bin\\polarity_model.pt")
                else:
                    torch.save(model.state_dict(), __file__ + "\\..\\bin\\polarity_model_2.pt")
                best_accuracy = accuracy
        else:
            ## Load the best model
            if raw_data:
                model_params = torch.load(__file__ + "\\..\\bin\\polarity_model.pt")
            else:
                model_params = torch.load(__file__ + "\\..\\bin\\polarity_model_2.pt")
            model.load_state_dict(model_params)

            ## EVAL
            accuracy = eval_loop(test_loader, model)
            scores_clf.append(accuracy)
            print(f'Accuracy (split {i+1}): {accuracy}')

    return round(sum(scores_clf)/len(scores_clf), 3)