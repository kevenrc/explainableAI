import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scripts.nn_train import TabularDataset, FeedForwardNN, binary_acc
from sklearn.preprocessing import LabelEncoder

class NeuralNet:
    def __init__(self):
        self.categorical_features = ["MaxDelq2PublicRecLast12M", "MaxDelqEver"]
        self.output_feature = "RiskPerformance"
        self.label_encoders = {}
        self.model_name = "Neural Net"

    def fit(self, X_train, y_train):

        train = pd.concat([X_train, y_train], axis=1)
        for cat_col in self.categorical_features:
            self.label_encoders[cat_col] = LabelEncoder()
            train[cat_col] = self.label_encoders[cat_col].fit_transform(train[cat_col])

        dataset_train = TabularDataset(data=train, cat_cols=self.categorical_features,
                output_col=self.output_feature)

        batchsize=64
        dataloader=DataLoader(dataset_train, batchsize, shuffle=True, num_workers=1)
        cat_dims = [int(train[col].nunique()) for col in self.categorical_features]
        emb_dims = [(x, min(50, (x+1)//2)) for x in cat_dims]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FeedForwardNN(emb_dims, no_of_cont=21, lin_layer_sizes=[50, 100],
                output_size=1, emb_dropout=0.04,
                lin_layer_dropouts=[0.001, 0.01]).to(self.device)
        no_of_epochs = 1
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model.train()
        for epoch in range(no_of_epochs):
            epoch_loss = 0
            epoch_acc = 0
            for y, cont_x, cat_x in dataloader:
                cat_x = cat_x.to(self.device)
                cont_x = cont_x.to(self.device)
                y = y.to(self.device)

                preds = self.model(cont_x, cat_x)
                loss = criterion(preds, y)
                acc = binary_acc(preds, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            print('Epoch {:03}: | Loss: {:.5f} | Acc: {:.3f}'.format(epoch, epoch_loss/len(dataloader), epoch_acc/len(dataloader)))

    def predict(self, X_test):
        cont_x = torch.tensor(X_test.drop(self.categorical_features, axis=1).values).to(self.device)
        cat_x = torch.tensor(X_test[self.categorical_features].values).to(self.device)
        y_pred = self.model(cont_x, cat_x)
        y_pred = torch.round(torch.sigmoid(y_pred))
        return y_pred

    def get_model(self):
        return self.model

    def get_model_name(self):
        return self.model_name

