import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 2, 100)  # Adjusted based on actual output size after flatten
        self.fc2 = nn.Linear(100, num_classes)  # 4 classes

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten the tensor for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class FeatTimeAttention(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super().__init__()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.latent_dim = latent_dim
        T, D_f = input_shape
        # Define Kernel and Bias for Feature Projection
        self.kernel = torch.zeros(
            (1, 1, D_f, self.latent_dim), requires_grad=True).to(self.device)
        nn.init.xavier_uniform_(self.kernel)
        self.bias = torch.zeros(
            (1, 1, D_f, self.latent_dim), requires_grad=True).to(self.device)
        nn.init.uniform_(self.bias)

        # Define Time aggregation weights for averaging over time.
        self.unnorm_beta = torch.zeros((1, T, 1), requires_grad=True)
        nn.init.uniform_(self.unnorm_beta)

    def forward(self, x, latent):
        o_hat, _ = self.generate_latent_approx(x, latent)
        weights = self.calc_weights(self.unnorm_beta)
        return torch.sum(torch.mul(o_hat.to(self.device), weights.to(self.device)), dim=1)

    def generate_latent_approx(self, x, latent):
        features = torch.mul(x.unsqueeze(-1), self.kernel) + self.bias
        features = F.relu(features)

        # calculate the score
        X_T, X = features, features.transpose(2, 3)
        # print(X_T.shape, X.shape)
        X_T_X_inv = torch.inverse(torch.matmul(X_T, X))
        # print(X_T.shape, latent.unsqueeze(-1).shape)
        X_T_y = torch.matmul(X_T, latent.unsqueeze(-1))

        score_hat = torch.matmul(X_T_X_inv, X_T_y)
        scores = torch.squeeze(score_hat)

        # print(scores.unsqueeze(-1).shape, features.shape)
        o_hat = torch.sum(torch.mul(scores.unsqueeze(-1), features), dim=2)

        return o_hat, scores

    def calc_weights(self, x):
        abs_x = torch.abs(x)
        return abs_x / torch.sum(abs_x, dim=1)

# Bi-LSTM Encoder

# class Encoder(nn.Module):
#     def __init__(self, input_shape, attention_hidden_dim, latent_dim, dropout):
#         super().__init__()
#         self.lstm1 = nn.LSTM(input_size=input_shape[1],
#                              hidden_size=attention_hidden_dim,
#                              num_layers=2,
#                              dropout=dropout,
#                              batch_first=True,
#                              bidirectional=True)
#         self.ln1 = nn.LayerNorm(attention_hidden_dim * 2)
        
#         self.lstm2 = nn.LSTM(input_size=attention_hidden_dim * 2,
#                              hidden_size=latent_dim,
#                              num_layers=1,
#                              batch_first=True,
#                              bidirectional=True)
#         self.ln2 = nn.LayerNorm(latent_dim * 2)
        
        
#         self.attention = FeatTimeAttention(latent_dim * 2, input_shape)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         latent_rep, _ = self.lstm1(x)
#         latent_rep = self.ln1(latent_rep)
#         latent_rep = self.dropout(latent_rep)
        
#         latent_rep, _ = self.lstm2(latent_rep)
#         latent_rep = self.ln2(latent_rep)
#         latent_rep = self.dropout(latent_rep)
        
#         output = self.attention(x, latent_rep)
#         return output






class Encoder(nn.Module):
    def __init__(self, input_shape, attention_hidden_dim, latent_dim, dropout):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_shape[1],
                             hidden_size=attention_hidden_dim,
                             num_layers=2,
                             dropout=dropout,
                             batch_first=True)
        self.lstm2 = nn.LSTM(input_size=attention_hidden_dim,
                             hidden_size=latent_dim,
                             num_layers=1,
                             batch_first=True)
        self.attention = FeatTimeAttention(latent_dim, input_shape)

    def forward(self, x):
        latent_rep, _ = self.lstm1(x)
        latent_rep, _ = self.lstm2(latent_rep)
        output = self.attention(x, latent_rep)
        return output



class Identifier(nn.Module):
    def __init__(self, input_dim, mlp_hidden_dim, dropout, output_dim):
        super().__init__()
        # Bi-LSTM Encoder
        # self.fc1 = nn.Linear(input_dim*2, mlp_hidden_dim)
        
        self.fc1 = nn.Linear(input_dim, mlp_hidden_dim)
        self.sigmoid1 = nn.Sigmoid()

        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.sigmoid2 = nn.Sigmoid()
        self.dropout1 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(mlp_hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid1(x)

        x = self.fc2(x)
        x = self.sigmoid2(x)
        x = self.dropout1(x)

        x = self.fc4(x)
        x = self.softmax(x)
        return x


class Predictor(nn.Module):
    def __init__(self, input_dim, mlp_hidden_dim, dropout, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_hidden_dim)
        # Bi-LSTM Encoder
        # self.fc1 = nn.Linear(input_dim*2, mlp_hidden_dim)
        self.sigmoid1 = nn.Sigmoid()

        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.sigmoid2 = nn.Sigmoid()
        self.dropout1 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.sigmoid3 = nn.Sigmoid()
        self.dropout2 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(mlp_hidden_dim, output_dim)
        
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid1(x)

        x = self.fc2(x)
        x = self.sigmoid2(x)
        x = self.dropout1(x)

        x = self.fc3(x)
        x = self.sigmoid3(x)
        x = self.dropout2(x)

        x = self.fc4(x)
        
        # Apply the noise adaptation layer before sotmax
        x = self.softmax(x)
        return x

