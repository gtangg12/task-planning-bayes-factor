import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from babyai.common import *


class ResidualBLockFilm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1);
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, input, gamma, beta):
        residual = F.relu(self.conv(input))
        x = F.relu(self.body(residual))
        x = gamma.unsqueeze(2).unsqueeze(3) * x + beta.unsqueeze(2).unsqueeze(3)
        return x + residual


class FilmCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(19, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, dim, 3, padding=1),
        )
        num_film_layers = 2
        self.film_layers = nn.ModuleList(
            [ResidualBLockFilm(dim=dim) for _ in range(num_film_layers)]
        )

    def forward(self, x, gamma, beta):
        x = self.cnn(x)
        for block in self.film_layers:
            x = block(x, gamma, beta)
        return x


class ClassifierFilmRNN(nn.Module):
    def __init__(self):
        super().__init__()

        n_temporal = EMBEDDING_DIM
        self.goal_encoder  = nn.LSTM(n_temporal, n_temporal, num_layers=1, batch_first=True)
        self.actor_encoder = nn.LSTM(n_temporal, n_temporal, num_layers=1, batch_first=True)

        n_kernels = 128
        self.film_param = nn.ModuleList([nn.Linear(n_temporal, n_kernels),
                                         nn.Linear(n_temporal, n_kernels)])

        self.cnn = FilmCNN(dim=n_kernels)

        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_kernels * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(1024, 1024)
        )

        self.lstm = nn.LSTM(1024, 1024, num_layers=1, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, goal, actor_info, images, goal_len, seq_len, label=None):
        #torch.Size([8, 32, 128]) torch.Size([8, 7, 128]) torch.Size([8, 7, 19, 7, 7]) torch.Size([8]) torch.Size([8])
        #print(goal.shape, actor_info.shape, images.shape, goal_len.shape, seq_len.shape)
        batch_size, padded_temporal_dim = images.shape[0], images.shape[1]

        goal = self.forward_rnn(goal, goal_len, self.goal_encoder)
        goal = torch.stack([goal[i, idx - 1, :] for i, idx in enumerate(goal_len)])
        goal = goal.unsqueeze(1)

        actor_info = torch.cat((goal, actor_info), dim=1)
        actor_info = self.forward_rnn(actor_info, seq_len + 1, self.actor_encoder)
        actor_info = actor_info[:, 1:, :]

        #print(padded_temporal_dim, actor_info.shape, images.shape)

        joint_encodings = []
        for i in range(padded_temporal_dim):
            x = self.cnn(images[:, i, ...], self.film_param[0](actor_info[:, i, :]),
                                            self.film_param[1](actor_info[:, i, :]))
            joint_encodings.append(self.flatten(x))

        joint_encodings = torch.stack(joint_encodings, dim=1)
        lstm_output = self.forward_rnn(joint_encodings, seq_len, self.lstm)
        lstm_output_last = torch.stack([
            lstm_output[i][seq_len[i] - 1] for i in range(batch_size)
        ])

        logits = self.classifier(lstm_output_last)

        return logits

    def forward_rnn(self, input, seq_len, rnn):
        input =  pack_padded_sequence(input,
                                      seq_len.cpu(),
                                      batch_first=True,
                                      enforce_sorted=False)
        rnn.flatten_parameters()
        output, _ = rnn(input)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output
