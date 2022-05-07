import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from babyai.common import *
from babyai_task_sequence_dataset import TaskSequenceBatch


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
    def __init__(self, num_channels, dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1),
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
    def __init__(self, num_channels, embedding_dim):
        super().__init__()

        hidden_dim = 128
        self.task_encoder  = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.actor_encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)

        num_kernels = 128
        self.film_param = nn.ModuleList([nn.Linear(hidden_dim, num_kernels),
                                         nn.Linear(hidden_dim, num_kernels)])

        self.cnn = FilmCNN(num_channels, num_kernels)

        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_kernels * 7 * 7, 1024),
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

    def forward(self, inputs): 
        #task, actor_info, images, task_len, seq_len
        #torch.Size([8, 32, 128]) torch.Size([8, 7, 128]) torch.Size([8, 7, 19, 7, 7]) torch.Size([8]) torch.Size([8])
        task_batch, images_batch, actions_batch = inputs['task'], inputs['images'], inputs['actions']
        task_lens, sequence_lens = inputs['task_len'], inputs['sequence_len']

        batch_size, padded_sequence_len = images_batch.shape[0], images_batch.shape[1]

        task_batch = self._forward_rnn(task_batch, task_lens, self.task_encoder) 
        task_batch = torch.stack([task_batch[i, idx - 1, :] for i, idx in enumerate(task_lens)])
        task_batch = task_batch.unsqueeze(1)

        #task: torch.Size([8, 1, 128])
        combined_batch = torch.cat((task_batch, actions_batch), dim=1)

        # actor_info: torch.Size([8, 8, 128])
        combined_batch = self._forward_rnn(combined_batch, sequence_lens + 1, self.actor_encoder)
        combined_batch = combined_batch[:, 1:, :]

        # actor_info: torch.Size([8, 7, 128])
        joint_encodings = []
        for i in range(padded_sequence_len): # 7 
            x = self.cnn(images_batch[:, i, ...], self.film_param[0](combined_batch[:, i, :]),
                                                  self.film_param[1](combined_batch[:, i, :]))
            joint_encodings.append(self.flatten(x))

        joint_encodings = torch.stack(joint_encodings, dim=1)
        lstm_output = self.forward_rnn(joint_encodings, sequence_lens, self.lstm)
        lstm_output_last = torch.stack([
            lstm_output[i][sequence_lens[i] - 1] for i in range(batch_size)
        ])

        logits = self.classifier(lstm_output_last)
        return logits

    def _forward_rnn(self, input, seq_len, rnn):
        input =  pack_padded_sequence(input,
                                      seq_len.cpu(),
                                      batch_first=True,
                                      enforce_sorted=False)
        rnn.flatten_parameters()
        output, _ = rnn(input)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output


if __name__ == '__main__':
    pass