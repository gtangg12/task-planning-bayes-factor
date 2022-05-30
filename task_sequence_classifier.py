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
            nn.BatchNorm2d(dim),
        )

    def forward(self, input, gamma, beta):
        residual = F.relu(self.conv(input))
        x = F.relu(self.body(residual))
        x = gamma.unsqueeze(2).unsqueeze(3) * x + beta.unsqueeze(2).unsqueeze(3)
        return x + residual


class CNNFilm(nn.Module):
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
    def __init__(self, num_channels, vocab_size, action_embedding_dim):
        super().__init__()

        # lstms for computing film params
        # task encoder maps task rnn output to same dim as action rnn input since task is fed to action rnn
        rnn_output_dim = 128
        self.task_encoder = \
            nn.LSTM(vocab_size, action_embedding_dim, num_layers=1, batch_first=True)
        self.actor_encoder = \
            nn.LSTM(action_embedding_dim, rnn_output_dim, num_layers=1, batch_first=True)

        # film param mappings
        num_cnn_kernels = 128
        self.film_param = nn.ModuleList([
            nn.Linear(rnn_output_dim, num_cnn_kernels),
            nn.Linear(rnn_output_dim, num_cnn_kernels)
        ])

        # film cnn for task sequence encodings
        self.cnn = CNNFilm(num_channels, num_cnn_kernels)

        # flatten film features into embedding
        embedding_dim = 1024
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_cnn_kernels * 7 * 7, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # lstm for encoding aggregation over time
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers=1, batch_first=True)

        # binary classifier to determine if task accomplished
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs): 
        # Example dims:
        # task, actor_info, images, task_len, seq_len
        # torch.Size([8, 5, 128]) torch.Size([8, 7, 128]) torch.Size([8, 7, 19, 7, 7]) torch.tensor([8]) torch.tensor([8])
        task_batch, images_batch, actions_batch = \
            inputs['task'].float(), inputs['images'].float(), inputs['actions'].float()
        task_lens, sequence_lens = \
            inputs['task_len'], inputs['sequence_len']
            
        batch_size, padded_sequence_len = images_batch.shape[0], images_batch.shape[1]

        #print(task_batch, actions_batch, images_batch)
        #print(batch_size, padded_sequence_len)

        # extract final task rnn output from each batch element
        task_batch = self._forward_rnn(task_batch, task_lens, self.task_encoder) 
        task_batch = torch.stack([task_batch[i, idx - 1, :] for i, idx in enumerate(task_lens)])
        task_batch = task_batch.unsqueeze(1)

        # append final task rnn output to beginning of action sequence
        combined_batch = torch.cat((task_batch, actions_batch), dim=1)
        combined_batch = self._forward_rnn(combined_batch, sequence_lens + 1, self.actor_encoder)
        # pop the first combined rnn output from the sequence
        combined_batch = combined_batch[:, 1:, :]   

        # using the combined rnn output as seeds for the film params, generate joint film embeddings 
        # for each timestamp of the task sequence
        joint_encodings = []
        for i in range(padded_sequence_len): # 7 
            x = self.cnn(images_batch[:, i, ...], self.film_param[0](combined_batch[:, i, :]),
                                                  self.film_param[1](combined_batch[:, i, :]))
            joint_encodings.append(self.flatten(x))
        joint_encodings = torch.stack(joint_encodings, dim=1)

        # aggregate the joint film embeddings over time 
        lstm_output = self._forward_rnn(joint_encodings, sequence_lens, self.lstm)
        
        # feed last rnn output into classifier
        lstm_output_last = torch.stack([
            lstm_output[i][sequence_lens[i] - 1] for i in range(batch_size)
        ])
        logits = self.classifier(lstm_output_last)
        return logits

    def _forward_rnn(self, input, seq_len, rnn):
        """ Helper function for evaluating rnn given input batch where every element has 
            different lengths 
        """
        input = pack_padded_sequence(input, 
                                     seq_len.cpu(), 
                                     batch_first=True, 
                                     enforce_sorted=False)
        rnn.flatten_parameters()
        output, _ = rnn(input)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output


if __name__ == '__main__':
    inputs = {
        'taskname': 'NAME',
        'task': torch.rand(8, 5, 32),
        'images': torch.rand(8, 7, 20, 7, 7),
        'actions': torch.rand(8, 7, 11),
        'task_len': torch.tensor([5, 5, 5, 5, 5, 5, 5, 5]),
        'sequence_len': torch.tensor([7, 4, 5, 3, 2, 1, 6, 5]),
    }
    model = ClassifierFilmRNN(num_channels=20, vocab_size=32, action_embedding_dim=11)
    print(model(inputs).shape)