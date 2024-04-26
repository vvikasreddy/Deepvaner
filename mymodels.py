import torch
import torch.nn as nn
import time


class BioFeatureExtractor(nn.Module):
    def __init__(self, input_size=32, feature_size=40):
        super(BioFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=24, kernel_size=5),
            nn.BatchNorm1d(num_features=24),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=24, out_channels=16, kernel_size=3),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(8 * 120, feature_size)

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        print(x.shape, "here done biofeature")
        return x


class Transformer1d(nn.Module):
    def __init__(self, input_size, n_classes, n_length, d_model, nhead, dim_feedforward, dropout, activation='relu'):
        super(Transformer1d, self).__init__()

        self.input_size = input_size
        self.n_classes = n_classes
        self.n_length = n_length
        self.d_model = d_model

        # Assuming input_size is not necessarily equal to d_model
        self.input_projection = nn.Linear(input_size, d_model)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=2,
            nhead=2,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )

        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=6)

        # Calculate the output size after the Transformer
        # This step is crucial and needs to be adjusted based on how you handle the sequence
        # For simplicity, assuming the output size remains d_model * n_length
        # This might need adjustment based on your actual sequence handling (e.g., pooling, averaging)
        self.fc_out_size = d_model * n_length

        # Final linear layer to match the desired feature size (n_classes)
        self.fc = nn.Linear(self.fc_out_size, n_classes)

    def forward(self, x):
        # Input x shape: (batch_size, input_size, n_length)

        # Project input to d_model dimension
        x = x.permute(2, 0, 1)  # Change shape to (n_length, batch_size, input_size)
        x = self.input_projection(x)  # Shape becomes (n_length, batch_size, d_model)

        print(x.shape)
        # Pass through the Transformer encoder
        x = self.transformer_encoder(x)  # Shape remains (n_length, batch_size, d_model)
        print("here")
        y_input = torch.tensor([0, 1])

        x = self.transformer_decoder()
        # Flatten the output
        x = x.permute(1, 2, 0)  # Change shape to (batch_size, d_model, n_length)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch_size, d_model * n_length)

        # Final linear layer
        x = self.fc(x)  # Shape becomes (ba||tch_size, n_classes)
        # print(x.shape)
        return x


class DeepVANetBio(nn.Module):
    def __init__(self, input_size=32, feature_size=64):
        super(DeepVANetBio, self).__init__()
        self.features = BioFeatureExtractor(input_size=input_size, feature_size=feature_size)
        self.features = Transformer1d(
            input_size,
            n_classes=64,
            n_length=200,
            d_model=32,
            nhead=8,
            dim_feedforward=128,
            dropout=0.1,
            activation='relu'
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(20, 3),
            nn.Softmax()
        )

    def forward(self, x):
        print(x.shape, "here")
        x = self.features(x)
        x = self.classifier(x)
        x = x.squeeze(-1)
        return x

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'physiological_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))
        # self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
