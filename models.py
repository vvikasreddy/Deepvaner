"""
Neural network models in DeepVANet
"""

import torch
import torch.nn as nn
import time


# The implementation of CONVLSTM are based on the code from
# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3, padding=1):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim=self.input_dim,
                                 hidden_dim=self.hidden_dim,kernel_size=kernel_size, padding=padding)

    def forward(self, input_tensor, time=None):

        b, _, _, h, w = input_tensor.size()

        hidden_state = self.cell.init_hidden(b,h,w)

        seq_len = input_tensor.size(1)

        h, c = hidden_state
        for t in range(seq_len):
            h, c = self.cell(input_tensor=input_tensor[:, t, :, :, :], cur_state=[h, c])
        return h

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
        self.fc = nn.Linear(8*120, feature_size)

    def forward(self,x):
        x = self.cnn(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        print(x.shape)
        return x
"""
Neural network models in DeepVANet
"""

import torch
import torch.nn as nn
import time


# class BioFeatureExtractor(nn.Module):
#     def __init__(self, input_size=32, feature_size=40):
#         super(BioFeatureExtractor, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv1d(in_channels=input_size, out_channels=24, kernel_size=5),
#             nn.BatchNorm1d(num_features=24),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=24, out_channels=16, kernel_size=3),
#             nn.BatchNorm1d(num_features=16),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3),
#             nn.BatchNorm1d(num_features=8),
#             nn.ReLU(inplace=True),
#         )
#         self.fc = nn.Linear(8*120, feature_size)

#     def forward(self,x):
#         x = self.cnn(x)
#         x = torch.flatten(x,1)
#         x = self.fc(x)
#         # print(x.shape)
#         return x

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
            d_model = 2,
            nhead = 2,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation = activation
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

        # Pass through the Transformer encoder
        x = self.transformer_encoder(x)  # Shape remains (n_length, batch_size, d_model)
        y_input = torch.tensor([0,1])
    
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
                                        n_length=128,
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
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
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




class FaceFeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FaceFeatureExtractorCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 2),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'FaceFeatureExtractorCNN_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        # self.load_state_dict(torch.load(path))
        self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))


class FaceFeatureExtractor(nn.Module):
    def __init__(self, feature_size=16, pretrain=True):
        super(FaceFeatureExtractor, self).__init__()
        cnn = FaceFeatureExtractorCNN()
        if pretrain:
            cnn.load('./pretrained_cnn.pth')
        self.cnn = cnn.net
        self.rnn = ConvLSTM(128, 128)
        self.fc = nn.Linear(128*6*6, feature_size)

    def forward(self, x):
        # input should be 5 dimension: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        cnn_output = self.cnn(x)
        rnn_input = cnn_output.view(b, t, 128, 6, 6)
        rnn_output = self.rnn(rnn_input)
        rnn_output = torch.flatten(rnn_output, 1)
        output = self.fc(rnn_output)
        return output

#
# class BioFeatureExtractor(nn.Module):
#     def __init__(self, input_size=32, feature_size=40):
#         super(BioFeatureExtractor, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv1d(in_channels=input_size, out_channels=24, kernel_size=5),
#             nn.BatchNorm1d(num_features=24),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=24, out_channels=16, kernel_size=3),
#             nn.BatchNorm1d(num_features=16),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3),
#             nn.BatchNorm1d(num_features=8),
#             nn.ReLU(inplace=True),
#         )
#         self.fc = nn.Linear(8*120, feature_size)
#
#     def forward(self,x):
#         x = self.cnn(x)
#         x = torch.flatten(x,1)
#         x = self.fc(x)
#         return x


class DeepVANetVision(nn.Module):
    def __init__(self,feature_size=16,pretrain=True):
        super(DeepVANetVision,self).__init__()
        self.features = FaceFeatureExtractor(feature_size=feature_size,pretrain=pretrain)
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.squeeze(-1)
        return x

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'face_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))
        # self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

#
# class DeepVANetBio(nn.Module):
#     def __init__(self, input_size=32, feature_size=64):
#         super(DeepVANetBio, self).__init__()
#         self.features = BioFeatureExtractor(input_size=input_size, feature_size=feature_size)
#         self.classifier = nn.Sequential(
#             nn.Linear(feature_size, 20),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(20, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         x = x.squeeze(-1)
#         return x
#
#     def save(self, name=None):
#         """
#         save the model
#         """
#         if name is None:
#             prefix = 'checkpoints/' + 'physiological_classifier_'
#             name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
#         torch.save(self.state_dict(), name)
#         return name
#
#     def load(self, path):
#         self.load_state_dict(torch.load(path))
#         # self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


class DeepVANet(nn.Module):
    def __init__(self, bio_input_size=32, face_feature_size=16, bio_feature_size=64,pretrain=True):
        super(DeepVANet,self).__init__()
        self.face_feature_extractor = FaceFeatureExtractor(feature_size=face_feature_size,pretrain=pretrain)

        self.bio_feature_extractor = BioFeatureExtractor(input_size=bio_input_size, feature_size=bio_feature_size)

        self.classifier = nn.Sequential(
            nn.Linear(face_feature_size + bio_feature_size, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        img_features = self.face_feature_extractor(x[0])
        bio_features = self.bio_feature_extractor(x[1])
        features = torch.cat([img_features,bio_features.float()],dim=1)
        output = self.classifier(features)
        output = output.squeeze(-1)
        return output

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'fusion_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))
        # self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

