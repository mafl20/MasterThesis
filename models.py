import torch.nn as nn

class AF_Autoencoder(nn.Module):
    def __init__(self):
        super(AF_Autoencoder, self).__init__()
        #> Encoder
        self.encoder = nn.Sequential(
            nn.Linear(640, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Linear(320, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Linear(160, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )

        #> Decoder
        self.decoder = nn.Sequential(
            nn.Linear(20, 40),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.Linear(40, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Linear(80, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Linear(160, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Linear(320, 640),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded



class BaselineAutoencoder(nn.Module):
    def __init__(self):
        super(BaselineAutoencoder, self).__init__()
        #> Encoder
        self.encoder = nn.Sequential(
            nn.Linear(640, 128), #input layer (640 features) to hidden layer (128 features)
            nn.BatchNorm1d(128), #batch normalization for the hidden layer
            nn.ReLU(), #activation function (ReLU)
            nn.Linear(128, 128), #first hidden layer (128 features)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 8), #from hidden layer (128 features) to bottleneck layer (8 features)
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )
        
        #> Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 128), #from bottleneck layer (8 features) to hidden layer (128 features)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 640), #output layer (640 features)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        
        decoded = self.decoder(encoded)

        #print(f"Input vector: {x}")

        return decoded