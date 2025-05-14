import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
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