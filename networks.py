import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from itertools import cycle
from collections import OrderedDict
from utils import reparameterize, transform_config
from data_loader import TrustpilotDataset  

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, style_dim, class_dim):
        super(TextEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 128, batch_first=True)

        # Style embeddings
        self.style_mu = nn.Linear(128, style_dim)
        self.style_logvar = nn.Linear(128, style_dim)

        # Class embeddings
        self.class_output = nn.Linear(128, class_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, (hn, cn) = self.lstm(x)  # Get the output of the last time step
        x = hn[-1]  # Use the last hidden state

        style_embeddings_mu = self.style_mu(x)
        style_embeddings_logvar = self.style_logvar(x)
        class_embeddings = self.class_output(x)

        return style_embeddings_mu, style_embeddings_logvar, class_embeddings


class Decoder(nn.Module):
    def __init__(self, style_dim, class_dim):
        super(Decoder, self).__init__()

        # Style embeddings input
        self.style_input = nn.Linear(style_dim, 128)

        # Class embeddings input
        self.class_input = nn.Linear(class_dim, 128)

        self.fc_model = nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(128, 256)),
            ('LeakyReLU_1', nn.LeakyReLU(negative_slope=0.2)),
            ('fc_2', nn.Linear(256, 1)),  # Output a single value (e.g., reconstructed rating)
            ('sigmoid_final', nn.Sigmoid())  # Assuming ratings are normalized between 0 and 1
        ]))

    def forward(self, style_embeddings, class_embeddings):
        style_embeddings = F.leaky_relu_(self.style_input(style_embeddings), negative_slope=0.2)
        class_embeddings = F.leaky_relu_(self.class_input(class_embeddings), negative_slope=0.2)

        x = style_embeddings + class_embeddings  # Combine embeddings
        x = self.fc_model(x)

        return x


class Classifier(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(Classifier, self).__init__()

        self.fc_model = nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(z_dim, 256)),
            ('fc_1_bn', nn.BatchNorm1d(256)),
            ('LeakyRelu_1', nn.LeakyReLU(negative_slope=0.2)),

            ('fc_2', nn.Linear(256, 256)),
            ('fc_2_bn', nn.BatchNorm1d(256)),
            ('LeakyRelu_2', nn.LeakyReLU(negative_slope=0.2)),

            ('fc_3', nn.Linear(256, num_classes))
        ]))

    def forward(self, z):
        x = self.fc_model(z)
        return x

if __name__ == '__main__':
    """
    Test for network outputs
    """
    # Example parameters
    vocab_size = 10000  # Adjust based on your tokenizer
    embedding_dim = 100
    style_dim = 16
    class_dim = 5  # Assuming 5 different ratings/classes

    encoder = TextEncoder(vocab_size, embedding_dim, style_dim, class_dim)
    decoder = Decoder(style_dim, class_dim)
    classifier = Classifier(z_dim=style_dim, num_classes=class_dim)

    # Load your Trustpilot dataset
    trustpilot_dataset = TrustpilotDataset('data/trustpilot.csv')  # Ensure the path is correct
    loader = cycle(DataLoader(trustpilot_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True))

    # Example input (you should preprocess your comments into token IDs)
    comment_batch, labels_batch = next(loader)  # Assuming comments are already tokenized

    # Convert comments to tensor (you may need to adjust this based on your preprocessing)
    comment_tensor = torch.tensor(comment_batch).long()  # Ensure it's long type for embedding

    mu, logvar, class_latent_space = encoder(Variable(comment_tensor))
    style_latent_space = reparameterize(training=True, mu=mu, logvar=logvar)

    reconstructed_rating = decoder(style_latent_space, class_latent_space)
    classifier_pred = classifier(style_latent_space)

    print(reconstructed_rating.size())
    print(classifier_pred.size())