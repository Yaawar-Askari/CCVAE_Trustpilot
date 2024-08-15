import os
import numpy as np
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import weights_init
from data_loader import TrustpilotDataset
from networks import TextEncoder, Decoder, Classifier
from torch.utils.data import DataLoader
from utils import mse_loss, reparameterize, l1_loss


def preprocess_text(comment):
    # This is a placeholder function. Implement actual preprocessing (tokenization, padding, etc.)
    # For example, tokenize and pad the text
    return comment


def training_procedure(FLAGS):
    """
    model definition
    """
    vocab_size = 10000  # Adjust based on your tokenizer
    embedding_dim = 100
    encoder = TextEncoder(vocab_size, embedding_dim, FLAGS.style_dim, FLAGS.class_dim)
    encoder.apply(weights_init)

    decoder = Decoder(FLAGS.style_dim, FLAGS.class_dim)
    decoder.apply(weights_init)

    classifier = Classifier(z_dim=FLAGS.style_dim, num_classes=FLAGS.class_dim)
    classifier.apply(weights_init)

    # Load saved models if load_saved flag is true
    if FLAGS.load_saved:
        encoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.encoder_save)))
        decoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.decoder_save)))
        classifier.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.classifier_save)))

    """
    variable definition
    """
    # Adjust input tensors to handle text data
    X = torch.LongTensor(FLAGS.batch_size, FLAGS.max_seq_length)  # Adjust max_seq_length accordingly
    style_latent_space = torch.FloatTensor(FLAGS.batch_size, FLAGS.style_dim)

    """
    loss definitions
    """
    cross_entropy_loss = nn.CrossEntropyLoss()

    '''
    add option to run on GPU
    '''
    if FLAGS.cuda:
        encoder.cuda()
        decoder.cuda()
        classifier.cuda()
        cross_entropy_loss.cuda()

        X = X.cuda()
        style_latent_space = style_latent_space.cuda()

    """
    optimizer and scheduler definition
    """
    auto_encoder_optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    reverse_cycle_optimizer = optim.Adam(
        list(encoder.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    classifier_optimizer = optim.Adam(
        list(classifier.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    # Divide the learning rate by a factor of 10 after 80 epochs
    auto_encoder_scheduler = optim.lr_scheduler.StepLR(auto_encoder_optimizer, step_size=80, gamma=0.1)
    reverse_cycle_scheduler = optim.lr_scheduler.StepLR(reverse_cycle_optimizer, step_size=80, gamma=0.1)
    classifier_scheduler = optim.lr_scheduler.StepLR(classifier_optimizer, step_size=80, gamma=0.1)

    """
    training
    """
    if torch.cuda.is_available() and not FLAGS.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if not os.path.exists('reconstructed_text'):
        os.makedirs('reconstructed_text')

    # load_saved is false when training is started from 0th iteration
    if not FLAGS.load_saved:
        with open(FLAGS.log_file, 'w') as log:
            log.write('Epoch\tIteration\tReconstruction_loss\tKL_divergence_loss\tReverse_cycle_loss\n')

    # Load dataset and create data loader instance
    print('Loading Trustpilot dataset...')
    trustpilot_dataset = TrustpilotDataset('data/trustpilot.csv', transform=preprocess_text)
    loader = cycle(DataLoader(trustpilot_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))

    # Initialize summary writer
    writer = SummaryWriter()

    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
        print('')
        print('Epoch #' + str(epoch) + '..........................................................................')

        # Update the learning rate scheduler
        auto_encoder_scheduler.step()
        reverse_cycle_scheduler.step()
        classifier_scheduler.step()

        for iteration in range(int(len(trustpilot_dataset) / FLAGS.batch_size)):
            # A. Run the auto-encoder reconstruction
            comment_batch, rating_batch = next(loader)

            auto_encoder_optimizer.zero_grad()

            X.copy_(torch.LongTensor(comment_batch))  # Convert to tensor if needed

            style_mu, style_logvar, class_latent_space = encoder(Variable(X))
            style_latent_space = reparameterize(training=True, mu=style_mu, logvar=style_logvar)

            kl_divergence_loss = FLAGS.kl_divergence_coef * (
                -0.5 * torch.sum(1 + style_logvar - style_mu.pow(2) - style_logvar.exp())
            )
            kl_divergence_loss /= FLAGS.batch_size
            kl_divergence_loss.backward(retain_graph=True)

            reconstructed_rating = decoder(style_latent_space, class_latent_space)
            reconstruction_error = FLAGS.reconstruction_coef * mse_loss(reconstructed_rating, Variable(torch.FloatTensor(rating_batch).unsqueeze(1)))
            reconstruction_error.backward()

            auto_encoder_optimizer.step()

            # B. Reverse cycle
            comment_batch, _ = next(loader)

            reverse_cycle_optimizer.zero_grad()

            X.copy_(torch.LongTensor(comment_batch))  # Convert to tensor if needed

            style_latent_space.normal_(0., 1.)
            _, __, class_latent_space = encoder(Variable(X))

            reconstructed_rating = decoder(Variable(style_latent_space), class_latent_space.detach())

            style_mu, style_logvar, _ = encoder(reconstructed_rating)
            style_latent_space = reparameterize(training=False, mu=style_mu, logvar=style_logvar)

            reverse_cycle_loss = FLAGS.reverse_cycle_coef * l1_loss(style_latent_space, Variable(torch.FloatTensor(style_latent_space)))
            reverse_cycle_loss.backward()

            reverse_cycle_optimizer.step()

            if (iteration + 1) % 10 == 0:
                print('')
                print('Epoch #' + str(epoch))
                print('Iteration #' + str(iteration))
                print('')
                print('Reconstruction loss: ' + str(reconstruction_error.data.storage().tolist()[0]))
                print('KL-Divergence loss: ' + str(kl_divergence_loss.data.storage().tolist()[0]))
                print('Reverse cycle loss: ' + str(reverse_cycle_loss.data.storage().tolist()[0]))

            # Write to log
            with open(FLAGS.log_file, 'a') as log:
                log.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
                    epoch,
                    iteration,
                    reconstruction_error.data.storage().tolist()[0],
                    kl_divergence_loss.data.storage().tolist()[0],
                    reverse_cycle_loss.data.storage().tolist()[0]
                ))

            # Write to TensorBoard
            writer.add_scalar('Reconstruction loss', reconstruction_error.data.storage().tolist()[0],
                              epoch * (int(len(trustpilot_dataset) / FLAGS.batch_size) + 1) + iteration)
            writer.add_scalar('KL-Divergence loss', kl_divergence_loss.data.storage().tolist()[0],
                              epoch * (int(len(trustpilot_dataset) / FLAGS.batch_size) + 1) + iteration)
            writer.add_scalar('Reverse cycle loss', reverse_cycle_loss.data.storage().tolist()[0],
                              epoch * (int(len(trustpilot_dataset) / FLAGS.batch_size) + 1) + iteration)

        # Save model after every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.end_epoch:
            torch.save(encoder.state_dict(), os.path.join('checkpoints', FLAGS.encoder_save))
            torch.save(decoder.state_dict(), os.path.join('checkpoints', FLAGS.decoder_save))
            torch.save(classifier.state_dict(), os.path.join('checkpoints', FLAGS.classifier_save))

            """
            Save reconstructed ratings to check progress
            """
            comment_batch, rating_batch = next(loader)
            X.copy_(torch.LongTensor(comment_batch))  # Convert to tensor if needed

            style_mu, style_logvar, class_latent_space = encoder(Variable(X))
            style_latent_space = reparameterize(training=False, mu=style_mu, logvar=style_logvar)

            reconstructed_rating = decoder(style_latent_space, class_latent_space)

            print("Original Ratings: ", rating_batch)
            print("Reconstructed Ratings: ", reconstructed_rating.data.cpu().numpy())
