import os
import numpy as np
import h5py
import json
import torch
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import pandas as pd
import imageio
from PIL import Image

def create_input_files(dataset, captions_per_image, min_word_freq, output_folder):

# Creates input files for training, validation, and test data.


    captions_per_image = 1

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()


    train_data = pd.read_csv(r"/Users/pranithred/Desktop/clef23/dataset/ImageCLEFmedical_Caption_2023_caption_prediction_train_labels.csv",delimiter='\t')
    train_captions = train_data[train_data.columns[1]]
    train_captions = train_captions.str.split()
    train_IDs=train_data[train_data.columns[0]]
    train_path = r"/Users/pranithred/Desktop/clef23/dataset/train"

    valid_data = pd.read_csv(r"/Users/pranithred/Desktop/clef23/dataset/ImageCLEFmedical_Caption_2023_caption_prediction_valid_labels.csv",delimiter='\t')
    valid_captions = valid_data[valid_data.columns[1]]
    valid_captions = valid_captions.str.split()
    valid_IDs=valid_data[valid_data.columns[0]]
    valid_path = r"/Users/pranithred/Desktop/clef23/dataset/valid"

    test_captions = train_captions[54000:]
    test_IDs=train_IDs[54000:]
    test_path = r"/Users/pranithred/Desktop/clef23/dataset/train"

    train_captions = train_captions[:54000]
    train_IDs=train_IDs[:54000]

    for img,caption in zip(train_IDs,train_captions):
        word_freq.update(caption)
        train_image_paths.append(train_path+'/'+img)
        train_image_captions.append(caption)
    
    for img,caption in zip(valid_IDs,valid_captions):
        word_freq.update(caption)
        val_image_paths.append(valid_path+'/'+img)
        val_image_captions.append(caption)

    for img,caption in zip(test_IDs,test_captions):
        word_freq.update(caption)
        test_image_paths.append(test_path+'/'+img)
        test_image_captions.append(caption)


    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = 1

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                captions=imcaps[i]


                # Read images
                img = imageio.v2.imread(impaths[i]+'.jpg')
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = np.array(Image.fromarray(obj=img).resize(size=(256, 256)))

                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in captions] + [
                word_map['<end>']] + [word_map['<pad>']] * (50 - len(captions))
                c_len = len(captions) + 2

                enc_captions.append(enc_c)
                caplens.append(c_len)


            # Sanity check
            print(images.shape[0], captions_per_image, len(enc_captions),len(caplens))
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

            print(max(caplens))


# def init_embedding(embeddings):

#     bias = np.sqrt(3.0 / embeddings.size(1))
#     torch.nn.init.uniform_(embeddings, -bias, bias)


# def load_embeddings(emb_file, word_map):
#     """
#     Creates an embedding tensor for the specified word map, for loading into the model.

#     :param emb_file: file containing embeddings (stored in GloVe format)
#     :param word_map: word map
#     :return: embeddings in the same order as the words in the word map, dimension of embeddings
#     """

#     # Find embedding dimension
#     with open(emb_file, 'r') as f:
#         embedding_dimension = len(f.readline().split(' ')) - 1

#     vocab = set(word_map.keys())

#     # Create tensor to hold embeddings, initialize
#     embeddings = torch.FloatTensor(len(vocab), embedding_dimension)
#     init_embedding(embeddings)

#     # Read embedding file
#     print("\nLoading embeddings...")
#     for line in open(emb_file, 'r'):
#         line = line.split(' ')

#         emb_word = line[0]
#         embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

#         # Ignore word if not in train_vocab
#         if emb_word not in vocab:
#             continue

#         embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

#     return embeddings, embedding_dimension



class AverageMeter(object):  #Keeps track of most recent, average, sum, and count of a metric.

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):


    print("\nshrinking learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
#param k: k in top-k accuracy

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def save_checkpoint(data_name, epoch, no_epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):

    state = {'epoch': epoch,
             'no_epochs_since_improvement': no_epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


def clip_gradient(optimizer, grad_clip):

    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
