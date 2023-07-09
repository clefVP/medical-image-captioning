import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasetsRenamed import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

#Adapted from repo: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

data_folder = r"/Users/pranithred/Desktop/clef23/captioning/caption_data"
data_name = r"cd1.2_1_cap_per_img_5_min_word_freq"


embedding_dimension = 512
attention_dimension = 512
decoder_dimension = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


start_epoch = 0
no_epochs = 20
no_epochs_since_improvement = 0
batch_size = 16
workers = 0
enc_learning_rate = 1e-4
dec_learning_rate = 4e-4
grad_clip = 5.
alpha_c = 1.
best_bleu4 = 0.
print_freq = 200
fine_tune_encoder = False
checkpoint = None


def main():
    """
    Training and validation.
    """

    global best_bleu4, no_epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, flip_map


    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)





    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dimension=attention_dimension,
                                       embed_dim=embedding_dimension,
                                       decoder_dimension=decoder_dimension,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=dec_learning_rate)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=enc_learning_rate) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        no_epochs_since_improvement = checkpoint['no_epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=enc_learning_rate)


    decoder = decoder.to(device)
    encoder = encoder.to(device)


    criterion = nn.CrossEntropyLoss().to(device)


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)


    for epoch in range(start_epoch, no_epochs):


        if no_epochs_since_improvement == 20:
            break
        if no_epochs_since_improvement > 0 and no_epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)


        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)


        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)


        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            no_epochs_since_improvement += 1
            print("\nno_epochs since last improvement: %d\n" % (no_epochs_since_improvement,))
        else:
            no_epochs_since_improvement = 0


        save_checkpoint(data_name, epoch, no_epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()


    loop = tqdm(train_loader)
    for i, (imgs, caps, caplens) in enumerate(loop):
        data_time.update(time.time() - start)


        #print(caps.size())
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)


        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)


        targets = caps_sorted[:, 1:]



        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data


        loss = criterion(scores, targets)


        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()


        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()


        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)


        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()


        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()


        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()
    hypotheses = list()



    with torch.no_grad():

        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):


            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)


            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)


            targets = caps_sorted[:, 1:]
            #print(sort_ind)


            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data


            loss = criterion(scores, targets)


            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()


            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))






            sort_ind = sort_ind.to('cpu')
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                #print(allcaps.shape[0])
                img_caps = allcaps[j].tolist()
                img_captions=list()
                for w in img_caps:
                    if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}:
                        img_captions.append(w)



                references.append([img_captions])

            #print(references[0])

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            #print(preds)
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)


        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
