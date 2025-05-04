from io import open
import re
import random
import numpy as np
import time
import math
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from transformers import GPT2Tokenizer, BertTokenizer
from sklearn.model_selection import train_test_split

SOS_token = 50256
EOS_token = 50256
MAX_LENGTH = 25

eng_tokenizer  = GPT2Tokenizer.from_pretrained('gpt2')
chi_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

ENG_EOS_TOKEN = eng_tokenizer.eos_token_id  # GPT2 EOS token (50256)

CHI_SOS_TOKEN = chi_tokenizer.cls_token_id  # Bert CLS token (101)
CHI_EOS_TOKEN = chi_tokenizer.sep_token_id  # Bert SEP token (102)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(eng_tokenizer.vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(chi_tokenizer.vocab_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, chi_tokenizer.vocab_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(CHI_SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

# drop some too long sentences.
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def get_dataloader(batch_size):
    lines = open('dataset.txt', encoding='utf-8').read().strip().split('\n')
    pairs = [[s for s in l.split('\t')[:2]] for l in lines]

    pairs = filterPairs(pairs)

    train_pairs, test_pairs = train_test_split(pairs, test_size=0.1, random_state=42)

    n = len(train_pairs)

    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (eng, chi) in enumerate(train_pairs):
        eng_ids = eng_tokenizer.encode(eng, add_special_tokens=False)[:MAX_LENGTH - 1] + [ENG_EOS_TOKEN]
        chi_ids = [CHI_SOS_TOKEN] + chi_tokenizer.encode(chi, add_special_tokens=False)[:MAX_LENGTH-2] + [CHI_EOS_TOKEN]

        input_ids[idx, :len(eng_ids)] = eng_ids
        target_ids[idx, :len(chi_ids)] = chi_ids

    # get training dataset
    train_dataset = TensorDataset(torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device))

    n = len(test_pairs)

    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (eng, chi) in enumerate(test_pairs):
        eng_ids = eng_tokenizer.encode(eng, add_special_tokens=False)[:MAX_LENGTH - 1] + [ENG_EOS_TOKEN]
        chi_ids = [CHI_SOS_TOKEN] + chi_tokenizer.encode(chi, add_special_tokens=False)[:MAX_LENGTH-2] + [CHI_EOS_TOKEN]

        input_ids[idx, :len(eng_ids)] = eng_ids
        target_ids[idx, :len(chi_ids)] = chi_ids

    # get testing dataset
    test_dataset = TensorDataset(torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device))

    return (train_pairs, test_pairs, DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size),
            DataLoader(test_dataset, batch_size=batch_size))


# one epoch for training
def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    attentions = None
    total_loss = 0

    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, attentions = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader), attentions


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("train_loss.png")

def plot_attention_maps(attention_weights):
    for i, attn_weights in enumerate(attention_weights[:15]):
        plt.figure(figsize=(15, 8))
        plt.imshow(attn_weights[0], aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Layer {i + 1}')
        plt.xlabel('Input Sequence')
        plt.ylabel('Output Sequence')
        plt.savefig("attention_maps.png")

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=1):
    start = time.time()
    plot_losses = []
    all_attention_weights = []
    attentions = None

    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=0)


    for epoch in range(1, n_epochs + 1):
        loss, attentions = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    all_attention_weights.append(attentions.cpu().detach().numpy())

    showPlot(plot_losses)

    # plot attention maps
    plot_attention_maps(all_attention_weights)

def evaluate_and_print_result(encoder, decoder, sentence, eng_tokenizer, chi_tokenizer):
    with torch.no_grad():
        input_ids = np.zeros(MAX_LENGTH, dtype=np.int32)
        eng_ids = eng_tokenizer.encode(sentence, add_special_tokens=False)[:MAX_LENGTH - 1] + [ENG_EOS_TOKEN]
        input_ids[:len(eng_ids)] = eng_ids

        encoder_outputs, encoder_hidden = encoder(torch.tensor(input_ids, dtype=torch.long, device=device).view(1, -1))
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        print(decoded_ids)

        decoded_words = []
        for idx in decoded_ids:
            if idx == CHI_EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(chi_tokenizer.decode(idx))
    return decoded_words, decoder_attn

def evaluateRandomly(pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate_and_print_result(encoder, decoder, pair[0], eng_tokenizer, chi_tokenizer)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

hidden_size = 128
batch_size = 32

train_pairs, test_pairs, train_dataloader, test_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size).to(device)

train(train_dataloader, encoder, decoder, 64, print_every=1)

encoder.eval()
decoder.eval()
evaluateRandomly(test_pairs, encoder, decoder)
# evaluate(test_dataloader, encoder, decoder)



