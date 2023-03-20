import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from transformers import *


class HR_RNN(nn.Module):
    def __init__(self, hidden_size, embedding_matrix):
        super(HR_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        if self.training == True:
            et = torch.tensor(embedding_matrix, dtype=torch.float32)
            self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True)

        self.attn = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)

        self.linear = nn.Linear(3 * 2 * hidden_size, 16)
        self.output_layer = nn.Linear(16, 1)
        self.weight_xt = nn.Parameter(torch.randn(1, 2 * hidden_size, 1), requires_grad=True)

    def forward(self, input, input_lengths):
        # print("before embedding",input.shape)
        embed = self.embedding(input)
        # print('embed shape',embed.shape)

        embed = torch.transpose(embed, 0, 1)

        h_0 = Variable(torch.zeros(2, embed.shape[1], self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(2, embed.shape[1], self.hidden_size).cuda())

        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, input_lengths, enforce_sorted=False)
        lstmOutput, _ = self.lstm(embed, (h_0, c_0))

        lstmOutput, _ = torch.nn.utils.rnn.pad_packed_sequence(lstmOutput)
        context_vector = self.attention(lstmOutput, self.weight_xt)
        max_pool, _ = torch.max(lstmOutput, 0)
        mean_pool = torch.mean(lstmOutput, 0)

        concat = torch.cat([context_vector, max_pool, mean_pool], dim=1)

        linearOutput = F.relu(self.linear(concat))
        out = self.output_layer(linearOutput)
        return torch.sigmoid(out)

    def attention(self, hidden_states, query_vector):
        uit = torch.tanh(self.attn(hidden_states))
        uit = torch.transpose(uit, 0, 1)
        multipliedWithQuery = torch.matmul(uit, query_vector)
        attentionWeights = F.softmax(multipliedWithQuery, dim=1)

        hidden_states = torch.transpose(hidden_states, 0, 1)
        attentionWeightedHidden = torch.mul(hidden_states, attentionWeights)
        context_vector = torch.sum(attentionWeightedHidden, dim=1)
        return context_vector

class HR_XML_CNN(nn.Module):
    def __init__(self, num_bottleneck_hidden,dynamic_pool_length,output_channel, embedding_matrix):
        super(HR_XML_CNN, self).__init__()
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        if self.training == True:
            et = torch.tensor(embedding_matrix, dtype=torch.float32)
            self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = True
        self.num_bottleneck_hidden = num_bottleneck_hidden
        self.dynamic_pool_length = dynamic_pool_length
        self.input_channel = 1
        self.output_channel = output_channel

        self.conv1 = nn.Conv2d(self.input_channel, self.output_channel, (3, embedding_matrix.shape[1]), padding=(1, 0))
        self.conv2 = nn.Conv2d(self.input_channel, self.output_channel, (5, embedding_matrix.shape[1]), padding=(2, 0))
        self.conv3 = nn.Conv2d(self.input_channel, self.output_channel, (9, embedding_matrix.shape[1]), padding=(4, 0))

        self.pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length)

        self.bottleneck = nn.Linear(3 * self.output_channel * self.dynamic_pool_length, self.num_bottleneck_hidden)
        self.fc1 = nn.Linear(self.num_bottleneck_hidden, 1)

    def forward(self, input, input_lengths):
        embed = self.embedding(input)
        embed = embed.unsqueeze(1)

        c1 = F.relu(self.conv1(embed)).squeeze(3)
        c2 = F.relu(self.conv2(embed)).squeeze(3)
        c3 = F.relu(self.conv3(embed)).squeeze(3)

        x = [c1, c2, c3]
        c1_pool = self.pool(c1)
        c2_pool = self.pool(c2)
        c3_pool = self.pool(c3)

        x = [c1_pool, c2_pool, c3_pool]

        x = torch.cat(x, 1) 
        x = F.relu(self.bottleneck(x.view(-1, 3 * self.output_channel * self.dynamic_pool_length)))

        output = torch.sigmoid(self.fc1(x)) 
        return output

class HR_Sci_BERT(nn.Module):
    def __init__(self):
        super(HR_Sci_BERT, self).__init__()
        self.bert = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', output_attentions = False, output_hidden_states = True, return_dict=False)
        self.linear = nn.Linear(768, 1)
    def forward(self, input, att_mask):
        _, pooled_output, _ = self.bert(input, attention_mask = att_mask)
        output = torch.sigmoid(self.linear(pooled_output))
        return output

class HR_BERT(nn.Module):
    def __init__(self):
        super(HR_BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions = False, output_hidden_states = True, return_dict=False)
        self.linear = nn.Linear(768, 1)
    def forward(self, input, att_mask):
        _, pooled_output, _ = self.bert(input, attention_mask = att_mask)
        output = torch.sigmoid(self.linear(pooled_output))
        return output

class HR_RNN_MultiTasking(nn.Module):
    def __init__(self, hidden_size, embedding_matrix):
        super(HR_RNN_MultiTasking, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        if self.training == True:
            et = torch.tensor(embedding_matrix, dtype=torch.float32)
            self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True)
        self.attn = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)
        self.linear = nn.Linear(3 * 2 * hidden_size, 16)
        self.output_layer = nn.Linear(16, 1)
        self.output_layer_seqlabeling = nn.Linear(2 * hidden_size, 1)
        self.weight_xt = nn.Parameter(torch.randn(1, 2 * hidden_size, 1), requires_grad=True)
        
    def forward(self, input, input_lengths):
        embed = self.embedding(input)
        embed = torch.transpose(embed, 0, 1)

        h_0 = Variable(torch.zeros(2, embed.shape[1], self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(2, embed.shape[1], self.hidden_size).cuda())

        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, input_lengths, enforce_sorted=False)

        lstmOutput, (h_0, c_0) = self.lstm(embed, (h_0, c_0))
        lstmOutput, _ = torch.nn.utils.rnn.pad_packed_sequence(lstmOutput)
        context_vector = self.attention(lstmOutput, self.weight_xt)
        
        max_pool, _ = torch.max(lstmOutput, 0)
        mean_pool = torch.mean(lstmOutput, 0)
        concat = torch.cat([context_vector, max_pool, mean_pool], dim=1)

        linearOutput = F.relu(self.linear(concat))
        out = self.output_layer(linearOutput)
        lstmOutput = torch.transpose(lstmOutput, 0, 1)
        out_seqlabeling = self.output_layer_seqlabeling(lstmOutput)
        return torch.sigmoid(out), torch.sigmoid(out_seqlabeling)

    def attention(self, hidden_states, query_vector):
        uit = torch.tanh(self.attn(hidden_states))
        uit = torch.transpose(uit, 0, 1)
        multipliedWithQuery = torch.matmul(uit, query_vector)
        attentionWeights = F.softmax(multipliedWithQuery, dim=1)
        hidden_states = torch.transpose(hidden_states, 0, 1)
        attentionWeightedHidden = torch.mul(hidden_states, attentionWeights)
        context_vector = torch.sum(attentionWeightedHidden, dim=1)
        return context_vector

class HR_XML_CNN_MultiTasking(nn.Module):
    def __init__(self, num_bottleneck_hidden,dynamic_pool_length,output_channel, embedding_matrix):
        super(HR_XML_CNN_MultiTasking, self).__init__()
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        if self.training == True:
            et = torch.tensor(embedding_matrix, dtype=torch.float32)
            self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = True
        self.num_bottleneck_hidden = num_bottleneck_hidden
        self.dynamic_pool_length = dynamic_pool_length
        self.input_channel = 1
        self.output_channel = output_channel

        self.conv1 = nn.Conv2d(self.input_channel, self.output_channel, (3, embedding_matrix.shape[1]), padding=(1, 0))
        self.conv2 = nn.Conv2d(self.input_channel, self.output_channel, (5, embedding_matrix.shape[1]), padding=(2, 0))
        self.conv3 = nn.Conv2d(self.input_channel, self.output_channel, (9, embedding_matrix.shape[1]), padding=(4, 0))

        self.pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length)

        self.bottleneck = nn.Linear(3 * self.output_channel * self.dynamic_pool_length, self.num_bottleneck_hidden)
        self.fc1 = nn.Linear(self.num_bottleneck_hidden, 1)
        self.output_layer_seqlabeling = nn.Linear(3 * output_channel, 1)

    def forward(self, input, input_lengths):
        embed = self.embedding(input)
        embed = embed.unsqueeze(1)
        c1 = F.relu(self.conv1(embed)).squeeze(3)
        c2 = F.relu(self.conv2(embed)).squeeze(3)
        c3 = F.relu(self.conv3(embed)).squeeze(3)

        x = [c1, c2, c3]
        c1_pool = self.pool(c1)
        c2_pool = self.pool(c2)
        c3_pool = self.pool(c3)

        x = [c1_pool, c2_pool, c3_pool]

        x = torch.cat(x, 1) 
        x = F.relu(self.bottleneck(x.view(-1, 3 * self.output_channel * self.dynamic_pool_length)))
        concat_for_seq_labeling = torch.cat([c1, c2, c3], dim=1)
        concat_for_seq_labeling = torch.transpose(concat_for_seq_labeling, 1, 2)
        keyword_output = torch.sigmoid(self.output_layer_seqlabeling(concat_for_seq_labeling))
        output = torch.sigmoid(self.fc1(x))  
        return output, keyword_output

class HR_Sci_BERT_MultiTasking(nn.Module):
    def __init__(self):
        super(HR_Sci_BERT_MultiTasking, self).__init__()
        self.bert = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", output_attentions = False, output_hidden_states = True, return_dict=False)
        self.linear = nn.Linear(768, 1)
        self.output_layer_seqlabeling = nn.Linear(768, 1)
    def forward(self, input, att_mask):
        hidden_states, pooled_output, _ = self.bert(input, attention_mask = att_mask)
        output = torch.sigmoid(self.linear(pooled_output))
        out_seqlabeling = torch.sigmoid(self.output_layer_seqlabeling(hidden_states))
        return output, out_seqlabeling

class HR_BERT_MultiTasking(nn.Module):
    def __init__(self):
        super(HR_BERT_MultiTasking, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", output_attentions = False, output_hidden_states = True, return_dict=False)
        self.linear = nn.Linear(768, 1)
        self.output_layer_seqlabeling = nn.Linear(768, 1)
    def forward(self, input, att_mask):
        hidden_states, pooled_output, _ = self.bert(input, attention_mask = att_mask)
        output = torch.sigmoid(self.linear(pooled_output))
        out_seqlabeling = torch.sigmoid(self.output_layer_seqlabeling(hidden_states))
        return output, out_seqlabeling

class Flat_XML_CNN(nn.Module):
    def __init__(self, num_bottleneck_hidden,dynamic_pool_length,output_channel, embedding_matrix, num_classes):
        super(Flat_XML_CNN, self).__init__()
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        if self.training == True:
            et = torch.tensor(embedding_matrix, dtype=torch.float32)
            self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = True
        self.num_bottleneck_hidden = num_bottleneck_hidden
        self.dynamic_pool_length = dynamic_pool_length
        self.input_channel = 1
        self.output_channel = output_channel

        self.conv1 = nn.Conv2d(self.input_channel, self.output_channel, (3, embedding_matrix.shape[1]), padding=(1, 0))
        self.conv2 = nn.Conv2d(self.input_channel, self.output_channel, (5, embedding_matrix.shape[1]), padding=(2, 0))
        self.conv3 = nn.Conv2d(self.input_channel, self.output_channel, (9, embedding_matrix.shape[1]), padding=(4, 0))

        self.pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length)

        self.bottleneck = nn.Linear(3 * self.output_channel * self.dynamic_pool_length, self.num_bottleneck_hidden)
        self.fc1 = nn.Linear(self.num_bottleneck_hidden, num_classes)

    def forward(self, input, input_lengths):
        embed = self.embedding(input)
        embed = embed.unsqueeze(1)

        c1 = F.relu(self.conv1(embed)).squeeze(3)
        c2 = F.relu(self.conv2(embed)).squeeze(3)
        c3 = F.relu(self.conv3(embed)).squeeze(3)

        x = [c1, c2, c3]
        c1_pool = self.pool(c1)
        c2_pool = self.pool(c2)
        c3_pool = self.pool(c3)

        x = [c1_pool, c2_pool, c3_pool]
        x = torch.cat(x, 1)  # (batch, channel_output * ks)
        x = F.relu(self.bottleneck(x.view(-1, 3 * self.output_channel * self.dynamic_pool_length)))

        output = torch.sigmoid(self.fc1(x))
        return output

class Flat_RNN(nn.Module):
    def __init__(self, hidden_size, embedding_matrix, num_classes):
        super(Flat_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        if self.training == True:
            et = torch.tensor(embedding_matrix, dtype=torch.float32)
            self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True)

        self.attn = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)

        self.linear = nn.Linear(3 * 2 * hidden_size, 72)
        self.output_layer = nn.Linear(72, num_classes)
        self.weight_xt = nn.Parameter(torch.randn(1, 2 * hidden_size, 1), requires_grad=True)

    def forward(self, input, input_lengths):
        embed = self.embedding(input)
        embed = torch.transpose(embed, 0, 1)

        h_0 = Variable(torch.zeros(2, embed.shape[1], self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(2, embed.shape[1], self.hidden_size).cuda())

        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, input_lengths, enforce_sorted=False)
        lstmOutput, _ = self.lstm(embed, (h_0, c_0))

        lstmOutput, _ = torch.nn.utils.rnn.pad_packed_sequence(lstmOutput)

        context_vector = self.attention(lstmOutput, self.weight_xt)
        max_pool, _ = torch.max(lstmOutput, 0)
        mean_pool = torch.mean(lstmOutput, 0)

        concat = torch.cat([context_vector, max_pool, mean_pool], dim=1)

        linearOutput = F.relu(self.linear(concat))
        out = self.output_layer(linearOutput)
        return torch.sigmoid(out)

    def attention(self, hidden_states, query_vector):
        uit = torch.tanh(self.attn(hidden_states))
        uit = torch.transpose(uit, 0, 1)
        multipliedWithQuery = torch.matmul(uit, query_vector)
        attentionWeights = F.softmax(multipliedWithQuery, dim=1)
        hidden_states = torch.transpose(hidden_states, 0, 1)
        attentionWeightedHidden = torch.mul(hidden_states, attentionWeights)
        context_vector = torch.sum(attentionWeightedHidden, dim=1)
        return context_vector

class Flat_RNN_Multitasking(nn.Module):
    def __init__(self, hidden_size, embedding_matrix, num_classes):
        super(Flat_RNN_Multitasking, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        if self.training == True:
            et = torch.tensor(embedding_matrix, dtype=torch.float32)
            self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True)

        self.attn = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)

        self.linear = nn.Linear(3 * 2 * hidden_size, 72)
        self.output_layer = nn.Linear(72, num_classes)
        self.output_layer_seqlabeling = nn.Linear(2 * hidden_size, 1)

        self.weight_xt = nn.Parameter(torch.randn(1, 2 * hidden_size, 1), requires_grad=True)

    def forward(self, input, input_lengths):
        embed = self.embedding(input)
        embed = torch.transpose(embed, 0, 1)

        h_0 = Variable(torch.zeros(2, embed.shape[1], self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(2, embed.shape[1], self.hidden_size).cuda())

        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, input_lengths, enforce_sorted=False)
        lstmOutput, (h_0, c_0) = self.lstm(embed, (h_0, c_0))

        lstmOutput, _ = torch.nn.utils.rnn.pad_packed_sequence(lstmOutput)

        context_vector = self.attention(lstmOutput, self.weight_xt)
        max_pool, _ = torch.max(lstmOutput, 0)
        mean_pool = torch.mean(lstmOutput, 0)

        concat = torch.cat([context_vector, max_pool, mean_pool], dim=1)

        linearOutput = F.relu(self.linear(concat))
        out = self.output_layer(linearOutput)

        lstmOutput = torch.transpose(lstmOutput, 0, 1)
        out_seqlabeling = self.output_layer_seqlabeling(lstmOutput)

        return torch.sigmoid(out), torch.sigmoid(out_seqlabeling)

    def attention(self, hidden_states, query_vector):
        uit = torch.tanh(self.attn(hidden_states))
        uit = torch.transpose(uit, 0, 1)
        multipliedWithQuery = torch.matmul(uit, query_vector)
        attentionWeights = F.softmax(multipliedWithQuery, dim=1)

        hidden_states = torch.transpose(hidden_states, 0, 1)
        attentionWeightedHidden = torch.mul(hidden_states, attentionWeights)
        context_vector = torch.sum(attentionWeightedHidden, dim=1)
        return context_vector

class Flat_XML_CNN_Multitasking(nn.Module):
    def __init__(self, num_bottleneck_hidden,dynamic_pool_length,output_channel, embedding_matrix, num_classes):
        super(Flat_XML_CNN_Multitasking, self).__init__()
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        et = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = False
        self.num_bottleneck_hidden = num_bottleneck_hidden
        self.dynamic_pool_length = dynamic_pool_length
        self.input_channel = 1
        self.output_channel = output_channel

        self.conv1 = nn.Conv2d(self.input_channel, self.output_channel, (3, embedding_matrix.shape[1]), padding=(1, 0))
        self.conv2 = nn.Conv2d(self.input_channel, self.output_channel, (5, embedding_matrix.shape[1]), padding=(2, 0))
        self.conv3 = nn.Conv2d(self.input_channel, self.output_channel, (9, embedding_matrix.shape[1]), padding=(4, 0))

        self.pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length)

        self.bottleneck = nn.Linear(3 * self.output_channel * self.dynamic_pool_length, self.num_bottleneck_hidden)
        self.fc1 = nn.Linear(self.num_bottleneck_hidden, num_classes)
        self.output_layer_seqlabeling = nn.Linear(3 * output_channel, 1)

    def forward(self, input, input_lengths):
        embed = self.embedding(input)
        embed = embed.unsqueeze(1)

        c1 = F.relu(self.conv1(embed)).squeeze(3)
        c2 = F.relu(self.conv2(embed)).squeeze(3)
        c3 = F.relu(self.conv3(embed)).squeeze(3)

        x = [c1, c2, c3]
        c1_pool = self.pool(c1)
        c2_pool = self.pool(c2)
        c3_pool = self.pool(c3)

        x = [c1_pool, c2_pool, c3_pool]

        x = torch.cat(x, 1)  
        x = F.relu(self.bottleneck(x.view(-1, 3 * self.output_channel * self.dynamic_pool_length)))

        concat_for_seq_labeling = torch.cat([c1, c2, c3], dim=1)
        concat_for_seq_labeling = torch.transpose(concat_for_seq_labeling, 1, 2)
        keyword_output = torch.sigmoid(self.output_layer_seqlabeling(concat_for_seq_labeling))

        output = torch.sigmoid(self.fc1(x)) 
        return output, keyword_output

class Flat_Sci_BERT(nn.Module):
    def __init__(self, num_classes):
        super(Flat_Sci_BERT, self).__init__()
        self.bert = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", output_attentions = False, output_hidden_states = True, return_dict = False)
        self.linear = nn.Linear(768, num_classes)
    def forward(self, input, att_mask):
        _, pooled_output, _ = self.bert(input, attention_mask = att_mask)
        output = torch.sigmoid(self.linear(pooled_output))
        return output

class Flat_BERT(nn.Module):
    def __init__(self, num_classes):
        super(Flat_BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions = False, output_hidden_states = True, return_dict = False)
        self.linear = nn.Linear(768, num_classes)
    def forward(self, input, att_mask):
        _, pooled_output, _ = self.bert(input, attention_mask = att_mask)
        output = torch.sigmoid(self.linear(pooled_output))
        return output

class Flat_BERT_Multitasking(nn.Module):
    def __init__(self, num_classes):
        super(Flat_BERT_Multitasking, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", output_attentions = False, output_hidden_states = True, return_dict=False)
        self.linear = nn.Linear(768, num_classes)
        self.output_layer_seqlabeling = nn.Linear(768, 1)
    def forward(self, input, att_mask):
        hidden_states, pooled_output, _ = self.bert(input, attention_mask = att_mask)
        output = torch.sigmoid(self.linear(pooled_output))
        out_seqlabeling = torch.sigmoid(self.output_layer_seqlabeling(hidden_states))
        return output, out_seqlabeling

class Flat_Sci_BERT_Multitasking(nn.Module):
    def __init__(self, num_classes):
        super(Flat_Sci_BERT_Multitasking, self).__init__()
        self.bert = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", output_attentions = False, output_hidden_states = True, return_dict=False)
        self.linear = nn.Linear(768, num_classes)
        self.output_layer_seqlabeling = nn.Linear(768, 1)
    def forward(self, input, att_mask):
        hidden_states, pooled_output, _ = self.bert(input, attention_mask = att_mask)
        output = torch.sigmoid(self.linear(pooled_output))
        out_seqlabeling = torch.sigmoid(self.output_layer_seqlabeling(hidden_states))
        return output, out_seqlabeling
