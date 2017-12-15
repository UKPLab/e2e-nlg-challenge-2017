# Based on: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

import random
import torch
from torch.autograd import Variable

from components.data.common import cuda_if_gpu
from components.constants import BOS_ID, EOS_ID
from components.model import E2ESeq2SeqModel
from components.model.modules.encoders.enc_mlp import EncoderMLP
from components.model.modules.decoders.dec_attn import DecoderRNNAttnBahd


class E2EMLPModel(E2ESeq2SeqModel):
    def set_encoder(self):
        encoder_params = self.config["encoder_params"]
        self.encoder = EncoderMLP(encoder_params)

    def set_decoder(self):
        decoder_rnn_params = self.config["decoder_params"]
        self.decoder = DecoderRNNAttnBahd(rnn_config=decoder_rnn_params,
                                          output_size=self.tgt_vocab_size,
                                          prev_y_dim=self.embedding_dim,
                                          enc_dim=self.encoder.hidden_size,
                                          enc_num_directions=1)

    def forward(self, datum):
        """
        Run the model on one data instance.
        :param datum:
        :return:
        """

        # batch_x_var: SL x B
        # batch_y_var: TL x B
        batch_x_var, batch_y_var = datum

        # Embedding lookup
        encoder_input_embedded = self.embedding_lookup(batch_x_var)  # SL x B x E
        encoder_input_embedded = self.embedding_dropout_layer(encoder_input_embedded)

        # Encode embedded input
        # shapes: SL x B x H; 1 x B x H
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded)

        # Decoding using one of two policies:
        #   - use gold standard label as output of the previous step (teacher forcing)
        #   - use previous prediction as output of the previous step (dynamic decoding)
        # See: http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf
        # and the official PyTorch tutorial: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        use_teacher_force = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_force:
            logits = self.decode_teacher(batch_y_var, encoder_hidden, encoder_outputs)
        else:
            logits = self.decode_dynamic(batch_y_var, encoder_hidden, encoder_outputs)

        return logits

    def decode_teacher(self, dec_input_var, encoder_hidden, encoder_outputs):
        """
        Decoding policy 1: feeding the ground truth label as a target
        :param dec_input_var: ground truth labels
        :param encoder_hidden: the last hidden state of the Encoder RNN; (num_layers * num_directions) x B x enc_dim
        :param encoder_outputs: SL x B x enc_dim
        :return:
        """

        dec_len = dec_input_var.size()[0]
        batch_size = dec_input_var.size()[1]
        dec_hidden = encoder_hidden
        dec_input = cuda_if_gpu(Variable(torch.LongTensor([BOS_ID] * batch_size)))
        predicted_logits = cuda_if_gpu(Variable(torch.zeros(dec_len, batch_size, self.tgt_vocab_size)))

        # Teacher forcing: feed the target as the next input
        for di in range(dec_len):
            prev_y = self.embedding_mat(dec_input)  # embedding lookup of a vector of length = B; result: B x E
            # prev_y = self.dropout(prev_y) # apply dropout
            dec_output, dec_hidden, attn_weights = self.decoder(prev_y, dec_hidden, encoder_outputs)
            # shapes:
            # - dec_output: B, TV
            # - dec_hidden: 1, B, dec_dim
            # - attn_weights: B x SL

            predicted_logits[di] = dec_output  # store this step's outputs
            dec_input = dec_input_var[di]  # next input

        return predicted_logits

    def decode_dynamic(self, dec_input_var, encoder_hidden, encoder_outputs):

        """
        Decoding policy 2: feeding the previous prediction as a target
        :param dec_input_var: ground truth labels (used only to get the shape, not for decoding)
        :param encoder_hidden: the last hidden state of the Encoder RNN; (num_layers * num_directions) x B x enc_dim
        :param encoder_outputs: SL x B x enc_dim
        :return:
        """
        dec_len = dec_input_var.size()[0]
        batch_size = dec_input_var.size()[1]
        dec_hidden = encoder_hidden
        dec_input = cuda_if_gpu(Variable(torch.LongTensor([BOS_ID] * batch_size)))
        predicted_logits = cuda_if_gpu(Variable(torch.zeros(dec_len, batch_size, self.tgt_vocab_size)))

        # Dynamic decoding: feed the previous prediction as the next input
        for di in range(dec_len):
            prev_y = self.embedding_mat(dec_input)  # B x E
            # prev_y = self.dropout(prev_y)
            dec_output, dec_hidden, attn_weights = self.decoder(prev_y, dec_hidden, encoder_outputs)
            # shapes:
            # - dec_output: B, TV
            # - dec_hidden: 1, B, dec_dim
            # - attn_weights: B x SL

            predicted_logits[di] = dec_output
            topval, topidx = dec_output.data.topk(1)
            dec_input = cuda_if_gpu(Variable(
                torch.LongTensor(topidx.squeeze().cpu().numpy())
            )
            )

        return predicted_logits

    def predict(self, input_var):

        # Embedding lookup
        encoder_input_embedded = self.embedding_lookup(input_var)

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded)

        # Decode
        dec_ids, attn_w = [], []
        curr_token_id = BOS_ID
        curr_dec_idx = 0
        dec_input_var = cuda_if_gpu(Variable(torch.LongTensor([curr_token_id])))
        dec_hidden = encoder_hidden[:1]  # 1 x B x enc_dim

        while (curr_token_id != EOS_ID and curr_dec_idx <= self.max_tgt_len):
            prev_y = self.embedding_mat(dec_input_var)
            # prev_y = self.dropout(prev_y)

            decoder_output, dec_hidden, decoder_attention = self.decoder(prev_y, dec_hidden, encoder_outputs)
            attn_w.append(decoder_attention.data)

            topval, topidx = decoder_output.data.topk(1)
            curr_token_id = topidx[0][0]
            dec_ids.append(curr_token_id)
            dec_input_var = cuda_if_gpu(Variable(torch.LongTensor([curr_token_id])))

            curr_dec_idx += 1

        return dec_ids, attn_w


component = E2EMLPModel
