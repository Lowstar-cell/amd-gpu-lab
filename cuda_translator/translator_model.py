import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqTranslator(nn.Module):
    """
    Character-level Seq2Seq model for CUDA → HIP translation.

    Architecture:
    - Encoder: Bi-GRU over input characters
    - Decoder: GRU with attention over encoder outputs
    - Output: Character probabilities for HIP code
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Decoder
        self.decoder_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.decoder = nn.GRU(
            embed_dim + hidden_dim * 2,  # embedding + context from attention
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention
        self.attention = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.attention_combine = nn.Linear(hidden_dim, 1)

        # Output projection
        self.out = nn.Linear(hidden_dim, vocab_size)

        # Bridge from encoder to decoder (bidirectional → unidirectional)
        self.bridge_h = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch, src_len) - CUDA code tokens
            tgt: (batch, tgt_len) - HIP code tokens (for teacher forcing)
            teacher_forcing_ratio: Probability of using ground truth vs. prediction

        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        # Encode
        encoder_out, h_n = self.encode(src)

        # Initialize decoder hidden state from encoder
        # h_n: (num_layers * 2, batch, hidden) → (num_layers, batch, hidden)
        h_0 = self._bridge_hidden(h_n)

        # Decode
        outputs = []
        decoder_input = tgt[:, 0].unsqueeze(1)  # <SOS> token
        hidden = h_0

        for t in range(1, tgt_len):
            # Decode one step
            output, hidden = self.decode_step(decoder_input, hidden, encoder_out)
            outputs.append(output)

            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(dim=-1)

        logits = torch.stack(outputs, dim=1)  # (batch, tgt_len-1, vocab_size)
        return logits

    def encode(self, src):
        """Encode source sequence."""
        # src: (batch, src_len)
        embedded = self.encoder_embed(src)  # (batch, src_len, embed_dim)
        encoder_out, h_n = self.encoder(embedded)
        # encoder_out: (batch, src_len, hidden * 2)
        # h_n: (num_layers * 2, batch, hidden)
        return encoder_out, h_n

    def _bridge_hidden(self, hidden):
        """Convert bidirectional encoder hidden to unidirectional decoder hidden."""
        # hidden: (num_layers * 2, batch, hidden)
        # Reshape: (num_layers, 2, batch, hidden)
        num_layers = self.num_layers
        batch_size = hidden.size(1)

        # Combine forward and backward
        hidden = hidden.view(num_layers, 2, batch_size, self.hidden_dim)
        hidden = hidden.transpose(1, 2).contiguous()  # (num_layers, batch, 2, hidden)
        hidden = hidden.view(num_layers, batch_size, -1)  # (num_layers, batch, hidden * 2)

        # Project
        if hidden.dim() == 3:
            return self.bridge_h(hidden)
        else:
            return self.bridge_h(hidden)

    def decode_step(self, input_token, hidden, encoder_out):
        """
        Decode one step with attention.

        Args:
            input_token: (batch, 1) - current input token
            hidden: h - decoder hidden state
            encoder_out: (batch, src_len, hidden * 2) - encoder outputs

        Returns:
            output: (batch, 1, vocab_size)
            hidden: updated decoder hidden
        """
        # Embed input
        embedded = self.decoder_embed(input_token)  # (batch, 1, embed_dim)

        # Attention
        # Repeat decoder hidden for each encoder position
        decoder_hidden = hidden[-1].unsqueeze(1).repeat(1, encoder_out.size(1), 1)  # (batch, src_len, hidden)

        # Compute attention scores
        attention_input = torch.cat([encoder_out, decoder_hidden], dim=2)  # (batch, src_len, hidden*3)
        attention_scores = self.attention(attention_input)  # (batch, src_len, hidden)
        attention_scores = self.attention_combine(attention_scores).squeeze(2)  # (batch, src_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, src_len)

        # Apply attention
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_out)  # (batch, 1, hidden*2)

        # Combine embedding with context
        decoder_input = torch.cat([embedded, context], dim=2)  # (batch, 1, embed_dim + hidden*2)

        # Decode
        decoder_out, hidden = self.decoder(decoder_input, hidden)

        # Project to vocabulary
        output = self.out(decoder_out)  # (batch, 1, vocab_size)

        return output, hidden

    def translate(self, src, max_len=200, sos_token=1, eos_token=2):
        """
        Translate CUDA code to HIP code (inference mode).

        Args:
            src: (batch, src_len) - CUDA code tokens
            max_len: Maximum output length
            sos_token: Start of sequence token ID
            eos_token: End of sequence token ID

        Returns:
            translated: (batch, output_len) - HIP code tokens
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device

            # Encode
            encoder_out, h_n = self.encode(src)
            hidden = self._bridge_hidden(h_n)

            # Start with SOS token
            decoder_input = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)

            outputs = [decoder_input]
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len):
                # Decode step
                output, hidden = self.decode_step(decoder_input, hidden, encoder_out)

                # Get prediction
                predicted = output.argmax(dim=-1)  # (batch, 1)
                outputs.append(predicted)

                # Check for EOS
                finished |= (predicted.squeeze(1) == eos_token)
                if finished.all():
                    break

                decoder_input = predicted

            translated = torch.cat(outputs, dim=1)  # (batch, output_len)
            return translated


class CharTokenizer:
    """Simple character-level tokenizer for code."""

    def __init__(self):
        self.char2id = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.id2char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.next_id = 4

    def build_vocab(self, texts):
        """Build vocabulary from list of code strings."""
        for text in texts:
            for char in text:
                if char not in self.char2id:
                    self.char2id[char] = self.next_id
                    self.id2char[self.next_id] = char
                    self.next_id += 1

    def encode(self, text, add_sos_eos=True):
        """Convert text to token IDs."""
        tokens = [self.char2id.get(c, self.char2id['<UNK>']) for c in text]
        if add_sos_eos:
            tokens = [self.char2id['<SOS>']] + tokens + [self.char2id['<EOS>']]
        return tokens

    def decode(self, tokens, skip_special=True):
        """Convert token IDs to text."""
        chars = []
        for token in tokens:
            char = self.id2char.get(token, '<UNK>')
            if skip_special and char in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                continue
            chars.append(char)
        return ''.join(chars)

    @property
    def vocab_size(self):
        return len(self.char2id)
