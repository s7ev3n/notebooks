from .transformer import create_transformer


src_vocab_size = 26
tgt_vocab_size = 26

model = create_transformer(src_vocab_size, tgt_vocab_size)
