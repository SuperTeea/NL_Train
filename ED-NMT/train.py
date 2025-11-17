from MTFraEng import MTFraEng
from Model.Encoder import Encoder
from Model.Decoder import Decoder
from Seq2Seq import Seq2Seq
from Trainer import Trainer

data = MTFraEng(batch_size=128)
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
encoder = Encoder(
    len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Decoder(
    len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                lr=0.005)
trainer = Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)