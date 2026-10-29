# gotham-LSTM
Reading order (suggested)
dataloader.ipynb, skim the section headers: see the whole pipeline in 2 minutes.
AWDLSTM.py → lstm_forward / lstm_backward: plain LSTM calculus. Gate order [i | f | g | o].
AWDLSTM.forward: where each dropout mask is applied.
AWDLSTM.loss_and_grads: softmax-CE gradient, AR/TAR terms, tied-embedding gradient (two contributions: decoder side + input side), DropConnect gradient masking.
gradcheck: why we trust step 4.
SGD / ASGD / averaged / fit: NT-ASGD, the non-monotone trigger.
train_epoch: variable-length BPTT and LR rescaling.

It is an implementation from the paper AWDLSTM:- 

AWD-LSTM (Merity et al. 2017, arXiv:1708.02182) - hand-written forward and backward passes, no torch, no autograd. Trained on WikiText-2 and this is done by me.