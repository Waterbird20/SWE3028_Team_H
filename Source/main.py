import torch
from common.FinanceTrainer import FinanceTrainer
from common.FinanceArguments import LSTMArgs, DataArgs, TrainerArgs
from dataset.FinanceDataset import FinanceDataset
from models.FinanceLSTM import FinanceLSTM


device = None

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model_args = LSTMArgs(output_length=1, num_layers=1, input_size=5, hidden_size=256, fc_hidden_size=128)
data_args = DataArgs(stock_id='samsung', seq_length=30, output_length=1, predict_type='high')
train_args = TrainerArgs(batch_size=8, learning_rate=1e-3, num_epoch=20, do_train=True, do_test=True)

model = FinanceLSTM(model_args)
dataset = FinanceDataset
trainer = FinanceTrainer(train_args, data_args, model, dataset)

trainer.train()
trainer.test()
