import torch
from common.FinanceTrainer import FinanceTrainer
from common.FinanceArguments import ModelArgs, DataArgs, TrainerArgs
from common.FinanceArgParser import FinanceParser
from dataset.FinanceDataset import FinanceDataset
from models.FinanceLSTM import FinanceLSTM
from models.FinanceGRU import FinanceGRU
from models.FinanceTransformer import FinanceTransformer


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = FinanceParser('./config/config.yaml')

model_args = parser.parse_model_args()
data_args = parser.parse_data_args()
trainer_args = parser.parse_trainer_args()

model = FinanceLSTM(model_args)
dataset = FinanceDataset
trainer = FinanceTrainer(trainer_args, data_args, model, dataset)

trainer.start()
