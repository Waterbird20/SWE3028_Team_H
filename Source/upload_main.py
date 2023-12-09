import torch
from common.FinanceCSVUploader import FinanceTrainer
from common.FinanceArgParser import FinanceParser
from dataset.FinanceDataset import FinanceDataset
from models.FinanceLSTM import FinanceLSTM
from models.FinanceGRU import FinanceGRU
from models.FinanceTransformer import FinanceTransformer


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = FinanceParser('./config/upload_config.yaml')

model_args = parser.parse_model_args()
data_args = parser.parse_data_args()
trainer_args = parser.parse_trainer_args()

if trainer_args.model_type == 'LSTM':
    state_high = torch.load(f'./savedmodels/{trainer_args.model_type}_{trainer_args.stock_type}_high.pth')
    model_high = FinanceLSTM(model_args)
    model_high.load_state_dict(state_high)

    state_low = torch.load(f'./savedmodels/{trainer_args.model_type}_{trainer_args.stock_type}_low.pth')
    model_low = FinanceLSTM(model_args)
    model_low.load_state_dict(state_low)

elif trainer_args.model_type == 'GRU':
    state_high = torch.load(f'./savedmodels/{trainer_args.model_type}_{trainer_args.stock_type}_high.pth')
    model_high = FinanceGRU(model_args)
    model_high.load_state_dict(state_high)

    state_low = torch.load(f'./savedmodels/{trainer_args.model_type}_{trainer_args.stock_type}_low.pth')
    model_low = FinanceGRU(model_args)
    model_low.load_state_dict(state_low)

elif trainer_args.model_type == 'TF':
    state_high = torch.load(f'./savedmodels/{trainer_args.model_type}_{trainer_args.stock_type}_high.pth')
    model_high = FinanceTransformer(model_args)
    model_high.load_state_dict(state_high)

    state_low = torch.load(f'./savedmodels/{trainer_args.model_type}_{trainer_args.stock_type}_low.pth')
    model_low = FinanceTransformer(model_args)
    model_low.load_state_dict(state_low)

dataset = FinanceDataset
trainer = FinanceTrainer(trainer_args, data_args, model_high, model_low, dataset)

trainer.start()
