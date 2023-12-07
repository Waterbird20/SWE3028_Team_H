from .FinanceArguments import *
import yaml


class FinanceParser:

    def __init__(self, file_path):
        self.arg = None
        with open(file_path) as f:
            self.arg = yaml.safe_load(f)

    def parse_model_args(self):
        return ModelArgs(output_length=self.arg['output_length'], num_layers=self.arg['num_layers'], input_size=self.arg['input_size'], hidden_size=self.arg['hidden_size'], fc_hidden_size=self.arg['fc_hidden_size'], device=self.arg['device'])

    def parse_data_args(self):
        return DataArgs(seq_length=self.arg['seq_length'], output_length=self.arg['output_length'], predict_type=self.arg['predict_type'], data_path=self.arg['data_path'])

    def parse_trainer_args(self):
        return TrainerArgs(batch_size=self.arg['batch_size'], learning_rate=self.arg['learning_rate'], num_epoch=self.arg['num_epoch'], do_train=self.arg['do_train'], do_test=self.arg['do_test'], device=self.arg['device'], is_transformer=self.arg['is_transformer'], resolution=self.arg['resolution'], stock_type=self.arg['stock_type'])
    
    def parse_transformer_args(self):
        return TransformerArgs(embed_dim=self.arg['embed_dim'], resolution=self.arg['resolution'], n_head=self.arg['n_head'], n_layer=self.arg['n_layer'], fc_hidden_size=self.arg['fc_hidden_size'], output_length=self.arg['output_length'], seq_length=self.arg['seq_length'], input_size=self.arg['input_size'])
    
    def parse_databuilder_args(self):
        return DataBuilderArgs(seq_length=self.arg['seq_length'], output_length=self.arg['output_length'], predict_type=self.arg['predict_type'])