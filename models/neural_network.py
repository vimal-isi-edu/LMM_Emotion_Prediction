import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def get_activation(function_type):
    if function_type == 'relu':
        return nn.ReLU()
    elif function_type == 'sigmoid':
        return nn.Sigmoid()


class Model(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(Model, self).__init__()

        self.layer_list = []
        self.input_channel = input_shape
        self.args = args

        assert isinstance(self.args, dict)
        self.activation_type = self.args['activation']
        self.num_layer = self.args['num_layer']
        latent_shape = self.args['latent_shape']
        down_sample = self.args['down_sample']

        assert self.num_layer - 1 == len(latent_shape)

        if self.num_layer == 1:
            self.layer_list.append(nn.Linear(input_shape, output_shape))
        else:
            assert self.num_layer - 1 == len(latent_shape)
            for index in range(self.num_layer - 1):
                output_channel = latent_shape[index]

                self.layer_list.append(nn.Linear(self.input_channel, output_channel))
                self.layer_list.append(get_activation(self.activation_type))

                self.input_channel = output_channel
            self.layer_list.append(nn.Linear(self.input_channel, output_shape))

        self.layer_list = nn.Sequential(*self.layer_list)

        if self.num_layer > 1 and down_sample:
            self.down_sample = nn.Linear(input_shape, output_shape)
        else:
            self.down_sample = None

    def forward(self, x):
        output = self.layer_list(x)
        if self.num_layer > 1 and self.down_sample:
            output += self.down_sample(x)
        return output


class EmbeddingDataset(Dataset):
    def __init__(self, input_data, input_label, norm=None):
        self.data = input_data
        self.label = input_label
        self.norm = norm

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.norm is None:
            return self.data[idx], self.label[idx]
        else:
            assert isinstance(self.norm, int)
            return F.normalize(self.data[idx], p=self.norm, dim=0), self.label[idx]


class NeuralNetwork:
    def __init__(self, train_data, test_data, train_label, test_label, topk=(1,)):
        self.train_data = torch.from_numpy(train_data)
        self.test_data = torch.from_numpy(test_data)
        self.train_label = train_label
        self.test_label = test_label
        self.topk = topk

        self.input_shape = self.train_data.shape[1]
        self.output_shape = self.train_label.shape[1]

        self.train_label = torch.from_numpy(np.argmax(self.train_label, axis=1))
        self.test_label = torch.from_numpy(np.argmax(self.test_label, axis=1))

    def get_loss(self):
        return nn.CrossEntropyLoss()

    def evaluate_model(self, input_model, input_device, input_dataloader):
        correct_item_count = 0

        total_item_count = 0
        input_model.eval()

        with torch.no_grad():
            for batch_data, batch_label in input_dataloader:
                batch_data = batch_data.to(input_device)
                batch_label = batch_label.to(input_device)

                logits = input_model(batch_data)

                _, predicted = torch.max(logits.data, 1)
                total_item_count += batch_label.size(0)
                correct_item_count += (predicted == batch_label).sum().item()

            result = correct_item_count / total_item_count

        return result

    def predict(self, args):
        if args['logger'] is not None:
            logger = args['logger']
        else:
            logger = None

        model_save_path = args['save_path']
        epoch_num = 50
        lr = 0.03
        device = 'cuda'
        loss_function = self.get_loss()

        model = Model(self.input_shape, self.output_shape, args).to(device)
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        train_set = EmbeddingDataset(self.train_data, self.train_label, norm=args['norm'])
        train_dataloader = DataLoader(train_set, batch_size=1024, shuffle=True)

        test_set = EmbeddingDataset(self.test_data, self.test_label, norm=args['norm'])
        test_dataloader = DataLoader(test_set, batch_size=1024, shuffle=False)

        output_dict = {}
        for epoch in range(epoch_num):
            model.train()
            for i, (batch_data, batch_label) in enumerate(train_dataloader):
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)

                logits = model(batch_data)
                loss = loss_function(logits, batch_label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_metric = self.evaluate_model(model, device, train_dataloader)
            test_metric = self.evaluate_model(model, device, test_dataloader)

            if logger:
                logger.log({
                    'epoch': epoch,
                    'loss': loss.item(),
                    'train_metric': train_metric,
                    'test_metric': test_metric})
            # else:
                # print(epoch, f'{train_metric:.3f}', f'{test_metric:.3f}', f'{lr:.4f}, loss:{loss.item()}')

            output_dict['result'] = [test_metric * 100]

        if model_save_path:
            torch.save(model.state_dict(), model_save_path)

        return output_dict
