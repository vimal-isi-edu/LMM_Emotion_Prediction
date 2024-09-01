import numpy as np
from sklearn.metrics import average_precision_score
import torch


class LinearModel:
    def __init__(self, train_data, test_dasta, train_label, test_label, topk=(1,), multi_label=False):
        self.train_data = train_data
        self.test_data = test_dasta
        self.train_label = train_label
        self.test_label = test_label
        self.topk = topk

        self.multi_label = multi_label

        if self.multi_label:
            assert max(self.topk) == 1

        # self.train_label = np.clip(self.train_label, 0.1, 0.9)
        # self.test_label = np.clip(self.test_label, 0.1, 0.9)

    def predict(self, args=None, to_numpy=True):
        a_pinv = np.linalg.pinv(self.train_data)
        transform_mat = np.dot(a_pinv, self.train_label)
        prediction_output = np.matmul(self.test_data, transform_mat)

        output_dict = {}
        if not self.multi_label:
            test_label_index = np.argmax(self.test_label, axis=-1)
            predict_label_index = np.argmax(prediction_output, axis=1)

            test_label_index_torch = torch.from_numpy(test_label_index)
            prediction_output = torch.from_numpy(prediction_output)
            maxk = max(self.topk)
            _, pred = torch.topk(prediction_output, k=maxk, dim=-1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(test_label_index_torch.reshape(1, -1).expand_as(pred))

            res = []
            for k in self.topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / test_label_index_torch.size(0)))

            if to_numpy:
                res = [item.item() for item in res]

            output_dict['result'] = res
            output_dict['true_list'] = test_label_index
            output_dict['predict_list'] = predict_label_index
        else:
            # prediction_output = np.clip(prediction_output, 0.0, 1.0)

            mAP_list = []
            for column_idx in range(self.test_label.shape[1]):
                mAP_list.append(average_precision_score(
                    self.test_label[:, column_idx], prediction_output[:, column_idx]))
            output_dict['result'] = [sum(mAP_list) / len(mAP_list)]

        return output_dict
