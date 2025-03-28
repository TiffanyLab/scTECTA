import argparse
import os
import glob
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from models import GradReverse
from models import Model
from utils import csv_load_data_together
from pprint import pprint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score




class Trainer:
    def __init__(self, params,args):
        self.args = args
        self.params = params
        self.prj_path = Path(__file__).parent.resolve()
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{params.gpu}')
        self.snum_cells, self.snum_genes, self.snum_labels, self.source_data, self.tnum_cells, self.tnum_genes, self.tnum_labels, self.target_data, self.label2id =csv_load_data_together(self, params)
        self.source_data.y = self.source_data.y.clone().detach().to(self.device).long()
        self.target_data.y = self.target_data.y.clone().detach().to(self.device).long()
        self.model = Model(self.params.dense_dim, self.params.hidden_dim, self.snum_labels,self.params.dropout_ratio).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self):
        min_loss = 1e10
        patience_cnt = 0
        loss_values = []
        best_epoch = 0

        for epoch in range(self.params.n_epochs):
            source_plogits = self.model(self.source_data.x, self.source_data.edge_index, params.source_pnum)
            train_loss = F.nll_loss(F.log_softmax(source_plogits, dim=1), self.source_data.y)
            loss = train_loss
            p = float(epoch) / params.n_epochs
            GradReverse.rate = 2. / (1. + np.exp(-10. * p)) - 1

            source_feature = self.model.feat_bottleneck(self.source_data.x, self.source_data.edge_index, params.source_pnum)
            target_feature = self.model.feat_bottleneck(self.target_data.x, self.target_data.edge_index, self.args.target_pnum)

            source_dlogits = self.model.domain_classifier(source_feature, self.source_data.edge_index)
            target_dlogits = self.model.domain_classifier(target_feature, self.target_data.edge_index)

            domain_label = torch.tensor([0] * source_feature.shape[0] + [1] * target_feature.shape[0]).to(self.device)
            domain_loss = F.cross_entropy(torch.cat([source_dlogits, target_dlogits], 0), domain_label)

            loss = loss + args.weight * domain_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.eval()

            with torch.no_grad():
                train_acc = self.evaluate(self.source_data, self.model, params.source_pnum)
                test_acc = self.evaluate(self.target_data, self.model, params.target_pnum)
                if epoch == self.params.n_epochs - 1:  # 保存模型参数
                    self.save_model()
                print(
                    f">>>>Epoch {epoch:04d}:Loss {loss:.4f}"
                    f":Train Acc:{train_acc:4f} Test Acc {test_acc:.4f}")

            loss_values.append(loss)
            torch.save(self.model.state_dict(), args.save_path + '{}.pth'.format(epoch))

            if loss_values[-1] < min_loss:
                min_loss = loss_values[-1]
                best_epoch = epoch
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt == args.patience:
                break

            files = glob.glob(args.save_path + '*.pth')

            for f in files:
                epoch_nb = int(f[2:].split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(f)
                    pass

        files = glob.glob(args.save_path + '*.pth')
        for f in files:
            epoch_nb = int(f[2:].split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(f)
        self.model.load_state_dict(torch.load(args.save_path + '{}.pth'.format(best_epoch), weights_only=False))

        acc2, macro_f1, micro_f1, weighted_f1, loss, macro_precision, micro_precision, macro_recall, micro_recall, weighted_recall,weighted_precison = self.evaluate_last(
            self.target_data, self.model, args.target_pnum)

        print(f"acc2:{acc2}, macro_f1:{macro_f1}, micro_f1:{micro_f1} , weighted_f1:{weighted_f1}, macro_precision:{macro_precision}, micro_precision:{micro_precision}, macro_recall:{macro_recall}, micro_recall:{micro_recall} , weighted_recall:{weighted_recall}, weighted_precision:{weighted_precison}")


    def evaluate(self, data, model, conv_time=0):
        model.eval()
        output = model(data.x, data.edge_index, conv_time)
        output = F.log_softmax(output, dim=1)
        pred = output.max(dim=1)[1]
        correct = pred.eq(data.y).sum().item()
        acc = correct * 1.0 / len(data.y)

        return acc


    def evaluate_last(self, data, model, conv_time=30):
        model.eval()
        output = model(data.x, data.edge_index, conv_time)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, data.y)
        pred = output.max(dim=1)[1]
        correct = pred.eq(data.y).sum().item()

        acc = correct * 1.0 / len(data.y)
        pred = pred.cpu().numpy()
        gt = data.y.cpu().numpy()

        macro_f1 = f1_score(gt, pred, average='macro')
        micro_f1 = f1_score(gt, pred, average='micro')
        weighted_f1 = f1_score(gt, pred, average='weighted')

        macro_precision = precision_score(gt, pred, average='macro', zero_division=0)
        micro_precision = precision_score(gt, pred, average='micro', zero_division=0)
        macro_recall = recall_score(gt, pred, average='macro', zero_division=0)
        micro_recall = recall_score(gt, pred, average='micro', zero_division=0)
        weighted_recall = recall_score(gt, pred, average='weighted', zero_division=0)
        weighted_precision = precision_score(gt,pred,average='weighted', zero_division=0)

        return acc, macro_f1, micro_f1, weighted_f1, loss, macro_precision, micro_precision, macro_recall, micro_recall, weighted_recall,weighted_precision

    def save_model(self):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(state, self.save_path / f"{self.params.source_disease}.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--n_epochs", type=int, default=2000,
                        help="number of training epochs")
    parser.add_argument("--dense_dim", type=int, default=30,
                        help="number of hidden gcn units")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="number of hidden gcn units")
    parser.add_argument("--threshold", type=float, default=0,
                        help="the threshold to connect edges between cells and genes")
    parser.add_argument("--exclude_rate", type=float, default=0.005,
                        help="exclude some cells less than this rate.")
    parser.add_argument("--source_disease", type=str, required=True)
    parser.add_argument("--k",type=float,default=0.002)
    parser.add_argument("--target_disease", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--unsure_rate", type=float, default=2.,
                        help="the threshold to predict unsure cell")
    parser.add_argument("--valid_rate", type=float, default=0.2)
    parser.add_argument("--use_pca", type=bool, required=True)
    parser.add_argument('--source_pnum', type=int, default=0,
                        help='the number of propagation layers on the source graph')
    parser.add_argument('--target_pnum', type=int, default=5,
                        help='the number of propagation layers on the target graph')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio')
    parser.add_argument('--weight', type=float, default=5,
                        help='trade-off parameter')
    parser.add_argument('--patience', type=int, default=15,
                        help='patience for early stopping')

    params = parser.parse_args()
    args = parser.parse_args()

    params.gpu = 0
    args.gpu = 0
    args.save_path = './'

    pprint(vars(params))

    trainer = Trainer(params , args)

    trainer.train()




