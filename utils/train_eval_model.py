import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score


class EarlyStopping:
    def __init__(self, patience=None, descend_mode=True, save_path=None):
        self.patience = patience
        self.descend_mode = descend_mode
        self.save_path = save_path
        self.monitor = np.inf
        self.stopping = False
        self.counter = 0

    def __call__(self, value, model=None):
        if not self.descend_mode:
            mvalue = -value
            mmonitor = -self.monitor
        else:
            mvalue = value
            mmonitor = self.monitor

        if self.monitor > mvalue:
            if (self.save_path is not None) & (model is not None):
                torch.save(model.state_dict(), self.save_path)
                print(
                    'Value improved from {:.5f} to {:.5f}, saving model to {}'.
                    format(mmonitor, value, self.save_path))
            else:
                print('Value improved from {:.5f} to {:.5f}'.format(
                    mmonitor, value))
            self.monitor = mvalue
            self.counter = 0
        else:
            self.counter += 1
            print('Value did not improved from {:.5f}. EarlyStopping: {}/{}'.
                  format(mmonitor, self.counter, self.patience))
            if self.counter == self.patience:
                print('[Info] EarlyStopping.')
                self.stopping = True

    def reset(self):
        self.stopping = False
        self.monitor = np.inf
        self.counter = 0


class Train_Eval_Model:
    def __init__(self,
                 model,
                 optim='adam',
                 lr=0.0001,
                 loss_func=None,
                 device='cpu',
                 lr_scheduler=True):

        self.model = model.to(device)
        self.lr = lr
        self.device = device
        self.lr_scheduler = lr_scheduler

        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.lr)
        elif optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.lr,
                                             momentum=0.9,
                                             weight_decay=0.0001,
                                             nesterov=True)
        if loss_func is not None:
            self.loss_func = loss_func
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()

        if lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[60], gamma=0.1)

        print('[Info] #Para: ', self.get_param_num())

    def get_param_num(self, require_grad_param=True):
        if require_grad_param:
            return sum(param.numel() for param in self.model.parameters()
                       if param.requires_grad)
        else:
            return sum(param.numel() for param in self.model.parameters())

    def _train_eval_step(self, dataloader, train=True, eval=False):
        losses, preds, labels = [], [], []

        if train:
            self.model.train()
        else:
            self.model.eval()

        for data in dataloader:
            y = data[-2].to(self.device)
            inputs = [
                d.to(self.device) for n, d in enumerate(data[:-1])
                if n in self.index
            ]

            if train:
                self.optimizer.zero_grad()
            outputs = self.model(*inputs)

            if type(outputs) is tuple:
                outputs = outputs[0]

            loss = self.loss_func(outputs, y)
            if train:
                loss.backward(retain_graph=True)
                self.optimizer.step()
            losses.append(loss.item())

            if eval:
                preds.extend(outputs.cpu().detach().numpy())
                labels.extend(y.cpu().numpy())

        avg_loss = np.average(losses)

        if eval:
            preds = np.array(preds).argmax(axis=1)
            labels = np.array(labels)
            accuracy = accuracy_score(labels, preds) * 100
            return avg_loss, accuracy, labels, preds
        else:
            return avg_loss

    def train_model(self,
                    train_loader,
                    val_loader,
                    test_loader,
                    index,
                    epoch_num,
                    earlystop=None,
                    writer=None,
                    gate_his=False):

        train_loss_his, train_acc_his, val_loss_his, val_acc_his, test_loss_his, test_acc_his = [], [], [], [], [], []
        self.index = index

        if earlystop is not None:
            earlystop.reset()

        for epoch in range(epoch_num):
            start_time = time.time()

            train_loss, train_acc, _, _ = self._train_eval_step(train_loader,
                                                                train=True,
                                                                eval=True)
            val_loss, val_acc, _, _ = self._train_eval_step(val_loader,
                                                            train=False,
                                                            eval=True)
            test_loss, test_acc, _, _ = self._train_eval_step(test_loader,
                                                              train=False,
                                                              eval=True)

            train_loss_his.append(train_loss)
            train_acc_his.append(train_acc)
            val_loss_his.append(val_loss)
            val_acc_his.append(val_acc)
            test_loss_his.append(test_loss)
            test_acc_his.append(test_acc)

            print('[Epoch:{}/{}] train_loss: {:.5f} train_acc: {:.2f} val_loss:{:.5f} val_acc: {:.2f} test_loss:{:.5f} test_acc: {:.2f} time: {:.2f}'.\
                format(epoch+1,epoch_num,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc,time.time()-start_time))

            if earlystop is not None:
                earlystop(val_acc, self.model)
                if earlystop.stopping:
                    break

            if writer is not None:
                writer.add_scalars(
                    'Summary', {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'test_loss': test_loss,
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                        'test_acc': test_acc
                    }, epoch)
                if gate_his is True:
                    for num, block in enumerate(self.model.blocks):
                        if block.ssgc.gate is True:
                            writer.add_scalars('gate', {
                                ('g' + str(num)):
                                float(block.ssgc.g.cpu().detach())
                            }, epoch)
                        else:
                            print('[Error] gate not activated.')
                            break

            if self.lr_scheduler:
                self.scheduler.step()

        if writer is not None:
            writer.close()

        print('[Info] Training finished.')
        print('[Info] Test performance: Acc: {}'.format(max(test_acc_his)))
        
        if earlystop is not None:
            self.model.load_state_dict(torch.load(earlystop.save_path))
            loss, accuracy, labels, preds = self._train_eval_step(test_loader,
                                                                  train=False,
                                                                  eval=True)
            recall = recall_score(labels, preds, average='macro') * 100
            print('*' * 20)
            print('Accuracy Score:', accuracy)
            print('Recall:', recall)
            return train_loss_his, train_acc_his, val_loss_his, val_acc_his, test_loss_his, test_acc_his, labels, preds

        return train_loss_his, train_acc_his, val_loss_his, val_acc_his, test_loss_his, test_acc_his