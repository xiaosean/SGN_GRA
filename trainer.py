
import time
import os
import os.path as osp
import csv
import numpy as np
import random

import shutil
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from data import NTUDataLoaders, PseudoDataset, collate_fn_ssl
from model_SGN import LabelSmoothingLoss, Linear_clf, save_checkpoint
from util import write_training_log , t_sne_vis

from torch.utils.data import Dataset, DataLoader

class Trainer():
    def __init__(self, backbone, train_loader, test_loader, train_subloader, tsne_loader, config, situation=1):
        """
        backbone:           Feature extractor of action recognition network
        args:               Setting during the setting
        """

        self.dataset = config["dataset"]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.backbone = backbone.to(self.device)
        self.num_epochs = config["max-epoches"]
        self.test_freq = config["test-freq"]
        self.data_volume = config["data-volume"]
        self.ckpt_name = config["ckpt-name"]
        self.clf_layer_num = 1
        self.config = config
        self.situation = situation # normal, jm, fm situation

        if self.dataset == "NTU":
            self.num_classes = 60
        elif self.dataset == "NUCLA":
            self.num_classes = 10
        

        # Get training set dataloader and testing dataloader
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_subloader = train_subloader
        self.tsne_loader = tsne_loader

        # Parameter of training
        self.batch_size = config["batch-size"]
        self.lr = config["lr"]
        self.weight_decay = config["weight-decay"]
        self.test_freq = config["test-freq"]
        self.fulldata = config["Full-Data"]

        # Record 
        self.best_acc = 0
        self.select_acc = 0
        self.best_epoch = 0

    def test(self, testloader, clf):
        """
        # Test the test dataset with backbone network and linear classifier for validation.
        """
        criterion = LabelSmoothingLoss(self.num_classes, smoothing=0.1).cuda()
        
        self.backbone.eval()
        clf.eval()
        test_clf_loss = 0
        correct = 0
        total = 0
        print(self.situation)
        if self.situation == 1:
            print("Testing on situtaion 1: Complete Skeleton Data.")
        elif self.situation == 2:
            print("Testing on situtaion 2: Joint Mask Situation.")
        elif self.situation == 3:
            print("Testing on situtaion 3: Frame Mask Situation.")

        with torch.no_grad():
            t = tqdm(enumerate(testloader), total=len(testloader), desc='Loss: **** | Test Acc: ****% ',
                    bar_format='{desc}{bar}{r_bar}')
            for batch_idx, (inputs, targets) in t:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                representation, _ = self.backbone(inputs)

                representation = representation.view(representation.shape[0], representation.shape[1]).contiguous()
                raw_scores = clf(representation)
                clf_loss = criterion(raw_scores, targets)
                test_clf_loss += clf_loss.item()*inputs.shape[0]

                _, predicted = raw_scores.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                t.set_description('Loss: %.3f | Test Acc: %.3f%% ' % (test_clf_loss / ((batch_idx + 1)*inputs.shape[0]), 100. * correct / total))
                testing_loss = test_clf_loss / ((batch_idx + 1)*inputs.shape[0])
                acc = 100. * correct / total

        return acc, testing_loss

    def train_semi(self, epochs):
        clf = Linear_clf(self.num_classes, self.backbone.representation_dim, self.clf_layer_num).to(self.device)
        criterion = LabelSmoothingLoss(self.num_classes, smoothing=0.1).cuda()
        semi_optimizer = optim.AdamW(list(self.backbone.parameters()) + list(clf.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        self.best_acc = 0
        self.best_epoch = 0
        for epoch in range(epochs):
            clf.train()
            self.backbone.train()
            #print('Train Backbone + CLF Epoch: %d' % epoch)
            
            train_loss = 0
            positive_loss = 0
            total = 0
            correct = 0
            rep_loss = 0
            clf_loss = 0

            if self.fulldata :
                print("Training on full data")
                t = tqdm(enumerate(self.train_loader), desc='Loss: **** ', total=len(self.train_loader), bar_format='{desc}{bar}{r_bar}')
            else:
                t = tqdm(enumerate(self.train_subloader), desc='Loss: **** ', total=len(self.train_subloader), bar_format='{desc}{bar}{r_bar}')
            
            for batch_idx, (x, target, x1, gt1, x2, gt2) in t:
                semi_optimizer.zero_grad()
                x1,  targets = x.to(self.device), target.to(self.device)
                (representation1, _) = self.backbone(x1)

                representation1 = representation1.view(representation1.shape[0], representation1.shape[1]).contiguous()
                raw_scores = clf(representation1)
                class_loss = criterion(raw_scores, targets)
                loss = class_loss # 66.265
                loss.backward()
                semi_optimizer.step()
                
                _, predicted = raw_scores.max(1)
                train_loss += loss.item()
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                clf_loss += class_loss.item()*x1.shape[0]

                train_loss = clf_loss / ((batch_idx + 1)*x1.shape[0])

                t.set_description('Semi %.0f | Train Loss: %.3f | Train Acc: %.3f%% ' % (epoch, train_loss, 100. * correct / total))
             
            if (self.test_freq > 0) and (epoch % self.test_freq == (self.test_freq - 1)):
                acc, testing_loss = self.test(self.test_loader, clf)
                if acc >= self.best_acc:
                    self.best_acc = acc
                    self.best_epoch = epoch
                    save_checkpoint(self.backbone, clf, epoch, self.config, os.path.basename(__file__), f"semi/best_{epoch}.pth")
                    save_checkpoint(self.backbone, clf, epoch, self.config, os.path.basename(__file__), f"semi/best.pth")
                    # write data in HackMD table foramt as log
                    write_training_log(
                        self.config,
                        100. * correct / total,
                        train_loss,
                        testing_loss,
                        self.best_acc,
                        self.best_epoch,
                        0,
                        0,
                        "semi"
                    )

                print(f"current acc:{acc}, best_acc: {self.best_acc} (epoch:{self.best_epoch})")

    
    def pseudo_label_new(self, num_epochs, cndf_ratio = 0.8):

        clf = Linear_clf(self.num_classes, self.backbone.representation_dim, self.clf_layer_num).to(self.device)
        clf.load_state_dict(torch.load(f'./checkpoint/{self.dataset}/{self.config["case"]}/{self.config["data-volume"]}/{self.ckpt_name}/semi/best.pth')["clf"])
        print(torch.load(f'./checkpoint/{self.dataset}/{self.config["case"]}/{self.config["data-volume"]}/{self.ckpt_name}/semi/best.pth')["epoch"])
        self.select_acc, testing_loss = self.test(self.test_loader, clf)
        save_checkpoint(self.backbone, clf, 0, self.config, os.path.basename(__file__), f"pseudo/select.pth")
        save_checkpoint(self.backbone, clf, 0, self.config, os.path.basename(__file__), f"pseudo/current.pth")
        
        criterion = LabelSmoothingLoss(self.num_classes, smoothing=0.1).cuda()
        mse = nn.MSELoss()
        semi_optimizer = optim.AdamW(list(self.backbone.parameters()) + list(clf.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        
        softmax = nn.Softmax(dim=-1)
        for epoch in range(num_epochs):
            
            if epoch > 0:
                #print(f"Load select backbone and clf...")
                self.backbone.load_state_dict(torch.load(f'./checkpoint/{self.dataset}/{self.config["case"]}/{self.config["data-volume"]}/{self.ckpt_name}/pseudo/select.pth')["net"])
                clf.load_state_dict(torch.load(f'./checkpoint/{self.dataset}/{self.config["case"]}/{self.config["data-volume"]}/{self.ckpt_name}/pseudo/select.pth')["clf"])
            clf.eval()
            self.backbone.eval()
            
            sample = 0
            correct = 0
            t = tqdm(enumerate(self.train_loader), desc='Loss: **** ', total=len(self.train_loader), bar_format='{desc}{bar}{r_bar}')
            pseudo_x = 0
            psize = 0
            for batch_idx, (x, target, x1, gt1, x2, gt2) in t:
                x, target = x.to(self.device), target.to(self.device)
                with torch.no_grad():
                    (representation1, _) = self.backbone(x)
                    representation1 = representation1.view(representation1.shape[0], representation1.shape[1]).contiguous()
                    pesudo_label = clf(representation1)
                
                pesudo_label = softmax(pesudo_label)
                pseudo_score, pesudo_label = pesudo_label.max(1)
                indices = torch.nonzero((pseudo_score>cndf_ratio))
                sample+= len(indices)
                
                if len(indices) == 0:
                    continue
                if type(pseudo_x) == int:
                    pseudo_x = torch.cat([i for i in x[indices]], 0)
                    pseudo_y = torch.cat([i for i in pesudo_label[indices]], 0)
                else:
                    temp = torch.cat([i for i in x[indices]], 0)
                    pseudo_x = torch.cat([pseudo_x, temp])
                    temp = torch.cat([i for i in pesudo_label[indices]], 0)
                    pseudo_y = torch.cat([pseudo_y, temp])


                correct += pesudo_label[indices].eq(target[indices]).sum().item()

                t.set_description('over %.2f%% Confidence Data: %.3f%% | correct rate: %.3f%%' % (cndf_ratio*100, sample/((batch_idx + 1)*x1.shape[0])*100, correct/sample*100))
                psize=sample/((batch_idx + 1)*x1.shape[0])*100
            if sample > 0:
                pseudo_data = PseudoDataset(pseudo_x, pseudo_y)
                pseudo_loader = DataLoader(pseudo_data, batch_size = self.batch_size, shuffle = True, collate_fn=collate_fn_ssl)

            #print(f"Load current backbone and clf...")
            self.backbone.load_state_dict(torch.load(f'./checkpoint/{self.dataset}/{self.config["case"]}/{self.config["data-volume"]}/{self.ckpt_name}/pseudo/current.pth')["net"])
            clf.load_state_dict(torch.load(f'./checkpoint/{self.dataset}/{self.config["case"]}/{self.config["data-volume"]}/{self.ckpt_name}/pseudo/current.pth')["clf"])
            
            clf.train()
            self.backbone.train()
            for e in range(200):
                train_loss = 0
                total = 0
                correct = 0
                rclf_loss = 0
                rconsis_loss = 0
                
                
                if sample > 0:
                    t = tqdm(enumerate(pseudo_loader), desc='Loss: **** ', total=len(pseudo_loader), bar_format='{desc}{bar}{r_bar}')
                    for pseudo_batch_idx, (x, target, x1, gt1, x2, gt2) in t:

                        x, target, x1, gt1, x2, gt2 = x.to(self.device), target.to(self.device), x1.to(self.device), gt1.to(self.device), x2.to(self.device), gt2.to(self.device)
                        
                        (rep, _) = self.backbone(x)
                        rep = rep.view(rep.shape[0], rep.shape[1]).contiguous()
                        raw_scores = clf(rep)
                        class_loss = criterion(raw_scores, target)

                        (rep1, _) = self.backbone(x1)
                        rep1 = rep1.view(rep1.shape[0], rep1.shape[1]).contiguous()

                        (rep2, _) = self.backbone(x2)
                        rep2 = rep2.view(rep2.shape[0], rep2.shape[1]).contiguous()

                        #consis_loss = mse(rep, rep1) + mse(rep, rep2) + mse(rep1, rep2)
                        consis_loss = mse(rep, rep1) + mse(rep, rep2)

                        semi_optimizer.zero_grad()
                        #loss = class_loss # 66.265
                        loss = class_loss + consis_loss # 66.265
                        loss.backward()
                        semi_optimizer.step()

                        rclf_loss = class_loss.item()
                        rconsis_loss = consis_loss.item()
                        _, predicted = raw_scores.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target).sum().item()
                        
                        #t.set_description('Pseudo Label %.0f | Clf Loss: %.3f | Train Acc: %.3f%% ' % (e, clf_loss / ((pseudo_batch_idx + 1)*x2.shape[0]), 100. * correct / total))
                        t.set_description('P-Label %.0f-%.0f | Clf Loss: %.3f | Consis Loss: %.3f | Train Acc: %.3f%% ' % (epoch+1, e+1, rclf_loss , rconsis_loss, 100. * correct / total))
                
                rclf_loss = 0
                rconsis_loss = 0
                t = tqdm(enumerate(self.train_subloader), desc='Loss: **** ', total=len(self.train_subloader), bar_format='{desc}{bar}{r_bar}')
                for batch_idx, (x, target, x1, gt1, x2, gt2) in t:
                    
                    x, target, x1, gt1, x2, gt2 = x.to(self.device), target.to(self.device), x1.to(self.device), gt1.to(self.device), x2.to(self.device), gt2.to(self.device)
                    
                    (rep, _) = self.backbone(x)
                    rep = rep.view(rep.shape[0], rep.shape[1]).contiguous()
                    raw_scores = clf(rep)
                    class_loss = criterion(raw_scores, target)

                    (rep1, _) = self.backbone(x1)
                    rep1 = rep1.view(rep1.shape[0], rep1.shape[1]).contiguous()

                    (rep2, _) = self.backbone(x2)
                    rep2 = rep2.view(rep2.shape[0], rep2.shape[1]).contiguous()

                    #consis_loss = mse(rep, rep1) + mse(rep, rep2) + mse(rep1, rep2)
                    consis_loss = mse(rep, rep1) + mse(rep, rep2)
                    semi_optimizer.zero_grad()
                    #loss = class_loss # 66.265
                    loss = (class_loss + consis_loss) # 66.265
                    loss.backward()
                    semi_optimizer.step()

                    rclf_loss = class_loss.item()
                    rconsis_loss = consis_loss.item()
                    _, predicted = raw_scores.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

                    t.set_description('Label %.0f-%.0f | Clf Loss: %.3f | Consis Loss: %.3f | Train Acc: %.3f%% ' % (epoch+1, e+1, rclf_loss, rconsis_loss ,100. * correct / total))

                save_checkpoint(self.backbone, clf, epoch, self.config, os.path.basename(__file__), f"pseudo/current.pth")
                if (self.test_freq > 0) and (epoch % self.test_freq == (self.test_freq - 1)):
                    acc, testing_loss = self.test(self.test_loader, clf)
                    if acc > self.select_acc:
                        self.select_acc = acc
                        print("Save select weights")
                        save_checkpoint(self.backbone, clf, epoch, self.config, os.path.basename(__file__), f"pseudo/select.pth")

                    if acc > self.best_acc:
                        self.best_acc = acc
                        self.best_epoch = epoch
                        save_checkpoint(self.backbone, clf, epoch, self.config, os.path.basename(__file__), f"pseudo/best_{epoch}.pth")
                        save_checkpoint(self.backbone, clf, epoch, self.config, os.path.basename(__file__), f"pseudo/best.pth")
                        # write data in HackMD table foramt as log
                        write_training_log(
                            self.config,
                            0.0000,
                            train_loss,
                            testing_loss,
                            self.best_acc,
                            self.best_epoch,
                            e+1,
                            psize,
                            "pseudo"
                        )
                        print(f"current acc:{acc}, best_acc: {self.best_acc}, select_acc: {self.select_acc} (epoch:{self.best_epoch+1})")
                        break
                    else:
                        print(f"current acc:{acc}, best_acc: {self.best_acc}, select_acc: {self.select_acc} (epoch:{self.best_epoch+1})")
    """
    def pseudo_label_normal(self, num_epochs, cndf_ratio = 0.8):

        clf = Linear_clf(self.num_classes, self.backbone.representation_dim, self.clf_layer_num).to(self.device)
                
        criterion = LabelSmoothingLoss(self.num_classes, smoothing=0.1).cuda()
        mse = nn.MSELoss()
        semi_optimizer = optim.AdamW(list(self.backbone.parameters()) + list(clf.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        
        softmax = nn.Softmax(dim=-1)
        for epoch in range(num_epochs):
           
            clf.eval()
            self.backbone.eval()
            
            sample = 0
            correct = 0
            t = tqdm(enumerate(self.train_loader), desc='Loss: **** ', total=len(self.train_loader), bar_format='{desc}{bar}{r_bar}')
            pseudo_x = 0
            #for batch_idx, (x, target) in t:
            for batch_idx, (x, target, x1, gt1, x2, gt2) in t:
                x, target = x.to(self.device), target.to(self.device)
                with torch.no_grad():
                    (representation1, _) = self.backbone(x)
                    representation1 = representation1.view(representation1.shape[0], representation1.shape[1]).contiguous()
                    pesudo_label = clf(representation1)
                
                pesudo_label = softmax(pesudo_label)
                pseudo_score, pesudo_label = pesudo_label.max(1)
                indices = torch.nonzero((pseudo_score>cndf_ratio))
                sample+= len(indices)
                
                if len(indices) == 0:
                    continue
                if type(pseudo_x) == int:
                    pseudo_x = torch.cat([i for i in x[indices]], 0)
                    pseudo_y = torch.cat([i for i in pesudo_label[indices]], 0)
                else:
                    temp = torch.cat([i for i in x[indices]], 0)
                    pseudo_x = torch.cat([pseudo_x, temp])
                    temp = torch.cat([i for i in pesudo_label[indices]], 0)
                    pseudo_y = torch.cat([pseudo_y, temp])


                correct += pesudo_label[indices].eq(target[indices]).sum().item()

                t.set_description('over %.2f%% Confidence Data: %.3f%% | correct rate: %.3f%%' % (cndf_ratio*100, sample/((batch_idx + 1)*x1.shape[0])*100, correct/sample*100))
            if sample > 0:
                pseudo_data = PseudoDataset(pseudo_x, pseudo_y)
                pseudo_loader = DataLoader(pseudo_data, batch_size = self.args.batch_size, shuffle = True, collate_fn=collate_fn_ssl)

            clf.train()
            self.backbone.train()
            for e in range(1):
                train_loss = 0
                total = 0
                correct = 0
                rclf_loss = 0
                rconsis_loss = 0
                
                if sample > 0:
                    t = tqdm(enumerate(pseudo_loader), desc='Loss: **** ', total=len(pseudo_loader), bar_format='{desc}{bar}{r_bar}')
                    for pseudo_batch_idx, (x, target, x1, gt1, x2, gt2) in t:

                        x, target, x1, gt1, x2, gt2 = x.to(self.device), target.to(self.device), x1.to(self.device), gt1.to(self.device), x2.to(self.device), gt2.to(self.device)
                        
                        (rep, _) = self.backbone(x)
                        rep = rep.view(rep.shape[0], rep.shape[1]).contiguous()
                        raw_scores = clf(rep)
                        class_loss = criterion(raw_scores, target)

                        (rep1, _) = self.backbone(x1)
                        rep1 = rep1.view(rep1.shape[0], rep1.shape[1]).contiguous()

                        (rep2, _) = self.backbone(x2)
                        rep2 = rep2.view(rep2.shape[0], rep2.shape[1]).contiguous()

                        consis_loss = mse(rep, rep1) + mse(rep, rep2) + mse(rep1, rep2)

                        semi_optimizer.zero_grad()
                        loss = class_loss # 66.265
                        #loss = class_loss + consis_loss # 66.265
                        loss.backward()
                        semi_optimizer.step()

                        rclf_loss = class_loss.item()
                        rconsis_loss = consis_loss.item()
                        _, predicted = raw_scores.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target).sum().item()
                        
                        #t.set_description('Pseudo Label %.0f | Clf Loss: %.3f | Train Acc: %.3f%% ' % (e, clf_loss / ((pseudo_batch_idx + 1)*x2.shape[0]), 100. * correct / total))
                        t.set_description('P-Label %.0f | Clf Loss: %.3f | Consis Loss: %.3f | Train Acc: %.3f%% ' % (e, rclf_loss , rconsis_loss, 100. * correct / total))

                rclf_loss = 0
                rconsis_loss = 0
                t = tqdm(enumerate(self.train_subloader), desc='Loss: **** ', total=len(self.train_subloader), bar_format='{desc}{bar}{r_bar}')
                for batch_idx, (x, target, x1, gt1, x2, gt2) in t:
                    
                    x, target, x1, gt1, x2, gt2 = x.to(self.device), target.to(self.device), x1.to(self.device), gt1.to(self.device), x2.to(self.device), gt2.to(self.device)
                    
                    (rep, _) = self.backbone(x)
                    rep = rep.view(rep.shape[0], rep.shape[1]).contiguous()
                    raw_scores = clf(rep)
                    class_loss = criterion(raw_scores, target)

                    (rep1, _) = self.backbone(x1)
                    rep1 = rep1.view(rep1.shape[0], rep1.shape[1]).contiguous()

                    (rep2, _) = self.backbone(x2)
                    rep2 = rep2.view(rep2.shape[0], rep2.shape[1]).contiguous()

                    consis_loss = mse(rep, rep1) + mse(rep, rep2) + mse(rep1, rep2)

                    semi_optimizer.zero_grad()
                    loss = class_loss # 66.265
                    #loss = (class_loss + consis_loss) # 66.265
                    loss.backward()
                    semi_optimizer.step()

                    rclf_loss = class_loss.item()
                    rconsis_loss = consis_loss.item()
                    _, predicted = raw_scores.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

                    t.set_description('Label %.0f | Clf Loss: %.3f | Consis Loss: %.3f | Train Acc: %.3f%% ' % (e, rclf_loss, rconsis_loss ,100. * correct / total))

                if (self.test_freq > 0) and (e % self.test_freq == (self.test_freq - 1)):
                    save_checkpoint(self.backbone, clf, self.critic, epoch, self.args, os.path.basename(__file__), f"pseudo/{epoch}.pth")
                    acc, testing_loss = self.test(self.test_loader, clf)
                    if acc > self.select_acc:
                        self.select_acc = acc
                        print("Save select weights")
                        save_checkpoint(self.backbone, clf, self.critic, epoch, self.args, os.path.basename(__file__), f"pseudo/select.pth")

                    if acc > self.best_acc:
                        self.best_acc = acc
                        self.best_epoch = epoch
                        save_checkpoint(self.backbone, clf, self.critic, epoch, self.args, os.path.basename(__file__), f"pseudo/best_{epoch}.pth")
                        save_checkpoint(self.backbone, clf, self.critic, epoch, self.args, os.path.basename(__file__), f"pseudo/best.pth")
                        # write data in HackMD table foramt as log
                        write_training_log(
                            self.args,
                            0.0000,
                            train_loss,
                            testing_loss,
                            self.best_acc,
                            self.best_epoch,
                            e,
                            "pseudo"
                        )
                        print(f"current acc:{acc}, best_acc: {self.best_acc}, select_acc: {self.select_acc} (epoch:{self.best_epoch})")
                        
                    else:
                        print(f"current acc:{acc}, best_acc: {self.best_acc}, select_acc: {self.select_acc} (epoch:{self.best_epoch})")
            """