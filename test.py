import torch
import numpy as np
import torch
from torch import nn
import warnings

class Consis_Reg(nn.Module):

    def __init__(self, metric="mse"):
        super(Consis_Reg, self).__init__()
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.MSELoss = nn.MSELoss(reduction='none')
        self.type = metric

    def forward(self, representations, targets):
        if self.type == "mse":
            A = representations
            L = targets
            A = A.unsqueeze(1)
            A_hat = A.transpose(0, 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mse = self.MSELoss(A, A_hat)
            
            L = L.expand(L.shape[0],L.shape[0])
            L_hat = torch.stack([torch.where(L[i] == L[i][i], 1, 0) for i in range(L.shape[0])]).float()

            mm = torch.mm(L_hat,mse.sum(dim=2))

            diagonal = torch.diagonal(mm, 0)

            consis_reg = (torch.div(diagonal, L_hat.sum(dim=-1))).sum()

            return consis_reg
        
        elif self.type == "cos":
            A = representations
            L = targets
            A = A.unsqueeze(1)
            A_hat = A.transpose(0, 1)
            cos = self.cossim(A,A_hat)
            d = cos.shape[0]
            cos[..., range(d), range(d)] = float('-inf')
            cos = torch.exp(cos)
            
            L = L.expand(L.shape[0],L.shape[0])
            L_hat = torch.stack([torch.where(L[i] == L[i][i], 1, 0) for i in range(L.shape[0])]).float()

            mm = torch.mul(L_hat,cos)
            consis_reg = -torch.log(mm.sum(dim=-1)/cos.sum(dim=-1))
            consis_reg[consis_reg == float('inf')] = 0
            return consis_reg.mean()


                
A = [[[1,2],[1,2]],
     [[3,4],[3,4]],
     [[5,6],[5,6]],
     [[7,8],[7,8]]]

B = [[[1,2],
     [1,2]],
     [[5,6],
     [5,6]],
     [[9,10],
     [9,10]]]
L = [0,1,0,0]

A = torch.tensor(A).float()
A = A.view(A.shape[0], -1)
B = torch.tensor(B).float()
B = B.view(B.shape[0], -1)
L = torch.tensor(L).float()

critic = Consis_Reg(metric="cos")
print(critic(A, L))


def pseudo_label_fixmatch(self, num_epochs, cndf_ratio = 0.8):

    clf = Linear_clf(self.num_classes, self.backbone.representation_dim, self.clf_layer_num).to(self.device)
    self.backbone.load_state_dict(torch.load(f"./checkpoint/{self.args.dataset}/{self.args.case}/{self.args.data_volume}/{self.args.ckpt_name}/semi/best.pth")["net"])
    clf.load_state_dict(torch.load(f"./checkpoint/{self.args.dataset}/{self.args.case}/{self.args.data_volume}/{self.args.ckpt_name}/semi/best.pth")["clf"])
    save_checkpoint(self.backbone, clf, self.critic, 0, self.args, os.path.basename(__file__), f"pseudo/select.pth")

    #self.select_acc, testing_loss = self.test(self.test_loader, clf)

    criterion = LabelSmoothingLoss(self.num_classes, smoothing=0.1).cuda()
    consis_criterion = Consis_Reg()
    semi_optimizer = optim.AdamW(list(self.backbone.parameters()) + list(clf.parameters()), lr=self.args.lr, weight_decay=self.args.weight_decay)
    
    softmax = nn.Softmax(dim=-1)

    for epoch in range(num_epochs):
        print()
        print(f"Load select backbone and clf...")
        self.backbone.load_state_dict(torch.load(f"./checkpoint/{self.args.dataset}/{self.args.case}/{self.args.data_volume}/{self.args.ckpt_name}/pseudo/select.pth")["net"])
        clf.load_state_dict(torch.load(f"./checkpoint/{self.args.dataset}/{self.args.case}/{self.args.data_volume}/{self.args.ckpt_name}/pseudo/select.pth")["clf"])

        # Fixmatch type
        t = tqdm(enumerate(zip(self.train10_loader, self.pseudo_loader_s)), desc='Loss: **** ', total=len(self.train10_loader), bar_format='{desc}{bar}{r_bar}')
        total = 0
        correct = 0
        label_loss = 0
        unlabel_loss = 0
        train_loss = 0

        for batch_idx, ((lx, _, l_target),(ux1, ux2, _)) in t:

            clf.train()
            self.backbone.train()
            lx, l_target = lx.to(self.device), l_target.to(self.device)
            ux1, ux2 = ux1.to(self.device), ux2.to(self.device)

            # label
            (l_representation, _) = self.backbone(lx)
            l_representation = l_representation.view(l_representation.shape[0], l_representation.shape[1]).contiguous()
            l_raw_scores = clf(l_representation)
            class_loss = criterion(l_raw_scores, l_target)
            l_loss = class_loss 
            label_loss += class_loss.item()*lx.shape[0]

            _, predicted = l_raw_scores.max(1)
            total += l_target.size(0)
            correct += predicted.eq(l_target).sum().item()

            
            #pseudo-label

            clf.eval()
            self.backbone.eval()
            with torch.no_grad():
                (representation1, _) = self.backbone(ux1)
                representation1 = representation1.view(representation1.shape[0], representation1.shape[1]).contiguous()
                pesudo_label = clf(representation1)
            
            pesudo_label = softmax(pesudo_label)
            pseudo_score, pesudo_label = pesudo_label.max(1)
            indices = torch.nonzero((pseudo_score>cndf_ratio))

            clf.train()
            self.backbone.train()

            (u_representation, _) = self.backbone(ux2)
            u_representation = u_representation.view(u_representation.shape[0], u_representation.shape[1]).contiguous()
            u_raw_scores = clf(u_representation)
            class_loss = criterion(torch.squeeze(u_raw_scores[indices], 1), torch.squeeze(pesudo_label[indices],1))

            u_loss = class_loss 
            unlabel_loss += class_loss.item()*lx.shape[0]

            _, predicted = torch.squeeze(u_raw_scores[indices], 1).max(1)
            total += torch.squeeze(pesudo_label[indices],1).size(0)
            correct += predicted.eq(torch.squeeze(pesudo_label[indices],1)).sum().item()
            
            semi_optimizer.zero_grad()
            loss = u_loss + l_loss
            loss.backward()
            semi_optimizer.step()

            train_loss = loss.item()

            t.set_description('FixMatch %.0f | Label Loss: %.3f | Unlabel Loss: %.3f  | Train Acc: %.3f%% ' % (epoch, l_loss.item(), u_loss.item(), 100. * correct / total))

        if (self.test_freq > 0) and (epoch % self.test_freq == (self.test_freq - 1)):
            acc, testing_loss = self.test(self.test_loader, clf)
            if acc >= self.best_acc:
                patience = 10
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
                    "pseudo"
                )
            print(f"current acc:{acc}, best_acc: {self.best_acc} (epoch:{self.best_epoch})")

        t_sne_vis(self.backbone, self.tsne_loader, self.device, f"{epoch+1}", "Cycle Pseudo-label", f"./fig/{self.args.ckpt_name}/pseudo/")

