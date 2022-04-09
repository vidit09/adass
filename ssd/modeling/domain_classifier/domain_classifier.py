import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # import pdb;pdb.set_trace()
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)



class DomainClassifierMultiLocalFEReweighted(nn.Module):

    def __init__(self,ins, weighting=True):
        super().__init__()
        self.weighting = weighting
        self.layer_mod = []
        if len(ins) == 7:
            N = 6
        else:
            N = 5
        for i,inchannel in enumerate(ins):

            module_in = nn.ModuleList([nn.Conv2d(inchannel, 256, kernel_size=3,padding=1),
                                       nn.GroupNorm(32, 256),
                                       nn.ReLU()])
            module = [nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(256, 256, kernel_size=3,padding=1),
                                    nn.GroupNorm(32,256),
                                    nn.ReLU(),
                                    ]
            module_mid = []
            for i in range(N-1):
                module_mid += copy.deepcopy(module)
            module_in.extend(module_mid)

            if not N==0:
                module_in.extend([nn.MaxPool2d(kernel_size=2, stride=2),nn.Flatten(),nn.Linear(256,2)])
            else:
                module_in.extend(
                    [ nn.Flatten(), nn.Linear(256, 2)])
            N=N-1
            self.layer_mod.append(module_in.cuda())

        self.layer_mod = nn.ModuleList(self.layer_mod)
        self.loss_domain = torch.nn.NLLLoss(reduction='none')

    def forward(self, inputs, constant, targets, attnw=None, lossw=None, unmodulated_feats=None):
        final_loss = 0
        if lossw is not None:
            lossw = lossw.detach()
        for i, vv in enumerate(zip(inputs,targets)):
            input, target = vv
            if attnw is not None:
                if not isinstance(attnw,list) :

                    if i ==0 :
                        val,ind = torch.max(attnw,1)
                        val = (val-val.min(1)[0].unsqueeze(1))/(val.max(1)[0].unsqueeze(1)-val.min(1)[0].unsqueeze(1))
                        val = val.reshape(input.shape[0],input.shape[2],input.shape[3]) + 1

                    val = F.interpolate(val.unsqueeze(0),size=(input.shape[2],input.shape[3])).squeeze(0)

                    input = input*val.unsqueeze(1)
                else:

                    if i>2:
                        val, ind = torch.max(attnw[2], 1)
                        val = (val - val.min(1)[0].unsqueeze(1)) / (val.max(1)[0].unsqueeze(1) - val.min(1)[0].unsqueeze(1))
                        # val = val.reshape(inputs[2].shape[0], inputs[2].shape[2], inputs[2].shape[3]) +1

                        val = val.reshape(inputs[2].shape[0], inputs[2].shape[2], inputs[2].shape[3])
                        val = F.interpolate(val.unsqueeze(0), size=(input.shape[2], input.shape[3])).squeeze(0)

                    else:
                        val, ind = torch.max(attnw[i], 1)
                        val = (val - val.min(1)[0].unsqueeze(1)) / (val.max(1)[0].unsqueeze(1) - val.min(1)[0].unsqueeze(1))
                        val = val.reshape(input.shape[0], input.shape[2], input.shape[3])

                    if constant[0]==0:
                        val = torch.ones(input.shape[0],input.shape[2],input.shape[3]).to(val.device)
                    else:
                        if constant[2] == 50000: ## city-->foggy
                            ratio = torch.tensor((constant[1]-5000)/45000.)
                        else:
                            ratio = torch.tensor((constant[1]-12000)/18000.)

                        gamma = -1+ 2/(1+torch.exp(-5*ratio))

                        val = gamma.cuda()*val+ 1-gamma.cuda()
                    if  self.weighting:
                        if unmodulated_feats is None:
                            input = input * val.unsqueeze(1)
                        else:
                            import pdb;pdb.set_trace()
                            input = input * (val.unsqueeze(1)-1) + unmodulated_feats[i]

            constant_ = constant[0]
            input = GradReverse.grad_reverse(input, constant_)
            layers = self.layer_mod[i]

            logits = input

            for layer in layers:

                logits = layer(logits)

            logits = F.log_softmax(logits, 1)

            if target.min() == 0 :
                target = torch.zeros(target.shape[0]).long().cuda()
            else:
                target = torch.ones(target.shape[0]).long().cuda()

            final_loss += self.loss_domain(logits, target).mean()

        final_loss = 0.5*final_loss/len(inputs)

        return final_loss




