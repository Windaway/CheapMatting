import torch
import torch.nn as nn
from timm.models.layers import drop_path as dpp
from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
ATLDAS_cifar100 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 3), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_5x5', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))



def drop_path(x, drop_prob):
    dpp(x, drop_prob, True)
    return x

class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out

OPS = {
    'none' : lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}





class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class ResBlock(nn.Module):
    def __init__(self, inc, midc, stride=1):
        super(ResBlock, self).__init__()
        if midc//16 == 0:
            gp=16
        else:
            gp=8
        self.conv1 = nn.Conv2d(inc, midc, kernel_size=1, stride=1, padding=0, bias=True)
        self.gn1=nn.GroupNorm(gp,midc)
        self.conv2 = nn.Conv2d(midc, midc, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn2=nn.GroupNorm(gp,midc)
        self.conv3=nn.Conv2d(midc,inc,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self,x):
        x_=x
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x+x_
        x = self.relu(x)
        return x



class PPMSM(nn.Module):
    def __init__(self, inc, midc, outc):
        super(PPMSM, self).__init__()
        self.ppm = []
        pool_scales = (5, 13, 19, 25)
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AvgPool2d(scale, scale // 2, scale // 2 - 1, count_include_pad=False),
            ))
        self.conv = []
        for x in range(5):
            self.conv.append(nn.Sequential(nn.Conv2d(inc, midc, 1, 1, 0, bias=True)))
        self.out = nn.Sequential(nn.Conv2d(midc * 4 + inc, outc, 1, 1, 0, bias=True))
        self.ppm = nn.ModuleList(self.ppm)
        self.conv = nn.ModuleList(self.conv)

    def forward(self, infeat):
        featin = infeat
        feats = []
        for conv in self.ppm:
            feat = conv(featin)
            feats.append(feat)
        outfeat = [featin]
        for x, conv in zip(feats, self.conv):
            feat = torch.nn.functional.interpolate(conv(x), size=featin.shape[2:], mode='bilinear', align_corners=False)
            outfeat.append(feat)
        outfeat = self.out(torch.cat(outfeat, 1))
        return outfeat



class CheapMatting(nn.Module):
    def __init__(self):
        super(CheapMatting, self).__init__()
        C=48
        num_classes=10
        layers=14
        genotype=ATLDAS_cifar100
        self.drop_path_prob=0

        self.stem00 = nn.Sequential(
            nn.Conv2d(6, C // 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.PReLU(C // 2),
            nn.Conv2d( C // 2, C // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(C // 2)
        )

        self.stem05=nn.Sequential(
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=True),
            ResBlock(C,32),ResBlock(C,32),
        )

        self.stem10 = nn.Sequential(
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )


        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        self.ppm=PPMSM(768,128,256)
        self.decoder_layer3 = nn.Sequential(nn.Conv2d(256+768+384,128,1,1,0),ResBlock(128,64),ResBlock(128,64))
        self.decoder_layer2 = nn.Sequential(nn.Conv2d(128+192,128,1,1,0),ResBlock(128,64),ResBlock(128,64))
        self.decoder_layer1 = nn.Sequential(nn.Conv2d(128+48,64,1,1,0),ResBlock(64,32),ResBlock(64,32))
        self.decoder_layer0 = nn.Sequential(nn.Conv2d(64+24,32,1,1,0),ResBlock(32,24),ResBlock(32,24))
        self.pred = nn.Sequential(nn.Conv2d(32+6,24,3,1,1),nn.PReLU(24),nn.Conv2d(24,16,3,1,1),nn.PReLU(16),nn.Conv2d(16,1,1,1,0))
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)

    def forward(self, input):
        s00 = self.stem00(input)
        s0a=self.stem05(s00)
        s0=s0a
        s1 = self.stem10(s0a)
        out=[]
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i in [3,8,13]:
                out.append(s1)
        x1,x2,x3=out
        x32=self.ppm(x3)
        x3_=self.up(torch.cat([x32,x3],1))
        x2=torch.cat([x3_,x2],dim=1)
        x2_=self.decoder_layer3(x2)
        x2_=self.up(x2_)
        x1=torch.cat([x2_,x1],dim=1)
        x1_=self.decoder_layer2(x1)
        x1_=self.up(x1_)
        x0=torch.cat([x1_,s0a],dim=1)
        x0=self.decoder_layer1(x0)
        x0_=self.up(x0)
        x0_=torch.cat([x0_,s00],dim=1)
        x0_=self.decoder_layer0(x0_)
        x0=self.up(x0_)
        x0=torch.cat([x0,input],dim=1)
        x0=self.pred(x0)
        x0=torch.clamp(x0,0,1)
        return x0



if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    import time
    model=CheapMatting().cuda()
    model.eval()
    model=torch.compile(model)
    x=torch.randn(8,6,1920,1088).cuda()
    with torch.amp.autocast(device_type="cuda"):
        with torch.no_grad():
            f1 = model(x)
            for i in range(5):
                f1= model(x)
            torch.cuda.synchronize()
            a = time.time()
            for i in range(15):
                f1 = model(x)
            torch.cuda.synchronize()
            print(120/ (time.time() - a))

