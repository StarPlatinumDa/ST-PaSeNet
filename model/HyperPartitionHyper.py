

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import clip
# from Text_Prompt import *
# from tools import *
# from einops import rearrange, repeat


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
from Text_Prompt import *
from tools import *



def type_1_partition(input, partition_size):  # partition_size = [N, L]
    # 输入为[B,3c/8,T,V]
    B, C, T, V = input.shape
    # print(B,C,T,V)
    # 前两维不变，T和V都各自分裂成2维，即4维变为6维，  [B,3c/8,T//T子集大小（即T分割的子集个数）,T的子集大小，V//V子集大小,V子集大小]
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], V // partition_size[1], partition_size[1])
    # 交换顺序变为[B,T//T子集大小,V//V子集大小,T的子集大小,V子集大小,3c/8]
    # 仅保留最后三维(两个子集大小和3c/8)，剩下的全部相乘
    # 返回为[B*T分组个数*V分组个数,T组内大小，V组内大小，3C/8]
    partitions = partitions.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_1_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    # 输入partitions为[B*partition1个数*partition2个数，partition1大小*partition2大小，C/8]
    T, V = original_size
    # 得到原始的B大小
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    # 变回为[B，partition1个数，partition2个数，partition1大小，partition2大小，C/8]
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    # 交换顺序为[B,C/8,partition1个数,partition1大小,partition2个数,partition2大小]
    # 再变成[B,C/8,T,V]
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, T, V)
    return output


def type_2_partition(input, partition_size):  # partition_size = [N, K]
    B, C, T, V = input.shape
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], partition_size[1], V // partition_size[1])
    partitions = partitions.permute(0, 2, 5, 3, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_2_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    output = output.permute(0, 5, 1, 3, 4, 2).contiguous().view(B, -1, T, V)
    return output


def type_3_partition(input, partition_size):  # partition_size = [M, L]
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], V // partition_size[1], partition_size[1])
    partitions = partitions.permute(0, 3, 4, 2, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_3_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    output = output.permute(0, 5, 3, 1, 2, 4).contiguous().view(B, -1, T, V)
    return output


def type_4_partition(input, partition_size):  # partition_size = [M, K]
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], partition_size[1], V // partition_size[1])
    partitions = partitions.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_4_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, -1, T, V)
    return output



class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)


### torch version too old for timm
### https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
### torch version too old for timm
### https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x









class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=False,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation
                ),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        # print(len(self.branches))
        # initialize
        self.apply(weights_init)



    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), groups=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class MHSAPe(nn.Module):
    def __init__(self, dim_in, dim, A, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., insert_cls_layer=0, pe=False, num_point=25,
                 outer=True, layer=0,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_point = num_point
        self.layer = layer



        h1 = A.sum(0)
        h1[h1 != 0] = 1
        h = [None for _ in range(num_point)]
        h[0] = np.eye(num_point)
        h[1] = h1
        self.hops = 0*h[0]
        for i in range(2, num_point):
            h[i] = h[i-1] @ h1.transpose(0, 1)
            h[i][h[i] != 0] = 1

        for i in range(num_point-1, 0, -1):
            if np.any(h[i]-h[i-1]):
                h[i] = h[i] - h[i - 1]
                self.hops += i*h[i]
            else:
                continue

        self.hops = torch.tensor(self.hops).long()
        #
        self.rpe = nn.Parameter(torch.zeros((self.hops.max()+1, dim)))

        self.w1 = nn.Parameter(torch.zeros(num_heads, head_dim))



        A = A.sum(0)
        A[:, :] = 0

        self.outer = nn.Parameter(torch.stack([torch.eye(A.shape[-1]) for _ in range(num_heads)], dim=0), requires_grad=True)

        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.kv = nn.Conv2d(dim_in, dim * 2, 1, bias=qkv_bias)
        self.q = nn.Conv2d(dim_in, dim, 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)


        self.proj = nn.Conv2d(dim, dim, 1, groups=6)

        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        self.insert_cls_layer = insert_cls_layer

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, e):
        N, C, T, V = x.shape
        kv = self.kv(x).reshape(N, 2, self.num_heads, self.dim // self.num_heads, T, V).permute(1, 0, 4, 2, 5, 3)
        k, v = kv[0], kv[1]

        ## n t h v c
        q = self.q(x).reshape(N, self.num_heads, self.dim // self.num_heads, T, V).permute(0, 3, 1, 4, 2)

        e_k = e.reshape(N, self.num_heads, self.dim // self.num_heads, T, V).permute(0, 3, 1, 4, 2)
        #
        #
        pos_emb = self.rpe[self.hops]
        #
        k_r = pos_emb.view(V, V, self.num_heads, self.dim // self.num_heads)
        #
        b = torch.einsum("bthnc, nmhc->bthnm", q, k_r)
        #
        c = torch.einsum("bthnc, bthmc->bthnm", q, e_k)
        d = torch.einsum("hc, bthmc->bthm", self.w1, e_k).unsqueeze(-2)


        a = q @ k.transpose(-2, -1)

        attn = a + b + c + d


        attn = attn * self.scale

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)


        x = (self.alpha * attn + self.outer) @ v
        # x = attn @ v


        x = x.transpose(3, 4).reshape(N, T, -1, V).transpose(1, 2)
        x = self.proj(x)

        x = self.proj_drop(x)
        return x


class MHSA(nn.Module):
    def __init__(self, dim_in, dim, A, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., insert_cls_layer=0, pe=False, num_point=25,
                 outer=True, layer=0,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_point = num_point
        self.layer = layer

        self.w1 = nn.Parameter(torch.zeros(num_heads, head_dim))



        A = A.sum(0)
        A[:, :] = 0

        # self.outer = nn.Parameter(torch.stack([torch.eye(A.shape[-1]) for _ in range(num_heads)], dim=0), requires_grad=True)

        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.kv = nn.Conv2d(dim_in, dim * 2, 1, bias=qkv_bias)
        self.q = nn.Conv2d(dim_in, dim, 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)


        # self.proj = nn.Conv2d(dim, dim, 1, groups=6)
        self.proj = nn.Conv2d(dim, dim, 1, groups=6)

        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        self.insert_cls_layer = insert_cls_layer

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        N, C, T, V = x.shape

        kv = self.kv(x).reshape(N, 2, self.num_heads, self.dim // self.num_heads, T, V).permute(1, 0, 4, 2, 5, 3)
        k, v = kv[0], kv[1]
        # print('k,v',k.shape,v.shape)

        ## n t h v c
        q = self.q(x).reshape(N, self.num_heads, self.dim // self.num_heads, T, V).permute(0, 3, 1, 4, 2)

        a = q @ k.transpose(-2, -1)

        # attn = a + b + c + d
        attn = a


        attn = attn * self.scale

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        # print("attn",attn.shape)
        # print("v", v.shape)
        # x = (self.alpha * attn + self.outer) @ v
        x = (self.alpha * attn) @ v


        x = x.transpose(3, 4).reshape(N, T, -1, V).transpose(1, 2)
        x = self.proj(x)

        x = self.proj_drop(x)
        return x

class unit_vit(nn.Module):
    def __init__(self, dim_in, dim, A, num_of_heads, add_skip_connection=True,  qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer=0,
                insert_cls_layer=0, pe=False, num_point=25, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim_in)
        self.dim_in = dim_in
        self.dim = dim
        self.add_skip_connection = add_skip_connection
        self.num_point = num_point
        # self.attn = MHSA(dim_in, dim, A, num_heads=num_of_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
        #                      proj_drop=drop, insert_cls_layer=insert_cls_layer, pe=pe, num_point=num_point, layer=layer, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn_pe = MHSAPe(dim_in//2, dim//2, A, num_heads=num_of_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                             proj_drop=drop, insert_cls_layer=insert_cls_layer, pe=pe, num_point=num_point, layer=layer, **kwargs)

        attn=[]
        for i in range(2):
            attn.append(
                MHSA(dim_in//4, dim//4, A, num_heads=num_of_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                     proj_drop=drop, insert_cls_layer=insert_cls_layer, pe=pe, num_point=num_point, layer=layer,
                     **kwargs)
            )
        self.attn=nn.ModuleList(attn)

        self.partition_size = [[8, 5], [4, 5], [8, 5], [4, 5]]
        self.partition_func = [type_1_partition, type_2_partition, type_3_partition, type_4_partition]
        self.reverse_func = [type_1_reverse, type_2_reverse, type_3_reverse, type_4_reverse]





        self.pe_proj = nn.Conv2d(dim_in//2, dim//2, 1, bias=False)
        self.pe = pe

    def forward(self, x, joint_label, groups):
        ## more efficient implementation


        x_in=self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        split_x=torch.chunk(x_in,chunks=2,dim=1)

        x_pe=split_x[0]


        label = F.one_hot(torch.tensor(joint_label)).float().to(x.device)
        z = x_pe @ (label / label.sum(dim=0, keepdim=True))
        #
        z = self.pe_proj(z).permute(3, 0, 1, 2)

        e = z[joint_label].permute(1, 2, 3, 0)
        # print('pe',x_pe.shape,e.shape)

        output = []

        output.append(self.attn_pe(x_pe,e))

        x_part=torch.chunk(split_x[1],chunks=2,dim=1)



        for i in range(2):
            part=x_part[i]

            # print('part',part.shape)
            B, C, T, V = part.shape

            # [B*partition1划分个数*partition2划分个数,partition1子集大小，partition2大小，3C/8]
            part = self.partition_func[i](part, self.partition_size[i])
            part = part.permute(0, 3, 1, 2).contiguous()
            # print('part and e[i]',part.shape)
            self.attn[i](part)
            # 还原成[B*partition1划分个数*partition2划分个数,partition1子集大小，partition2大小，3C/8]
            # 再到
            part=part.permute(0,2,3,1).contiguous()
            part=part.view(B,T//self.partition_size[i][0],V//self.partition_size[i][1],self.partition_size[i][0]
                           ,self.partition_size[i][1],-1)
            part= part.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, T, V)
            output.append(part)

        output=torch.cat(output,dim=1)

        x = x + self.drop_path(output)

            # part = part.view(-1, self.partition_size[i][0] * self.partition_size[i][1], C)


        # print('x:',x.shape)
        # print('e',e.shape)
        # print('---------------------')

        # x = x + self.drop_path(self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), e))



        return x


class TCN_ViT_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, num_of_heads=6, residual=True, kernel_size=5, dilations=[1,2], pe=False, num_point=25, layer=0):
        super(TCN_ViT_unit, self).__init__()
        self.vit1 = unit_vit(in_channels, out_channels, A, add_skip_connection=residual, num_of_heads=num_of_heads, pe=pe, num_point=num_point, layer=layer)
        # self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            # redisual=True has worse performance in the end
                                            residual=False)
        self.act = nn.ReLU(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, joint_label, groups):
        y = self.act(self.tcn1(self.vit1(x, joint_label, groups)) + self.residual(x))
        return y



class Model_lst_4part(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, head=['ViT-B/32'], k=0,
                 num_of_heads=9, joint_label=[]):
        super(Model_lst_4part, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25

        self.num_of_heads = num_of_heads

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.joint_label = joint_label

        # base_channel = 64
        # self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        # self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        # self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        # self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        # self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        # self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        # self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        # self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        # self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        # self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.proj_up = nn.Conv2d(in_channels=in_channels, out_channels=24 * num_of_heads, kernel_size=(1, 1), stride=(1, 1),
                                 padding=(0, 0))

        self.l1 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True,
                               num_point=num_point, layer=1)
        # * num_heads, effect of concatenation following the official implementation
        self.l2 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
                               pe=True, num_point=num_point, layer=2)
        self.l3 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
                               pe=True, num_point=num_point, layer=3)
        self.l4 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
                               pe=True, num_point=num_point, layer=4)
        self.l5 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, stride=2,
                               num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=5)
        self.l6 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
                               pe=True, num_point=num_point, layer=6)
        self.l7 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
                               pe=True, num_point=num_point, layer=7)
        # self.l8 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, num_of_heads=num_of_heads)
        self.l8 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, stride=2,
                               num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=8)
        self.l9 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
                               pe=True, num_point=num_point, layer=9)
        self.l10 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
                                pe=True, num_point=num_point, layer=10)




        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            # self.part_list.append(nn.Linear(256,512))
            self.part_list.append(nn.Linear(216,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            # self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            self.linear_head['ViT-B/32'] = nn.Linear(216,512)
            conv_init(self.linear_head['ViT-B/32'])
        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        # self.fc = nn.Linear(base_channel*4, num_class)
        self.fc = nn.Linear(24*num_of_heads, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def forward(self, x):


        groups = []
        for num in range(max(self.joint_label) + 1):
            groups.append([ind for ind, element in enumerate(self.joint_label) if element == num])

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)

        ## n, c, t, v
        x = x.view(N, M, V, C, T).contiguous().view(N * M, V, C, T).permute(0, 2, 3, 1)

        x = self.proj_up(x)

        x = self.l1(x, self.joint_label, groups)
        x = self.l2(x, self.joint_label, groups)
        x = self.l3(x, self.joint_label, groups)
        x = self.l4(x, self.joint_label, groups)
        x = self.l5(x, self.joint_label, groups)
        x = self.l6(x, self.joint_label, groups)
        x = self.l7(x, self.joint_label, groups)
        x = self.l8(x, self.joint_label, groups)
        x = self.l9(x, self.joint_label, groups)
        x = self.l10(x, self.joint_label, groups)

        # x = self.l1(x)
        # x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        # x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        # x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2,3,20]).long()
        hand_list = torch.Tensor([4,5,6,7,8,9,10,11,21,22,23,24]).long()
        foot_list = torch.Tensor([12,13,14,15,16,17,18,19]).long()
        hip_list = torch.Tensor([0,1,2,12,16]).long()
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))


        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)
        
        x = self.drop_out(x)

        return self.fc(x), feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]



class Model_lst_4part_bone(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, head=['ViT-B/32'], k=1):
        super(Model_lst_4part_bone, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25
        # self.A_vector = self.get_A(graph, k).float()

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)



        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2,3]).long()
        hand_list = torch.Tensor([4,5,6,7,8,9,10,11,20,22,23,24]).long()
        foot_list = torch.Tensor([12,13,14,15,16,17,18,19]).long()
        hip_list = torch.Tensor([0,1,12,16]).long()
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))


        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)
        
        x = self.drop_out(x)

        return self.fc(x), feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]


class Model_lst_4part_ucla(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, head=['ViT-B/32'], k=1):
        super(Model_lst_4part_ucla, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25
        # A = np.stack([np.eye(num_point)] * 3, axis=0)
        self.A_vector = self.get_A(graph, k).float()


        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)



        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])
        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
    
    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2,3]).long()
        hand_list = torch.Tensor([10,11,6,7,8,9,4,5]).long()
        foot_list = torch.Tensor([16,17,18,19,12,13,14,15]).long()
        hip_list = torch.Tensor([0,1,12,16]).long()
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))


        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)
        
        x = self.drop_out(x)

        return self.fc(x), feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]


class Model_lst_4part_bone_ucla(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, head=['ViT-B/32'], k=1):
        super(Model_lst_4part_bone_ucla, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25
        self.A_vector = self.get_A(graph, k).float()


        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)



        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])
        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2]).long()
        hand_list = torch.Tensor([7,8,9,10,3,4,5,6]).long()
        foot_list = torch.Tensor([11,12,13,14,15,16,17,18]).long()
        hip_list = torch.Tensor([0,1,11,15]).long()
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))


        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)
        
        x = self.drop_out(x)

        return self.fc(x), feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]
