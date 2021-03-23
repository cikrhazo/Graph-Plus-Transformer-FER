import math
import torch
from torch import nn
from net_utilz.resnet import resnet18

from net_utilz.vit_pytorch import Transformer, ST_Transformer
from net_utilz.modules import ResGCN_Input_Branch, AttGCN_Module, ResGCN_Module

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast


# classes
def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(x.shape[1])))


# main class
class GraphViT(nn.Module):
    def __init__(
        self, *, num_classes, dim, depth, heads, mlp_dim, A, pool='mean',
        dim_head=64, dropout=0., emb_dropout=0.3, num_frames=16, num_nodes=51
    ):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.register_buffer('A', A)
        # cnn encoder
        self.to_vis_embedding = resnet18(num_classes=num_classes, pretrained=False)
        # gcn encoder
        self.to_geo_embedding = nn.ModuleList([
            ResGCN_Input_Branch(block='Bottleneck', num_channel=2, A=A)
            for _ in range(3)
        ])

        # self.dropout = nn.Dropout(emb_dropout)
        self.pos_emb = nn.Embedding(num_frames * num_nodes, dim)

        module_list = [AttGCN_Module(512 + 66 * 3, dim, block='Bottleneck', A=A)]
        for i in range(depth):
            module_list += [AttGCN_Module(dim, dim, block='Bottleneck', A=A) for _ in range(3)]
            module_list += [ST_Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout)]
        self.main = nn.ModuleList(module_list)

        self.to_latent = nn.Sequential(
            nn.Identity(),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, 2 * dim),
            nn.PReLU(),
        )
        self.mlp_head = nn.Linear(2 * dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, vis, geo):

        b, n, c, h, w, t = vis.size()
        # input branches
        vis = rearrange(vis, 'b n c h w t -> (b n t) c h w').contiguous()
        # v_cat = []
        # for i in range(t):
        #     v_cat.append(self.to_vis_embedding(vis[:, :, :, :, i]).squeeze(-1).squeeze(-1))
        # v = torch.stack(v_cat, dim=2)
        v = self.to_vis_embedding(vis).squeeze(-1).squeeze(-1)
        v = rearrange(v, '(b n t) c -> b c t n', n=n, t=t).contiguous()  # b * c * t * n

        g_cat = []
        # Todo multiprocess
        for i, branch in enumerate(self.to_geo_embedding):
            g_cat.append(branch(geo[:, i, :, :, :]))
        g = torch.cat(g_cat, dim=1).contiguous()  # b * c * t * n

        x_token = torch.cat((v, g), dim=1)  # b * c * t * n
        # x_token = rearrange(x_token, 'b c t n -> b (t n) c').contiguous()  # b tn c
        # x_token = self.dropout(x_token)
        # x_token = rearrange(x_token, 'b (t n) c -> b c t n', t=t, n=n).contiguous()  # b c t n

        for i, layer in enumerate(self.main):
            if i == 0:
                x_token = layer(x_token, self.A)  # b * c * t * n
            elif (i-1) % 4 == 3:
                # To Tokens
                x_token = rearrange(x_token, 'b c t n -> b (t n) c').contiguous()  # b tn c
                if i == 3:
                    # Position Embedding
                    x_token = x_token + self.pos_emb(torch.arange(x_token.shape[1]).to(x_token.device))
                x_token = layer(x_token, t=t, n=n)
                # To Vertexes
                x_token = rearrange(x_token, 'b (t n) c -> b c t n', t=t, n=n).contiguous()  # b c t n
            else:
                x_token = layer(x_token, self.A)  # b * c * t * n

        cls_token = self.to_latent(x_token.mean(-1).mean(-1))  # b c
        out = self.mlp_head(cls_token)

        return cls_token, out


if __name__ == "__main__":
    from data_utilz.graphs import Graph
    vis_tensor = torch.randn(size=(2, 51, 3, 49, 49, 16))  # batch * point * channel * size * size * frame
    geo_tensor = torch.randn(size=(2, 3, 2, 16, 51))  # batch * branch * coordinate * frame * point
    A = torch.from_numpy(Graph().A).float()
    net = GraphViT(
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        num_classes=6,
        A=A
    )
    net.cuda()

    vis_tensor = vis_tensor.cuda()
    geo_tensor = geo_tensor.cuda()
    # seq_tensor = seq_tensor.cuda()
    output, logits = net(vis_tensor, geo_tensor)

    print(output.size())
    print(logits.size())
