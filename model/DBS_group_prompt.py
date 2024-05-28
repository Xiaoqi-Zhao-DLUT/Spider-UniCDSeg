import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.FoldConv import ASPP

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, x_dim, y_dim=None, heads=8, hid_dim=64, dropout=0., use_sdpa=True):
        super().__init__()
        y_dim = y_dim if y_dim else x_dim
        self.heads = heads
        assert hid_dim % heads == 0
        dim_head = hid_dim // heads
        self.scale = dim_head ** -0.5
        self.use_sdpa = use_sdpa
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(x_dim, hid_dim, bias=False)
        self.to_k = nn.Linear(y_dim, hid_dim, bias=False)
        self.to_v = nn.Linear(y_dim, hid_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(hid_dim, x_dim), nn.Dropout(dropout))

    def forward(self, q, kv):
        # q, kv: L,B,C
        q = self.to_q(q)
        k = self.to_k(kv)
        v = self.to_v(kv)
        q, k, v = map(lambda t: rearrange(t, 'n b (h d) -> b h n d', h=self.heads), (q, k, v))
        
        if self.use_sdpa:
            # q = q * self.scale
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=False)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        
        out = rearrange(out, 'b h n d -> n b (h d)')
        return self.to_out(out)


class CrossTransformer(nn.Module):
    def __init__(self, dim, heads, hid_dim, dropout=0.):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, hid_dim=hid_dim, dropout=dropout)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim=hid_dim, dropout=dropout)

    def forward(self, tgt, memory):
        tgt = tgt + self.attn(tgt, memory)
        tgt = self.attn_norm(tgt)
        tgt = tgt + self.ffn(tgt)
        tgt = self.ffn_norm(tgt)
        return tgt


    
class Spider_ConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        # timm.list
        ###############################Transition Layer########################################
        # self.bkbone = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', features_only=True, pretrained=True)
        # self.bkbone_prompt = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', features_only=True, pretrained=True)
         #128, 256, 512, 1024
            
        # self.bkbone = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', features_only=True, pretrained=True)
        # self.bkbone_prompt = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', features_only=True, pretrained=True)
        # #192, 384, 768, 1536

        self.bkbone = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', features_only=True, pretrained=True)
        self.bkbone_prompt = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', features_only=True, pretrained=True)
        #192, 384, 768, 1536
        ###############################Transition Layer########################################
        self.dem5 = ASPP(1536, 64)
        # self.dem5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem4 = nn.Sequential(nn.Conv2d(768, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(384, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.dem2 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        ################################FPN branch#######################################
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        ################################FPN branch#######################################
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))

        self.emb_dim = 64
        self.output_dim = 1

        total_dim =  self.emb_dim * self.output_dim
        self.proj = nn.Sequential(nn.Linear(1536, total_dim), nn.LayerNorm(total_dim))
        self.cross_atten3 = CrossTransformer(self.emb_dim, 4, hid_dim=self.emb_dim, dropout=0.1)


    def forward(self, x, filter_list, mask_list):
        input = x
        B1, _, _, _ = input.size()
        E2, E3, E4, E5 = self.bkbone(x)
        kernels_3 = []
        # for filter_image in filter_list:  # => task,3,H,W
        for i in range(len(filter_list)):  # => 3(within task),3,H,W
            with torch.no_grad():
                self.bkbone_prompt.eval()
                _, _, _, feat = self.bkbone_prompt(filter_list[i])  # B,C,H,W
            B, C, H, W = feat.shape
            memory = feat.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(1)  # BHW,1,2048
            memory = self.proj(memory)  # BHW,1,C

            query = memory.reshape(B, H, W, -1)
            mask = F.upsample(mask_list[i], size=feat.size()[2:], mode='nearest')
            mask = mask.reshape(B, H, W, 1)
            query_fore = (mask * query).sum((0, 1, 2)) / (1 + mask.sum((0, 1, 2)))  # C
            query_back = ((1 - mask) * query).sum((0, 1, 2)) / (1 + (1 - mask).sum((0, 1, 2)))  # C

            query = torch.stack((query_fore, query_back), dim=0).unsqueeze(1)  # 2,1,C
            query_3 = self.cross_atten3(query, memory)
            query_3 = query_3.reshape(2, 1, 64, 1, 1)  # 2,Cin,Kh,Kw
            kernels_3.append(query_3)

        ################################Transition Layer#######################################
        T5 = self.dem5(E5)
        T4 = self.dem4(E4)
        T3 = self.dem3(E3)
        T2 = self.dem2(E2)
      
        ################################Gated FPN#######################################
        D4 = self.output4(F.upsample(T5, size=E4.size()[2:], mode='bilinear') + T4)
        D3 = self.output3(F.upsample(D4, size=E3.size()[2:], mode='bilinear') + T3)
        D2 = F.upsample(D3, size=E2.size()[2:], mode='bilinear') + T2
        output_fpn = []
        for k3 in kernels_3:
            out = F.conv2d(input=D2, weight=k3[0], bias=k3[1].mean(1).reshape(-1), stride=1, padding=0)
            output_fpn.append(F.upsample(out, size=input.size()[2:], mode='bilinear'))
        output_fpn = torch.cat(output_fpn, dim=1)
        return output_fpn


    

class Spider_Swin(nn.Module):
    def __init__(self):
        super().__init__()
        # timm.list
        ###############################Transition Layer########################################
        self.bkbone = timm.create_model('swin_base_patch4_window12_384.ms_in22k_ft_in1k', features_only=True, pretrained=True)
        self.bkbone_prompt = timm.create_model('swin_base_patch4_window12_384.ms_in22k_ft_in1k', features_only=True, pretrained=True)
        # timm=0.9.7
        # self.bkbone = timm.create_model('swin_large_patch4_window12_384.ms_in22k_ft_in1k', features_only=True,
        #                                 pretrained=True)
        # self.bkbone_prompt = timm.create_model('swin_large_patch4_window12_384.ms_in22k_ft_in1k', features_only=True,
        #                                        pretrained=True)

        ###############################Transition Layer########################################
        self.dem5 = ASPP(1024, 64)
        # self.dem5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem4 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.dem2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        ################################FPN branch#######################################
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))


        self.emb_dim = 64
        self.output_dim = 1

        total_dim =  self.emb_dim * self.output_dim
        self.proj = nn.Sequential(nn.Linear(1024, total_dim), nn.LayerNorm(total_dim))
        self.cross_atten3 = CrossTransformer(self.emb_dim, 4, hid_dim=self.emb_dim, dropout=0.1)


    def forward(self, x, filter_list, mask_list):
        input = x
        B1, _, _, _ = input.size()
        E2, E3, E4, E5 = self.bkbone(x)
        E2 = E2.permute(0, 3, 1, 2)
        E3 = E3.permute(0, 3, 1, 2)
        E4 = E4.permute(0, 3, 1, 2)
        E5 = E5.permute(0, 3, 1, 2)
        kernels_3 = []

        for i in range(len(filter_list)):  # => 3(within task),3,H,W
            with torch.no_grad():
                self.bkbone_prompt.eval()
                _, _, _, feat = self.bkbone_prompt(filter_list[i])  # B,C,H,W
                feat = feat.permute(0, 3, 1, 2)
            B, C, H, W = feat.shape
            memory = feat.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(1)  # BHW,1,2048
            memory = self.proj(memory)  # BHW,1,C

            query = memory.reshape(B, H, W, -1)
            mask = F.upsample(mask_list[i], size=feat.size()[2:], mode='nearest')
            mask = mask.reshape(B, H, W, 1)
            query_fore = (mask * query).sum((0, 1, 2)) / (1 + mask.sum((0, 1, 2)))  # C
            query_back = ((1 - mask) * query).sum((0, 1, 2)) / (1 + (1 - mask).sum((0, 1, 2)))  # C

            query = torch.stack((query_fore, query_back), dim=0).unsqueeze(1)  # 2,1,C
            query_3 = self.cross_atten3(query, memory)
            query_3 = query_3.reshape(2, 1, 64, 1, 1)  # 2,Cin,Kh,Kw


            kernels_3.append(query_3)

        ################################Transition Layer#######################################
        T5 = self.dem5(E5)
        T4 = self.dem4(E4)
        T3 = self.dem3(E3)
        T2 = self.dem2(E2)
      
        ################################Gated FPN#######################################
        D4 = self.output4(F.upsample(T5, size=E4.size()[2:], mode='bilinear') + T4)
        D3 = self.output3(F.upsample(D4, size=E3.size()[2:], mode='bilinear') + T3)
        D2 = F.upsample(D3, size=E2.size()[2:], mode='bilinear') + T2

        output_fpn = []
        for k3 in kernels_3:
            out = F.conv2d(input=D2, weight=k3[0], bias=k3[1].mean(1).reshape(-1), stride=1, padding=0)
            output_fpn.append(F.upsample(out, size=input.size()[2:], mode='bilinear'))
        output_fpn = torch.cat(output_fpn, dim=1)
        return output_fpn



###prompt时推理使用
class Spider_fast_prompt_infer(nn.Module):
    def __init__(self):
        super().__init__()
        # timm.list
        ###############################Transition Layer########################################
        # self.bkbone = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', features_only=True, pretrained=True)
        # self.bkbone_prompt = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', features_only=True, pretrained=True)
         # 128, 256, 512, 1024
            
        # self.bkbone = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', features_only=True, pretrained=True)
        # self.bkbone_prompt = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', features_only=True, pretrained=True)
        # #192, 384, 768, 1536
        
        # self.bkbone = timm.create_model('convnext_tiny.in12k_ft_in1k_384', features_only=True, pretrained=True)
        # self.bkbone_prompt = timm.create_model('convnext_tiny.in12k_ft_in1k_384', features_only=True, pretrained=True)
        #96, 192, 384, 768

        
        # self.bkbone = timm.create_model('swin_base_patch4_window12_384.ms_in22k_ft_in1k', features_only=True, pretrained=True)
        # self.bkbone_prompt = timm.create_model('swin_base_patch4_window12_384.ms_in22k_ft_in1k', features_only=True, pretrained=True)
        # timm=0.9.7
        self.bkbone = timm.create_model('swin_large_patch4_window12_384.ms_in22k_ft_in1k', features_only=True, pretrained=True)
        self.bkbone_prompt = timm.create_model('swin_large_patch4_window12_384.ms_in22k_ft_in1k', features_only=True, pretrained=True)
        
        ###############################Transition Layer########################################
        self.dem5 = ASPP(1536, 64)
        # self.dem5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem4 = nn.Sequential(nn.Conv2d(768, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(384, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.dem2 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))

        ################################FPN branch#######################################
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))


        self.emb_dim = 64
        self.output_dim = 1

        total_dim = self.emb_dim * self.output_dim
        self.proj = nn.Sequential(nn.Linear(1536, total_dim), nn.LayerNorm(total_dim))

        self.cross_atten3 = CrossTransformer(self.emb_dim, 4, hid_dim=self.emb_dim, dropout=0.1)


    def forward(self, x, filter_list, mask_list,generate_filter,prompt_kernel,backbone_name):
        if generate_filter:
            kernels_3 = []
            for i in range(len(filter_list)):  # => 3(within task),3,H,W
                with torch.no_grad():
                    self.bkbone_prompt.eval()
                    _, _, _, feat = self.bkbone_prompt(filter_list[i])  # B,C,H,W
                if backbone_name == 'swin':
                     feat = feat.permute(0, 3, 1, 2)
                B, C, H, W = feat.shape
                memory = feat.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(1)  # BHW,1,2048
                memory = self.proj(memory)  # BHW,1,C

                query = memory.reshape(B, H, W, -1)
                mask = F.upsample(mask_list[i], size=feat.size()[2:], mode='nearest')
                mask = mask.reshape(B, H, W, 1)
                query_fore = (mask * query).sum((0, 1, 2)) / (1 + mask.sum((0, 1, 2)))  # C
                query_back = ((1 - mask) * query).sum((0, 1, 2)) / (1 + (1 - mask).sum((0, 1, 2)))  # C

                query = torch.stack((query_fore, query_back), dim=0).unsqueeze(1)  # 2,1,C
                query_3 = self.cross_atten3(query, memory)  # 如何汇总所有的query_3?得到一个query_3?
                query_3 = query_3.reshape(2, 1, 64, 1, 1)  # 2,Cin,Kh,Kw

                kernels_3.append(query_3)
            return kernels_3
        else:
            input = x
            B1, _, _, _ = input.size()
            E2, E3, E4, E5 = self.bkbone(x)
            if backbone_name == 'swin':
                E2 = E2.permute(0, 3, 1, 2)
                E3 = E3.permute(0, 3, 1, 2)
                E4 = E4.permute(0, 3, 1, 2)
                E5 = E5.permute(0, 3, 1, 2)
            ################################Transition Layer#######################################
            T5 = self.dem5(E5)
            T4 = self.dem4(E4)
            T3 = self.dem3(E3)
            T2 = self.dem2(E2)
            ################################Gated FPN#######################################
            D4 = self.output4(F.upsample(T5, size=E4.size()[2:], mode='bilinear') + T4)
            D3 = self.output3(F.upsample(D4, size=E3.size()[2:], mode='bilinear') + T3)
            D2 = F.upsample(D3, size=E2.size()[2:], mode='bilinear') + T2

            output_fpn = []
            for k3 in prompt_kernel:
                out = F.conv2d(input=D2, weight=k3[0], bias=k3[1].mean(1).reshape(-1), stride=1, padding=0)
                output_fpn.append(F.upsample(out, size=input.size()[2:], mode='bilinear'))
            output_fpn = torch.cat(output_fpn, dim=1)
            return output_fpn


