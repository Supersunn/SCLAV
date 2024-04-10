import torch.nn as nn
from ..builder import HEADS
import torch
from .base import BaseHead
from ..components import CoAttention, SelfAttention, conv1d
from .fusion import AGV, VGA
import torch.nn.functional as F

@HEADS.register()
class CoLSTM(BaseHead):
    def __init__(self,
                 n_video_feature: int = 1024,
                 n_audio_feature: int = 128,
                 n_hidden_feature: int = 128, beta: float = 1.0,
                 skip_connect: str = 'cross',
                 use_layer_norm: bool = True,
                 use_sqrt: bool = True,
                 **kwargs) \
            -> None:
        super().__init__(**kwargs)
        self.conv_v = nn.Sequential(
            conv1d(n_video_feature, n_hidden_feature, 3), nn.ReLU())
        self.conv_a = nn.Sequential(
            conv1d(n_audio_feature, n_hidden_feature, 3), nn.ReLU())
        
        self.co_attn = CoAttention(n_hidden_feature ,
                                   skip_connection=skip_connect,
                                   use_layer_norm=use_layer_norm,
                                   use_sqrt=use_sqrt)
        self.agv = AGV()
        self.vga = VGA()
        self.lstm_v = nn.LSTM(input_size=128,
                              hidden_size=n_hidden_feature, num_layers=1,
                              bidirectional=True)
        self.lstm_a = nn.LSTM(input_size=128,
                              hidden_size=n_hidden_feature, num_layers=1,
                              bidirectional=True)

        self.self_attn_v_mutual = SelfAttention(n_hidden_feature,
                                                use_layer_norm=use_layer_norm,
                                                use_sqrt=use_sqrt)
        
        self.self_attn_a_mutual = SelfAttention(n_hidden_feature,
                                                use_layer_norm=use_layer_norm,
                                                use_sqrt=use_sqrt)

        self.fuse_v = conv1d(n_hidden_feature * 2 , n_hidden_feature, 3)
        self.fuse_a = conv1d(n_hidden_feature * 2 , n_hidden_feature, 3)
        self.fuse_v1 = conv1d(n_hidden_feature, n_hidden_feature, 3)

        self.affine_v = nn.Linear(n_hidden_feature, n_hidden_feature, bias=False)
        self.affine_g = nn.Linear(n_hidden_feature, n_hidden_feature, bias=False)
        self.affine_h = nn.Linear(n_hidden_feature, n_hidden_feature, bias=False)

        self.conv5 = conv1d(n_hidden_feature * 2, 1, 3)
        self.conv6 = conv1d(n_hidden_feature , 1, 3)

        self.beta = beta

    def forward(self, x, seq_len):
        video = x['video']   # [128,200,1024]
        audio = x['audio']   # [128,200,128]

        v = video.permute(0, 2, 1)  # BxC1xT
        a = audio.permute(0, 2, 1)  # BxC2xT

        v = self.conv_v(v)  # BxCxT
        a = self.conv_a(a)  # BxCxT

        # 20
        # x_v = v.permute(0, 2, 1)
        # x_a = a.permute(0, 2, 1)  #   

        # ================【前融合:AGV,VGA,None】================
        # v = self.agv(x_a,x_v)     # B,T,C

        # Audio-guided visual attention
        # content_v = self.affine_v(x_v) + self.affine_g(x_a)  # b,200,128

        # z_t = self.affine_h((F.tanh(content_v)))      # b,200,1024
        # alpha_t = F.softmax(z_t, dim=-1)              # attention map
        # video_t = torch.mul(alpha_t, x_v)               #attended visual features

        # a = self.vga(x_a,x_v)
        # ======================================================
        # 20
        # x_lv, _ = self.lstm_v(x_v)     #   B,T,C  双边LSTM   B,T,2C
        # x_la, _ = self.lstm_a(x_a)
        # x_lv = x_lv.permute(0, 2, 1)
        # x_la = x_la.permute(0, 2, 1)
        # x_v1 = self.fuse_v (x_lv)    # LSTM     3

        x_v1 = self.fuse_v1(v)    # LSTM     3   
        # x_v1 = self.fuse_v1(a)  # 【0427】
        
        # --------------
        # # x_a1 = self.fuse_v (x_a)   # LSTM     2
        # x_v1 = self.fuse_v1 (v)      # LSTM     7
        # =====================【中融合】========================
        x_cv, x_ca = self.co_attn(v, a)   # input:B*C*T   output:B*C*T 
        # x_cv, x_ca = self.co_attn(x_lv, x_la)   # input:B*C*T   output:B*C*T  
        # ======================================================
        # 20
        # x_cv = self.fuse_v(x_cv)
        # x_ca = self.fuse_a(x_ca)

        # 
        x_cv = v.permute(0, 2, 1)
        x_ca = a.permute(0, 2, 1)  # 
        # x_cv = x_cv.permute(0, 2, 1)
        # x_ca = x_ca.permute(0, 2, 1)  #  
        x_lv, _ = self.lstm_v(x_cv)     #   B,T,C  双边LSTM   B,T,2C
        x_la, _ = self.lstm_a(x_ca)
        x_lv = x_lv.permute(0, 2, 1)
        x_la = x_la.permute(0, 2, 1)  #  
        f_v = self.fuse_v(x_lv)
        f_a = self.fuse_a(x_la)
        # s_v = self.fuse_v(x_lv)
        # s_a = self.fuse_a(x_la)

        s_v = self.self_attn_v_mutual(f_v)    # input:B*C*T   output:B*C*T 
        s_a = self.self_attn_a_mutual(f_a)

        # # ------【A+V】
        # logits = self.conv6(s_v)
        # logits2 = self.conv6(s_a)             # 监督支路

        # ------【融合+V或A】
        # z = torch.cat((f_v, f_a), dim=1)
        z = torch.cat((s_v, s_a), dim=1)
        logits = self.conv5(z)

        v2 = self.self_attn_v_mutual(x_v1)       # 3
        # v2 = self.self_attn_a_mutual(x_a1)     # 2
        logits2 = self.conv6(v2)


        prob = torch.sigmoid(logits).squeeze(1)      # co支路
        prob2 = torch.sigmoid(logits2).squeeze(1)    # v支路

        # ==================【后融合:cat,dot,plus】==============
        pred = prob * self.beta + (1 - self.beta) * prob2   #  self.beta预测时为0最优，训练时有作用
        # ======================================================
        
        return {'prob': prob, 'prob2': prob2, 'pred': pred}
