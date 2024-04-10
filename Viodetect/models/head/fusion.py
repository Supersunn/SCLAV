import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

class AVGA(nn.Module):
    """Audio-guided visual attention used in AVEL.
    AVEL:Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chen-liang Xu. Audio-visual event localization in unconstrained videos. InECCV, 2018
    """
    def __init__(self, hidden_size=512):
        super(AVGA, self).__init__()
        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(128, hidden_size)
        self.affine_video = nn.Linear(512, hidden_size)
        self.affine_v = nn.Linear(hidden_size, 49, bias=False)
        self.affine_g = nn.Linear(hidden_size, 49, bias=False)
        self.affine_h = nn.Linear(49, 1, bias=False)

        init.xavier_uniform(self.affine_v.weight)
        init.xavier_uniform(self.affine_g.weight)
        init.xavier_uniform(self.affine_h.weight)
        init.xavier_uniform(self.affine_audio.weight)
        init.xavier_uniform(self.affine_video.weight)

    def forward(self, audio, video):
        #  audio: [bs, 10, 128]
        # video: [bs, 10, 7, 7, 512]
        v_t = video.view(video.size(0) * video.size(1), -1, 512)
        V = v_t

        # Audio-guided visual attention
        v_t = self.relu(self.affine_video(v_t))
        a_t = audio.view(-1, audio.size(-1))
        a_t = self.relu(self.affine_audio(a_t))
        content_v = self.affine_v(v_t) + self.affine_g(a_t).unsqueeze(2)

        z_t = self.affine_h((F.tanh(content_v))).squeeze(2)
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1)) # attention map
        c_t = torch.bmm(alpha_t, V).view(-1, 512)
        video_t = c_t.view(video.size(0), -1, 512) #attended visual features

        return video_t
#---------------------------------------------------------------------------------------------    
class AGV(nn.Module):
    def __init__(self, hidden_size=128):
        super(AGV, self).__init__()
        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(hidden_size, hidden_size)
        self.affine_video = nn.Linear(hidden_size, hidden_size)
        self.affine_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.affine_g = nn.Linear(hidden_size, hidden_size, bias=False)
        self.affine_h = nn.Linear(hidden_size, hidden_size, bias=False)

        init.xavier_uniform(self.affine_v.weight)
        init.xavier_uniform(self.affine_g.weight)
        init.xavier_uniform(self.affine_h.weight)
        init.xavier_uniform(self.affine_audio.weight)
        init.xavier_uniform(self.affine_video.weight)

    def forward(self, audio, video):
        #  audio: [bs, 200, 128]         B,T,C
        #  video: [bs, 200, 128]         B,T,C
        V = video    # b,200,128

        # Audio-guided visual attention
        v_t = self.relu(self.affine_video(video))       # b,200,128
        a_t = self.relu(self.affine_audio(audio))       # b,200,128
        content_v = self.affine_v(v_t) + self.affine_g(a_t)  # b,200,128

        z_t = self.affine_h((F.tanh(content_v)))      # b,200,1024
        alpha_t = F.softmax(z_t, dim=-1)              # attention map
        video_t = torch.mul(alpha_t, V)               #attended visual features

        return video_t    # B,T,C


class VGA(nn.Module):
    def __init__(self, hidden_size=128):
        super(VGA, self).__init__()
        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(hidden_size, hidden_size)
        self.affine_video = nn.Linear(hidden_size, hidden_size)
        self.affine_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.affine_g = nn.Linear(hidden_size, hidden_size, bias=False)
        self.affine_h = nn.Linear(hidden_size, hidden_size, bias=False)

        init.xavier_uniform(self.affine_v.weight)
        init.xavier_uniform(self.affine_g.weight)
        init.xavier_uniform(self.affine_h.weight)
        init.xavier_uniform(self.affine_audio.weight)
        init.xavier_uniform(self.affine_video.weight)

    def forward(self, audio, video):
        #  audio: [bs, 200, 128]         B,T,C
        #  video: [bs, 200, 128]         B,T,C
        A = audio    # b,200,128

        # Video-guided audio attention
        v_t = self.relu(self.affine_video(video))       # b,200,128       
        a_t = self.relu(self.affine_audio(audio))       # b,200,128
        content_v = self.affine_v(v_t) + self.affine_g(a_t)  # b,200,128

        z_t = self.affine_h((F.tanh(content_v)))      # b,200,128
        alpha_t = F.softmax(z_t, dim=-1)              # attention map
        audio_t = torch.mul(alpha_t, A)               #attended visual features

        return audio_t

