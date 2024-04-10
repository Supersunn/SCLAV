from configs.xd_vio.base import *

model = dict(type="VAWeakDetectionFromFeature",
             fuse=dict(type="ModalSelection", modal='mix'),
             head=dict(type="CoLSTM", n_video_feature=1024, n_audio_feature=128,
                       n_hidden_feature=128, ),
             loss=dict(type="SupConLoss", gamma=0.05,gamma2=1.0, mil_loss='mse',
                       threshold=0.5))
