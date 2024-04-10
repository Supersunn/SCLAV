import os

base_path = os.path.dirname(__file__) + '/../../'

annotation_train = os.path.join(base_path, 'data/xd-vio/annotations/train.json')

annotation_test = os.path.join(base_path, 'data/xd-vio/annotations/test.json')

gt_label = os.path.join(base_path, 'data/xd-vio/annotations/gt.npy')


log_config = dict(interval=50)


optimizer = dict(type="Adam", lr=0.0005, weight_decay=10e-4)

lr_scheduler = dict(type="CosineAnnealingWarmRestarts", T_0=10)

total_epochs = 20  # 30

work_dir = os.path.join(base_path, 'workdirs')

log_dir = os.path.join(work_dir, 'logs')

save_dir = ''

name = ''

max_seq_len = 200

rand_seed = 2024
gpus = [7]

dataset_train = dict(type="VAFeatureDataset",
                     annotation_file=annotation_train,
                     max_seq_len=max_seq_len)
dataset_val = dict(type="VAFeatureDataset",
                   annotation_file=annotation_test,
                   max_seq_len=max_seq_len,
                   train_mode=False)

dataloader_train = dict(type="DataLoader",
                        dataset=dataset_train,
                        batch_size=128,
                        pin_memory=True,
                        shuffle=True,
                        num_workers=8)

dataloader_val = dict(type="DataLoader",
                      dataset=dataset_val,
                      batch_size=5,
                      pin_memory=True,
                      shuffle=False,
                      num_workers=8)

data = dict(train=dataloader_train, val=dataloader_val)