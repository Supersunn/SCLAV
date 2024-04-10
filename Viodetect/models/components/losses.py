import torch
import torch.nn.functional as F
import numpy as np

def select_pseudo_labels(prob, threshold, seq_mask):
    detach_prob = prob.detach()
    select_positive = torch.logical_and(
        detach_prob >= threshold,
        seq_mask).nonzero(as_tuple=False).squeeze(1)
    select_negative = torch.logical_and(
        detach_prob < 1 - threshold,
        seq_mask).nonzero(as_tuple=False).squeeze(1)
    return select_positive, select_negative

# ---------------------------------------------------------------------------------------------------------------
def sup_con_loss(features,device, labels=None, mask=None,temperature:float=0.5,scale_by_temperature:bool=True):
    features = torch.unsqueeze(features,1)
    labels = torch.unsqueeze(labels,0)
    features = F.normalize(features, p=2, dim=1)
    batch_size = features.shape[0]

    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`') 
    elif labels is None and mask is None: 
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)  
    elif labels is not None: 
        labels = labels.contiguous().view(-1, 1) 
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device) 
    else:
        mask = mask.float().to(device)

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(features, features.T),temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    exp_logits = torch.exp(logits)


    logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)     
    positives_mask = mask * logits_mask
    negatives_mask = 1. - mask

    num_positives_per_row  = torch.sum(positives_mask , axis=1) 
    denominator = torch.sum(
    exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
        exp_logits * positives_mask, axis=1, keepdims=True)  
    
    log_probs = logits - torch.log(denominator)
    if torch.any(torch.isnan(log_probs)):
        raise ValueError("Log_prob has nan!")
    
    log_probs = torch.sum(
        log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]

    # loss
    loss = -log_probs
    if scale_by_temperature:
        loss *= temperature
    loss = loss.mean()
    return loss

# ---------------------------------------------------------------------------------------------------------------
def get_sup_con_loss(teacher,student,seq_mask,device,threshold:float,temperature:float=0.5,scale_by_temperature:bool=True):
    t_p, t_n = select_pseudo_labels(teacher, threshold, seq_mask)
    selected = torch.cat((t_p, t_n))
    i_0 = len(t_n)
    i_1 = len(t_p)
    max_i = max(i_0, i_1)
    if max_i > 0:
        target = torch.cat((torch.ones(i_0), torch.zeros(i_1))).to(device)
        pred = torch.index_select(student, dim=0, index=selected)
        return sup_con_loss(pred,device,labels=target,temperature=temperature,scale_by_temperature=scale_by_temperature)

    return torch.tensor(0.0).to(device)


# ---------------------------------------------------------------------------------------------------------------
def get_cross_con_loss(a_fea, v_fea, device, labels=None, mask=None,threshold:float=0.5, temperature:float=0.5, scale_by_temperature:bool=True):
    a_fea = torch.unsqueeze(a_fea,1)         # co-branch
    v_fea = torch.unsqueeze(v_fea,1)
    # labels = torch.unsqueeze(labels,0)
    a_fea = F.normalize(a_fea, p=2, dim=0)
    v_fea = F.normalize(v_fea, p=2, dim=0)
    batch_size = a_fea.shape[0]

    if labels is not None and mask is not None:  
        raise ValueError('Cannot define both `labels` and `mask`') 
    elif labels is None and mask is None: 
        v_labels = torch.where(v_fea > threshold,1.0,0.0)
        a_labels = torch.where(a_fea > threshold,1.0,0.0)
        labels = torch.matmul(v_labels, a_labels.T)
        mask = torch.eq(labels, labels.T).float().to(device)  

    elif labels is not None: 
        labels = labels.contiguous().view(-1, 1) 
        mask = torch.eq(labels, labels.T).float().to(device)  
    else:
        mask = mask.float().to(device)

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(v_fea, a_fea.T),temperature) 
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    exp_logits = torch.exp(logits)
    
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)     
    positives_mask = mask * logits_mask
    negatives_mask = 1. - mask

    num_positives_per_row  = torch.sum(positives_mask , axis=1)    
    denominator = torch.sum(
    exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
        exp_logits * positives_mask, axis=1, keepdims=True)  
    
    log_probs = logits - torch.log(denominator)
    if torch.any(torch.isnan(log_probs)):
        raise ValueError("Log_prob has nan!")
    
    log_probs = torch.sum(
        log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]

    loss = -log_probs
    if scale_by_temperature:
        loss *= temperature
    loss = loss.mean()
    return loss
# ---------------------------------------------------------------------------------------------------------------

def get_seq_mask(input_shape, seq_len, device):
    seq_len_ = seq_len.unsqueeze(1).repeat((1, input_shape[1])).flatten()

    return (torch.arange(0, input_shape[1]).repeat(
        input_shape[0]).to(device) < seq_len_)


def get_flatten_label(input_shape,labels):
    labels = labels.unsqueeze(1).repeat((1, input_shape[1])).flatten()
    return labels

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


