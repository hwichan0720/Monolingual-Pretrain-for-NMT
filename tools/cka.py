import torch
import numpy as np
from generate import generate, sent_to_ids

def dec_mean(decoder_outs, sent_order, dim=768):
    '''
        mean_layers = [
            the mean vector of embedding layer,
            the mean vector of the first layer,
            ....
            the mean vector of the six (final) layer,
        ]
        mean_layers: 7 * sent_num * dim

    '''
    mean_layers = []
    
    sent_num = len(decoder_outs[0][1]['inner_states'][0][0])
    layer_num = len(decoder_outs[0][1]['inner_states'])
    max_length = len(decoder_outs)
    dim = 768

    for layer in range(layer_num):
        layer_outs = []
        for idx in range(sent_num):
            vecs = [[0] * dim] * max_length 
            for n, x in enumerate(decoder_outs): # x: output tokens of each iteration
                x = x[1]['inner_states'][layer][0] # token vecors of each sentence (total_sent_num * dim)
                if len(x) <= idx: # idx が小さくなるほど sent length も短い。idx=0 の文は最後までトークンがある。
                    break
                vecs[n] = x[idx].tolist()
            layer_outs.append(vecs)
        layer_outs = torch.tensor(layer_outs)
        mean_layers.append([
            vec for _, vec in sorted(zip(sent_order, layer_outs.mean(dim=1).tolist()))
        ])
        
    return mean_layers

def dec_up_to_self_attn_mean(decoder_outs, sent_order, dim=768):
    '''
        mean_layers = [
            the mean vector of embedding layer,
            the mean vector of the first layer,
            ....
            the mean vector of the six (final) layer,
        ]
        mean_layers: 7 * sent_num * dim

    '''
    mean_layers = []
    
    sent_num = len(decoder_outs[0][1]['self_dec_attns'][0][0])
    layer_num = len(decoder_outs[0][1]['self_dec_attns'])
    max_length = len(decoder_outs)
    dim = 768

    for layer in range(layer_num):
        layer_outs = []
        for idx in range(sent_num):
            vecs = [[0] * dim] * max_length 
            for n, x in enumerate(decoder_outs): # x: output tokens of each iteration
                x = x[1]['self_dec_attns'][layer][0] # token vecors of each sentence (total_sent_num * dim)
                if len(x) <= idx: # idx が小さくなるほど sent length も短い。idx=0 の文は最後までトークンがある。
                    break
                vecs[n] = x[idx].tolist()
            layer_outs.append(vecs)
        layer_outs = torch.tensor(layer_outs)
        mean_layers.append([
            vec for _, vec in sorted(zip(sent_order, layer_outs.mean(dim=1).tolist()))
        ])
        
    return mean_layers

def enc_mean(encoder_layers, sent_order):
    '''
        mean_layers = [
            the mean vector of embedding layer,
            the mean vector of the first layer,
            ....
            the mean vector of the six (final) layer,
            the mean vector of the final layer after layer normalization,
        ]
        mean_layers: 8 * sent_num * dim
    '''
    embedding = encoder_layers.encoder_embedding
    encoder_out = encoder_layers.encoder_out
    all_layers = encoder_layers.encoder_states
    
    mean_layers = [
        [vec for _, vec in sorted(zip(sent_order, embedding.mean(dim=1).tolist()))]
    ]
    for x in all_layers:
        mean_layers.append([
            vec for _, vec in sorted(zip(sent_order, x.mean(dim=0).tolist()))
        ])
    mean_layers.append([
        vec for _, vec in sorted(zip(sent_order, encoder_out.mean(dim=0).tolist()))
    ])
        
    return mean_layers

def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)

def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)

def cal_encdec_mean(model, d, sents):
    lengths = [len(sent.strip().split()) for sent in sents]
    max_length = np.amax(lengths)
    token_ids = torch.tensor([
        sent_to_ids(d, sent.strip(), max_length).tolist() for sent in sents
    ])
    encoder_outs, decoder_outs, _, _, order = generate(
        token_ids, model, d.index('<s>')
    )
    enc = enc_mean(encoder_outs[0], order.tolist())
    dec = dec_mean(decoder_outs, order.tolist())
    self_dec = dec_up_to_self_attn_mean(decoder_outs, order.tolist())
    return np.array(enc), np.array(dec), np.array(self_dec)

def encdec_cka_sim(
    pre, ft, 
    pre_d, ft_d, 
    pre_sents, ft_sents,
    batch_size
):
    pre_sents_per_batch = [
        pre_sents[n*batch_size:n*batch_size+batch_size] 
        for n in range(-(-len(pre_sents)//batch_size))
    ]
    pre_enc, pre_dec, pre_self_dec = [], [], []
    for pre_batch in pre_sents_per_batch:
        if len(pre_enc) == 0:
            pre_enc, pre_dec, pre_self_dec = cal_encdec_mean(pre, pre_d, pre_batch)
            continue
        pre_enc_, pre_dec_, pre_self_dec_ = cal_encdec_mean(pre, pre_d, pre_batch)
        pre_enc = np.concatenate([pre_enc, pre_enc_], 1)
        pre_dec = np.concatenate([pre_dec, pre_dec_], 1)
        pre_self_dec = np.concatenate([pre_self_dec, pre_self_dec_], 1)
    print(pre_enc.shape, pre_dec.shape, pre_self_dec.shape)

    ft_sents_per_batch = [
        ft_sents[n*batch_size:n*batch_size+batch_size] 
        for n in range(-(-len(ft_sents)//batch_size))
    ]
    ft_enc, ft_dec, ft_self_dec = [], [], []
    for ft_batch in ft_sents_per_batch:
        if len(ft_enc) == 0:
            ft_enc, ft_dec, ft_self_dec = cal_encdec_mean(ft, ft_d, ft_batch)
            continue
        ft_enc_, ft_dec_, ft_self_dec_ = cal_encdec_mean(ft, ft_d, ft_batch)
        ft_enc = np.concatenate([ft_enc, ft_enc_], 1)
        ft_dec = np.concatenate([ft_dec, ft_dec_], 1)
        ft_self_dec = np.concatenate([ft_self_dec, ft_self_dec_], 1)
    print(ft_enc.shape, ft_dec.shape)
    
    print("Encoder CKA")
    for n in range(len(pre_enc)):
        print("Layer", n, linear_CKA(pre_enc[n], ft_enc[n]))
    
    print("\nDecoder CKA")
    for n in range(len(pre_dec)):
        print("Layer", n, linear_CKA(pre_dec[n], ft_dec[n]))
    
    print("\nDecoder up to self attention CKA")
    for n in range(len(pre_self_dec)):
        print("Layer", n, linear_CKA(pre_self_dec[n], ft_self_dec[n]))