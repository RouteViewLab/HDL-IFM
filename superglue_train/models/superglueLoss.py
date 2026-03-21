import torch


def superglueLoss(pred, data):
    epsilon = 1e-6
    all_matches = data['all_matches']  # shape=torch.Size([1, 87, 2])
    nMatch = data['num_match_list']
    scores = pred['scores'].exp()  # shape=(1,N,M)

    indices = [[batchIdx, x, y] for batchIdx, all_match in enumerate(all_matches)
               for x, y in all_match[:int(nMatch[batchIdx][0])]]
    indices = torch.LongTensor(indices).t().contiguous()  # batch index selection
    selected_scores = scores[indices[0], indices[1], indices[2]]  # Select the correct match scores
    loss = -torch.log(selected_scores + epsilon)
    loss_mean = torch.mean(loss).reshape(1, -1)

    # Return the mean loss, the number of correct matches, and the selected scores
    return loss_mean, torch.sum(nMatch).item(), selected_scores