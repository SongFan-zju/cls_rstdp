import numpy as np
import pickle as pkl

num_classes = 5
# weights = np.load("weights.npy")
# weights = np.round(weights * 127)
# axon = []
# for i in range(num_classes + weights.shape[0]):
#     tmp = []
#     if i < num_classes:
#         axon.append([tmp])
#         continue
#     for j in range(num_classes):
#         tmp.append([j, i - num_classes])
#     axon.append(tmp)
# print(axon)
# weights = np.full([16, num_classes], fill_value=0.1)
# weights = np.round(weights * 127)
# all_weights = np.zeros([num_classes + weights.shape[0], num_classes + weights.shape[0]])
# weights = weights.T
# all_weights[:num_classes, :weights.shape[1]] = weights
# print(all_weights)
# with open('pkl/axon.pkl', 'wb') as f:
#     pkl.dump(axon, f)
# with open('pkl/weight_init.pkl', 'wb') as f:
#     pkl.dump(all_weights, f)
with open('pkl/axon.pkl', 'rb') as f:
    axon = pkl.load(f)
with open('pkl/weight_init.pkl', 'rb') as f:
    weights = pkl.load(f)
    print(weights)
print("\n")
with open('pkl/weights.pkl', 'rb') as f:
    weights = pkl.load(f)
    print(weights)
