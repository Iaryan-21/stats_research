# stats_research
## OUTPUT 
```
(base) aryanmishra@Aryans-MacBook-Air Niyogi_Paper % python train.py
Epoch 0
Classification Loss: 0.5965
Manifold Loss: 4.6981
Gradient for fc1.weight: 0.0665
Gradient for fc1.bias: 0.0000
Gradient for bn1.weight: 0.0954
Gradient for bn1.bias: 0.0803
Gradient for res_block1.fc.weight: 0.0677
Gradient for res_block1.fc.bias: 0.0000
Gradient for res_block1.bn.weight: 0.0508
Gradient for res_block1.bn.bias: 0.0332
Gradient for res_block2.fc.weight: 0.0663
Gradient for res_block2.fc.bias: 0.0000
Gradient for res_block2.bn.weight: 0.0505
Gradient for res_block2.bn.bias: 0.0324
Gradient for fc_out.weight: 1.1100
Gradient for fc_out.bias: 0.0996
Gradient for graph_laplacian: 0.0006
Epoch [50/1000], Class. Loss: 0.5067, Manifold Loss: 0.1965, Total Loss: 0.7032, Test Accuracy: 0.9433
Epoch [100/1000], Class. Loss: 0.4553, Manifold Loss: 0.1391, Total Loss: 0.5945, Test Accuracy: 0.9733
Epoch 100
Classification Loss: 0.4567
Manifold Loss: 0.1458
Gradient for fc1.weight: 0.0043
Gradient for fc1.bias: 0.0000
Gradient for bn1.weight: 0.0017
Gradient for bn1.bias: 0.0021
Gradient for res_block1.fc.weight: 0.0021
Gradient for res_block1.fc.bias: 0.0000
Gradient for res_block1.bn.weight: 0.0011
Gradient for res_block1.bn.bias: 0.0009
Gradient for res_block2.fc.weight: 0.0025
Gradient for res_block2.fc.bias: 0.0000
Gradient for res_block2.bn.weight: 0.0016
Gradient for res_block2.bn.bias: 0.0011
Gradient for fc_out.weight: 0.0443
Gradient for fc_out.bias: 0.0041
Gradient for graph_laplacian: 0.0121
Epoch [150/1000], Class. Loss: 0.4171, Manifold Loss: 0.1305, Total Loss: 0.5475, Test Accuracy: 0.9767
Epoch [200/1000], Class. Loss: 0.3805, Manifold Loss: 0.1182, Total Loss: 0.4987, Test Accuracy: 0.9767
Epoch 200
Classification Loss: 0.3793
Manifold Loss: 0.1271
Gradient for fc1.weight: 0.0040
Gradient for fc1.bias: 0.0000
Gradient for bn1.weight: 0.0015
Gradient for bn1.bias: 0.0021
Gradient for res_block1.fc.weight: 0.0015
Gradient for res_block1.fc.bias: 0.0000
Gradient for res_block1.bn.weight: 0.0008
Gradient for res_block1.bn.bias: 0.0010
Gradient for res_block2.fc.weight: 0.0018
Gradient for res_block2.fc.bias: 0.0000
Gradient for res_block2.bn.weight: 0.0012
Gradient for res_block2.bn.bias: 0.0009
Gradient for fc_out.weight: 0.0379
Gradient for fc_out.bias: 0.0000
Gradient for graph_laplacian: 0.0375
Epoch [250/1000], Class. Loss: 0.3457, Manifold Loss: 0.1246, Total Loss: 0.4703, Test Accuracy: 0.9833
Epoch [300/1000], Class. Loss: 0.3213, Manifold Loss: 0.1203, Total Loss: 0.4416, Test Accuracy: 0.9833
Epoch 300
Classification Loss: 0.3230
Manifold Loss: 0.1285
Gradient for fc1.weight: 0.0045
Gradient for fc1.bias: 0.0000
Gradient for bn1.weight: 0.0011
Gradient for bn1.bias: 0.0019
Gradient for res_block1.fc.weight: 0.0013
Gradient for res_block1.fc.bias: 0.0000
Gradient for res_block1.bn.weight: 0.0009
Gradient for res_block1.bn.bias: 0.0007
Gradient for res_block2.fc.weight: 0.0014
Gradient for res_block2.fc.bias: 0.0000
Gradient for res_block2.bn.weight: 0.0009
Gradient for res_block2.bn.bias: 0.0007
Gradient for fc_out.weight: 0.0290
Gradient for fc_out.bias: 0.0045
Gradient for graph_laplacian: 0.0831
Epoch [350/1000], Class. Loss: 0.2978, Manifold Loss: 0.1248, Total Loss: 0.4226, Test Accuracy: 0.9800
Epoch [400/1000], Class. Loss: 0.2852, Manifold Loss: 0.1223, Total Loss: 0.4075, Test Accuracy: 0.9867
Epoch 400
Classification Loss: 0.2821
Manifold Loss: 0.1266
Gradient for fc1.weight: 0.0033
Gradient for fc1.bias: 0.0000
Gradient for bn1.weight: 0.0010
Gradient for bn1.bias: 0.0016
Gradient for res_block1.fc.weight: 0.0012
Gradient for res_block1.fc.bias: 0.0000
Gradient for res_block1.bn.weight: 0.0007
Gradient for res_block1.bn.bias: 0.0009
Gradient for res_block2.fc.weight: 0.0012
Gradient for res_block2.fc.bias: 0.0000
Gradient for res_block2.bn.weight: 0.0006
Gradient for res_block2.bn.bias: 0.0006
Gradient for fc_out.weight: 0.0241
Gradient for fc_out.bias: 0.0009
Gradient for graph_laplacian: 0.1479
Epoch [450/1000], Class. Loss: 0.2711, Manifold Loss: 0.1262, Total Loss: 0.3973, Test Accuracy: 0.9867
Epoch [500/1000], Class. Loss: 0.2614, Manifold Loss: 0.1311, Total Loss: 0.3925, Test Accuracy: 0.9800
Epoch 500
Classification Loss: 0.2663
Manifold Loss: 0.1274
Gradient for fc1.weight: 0.0029
Gradient for fc1.bias: 0.0000
Gradient for bn1.weight: 0.0009
Gradient for bn1.bias: 0.0018
Gradient for res_block1.fc.weight: 0.0011
Gradient for res_block1.fc.bias: 0.0000
Gradient for res_block1.bn.weight: 0.0006
Gradient for res_block1.bn.bias: 0.0007
Gradient for res_block2.fc.weight: 0.0011
Gradient for res_block2.fc.bias: 0.0000
Gradient for res_block2.bn.weight: 0.0005
Gradient for res_block2.bn.bias: 0.0007
Gradient for fc_out.weight: 0.0237
Gradient for fc_out.bias: 0.0011
Gradient for graph_laplacian: 0.2263
Epoch [550/1000], Class. Loss: 0.2567, Manifold Loss: 0.1295, Total Loss: 0.3862, Test Accuracy: 0.9833
Epoch [600/1000], Class. Loss: 0.2524, Manifold Loss: 0.1190, Total Loss: 0.3715, Test Accuracy: 0.9800
Epoch 600
Classification Loss: 0.2531
Manifold Loss: 0.1258
Gradient for fc1.weight: 0.0032
Gradient for fc1.bias: 0.0000
Gradient for bn1.weight: 0.0007
Gradient for bn1.bias: 0.0016
Gradient for res_block1.fc.weight: 0.0010
Gradient for res_block1.fc.bias: 0.0000
Gradient for res_block1.bn.weight: 0.0005
Gradient for res_block1.bn.bias: 0.0008
Gradient for res_block2.fc.weight: 0.0010
Gradient for res_block2.fc.bias: 0.0000
Gradient for res_block2.bn.weight: 0.0006
Gradient for res_block2.bn.bias: 0.0005
Gradient for fc_out.weight: 0.0210
Gradient for fc_out.bias: 0.0020
Gradient for graph_laplacian: 0.3128
Epoch [650/1000], Class. Loss: 0.2490, Manifold Loss: 0.1231, Total Loss: 0.3721, Test Accuracy: 0.9800
Epoch [700/1000], Class. Loss: 0.2431, Manifold Loss: 0.1254, Total Loss: 0.3685, Test Accuracy: 0.9800
Epoch 700
Classification Loss: 0.2475
Manifold Loss: 0.1231
Gradient for fc1.weight: 0.0030
Gradient for fc1.bias: 0.0000
Gradient for bn1.weight: 0.0008
Gradient for bn1.bias: 0.0017
Gradient for res_block1.fc.weight: 0.0009
Gradient for res_block1.fc.bias: 0.0000
Gradient for res_block1.bn.weight: 0.0005
Gradient for res_block1.bn.bias: 0.0006
Gradient for res_block2.fc.weight: 0.0009
Gradient for res_block2.fc.bias: 0.0000
Gradient for res_block2.bn.weight: 0.0005
Gradient for res_block2.bn.bias: 0.0005
Gradient for fc_out.weight: 0.0207
Gradient for fc_out.bias: 0.0019
Gradient for graph_laplacian: 0.4056
Epoch [750/1000], Class. Loss: 0.2445, Manifold Loss: 0.1260, Total Loss: 0.3705, Test Accuracy: 0.9833
Epoch [800/1000], Class. Loss: 0.2387, Manifold Loss: 0.1173, Total Loss: 0.3560, Test Accuracy: 0.9800
Epoch 800
Classification Loss: 0.2424
Manifold Loss: 0.1251
Gradient for fc1.weight: 0.0033
Gradient for fc1.bias: 0.0000
Gradient for bn1.weight: 0.0008
Gradient for bn1.bias: 0.0016
Gradient for res_block1.fc.weight: 0.0010
Gradient for res_block1.fc.bias: 0.0000
Gradient for res_block1.bn.weight: 0.0005
Gradient for res_block1.bn.bias: 0.0007
Gradient for res_block2.fc.weight: 0.0009
Gradient for res_block2.fc.bias: 0.0000
Gradient for res_block2.bn.weight: 0.0004
Gradient for res_block2.bn.bias: 0.0005
Gradient for fc_out.weight: 0.0172
Gradient for fc_out.bias: 0.0005
Gradient for graph_laplacian: 0.5032
Epoch [850/1000], Class. Loss: 0.2348, Manifold Loss: 0.1183, Total Loss: 0.3531, Test Accuracy: 0.9833
Epoch [900/1000], Class. Loss: 0.2354, Manifold Loss: 0.1259, Total Loss: 0.3613, Test Accuracy: 0.9867
Epoch 900
Classification Loss: 0.2368
Manifold Loss: 0.1211
Gradient for fc1.weight: 0.0025
Gradient for fc1.bias: 0.0000
Gradient for bn1.weight: 0.0006
Gradient for bn1.bias: 0.0018
Gradient for res_block1.fc.weight: 0.0008
Gradient for res_block1.fc.bias: 0.0000
Gradient for res_block1.bn.weight: 0.0004
Gradient for res_block1.bn.bias: 0.0006
Gradient for res_block2.fc.weight: 0.0009
Gradient for res_block2.fc.bias: 0.0000
Gradient for res_block2.bn.weight: 0.0005
Gradient for res_block2.bn.bias: 0.0005
Gradient for fc_out.weight: 0.0181
Gradient for fc_out.bias: 0.0013
Gradient for graph_laplacian: 0.6048
Epoch [950/1000], Class. Loss: 0.2294, Manifold Loss: 0.1238, Total Loss: 0.3531, Test Accuracy: 0.9800
Epoch [1000/1000], Class. Loss: 0.2355, Manifold Loss: 0.1171, Total Loss: 0.3526, Test Accuracy: 0.9800
Final Test Accuracy: 0.9800
```
