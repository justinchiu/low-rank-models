import sys, os
import torch
import matplotlib.pyplot as plt
import seaborn
filename = sys.argv[1]
output_dir = sys.argv[2]
seaborn.set(font_scale=1.5) 
print (filename)
rules_ll = torch.load(filename, map_location=torch.device('cpu'))

p = rules_ll.exp()

p_portion = p[:, :30, :30].sum() / p.sum()
print (p_portion)
entropies = -(rules_ll.exp() * rules_ll).sum(-1).sum(-1).numpy()

plt.figure()
plt.xlabel('Entropy of Each Head')




#_, bins, _ = plt.hist(entropies, bins=8, range=[0, 4], density=True)
#_ = plt.hist(bar, bins=bins, alpha=0.5, normed=True)
seaborn.histplot(entropies, bins=8, binrange=[0,4], stat='density' )
plt.ylim(0, 0.7)
plt.tight_layout()
#plt.xlim(0, 4)

plt.savefig(os.path.join(output_dir, f'entropy.png'))


#entropy = -(rules_ll.exp() * rules_ll).sum(-1).sum(-1).mean()
#print (entropy)
#
#num_samples = 10
##rules_ll = rules_ll.exp()
#NT, NT_T, _ = rules_ll.shape
#inds = list(range(NT))
#print (NT)
#import random
#random.seed(1234)
#random.shuffle(inds)
#os.makedirs(output_dir, exist_ok=True)
##for ind in inds:
##    img = rules_ll[ind]
##    ax=seaborn.heatmap(img.cpu().numpy())
##    plt.savefig(os.path.join(output_dir, f'{ind}.png'))
#
#for ind in inds:
#    img = rules_ll[ind]
#    plt.figure()
#    #import pdb; pdb.set_trace()
#    ax = seaborn.heatmap(img.cpu().exp().numpy())
#    #plt.imshow(img.cpu().exp().numpy())
#    plt.tight_layout()
#    plt.savefig(os.path.join(output_dir, f'{ind}_exp.png'))
