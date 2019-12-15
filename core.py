import torch
import numpy as np
import faiss

def index_features_at_layer(layer):
	index = faiss.IndexFlatL2(2048)
	test = []
	l = []
	vectors = torch.load("query_fea.pth")
	i = 0
	j = 0

	for img in vectors["img_id"]:
		if (i == layer):
			l = np.asarray(img)
		i += 1

	for feature in vectors["features"]:
		if (j == layer):
			test = feature.data.cpu().numpy()
			index.add(feature.data.cpu().numpy())
		j += 1

	t = 0
	for name in l:
		print(name)
		print(test[t])
		print()
		t += 1

	k = 4
	D, I = index.search(test,k)
	print(I)

index_features_at_layer(19)
