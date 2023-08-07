# Assuming we have a batch of 2 images. 
# For the first image, we have 3 objects with class labels 1, 2, 3. 
# For the second image, we have 2 objects with class labels 2, 3. 
targets = [{"labels": torch.tensor([1, 2, 3])}, {"labels": torch.tensor([2, 3])}]

# This is the output of the Hungarian algorithm. It's a list of tuples where each tuple corresponds to an image in the batch.
# For each tuple, the first tensor are the indices of the queries in descending order of matching scores, 
# and the second tensor are the indices of the ground truth boxes that these queries are matched to.
indices = [(torch.tensor([0, 1, 2]), torch.tensor([1, 0, 2])), (torch.tensor([0, 1]), torch.tensor([1, 0]))]

# Here we concatenate the true class labels of the matched objects for all images in the batch.
target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

# Let's assume we have 4 classes (class IDs: 0, 1, 2, 3) and an additional "no object" class (class ID: 4).
# src_logits has the shape (2, 4, 5) - 2 images, 4 queries, 5 classes.
# So, we create a tensor of shape (2, 4) filled with the class ID of "no object".
target_classes = torch.full((2, 4), 4, dtype=torch.int64)

# Here we create the indices for the queries that have been matched to objects. 
# The first tensor in the idx tuple corresponds to the batch dimension (image indices), 
# and the second tensor corresponds to the query indices within each image.
idx = (torch.tensor([0, 0, 0, 1, 1]), torch.tensor([0, 1, 2, 0, 1]))

# We assign the class labels of the matched objects to the corresponding queries.
target_classes[idx] = target_classes_o

# Now, target_classes contains the class IDs of the objects that each query is matched with in each image.
# Queries that are not matched with any object are assigned the "no object" class.
print(target_classes)
# tensor([[2, 1, 3, 4],
#         [3, 2, 4, 4]])
