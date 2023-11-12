import deeplake
ds = deeplake.load("hub://activeloop/imagenet-train")

print(ds.shape)