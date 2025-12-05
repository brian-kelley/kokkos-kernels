import mala
import torch
from time import perf_counter

#be2 = torch.load('be_model.network.pth')
#print(be2)

parameters, network, data_handler, predictor = mala.Predictor.load_run("Be_model")
print(network)
if torch.cuda.is_available():
    network.to("cuda")
network.eval()

batch = 100000
input = torch.rand((batch, 91), dtype=torch.float)

if torch.cuda.is_available():
    img = img.to("cuda")

# Warmup
resultWarmup = network.forward(input)

print("Warmup result shape:", resultWarmup.shape)

#torch.backends.cudnn.enabled = False

trials = 1000
time_start = perf_counter()

for i in range(trials):
    result = network.forward(input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

time_end = perf_counter()
print("torch interp, averaged over", trials, "trials:", (time_end - time_start) / trials)

model_compiled = torch.compile(network)
resultWarmup = model_compiled.forward(input)
print("Warmup result shape:", resultWarmup.shape)

time_start = perf_counter()

for i in range(trials):
    result = model_compiled.forward(input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

time_end = perf_counter()

print("torch.compile, averaged over", trials, "trials:", (time_end - time_start) / trials)
