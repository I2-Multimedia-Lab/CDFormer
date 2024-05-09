import torch
from model.dasr import BlindSR
from option import args

# from torchvision.models.resnet import resnet101

iterations = 300   # 重复计算的轮次

dir = './experiment/blindsr_x4_bicubic_iso'
model = BlindSR(args).cuda()
model.load_state_dict(torch.load(dir + '/model/model_' + str(600) + '.pt',map_location='cuda:0'), strict=False)
model.eval()
print(dir + '/model/model_' + str(600) + '.pt')
device = torch.device("cuda:0")
model.to(device)

# x = torch.randn(size=(2, 2, 3, 64, 64))
random_input = torch.randn(1, 3, 64, 64).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))
