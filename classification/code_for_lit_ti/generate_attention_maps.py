import torch
import torch.nn as nn
import numpy as np
import utils
from timm.models import create_model
import pvt_full_msa
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set_theme()
cmap = sns.light_palette("#A20000", as_cmap=True)
from params import args
import torch.backends.cudnn as cudnn
from datasets import build_dataset

@torch.no_grad()
def get_attention_data(data_loader, model, device, base_path):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    attention_store = []
    samples = 0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        samples += images.size()[0]
        # compute output
        with torch.cuda.amp.autocast():
            output, attention_maps = model(images)
            loss = criterion(output, target)
            if len(attention_store) == 0:
                for i, stage_maps in enumerate(attention_maps):
                    stage_attns = []
                    for j, block_maps in enumerate(stage_maps):
                        # Simply use a summation to aggregate the attention probabilities from all batches,
                        # you can also try to use average or some other scaling methods
                        stage_attns.append(block_maps.sum(dim=0))
                    attention_store.append(stage_attns)
            else:
                for i, stage_maps in enumerate(attention_maps):
                    for j, block_maps in enumerate(stage_maps):
                        attention_store[i][j] += block_maps.sum(dim=0)

        np_attns = []
        for i, stage_maps in enumerate(attention_store):
            stage_attns = []
            for j, block_maps in enumerate(stage_maps):
                block_maps /= samples
                stage_attns.append(block_maps.numpy())
            np_attns.append(stage_attns)
        np.save(os.path.join(base_path, 'full_msa_eval_maps.npy'), np.array(np_attns))
        break

def visualize_attentions(base_path):

    save_path = os.path.join(base_path, 'attention_map')
    attention_maps = np.load(os.path.join(base_path, 'full_msa_eval_maps.npy'), allow_pickle=True)

    linewidths = [1, 1, 2, 2]
    # Remember that PVT has 4 stages
    for stage_id, stage_attn_map in enumerate(attention_maps):
        # each stage has several Transformer blocks
        for block_id, block_attn_map in enumerate(stage_attn_map):

            block_attn_map = torch.from_numpy(block_attn_map) # size: num_head * seq_len * seq_len

            # PVT has the CLS token at the last stage, here we exclude it for better visualization.
            if stage_id == 3:
                test = block_attn_map[:, 1:, :]
                block_attn_map = test[:, :, 1:]

            H, N, _ = block_attn_map.shape
            width = int(N ** (1 / 2))

            # iterate each self-attention head
            for head_id in range(H):
                head_atth_map = block_attn_map[head_id, ...]
                map_save_dir = os.path.join(save_path, 'stage-'+str(stage_id), 'block'+str(block_id))

                if not os.path.exists(map_save_dir):
                    os.makedirs(map_save_dir, exist_ok=True)

                for pixel_id in range(N):
                    # some random pixel indices, just want to make sure the visualized pixel is near the centre.
                    if stage_id == 0 and pixel_id != 1260:
                        continue
                    if stage_id == 1 and pixel_id != 294:
                        continue
                    if stage_id == 2 and pixel_id != 92:
                        continue
                    if stage_id == 3 and pixel_id != 17:
                        continue

                    plt.clf()
                    f, ax = plt.subplots(1, 1, figsize=(4, 4))
                    ax.set_aspect('equal')

                    print(stage_id, block_id, head_id, pixel_id)

                    pixel_attn_map = head_atth_map[pixel_id, ...].reshape(int(N ** (1 / 2)), int(N ** (1 / 2))).numpy()

                    x = int(pixel_id % width)
                    y = int(pixel_id / width)

                    # visualize the attention map with seaborn heatmap
                    ax = sns.heatmap(pixel_attn_map, cmap="OrRd", cbar=False, xticklabels=False, yticklabels=False, ax=ax)
                    patch = patches.Rectangle((x, y), 1, 1, linewidth=linewidths[stage_id], edgecolor='lime', facecolor='None')
                    ax.add_patch(patch)
                    image_name = 'pixel-{}-block-{}-head-{}.png'.format(pixel_id, block_id, head_id)
                    plt.savefig(os.path.join(map_save_dir, image_name), transparent=True)



if __name__ == '__main__':
    # You may change the path for saving the results.
    save_path = 'attn_results'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    dataset_val, _ = build_dataset(is_train=False, args=args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=100,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = create_model(
        'pvt_small_full_msa',
        pretrained=False,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    model.to(device)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)
    get_attention_data(data_loader_val, model, device, save_path)
    visualize_attentions(save_path)



