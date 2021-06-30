import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import os

def show_boxes_simple(bbox, color='r', lw=2):
    rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False, edgecolor=color, linewidth=lw)
    plt.gca().add_patch(rect)

def kernel_inv_map(vis_attr, target_point, map_h, map_w):
    ratio = 2
    pos_shift = [0, 1]
    source_point = []
    for idx in range(vis_attr['filter_size']**2):
        cur_source_point = np.array([target_point[0] * ratio + pos_shift[int(idx / 2)],
                                     target_point[1] * ratio + pos_shift[int(idx % 2)]])
        source_point.append(cur_source_point.astype('f'))
    return source_point

def offset_inv_map(source_points, offset):
    offset = offset.reshape(np.prod(offset.shape))
    for idx, _ in enumerate(source_points):
        source_points[idx][0] += offset[2*idx]
        source_points[idx][1] += offset[2*idx + 1]
    return source_points

def get_bottom_position(vis_attr, top_points, all_offset):

    for level in range(vis_attr['plot_level']):
        map_h = all_offset[level].shape[2]
        map_w = all_offset[level].shape[3]
        source_points = []
        for idx, cur_top_point in enumerate(top_points):
            cur_top_point = np.round(cur_top_point)
            if cur_top_point[0] < 0 or cur_top_point[1] < 0 \
                    or cur_top_point[0] > map_h - 1 or cur_top_point[1] > map_w - 1:
                continue
            cur_source_point = kernel_inv_map(vis_attr, cur_top_point, map_h, map_w)
            cur_offset = np.squeeze(all_offset[level][:, :, int(cur_top_point[0]), int(cur_top_point[1])])
            cur_source_point = offset_inv_map(cur_source_point, cur_offset)
            source_points = source_points + cur_source_point

        top_points = source_points
    return source_points

def plot_according_to_point(vis_attr, im, source_points, map_h, map_w, color=[255,0,0]):
    plot_area = vis_attr['plot_area']
    for idx, cur_source_point in enumerate(source_points):
        y = np.round((cur_source_point[0] + 0.5) * 4).astype('i')
        x = np.round((cur_source_point[1] + 0.5) * 4).astype('i')

        y = min(y, im.shape[0] - vis_attr['plot_area'] - 1)
        x = min(x, im.shape[1] - vis_attr['plot_area'] - 1)
        y = max(y, vis_attr['plot_area'])
        x = max(x, vis_attr['plot_area'])
        im[y-plot_area:y+plot_area+1, x-plot_area:x+plot_area+1, :] = np.tile(
            np.reshape(color, (1, 1, 3)), (2*plot_area+1, 2*plot_area+1, 1)
        )
    return im

def get_sample_points(vis_attr, im, source_points, map_h, map_w, color=[255,0,0]):
    # plot_area = vis_attr['plot_area']
    sample_points = []
    for idx, cur_source_point in enumerate(source_points):
        y = np.round((cur_source_point[0]) * 4 + 2).astype('i')
        x = np.round((cur_source_point[1]) * 4 + 2).astype('i')

        if x < 0 or y < 0 or x > im.shape[1]-1 or y > im.shape[0]-1:
            continue
        y = min(y, im.shape[0] - vis_attr['plot_area'] - 1)
        x = min(x, im.shape[1] - vis_attr['plot_area'] - 1)
        y = max(y, vis_attr['plot_area'])
        x = max(x, vis_attr['plot_area'])
        sample_points.append([x,y])
    return sample_points

def show_dpsroi_offset(im, boxes, offset, classes, trans_std=0.1):
    plt.cla
    for idx, bbox in enumerate(boxes):
        plt.figure(idx+1)
        plt.axis("off")
        plt.imshow(im)

        offset_w = np.squeeze(offset[idx, classes[idx]*2, :, :]) * trans_std
        offset_h = np.squeeze(offset[idx, classes[idx]*2+1, :, :]) * trans_std
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        roi_width = x2-x1+1
        roi_height = y2-y1+1
        part_size = offset_w.shape[0]
        bin_size_w = roi_width / part_size
        bin_size_h = roi_height / part_size
        show_boxes_simple(bbox, color='b')
        for ih in range(part_size):
            for iw in range(part_size):
                sub_box = np.array([x1+iw*bin_size_w, y1+ih*bin_size_h,
                                    x1+(iw+1)*bin_size_w, y1+(ih+1)*bin_size_h])
                sub_offset = offset_h[ih, iw] * np.array([0, 1, 0, 1]) * roi_height \
                             + offset_w[ih, iw] * np.array([1, 0, 1, 0]) * roi_width
                sub_box = sub_box + sub_offset
                show_boxes_simple(sub_box)
        plt.show()

def show_dconv_offset(im, all_offset, step=[2, 2], filter_size=3,
                      dilation=2, pad=2, plot_area=2, plot_level=3, image_name=None):
    vis_attr = {'filter_size': filter_size, 'dilation': dilation, 'pad': pad,
                'plot_area': plot_area, 'plot_level': plot_level}

    map_h = all_offset[0].shape[2]
    map_w = all_offset[0].shape[3]

    step_h = step[0]
    step_w = step[1]
    start_h = int(np.round(step_h / 2))
    start_w = int(np.round(step_w / 2))

    plt.figure()
    for im_h in range(start_h, map_h, step_h):
        for im_w in range(start_w, map_w, step_w):
            target_point = np.array([im_h, im_w])
            cur_im = np.copy(im)
            source_points = get_bottom_position(vis_attr, [target_point], all_offset)
            sample_points = get_sample_points(vis_attr, cur_im, source_points, map_h, map_w)
            plt.clf()
            fig, ax = plt.subplots()
            ax.imshow(cur_im)

            for point in sample_points:
                patch = patches.Rectangle(point, plot_area, plot_area, linewidth=1, edgecolor='red', facecolor='red')
                ax.add_patch(patch)

            patch_size = 32
            patch = patches.Rectangle((im_w*patch_size, im_h*patch_size), patch_size, patch_size, linewidth=1, edgecolor='lime',
                                      facecolor='None')
            ax.add_patch(patch)
            plt.axis("off")
            plt.imshow(cur_im)
            plt.tight_layout()
            save_dir = os.path.join(os.getcwd(), 'visualization', image_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/{im_h}-{im_w}.png')
            plt.clf()
            plt.close()