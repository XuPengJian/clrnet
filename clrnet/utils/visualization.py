import cv2
import os
import os.path as osp

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]


def get_first_point(lanes):
    lanes_xys = []
    first_point_list = []
    # 检查轨迹线的合法性，并抽离出来
    for _, lane in enumerate(lanes):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
        lanes_xys.append(xys)
    lanes_xys.sort(key=lambda xys: xys[0][0])

    # 遍历每一条轨迹
    for idx, xys in enumerate(lanes_xys):
        # 遍历每一个点
        for i in range(1, len(xys)):
            # 看看第一个点的位置是否就是我要的点
            if i == 1:
                first_point_list.append(xys[0])
    return first_point_list


def imshow_lanes(img, lanes, show=False, out_file=None, width=4):
    lanes_xys = []
    first_point_list = []
    # 检查轨迹线的合法性，并抽离出来
    for _, lane in enumerate(lanes):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
        lanes_xys.append(xys)
    lanes_xys.sort(key=lambda xys: xys[0][0])

    # 遍历每一条轨迹
    for idx, xys in enumerate(lanes_xys):
        # 遍历每一个点
        for i in range(1, len(xys)):
            cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)

            # 看看第一个点的位置是否就是我要的点
            if i == 1:
                first_point_list.append(xys[0])
                cv2.circle(img, xys[0], 6, (255, 0, 0), 2)

    # print(first_point_list)

    if show:
        cv2.imshow('view', img)
        cv2.waitKey(0)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)
