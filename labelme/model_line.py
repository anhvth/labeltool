import sys

sys.path.append('..')
from pyson.utils import *
from skimage import measure
from pyson.unet import *

model = unet_model('labelme/model/')


def imgpath2lines(path):
    img = read_img(path)
    out = model.run(img)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 100, 255,
                           cv2.THRESH_BINARY_INV)[1]

    ngang = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones([1, 50]))
    doc = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones([50, 1]))
    pngang, lines_ngang = process(ngang, 'h', doc)
    pdoc, lines_doc = process(doc, 'v', ngang)
    points = lines_ngang + lines_doc
    d = points2dict(path, points)
    return d


def process(img, mode='h', img_2=None):
    if mode == 'h':
        channel = 1
    elif mode == 'v':
        channel = 0
    else:
        assert mode == 'h' or mode == 'v'

    labels = measure.label(img, neighbors=8, background=0)
    pad = np.zeros_like(img)
    lines = []
    for label in np.unique(labels):
        if label == 0:
            continue

        mask = labels == label
        mask = (mask * 255).astype('uint8')
        if img_2 is not None:
            kernel = np.ones((100, 15)) if mode == 'h' else np.ones((15, 100))
            mask_dilate = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
            mask_2 = mask * img_2
            non_zeros = np.count_nonzero(mask_2)
            if non_zeros == 0:
                continue
        idxs = np.column_stack(np.where(mask > 0))
        min_idx, max_idx = np.argmin(idxs[:, channel]), np.argmax(idxs[:, channel])
        a, b = tuple(idxs[min_idx][::-1]), tuple(idxs[max_idx][::-1])
        #         show(mask)
        #         print(a, b)
        cv2.line(pad, a, b, 255, 3)
        lines.append((a, b))
    return pad, lines


def points2dict(file_path, points, old_dict=None):
    def point2dict(a, b):
        xa, ya = a
        xb, yb = b
        d = {'label': 'line',
             'line_color': [255, 0, 0],
             'fill_color': None,
             'points': [[int(xa), int(ya)], [int(xb), int(yb)]]
             }
        return d

    file_name = os.path.split(file_path)[-1]

    if old_dict is None:

        d = {'flags': {},
             'lineColor': [0, 255, 0, 128],
             'fillColor': [0, 0, 0, 128],
             'imagePath': file_name}
        #         d['imageData'] = self.imageData
        point_list = [point2dict(p[0], p[1]) for p in points]
    else:
        d = old_dict
        point_list = d['shape']
        for p in points:
            point_list.append(point2dict(p[0], p[1]))

    d['shapes'] = point_list
    d['imageData'] = None
    return d
