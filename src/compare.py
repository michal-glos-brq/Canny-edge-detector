from argparse import ArgumentParser
import cv2
import os
import numpy as np
import sys

TRESHOLD = 40

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('input', type=str,
                        help='Image detected with edge detector')
    parser.add_argument('source', type=str,
                        help='Source image')
    parser.add_argument('--debug', action='store_true', help="Show compared images")
    parser.add_argument('--compare-level', '-c', choices=[0,1,2,3,4,5], default=2, type=int,
        help='Choose the level of consensus to be used as reference image.')
    return parser.parse_args()

def read_input(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return (img != 0)

def sobel(img):
    x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    return cv2.addWeighted(cv2.convertScaleAbs(x), 0.5, cv2.convertScaleAbs(y), 0.5, 0)

def prewit(img):
    kx = np.array([[ 1, 1, 1], [ 0, 0, 0], [-1, -1, -1]])
    ky = np.array([[-1, 0, 1], [-1, 0, 1], [-1,  0,  1]])

    x = cv2.filter2D(img, cv2.CV_64F, kx)
    y = cv2.filter2D(img, cv2.CV_64F, ky)

    return cv2.addWeighted(cv2.convertScaleAbs(x), 0.5, cv2.convertScaleAbs(y), 0.5, 0)

def canny(img):
    return cv2.Canny(img, 100, 200)

def laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F, ksize=3)

def roberts(img):
    kx = np.array([[0, 1], [-1, 0]])
    ky = np.array([[1, 0], [0, -1]])

    x = cv2.filter2D(img, cv2.CV_64F, kx)
    y = cv2.filter2D(img, cv2.CV_64F, ky)

    return cv2.addWeighted(cv2.convertScaleAbs(x), 0.5, cv2.convertScaleAbs(y), 0.5, 0)

def generate(path):
    src = cv2.imread(path)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_blurred = cv2.GaussianBlur(src_gray, (3, 3), 0)

    EDGE_DETECTORS = {
        'sobel':     sobel,
        'prewit':    prewit,
        'canny':     canny,
        'laplacian': laplacian,
        'roberst':   roberts
    }

    imgs = []

    for _, ed in EDGE_DETECTORS.items():
        img = cv2.normalize(ed(src_blurred), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        imgs.append(img)

    return imgs
        
def create_concensus_image(imgs, lvl):
    h, w = imgs[0].shape[:2]
    imgs = np.array(imgs) > TRESHOLD
    imgs = np.sum(imgs, axis=0)
    return imgs > lvl

def distance2(ref, img):
    assert ref.shape == img.shape
    # Get coordinates of outliers only in order to speed it up (correct edge would have 0 distance)
    false_positive = np.stack(np.where((img & np.logical_not(ref)))).transpose()
    false_negative = np.stack(np.where((ref & np.logical_not(img)))).transpose()
    # Get coordinates of true edges
    gt_edges = np.stack(np.where(ref)).transpose()
    predicted_edges = np.stack(np.where(img)).transpose()
    # Now, calculate distances between:
    #     false positive detections and gt edges
    #     false negative detections and predicted edges
    if len(false_positive):
        print("Processing false positives ...", file=sys.stderr)
        fpd = np.array([(gt_edges - coord) for coord in false_positive]).transpose(2,0,1)
        _, fpdx, fpdy = fpd.shape
        fpd = np.amin(((fpd[0]**2 + fpd[1]**2)**(1./2.)).reshape((fpdx, fpdy)), axis=1)
    else:
        print("No false positive found ...", file=sys.stderr)
        fpd = np.array([0])
    if len(false_negative):
        print("Processing false negatives ...", file=sys.stderr)
        fnd = np.array([(predicted_edges - coord) for coord in false_negative]).transpose(2,0,1)
        _, fndx, fndy = fnd.shape
        fnd = np.amin(((fnd[0]**2 + fnd[1]**2)**(1./2.)).reshape((fndx, fndy)), axis=1)
    else:
        print("No false negative found ...", file=sys.stderr)
        fnd = np.array([0])
    fall = np.concatenate((fpd, fnd))
    # Compose nice dictionary with neat structure :)
    return {
        'all': {
            'avg':   np.mean(fall),
            'max':   np.max(fall),
            'sigma': np.std(fall),
            'pix':   len(fall)
        },
        'fn':  {
            'avg':   np.mean(fnd),
            'max':   np.max(fnd),
            'sigma': np.std(fnd),
            'pix':   len(fnd)
        },
        'fp':  {
            'avg':   np.mean(fpd),
            'max':   np.max(fpd),
            'sigma': np.std(fpd),
            'pix':   len(fpd)
        }
    }  
    
    
def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f'Invalid input: {args.input}')
    
    if not os.path.isfile(args.source):
        print(f'Invalid source: {args.source}')

    img = read_input(args.input)
    ed_imgs = generate(args.source)
    cons_img = create_concensus_image(ed_imgs, args.compare_level)

    if args.debug:
        c = np.zeros(cons_img.shape)
        c[cons_img] = 255

        x, y = c.shape
        cv2.imshow("ref", cv2.resize(c, (2*y, 2*x)))
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

        i = np.zeros(img.shape)
        i[img] = 255

        cv2.imshow("img", cv2.resize(i, (2*y, 2*x)))
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    result = distance2(cons_img, img)
    for type, data in result.items():
        print(type)
        for stat, val in data.items():
            print(f'\t{stat}: {val}')

if __name__ == '__main__':
    main()