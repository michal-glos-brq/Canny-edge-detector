import cv2
import argparse
from os.path import exists
import sys
import numpy as np

def parse_args():
    '''Parse CLI arguments'''
    parser = argparse.ArgumentParser()
    # Input image (only argument without a flag before)
    parser.add_argument('input', type=str,
                       help='Path to a image you want to have processed.')

    # Configure blurring with sigma of gaussian filter applied
    b_sigma = parser.add_mutually_exclusive_group()
    b_sigma.add_argument('--b-sigma', '--bs', type=float,
                        help="Sigma of gaussian kernel used to blur the input image. When not provided, " \
                             "pseudo optimal sigma would be computed.")
    b_sigma.add_argument('--b-sigma-c', '--bsc', type=float, default=1,
                        help="Coefficient of automatic determinated sigma. Estimated sigma would be multiplied by this coefficient, same rules as above apply. " \
                              "If there is still some noise left, try to set it higher, if some edges vanish, set it lower.")

    # Configure edge detection
    parser.add_argument('--rgb', action='store_true',
                        help='Calculate gradients on RGB channels. Would be calculated on'\
                             ' greyscale image otherwise.')
    parser.add_argument('--g-ksize', '--gk', type=int, default=3,
                        help='Size of gaussian kernel used to blur the input image.')

    # Configure double treshold ratio
    parser.add_argument('--ltr', type=float, default=0.033,
                        help='Define low treshold ratio for double tresholding phase.')
    parser.add_argument('--htr', type=float, default=0.1,
                        help='Define high treshold ratio for double tresholding phase.')

    # Add some output options
    parser.add_argument('--max-resolution', '--mr', type=int, nargs=2, default=[1920, 1080],
                        help="Provide your resolution (width, height) to resize images in order to fit them inside your screen")
    parser.add_argument('--output', '-o', type=str, default="output",
                        help='Define name od output file (file extension excluded).')
    parser.add_argument('--pdf', action='store_true',
                        help='Save all outputs as one pdf.')
    parser.add_argument('--save', action='store_true',
                        help='Save the output images.')
    parser.add_argument('--show', action='store_true',
                        help='Show the output on screen.')
    parser.add_argument('--step', action='store_true',
                        help='Output picture after each phase of canny edge detector.')
    parser.add_argument('--grid', action='store_true',
                        help='Show or save a grid composed of image in each step during canny edga finding operation.')

    return parser.parse_args()

class Canny:

############################################################################
############### Initialize class and load it's configuration ###############
############################################################################

    def guess_sigma(self, c):
        '''
        The rule of thumb may be setting the sigma in order to create a kernel with
        one side of lenght 1/100 of the shorter side of input image
        '''
        h, w, _ = self.image_input.shape
        return c * min(h, w) / 600.

    def __init__(self, args):
        '''
        Initialize canny object, parse the configuration

        args:
            args:   Arguments provided on command line interface
        '''      
        self.config = {
            # Paths to image
            "image_path": args.input,
            "output": "canny-" + args.input.split("/")[-1].split(".")[0] if args.output is None else args.output,
        }
        # Load the image to use it's data to dynamically obtain the optimal parameters
        self.load_image()
        # Blur
        # Guess the sigma
        self.config['b_sigma'] = args.b_sigma if args.b_sigma else self.guess_sigma(args.b_sigma_c)
        # This could be just cut when values outside the kernel would be pretty low
        self.config['b_ksize'] = int(6 * self.config['b_sigma']) + 1
        self.config.update({
            # Edge detection
            "rgb_grads": args.rgb,
            "g_ksize": args.g_ksize,
            # Tresholds
            "lti": 0.2,
            "hti": 1,
            "ltr": args.ltr,
            "htr": args.htr,
            # Settings
            "max_resolution": args.max_resolution,
            "show": args.show,
            "save": args.save,
            "step": args.step,
            "grid": args.grid
        })

###########################################################################
############### Composing, showing and saving output images ###############
###########################################################################

    def _create_grid(self):
        '''
        Creates grid of images:
        |    input rgb   |  blurred  |        angles      | double treshold |
        | input gryscale | gradients | non-max supression |    hysterezis   |
        or transposed (what better fits the screen)
        '''
        # Create grayscale image to be shown
        grayscale = cv2.cvtColor(self.image_input, cv2.COLOR_BGR2GRAY)
        grayscale = self.img_normalize2int(grayscale)
        # Compose images into a single grid of images:
        h, w, _ = self.image_input.shape
        # Try to optimize a bit the grid constitutiuon
        vertical_coef = (2*w) / (4*h) - (self.config['max_resolution'][0]/self.config['max_resolution'][1])
        horizontal_coef = (4*w) / (2*h) - (self.config['max_resolution'][0]/self.config['max_resolution'][1])

        if len(grayscale.shape) < 3:
            grayscale = np.array([grayscale, grayscale, grayscale]).transpose((1,2,0))
        if len(self.image_input.shape) < 3:
            self.image_input = np.array([self.image_input, self.image_input, self.image_input]).transpose((1,2,0))
        if len(self.image_blurred.shape) < 3:
            self.image_blurred = np.array([self.image_blurred, self.image_blurred, self.image_blurred]).transpose((1,2,0))
        if len(self.image_angles.shape) < 3:
            self.image_angles = np.array([self.image_angles, self.image_angles, self.image_angles]).transpose((1,2,0))
        if len(self.image_double_tresholded.shape) < 3:
            self.image_double_tresholded = np.array([self.image_double_tresholded, self.image_double_tresholded, self.image_double_tresholded]).transpose((1,2,0))
        if len(self.image_hysterezis.shape) < 3:
            self.image_hysterezis = np.array([self.image_hysterezis, self.image_hysterezis, self.image_hysterezis]).transpose((1,2,0))
        if len(self.image_gradients.shape) < 3:
            self.image_gradients = np.array([self.image_gradients, self.image_gradients, self.image_gradients]).transpose((1,2,0))
        if len(self.image_non_max_supressed.shape) < 3:
            self.image_non_max_supressed = np.array([self.image_non_max_supressed, self.image_non_max_supressed, self.image_non_max_supressed]).transpose((1,2,0))
        self.image_input = self.image_input / 255
        grayscale = grayscale / 255
        self.image_blurred = self.image_blurred / 255
        if vertical_coef > horizontal_coef:
            img = np.concatenate((
                np.concatenate((self.image_input, self.image_blurred, self.image_angles, self.image_double_tresholded)),
                np.concatenate((grayscale, self.image_gradients, self.image_non_max_supressed, self.image_hysterezis))
            ), axis=1)    
        else:
            img = np.concatenate((
                np.concatenate((self.image_input, grayscale)),
                np.concatenate((self.image_blurred, self.image_gradients)),
                np.concatenate((self.image_angles, self.image_non_max_supressed)),
                np.concatenate((self.image_double_tresholded, self.image_hysterezis))
            ), axis=1)
        return img

    def show(self, img, title):
        '''
        Resizes image to fit it into the screen defined in max_resolution config variable
        '''
        c = 1
        size = img.shape
        while (size[1]/c > self.config['max_resolution'][0] or size[0]/c > self.config['max_resolution'][1]):
            c += 1
        img = cv2.resize(img, (int(size[1]/c), int(size[0]/c)), interpolation = cv2.INTER_AREA)
        cv2.imshow(title, img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    def save(self, img, content):
        '''
        Save image nicely

        Find the correct file extension and compose it's file name
        '''
        # choose the correct suffix
        suffix = self.config['image_path'].split('.')[-1]
        file_safe_phase_name = content.replace(' ', '_').lower()
        # Start with output without prefix, add it's operation order (for sorting) and it's phase.
        # Finally, get it's correct suffix for saving it.
        output = self.config['output']
        cv2.imwrite(f'{output}-{file_safe_phase_name}.{suffix}', img)

    def show_steps(self):
        '''
        If step option in args is True, show or/and save the image after each step
        '''
        # Organize them into an iterable
        STEPS = [
            ("Original image", self.image_input),
            ("Blurred image", self.image_blurred),
            ("Gradients", self.image_grad_normalized),
            ("Angles", self.image_angles_normalized),
            ("Non max supression", self.image_non_max_supressed_normalized),
            ("Double tresholded", self.image_tresholded_normalized),
            ("Hysterezis", self.image_hysterized_normalized)
        ]
        for phase, img in STEPS:
            if self.config['show']:
                self.show(img, phase)
            if self.config['save']:
                self.save(img, phase)

################################################
############### Image processing ###############
################################################ 

    def apply_canny(self):
        '''
        Apply the canny operator, store image from each phase as class property
        '''
        # Aplly all phases of canny detector
        for d, f in zip(
                ["Blurring ...", "Detecting edges ...", "Supressing ...", "Tresholding ...", "Hysterezis ..."],
                [self.apply_blur, self.apply_edge_detection, self.apply_non_max_supression, self.apply_double_treshold, self.apply_hysterezis]
            ):
            print(d)
            f()

        # Make'em 3 channels to concatenate and show correctly
        self.image_grad_normalized = self.img_normalize2int(self.image_gradients, norm_value=255)
        self.image_angles_normalized = self.img_normalize2int(self.image_angles, norm_value=255/np.pi)
        self.image_non_max_supressed_normalized = self.img_normalize2int(self.image_non_max_supressed, norm_value=255)
        self.image_tresholded_normalized = self.img_normalize2int(self.image_double_tresholded, norm_value=255)
        self.image_hysterized_normalized = self.img_normalize2int(self.image_hysterezis, norm_value=255)

        if args.step:
            self.show_steps()
        else:
            img = self.image_hysterized_normalized
            title = "Edges detected by canny"
            if args.save:
                self.save(img, title)
            if args.show:
                self.show(img, title)
        if args.grid:
            img = self._create_grid()
            title = "All canny phases"
            if args.save:
                self.save(img, title)
            if args.show:
                self.show(img, title)
            
        # Save all images in a grid
        

    def load_image(self):
        '''
        Read input image and save it as a property, throw an error if image does not exist

        args:
            img_path:   Path to a image to be processed
        returns:
            3D numpy array (Image from img_path with 3 RGB channels)
        '''
        if exists(self.config['image_path']):
            self.image_input = cv2.imread(self.config['image_path'])
        else:
            sys.exit(f"ERROR: Provided image on path '{self.config['image_path']}' does not exist.")

    def apply_blur(self):
        '''
        Apply gaussian filter on input image to reduce it's noise
        '''
        kernel = cv2.getGaussianKernel(self.config['b_ksize'], self.config['b_sigma'])
        self.image_blurred = cv2.filter2D(self.image_input, cv2.CV_8U, kernel)

    def apply_edge_detection(self):
        '''
        Detect edges in input image and calculate it's angles.
        '''
        # If we do not want to apply Sobel filter through all channels, convert image to greyscale
        if not self.config['rgb_grads']:
            img = cv2.cvtColor(self.image_blurred, cv2.COLOR_BGR2GRAY)
        # Calculate the gradients
        gx = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.config['g_ksize'])
        gy = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.config['g_ksize'])
        # Combine gradients of x and y
        self.image_gradients = np.hypot(gx, gy)
        # If we want to apply Sobel filter through all channels, sum the results for each channel
        if self.config['rgb_grads']:
            self.image_gradients = self.image_gradients.sum(axis=2)
            gx = gx.sum(axis=2)
            gy = gy.sum(axis=2)
        # Calculate the gradient for both (x,y) coordinates
        self.image_angles = np.arctan2(gx, gy)
        # Normalize the image (float: <0,1>)
        self.image_gradients /= img.max()

    def apply_non_max_supression(self):
        '''
        Apply non-max supression. Make the lines 1px thick.
        '''
        # Pad the input image with zeros, create new empty instance of image
        h, w = self.image_gradients.shape
        padded_img = np.zeros((h+2, w+2))
        padded_img[1:-1, 1:-1] = self.image_gradients
        new_img = np.zeros_like(self.image_gradients)

        # Normalize the angles
        self.image_angles /= np.pi
        self.image_angles[self.image_angles < 0] += 1

        # Apply the thinning procedure - each pixel left as edge should have the highest intensity
        # among it's neighbours in the direction of normal vector of the edge
        for x in range(w):
            for y in range(h):
                # For each pixel, check it's neighbours. Write the pixel only if it's neighbours have less intensity
                angle = self.image_angles[y, x]
                dy = (-1 if angle > 0.5 else 1) if angle > 0.125 and angle < 0.875 else 0
                dx = 1 if angle < 0.375 or angle > 0.625 else 0
                if self.image_gradients[y,x] > padded_img[y+dy+1,x+dx+1] and self.image_gradients[y,x] > padded_img[y-dy+1,x-dx+1]:
                    new_img[y,x] = self.image_gradients[y,x]
        self.image_non_max_supressed = new_img

    def apply_double_treshold(self):
        '''
        Apply double tresholding
        '''
        # Compute treshold absolute value from treshold ratio and intensity of max valued pixel from input image
        ht = self.image_non_max_supressed.max() * self.config['htr']
        lt = self.image_non_max_supressed.max() * self.config['ltr']
        print(f'High treshold: {ht}\nLow treshold {lt}', file=sys.stderr)
        new_img = np.zeros_like(self.image_non_max_supressed)

        # Pixel coordinates with intensity higher then high treshold
        hy, hx = np.where(self.image_non_max_supressed > ht)
        # Pixel coordinates with intensity higher then low treshold
        ly, lx = np.where(self.image_non_max_supressed > lt)

        # Set low treshold intensity to pixels surapssing the lower treshold
        new_img[ly, lx] = self.config['lti']
        # Set high treshold intensity to pixels surapssing the higher treshold
        new_img[hy, hx] = self.config['hti']
        self.image_double_tresholded = new_img

    def apply_hysterezis(self):
        '''
        Apply hysterezis on edges

        args:
            img:    Input image
            hti:    Intensity of pixels surpassing higher treshold in double_treshold function
        returns:
            Hysterized image (2D numpy array)
        '''
        # Get the image shape in order to iterate through it's pixels
        h, w, _ = self.image_input.shape
        # Create padded img
        padded_img = np.zeros((h+2, w+2))
        padded_img[1:-1, 1:-1] = self.image_double_tresholded
        new_img = np.zeros_like(self.image_double_tresholded)

        # From left up -> right down
        for x in range(w):
            for y in range(h):
                if self.image_double_tresholded[y,x] != 0 and \
                    ((padded_img[y:y+3, x:x+3] == self.config['hti']).any() or \
                    (new_img[max(y-1, 0):min(y+2, len(new_img) - 1), max(0, x-1):min(x+2, len(new_img)-1)] == self.config['hti']).any()):
                    new_img[y, x] = self.config['hti']
        # From right up -> left down
        for x in list(range(w))[::-1]:
            for y in range(h):
                if self.image_double_tresholded[y,x] != 0 and \
                    ((padded_img[y:y+3, x:x+3] == self.config['hti']).any() or \
                    (new_img[max(y-1, 0):min(y+2, len(new_img) - 1), max(0, x-1):min(x+2, len(new_img)-1)] == self.config['hti']).any()):
                    new_img[y, x] = self.config['hti']
        # From left down -> right up
        for x in range(w):
            for y in list(range(h))[::-1]:
                if self.image_double_tresholded[y,x] != 0 and \
                    ((padded_img[y:y+3, x:x+3] == self.config['hti']).any() or \
                    (new_img[max(y-1, 0):min(y+2, len(new_img) - 1), max(0, x-1):min(x+2, len(new_img)-1)] == self.config['hti']).any()):
                    new_img[y, x] = self.config['hti']
        # From right down -> left up
        for x in list(range(w))[::-1]:
            for y in list(range(h))[::-1]:
                if self.image_double_tresholded[y,x] != 0 and \
                    ((padded_img[y:y+3, x:x+3] == self.config['hti']).any() or \
                    (new_img[max(y-1, 0):min(y+2, len(new_img) - 1), max(0, x-1):min(x+2, len(new_img)-1)] == self.config['hti']).any()):
                    new_img[y, x] = self.config['hti']
        self.image_hysterezis = new_img

    def img_normalize2int(self, img, norm_value=1):
        '''
        Convert an image into int type and normalize it's values
    
        args:
            img:        Image to be normalized and converted
            norm_value: Normalization value to be multiplied with input image
        returns:
            Normalized image with 3 channels
        '''
        return (np.array([img ,img, img]) * norm_value).transpose((1,2,0)).astype(np.uint8)
    


if __name__ == "__main__":
    # Parse CLI args
    args = parse_args()
    canny = Canny(args)
    canny.apply_canny()
    