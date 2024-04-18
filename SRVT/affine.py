import cv2
import numpy as np
import matplotlib.pyplot as plt

def tilt(img):
    new_imgs = []
    h, w = img.shape[:2]
    for t in 1.5 ** (0.5 * np.arange(1, 5)):
        for phi in np.arange(0, 180, 22.5):
            new_img = img.copy()
            A = np.float32([[1, 0, 0], [0, 1, 0]])
            # Rotate image
            if phi != 0.0:
                phi = np.deg2rad(phi)
                s, c = np.sin(phi), np.cos(phi)
                A = np.float32([[c, -s], [s, c]])
                corners = [[0, 0], [w, 0], [w, h], [0, h]]
                tcorners = np.int32(np.dot(corners, A.T))
                x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
                A = np.hstack([A, [[-x], [-y]]])
                new_img = cv2.warpAffine(new_img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # Tilt image (resizing after rotation)
            if t != 1.0:
                s = 0.7 * np.sqrt(t * t - 1)
                new_img = cv2.GaussianBlur(new_img, (0, 0), sigmaX=s, sigmaY=0.01)
                new_img = cv2.resize(new_img, (0, 0), fx=1.0 / t, fy=1.0, interpolation=cv2.INTER_NEAREST)
                A[0] /= t

            new_imgs.append(new_img)

    return new_imgs
            
            

def apply_tilt(image, tilt_factor):
    h, w = image.shape[:2]
    # A tilt factor > 1 means the image is viewed more steeply
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 / np.sqrt(tilt_factor))
    tilted_image = cv2.warpAffine(image, matrix, (w, h))
    return tilted_image


def corner_detection():
# Load image
    image = cv2.imread('..\\data\\10testers\\1book\\2.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # Find corners
    corners = cv2.goodFeaturesToTrack(gray,4,0.01,250)
    corners = np.int0(corners)
    
    for i in corners:
        x,y = i.ravel()
        cv2.circle(gray,(x,y),30,255,-1)
    
    plt.imshow(gray),plt.show()


if __name__ == '__main__':
    # Load an image
    image = cv2.imread('..\\data\\10testers\\1book\\2.jpg', cv2.IMREAD_GRAYSCALE)

    # Apply tilt
    tilted_iamges = tilt(image)

    #plot all images together
    fig, axs = plt.subplots(2, 4)
    for i, ax in enumerate(axs.flat):
        ax.imshow(tilted_iamges[i], cmap='gray')
        ax.set_title(f'Tilt factor: {i}')
    plt.show()


    # for i in range(1, 6):
    #     tilted_image = apply_tilt(image, i)
    #     plt.imshow(tilted_image, cmap='gray')
    #     plt.title(f'Tilt factor: {i}')
    #     plt.show()
