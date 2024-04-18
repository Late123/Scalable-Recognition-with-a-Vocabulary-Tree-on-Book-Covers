from feature import *
from homography import *
from database import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from variables import LABEL_MAP
# Define the test path and DVD cover path
def FlannMatch():
    test_path = '..\\data\\demo'

    # query a image
    test1 = test_path + '\\1.jpg'
    test1 = cv2.imread(test1)

    test2 = test_path + '\\2.jpg'
    test2 = cv2.imread(test2)

    # maxpooling the image
    test2 = cv2.resize(test2, (0,0), fx=0.25, fy=0.25)

    print('size of the test image: {}'.format(test1.shape))
    print('size of the test image: {}'.format(test2.shape))
    print('Querying the image')

    fd = FeatureDetector()
    fd2 = FeatureDetector(n_features=50)
    kp1, des1 = fd.detect(test1, method='SIFT')
    print('Number of keypoints for image 1: {}'.format(len(kp1)))
    kp2, des2 = fd2.detect(test2, method='SIFT')
    print('Number of keypoints for image 2: {}'.format(len(kp2)))
    correspondences = fd.detect_and_match(test1, test2)
    print('Number of correspondences: {}'.format(len(correspondences)))
    flann = cv2.FlannBasedMatcher() 
    Matches = flann.knnMatch(des1, des2, k=2) 

    num_good_matches = 0
    good_matches = [[0, 0] for i in range(len(Matches))] 
    for i, (m, n) in enumerate(Matches): 
        if m.distance < 0.8*n.distance: 
            good_matches[i] = [1, 0]
            num_good_matches += 1

    print('Number of matches: {}'.format(len(Matches)))
    print('Number of good matches: {}'.format(num_good_matches))
    plt.imshow(cv2.drawKeypoints(test1, kp1, None, color=(0,255,0), flags=0))
    plt.show()
    plt.imshow(cv2.drawKeypoints(test2, kp2, None, color=(0,255,0), flags=0))
    plt.show()
    # Draw the correspondences
    img3 = cv2.drawMatchesKnn(test1, kp1, test2, kp2, Matches, None, matchesMask=good_matches, flags=2)
    plt.imshow(img3)
    plt.show()


def import_model(model_dir):
    

    # Load the model
    model = tf.saved_model.load(model_dir)
    return model


def image_preprocess(model, img_name, threshold = 0.5):

    target_width = 256
    source_image = cv2.imread(img_name)
    resized_image = source_image

    # target_width = 256
    # # inport a image
    # ratio = target_width / source_image.shape[1]
    # target_height = int(source_image.shape[0] * ratio)
    # resized_image = cv2.resize(source_image, (target_width, target_height))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    img = np.array(resized_image)
    print('size of the image: {}'.format(img.shape))

    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...]
    # Run detection
    detections = model(input_tensor)

    # Explore `detections` to see what you got
    # print(detections)

    # Suppose `detections` is the output from the model
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    category_index = LABEL_MAP
    labels = [category_index.get(c, 'Unknown') for c in classes]

    image_with_boxes = draw_boxes(img, boxes, labels, scores, threshold = threshold)

    # # Display the image
    # plt.figure(figsize=(12, 8))
    # plt.imshow(image_with_boxes)
    # plt.axis('off')
    # plt.show()

    target_label = 84  # Label for 'book'
    cropped_images = crop_images(resized_image, boxes, classes, scores, target_label, threshold=threshold)

    print('Number of books found: {}'.format(len(cropped_images)))
    if len(cropped_images) == 0:
        print('No objects detected with label {}, using lower threshold'.format(target_label))
        
        cropped_images = crop_images(resized_image, boxes, classes, scores, target_label, threshold=threshold*0.7)

    if len(cropped_images) == 0:
        print('No objects detected with label {}, using original image'.format(target_label))
        cropped_images = [resized_image]
    
    # cvt back to RGB
    result_images = []
    for image in cropped_images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # inport a image
        ratio = target_width / image.shape[1]
        target_height = int(image.shape[0] * ratio)
        image = cv2.resize(image, (target_width, target_height))
        result_images.append(image)
    # fig, axs = plt.subplots(1, len(cropped_images), figsize=(15, 5))
    
    # # If there's only one image, axs won't be an array, so we make it into one for uniformity
    # if len(cropped_images) == 1:
    #     axs = [axs]

    # # Display each image
    # for img, ax in zip(cropped_images, axs):
    #     ax.imshow(img)
    #     ax.axis('off')  # Hide axes

    # plt.show()
    
    return result_images



def crop_images(image, boxes, classes, scores, target_label, threshold=0.5):
    """
    Crops and saves images based on the detection of a specific label.

    Parameters:
    - image: The original image.
    - boxes: Bounding boxes for detected objects.
    - classes: Labels for each bounding box.
    - scores: Confidence scores for each detection.
    - target_label: The label of interest to crop.
    - threshold: Score threshold to consider a detection.
    """
    h, w, _ = image.shape
    saved_count = 0

    cropped_imgs = []
    for i, box in enumerate(boxes):
        if classes[i] == target_label and scores[i] >= threshold:
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * w, xmax * w, ymin * h, ymax * h)
            cropped_img = image[int(top):int(bottom), int(left):int(right)]
            cropped_imgs.append(cropped_img)

    return cropped_imgs  # Number of images saved


def draw_boxes(img, boxes, classes, scores, threshold=0.5):
    # Set the colors for the bounding boxes
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # Read the image with OpenCV
    # img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    
    h, w, _ = img.shape
    for i, box in enumerate(boxes):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * w, xmax * w, ymin * h, ymax * h)

            # Draw the box
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), COLORS[i % len(COLORS)], 2)

            # Draw label
            label = f'{classes[i]}: {int(scores[i]*100)}%'
            cv2.putText(img, label, (int(left), int(top-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[i % len(COLORS)], 2)
    
    return img

def draw_keypoints(output_dict, image_np):
    # Visualization of the results of a detection.
    num_detections = int(output_dict.pop('num_detections'))
    detection_boxes = output_dict['detection_boxes'][0]
    detection_classes = output_dict['detection_classes'][0]
    detection_boxes = output_dict['detection_boxes'][0]
    detection_scores = output_dict['detection_scores'][0]
    detection_keypoints = output_dict['detection_keypoints'][0]
    detection_keypoint_scores = output_dict['detection_keypoint_scores'][0]
    
    print('Number of detections: {}'.format(num_detections))
    print('boxes: {}'.format(detection_boxes))
    print('Classes: {}'.format(detection_classes))
    print('Boxes: {}'.format(detection_boxes))
    print('Scores: {}'.format(detection_scores))
    print('Keypoints: {}'.format(detection_keypoints))
    print('Keypoint scores: {}'.format(detection_keypoint_scores))
    
    # Visualization of the bounding boxes
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)

    ax = plt.gca()

    for i in range(num_detections):
        if detection_scores[i] > 0.5:  # threshold to filter weak detections
            # Extract the bounding box
            y_min, x_min, y_max, x_max = detection_boxes[i]
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # Plot keypoints if they exist
            if detection_keypoints is not None:
                for kp_idx, kp in enumerate(detection_keypoints[i]):
                    if detection_keypoint_scores[i][kp_idx] > 0.5:  # threshold for keypoint visibility
                        plt.plot(kp[1], kp[0], 'ro')  # keypoints are (y, x)

    plt.axis('off')
    plt.show()

def get_image(img_path):
    model_dir = '..\\model\\efficientdet_d6_coco17_tpu-32\\saved_model'
    # model_dir = '..\\model\\centernet_hg104_512x512_coco17_tpu-8\\saved_model'
    model = import_model(model_dir)
    image_preprocess(model, img_path)


if __name__ == '__main__':
    
    # model_dir = '..\\model\\efficientdet_d6_coco17_tpu-32\\saved_model'
    # model_dir = '..\\model\\centernet_hg104_512x512_coco17_tpu-8\\saved_model'
    model_dir = '..\\model\\centernet_hg104_1024x1024_kpts_coco17_tpu-32\\saved_model'
    img_path = '..\\data\\10testers\\1book'
    model = import_model(model_dir)
    image_name = '\\2.jpg'
    source_image = img_path + image_name
    # image_preprocess(model, source_image)

    
    source_image = cv2.imread(source_image)
    # resize to 1024x1024
    resized_image = cv2.resize(source_image, (1024, 1024))

    # target_width = 256
    # # inport a image
    # ratio = target_width / source_image.shape[1]
    # target_height = int(source_image.shape[0] * ratio)
    # resized_image = cv2.resize(source_image, (target_width, target_height))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    img = np.array(resized_image)
    print('size of the image: {}'.format(img.shape))

    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...]
    # Run detection
    detections = model(input_tensor)

    draw_keypoints(detections, resized_image)