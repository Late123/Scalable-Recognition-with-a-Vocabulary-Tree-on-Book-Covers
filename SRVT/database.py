import os
import sys
import cv2
import imghdr
import time
from feature import *
from homography import *
from preprocess import *
from affine import *
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import time
import gc
from pympler import asizeof

class Node:
    def __init__(self):
        self.value = None
        self.kmeans = None
        self.children = []
        self.occurrences_in_img = {}
        self.index = None

class Database:
    def __init__(self):
        self.data_path = ''
        self.num_imgs = 0
        self.word_to_img = {}
        self.BoW = {}
        self.word_count = []
        self.img_to_histgram = {}
        self.all_des = [] # store all the descriptors for all the image in the database
        self.all_image = [] # store all the image paths ...
        self.num_feature_per_image = [] # store number of features for each images, we use it extra corresponding kpts/des
        self.feature_start_idx = [] # feature_start_idx[i] store the start index of img i's descriptor in all_des
        self.kmeans = None
        self.word_idx_count = 0
        self.vocabulary_tree = None
        self.extra_image = []
        


    def loadImgs(self, data_path, method='SIFT'):
        self.data_path = data_path
        fd = FeatureDetector(n_features=130)
        for subdir, dirs, files in os.walk(self.data_path):
            print('Loading images from {}'.format(subdir))
            filecount = 0

            for f in files:
                img_path = os.path.join(subdir, f)
                img_type = imghdr.what(img_path)
                if (imghdr.what(img_path) != None and img_type in 'png/jpg/jpeg/'):
                    # if filecount > 250:
                    #     self.extra_image.append(img_path)
                    #     continue
                    img = cv2.imread(img_path)
                    # get all the kpts and des for each images.
                    kpts, des = fd.detect(img, method)
                    if des is None:
                        print('Cannot compute the features for the image: {}'.format(img_path))
                        continue
                    self.all_image.append(img_path)
                    # print(len(des))
                    self.all_des += [[d, img_path] for d in des.tolist()]
                    # self.all_des += [d for d in des.tolist()]
                    idx = 0 if len(self.num_feature_per_image) == 0 else self.num_feature_per_image[-1] + self.feature_start_idx[-1]
                    self.num_feature_per_image.append(des.shape[0])
                    self.feature_start_idx.append(idx)
                    filecount += 1
        # turn the list into a np.array
        self.num_imgs = len(self.all_image)
        self.all_des = np.array(self.all_des, dtype=object)
    
    def run_KMeans(self, k, L, max_iter=300):
        total_nodes = (k*(k**L)-1)/(k-1)
        n_leafs = k**L
        self.word_count = np.zeros(n_leafs)
        self.vocabulary_tree = self.hierarchical_KMeans(k,L, self.all_des, max_iter=max_iter)


    def print_tree(self, node):
        children = node.children
        if len(children) == 0:
            print(node.index)
        else:
            for c in children:
                self.print_tree(c)

    

    def hierarchical_KMeans(self, k, L, des_and_path, max_iter=300):
        # devide the given des vector in to k cluster
        des = [ pair[0] for pair in des_and_path]
        root = Node()
        if len(des) < k:
            if len(des) == 0:
                print('Empty cluster')
                return None
            root.index = self.word_idx_count
            self.word_idx_count += 1
            for pair in des_and_path:
                img_path = pair[1]
                if img_path not in root.occurrences_in_img:
                    root.occurrences_in_img[img_path] = 1
                else:
                    root.occurrences_in_img[img_path] += 1
            self.word_count[root.index] = len(root.occurrences_in_img)
            if self.word_idx_count % 5000 == 0:
                print('Building Vocabulary Tree: {}'.format(self.word_idx_count))
            return root
            

        root.kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto', max_iter = max_iter).fit(des)
        root.value = des_and_path

        # we reach the leaf node
        if L == 0:
            if self.word_idx_count % 5000 == 0:
                print('Building Vocabulary Tree: {}'.format(self.word_idx_count))
            # assign the index to the leaf nodes.
            root.index = self.word_idx_count
            self.word_idx_count += 1

            # count the number of occurrences of a word in a image used in tf-idf
            for pair in root.value:
                img_path = pair[1]
                if img_path not in root.occurrences_in_img:
                    root.occurrences_in_img[img_path] = 1
                else:
                    root.occurrences_in_img[img_path] += 1
            
            self.word_count[root.index] = len(root.occurrences_in_img)
            return root

        # if we are not on the leaf level, then for each cluster, 
        # we recursively run KMean
        for i in range(k):
            cluster_i = des_and_path[root.kmeans.labels_ == i]
            node_i = self.hierarchical_KMeans(k, L-1, cluster_i)
            if node_i is not None:
                root.children.append(node_i)
        return root
        
    
    def build_histgram(self, node):
        '''
        build the histgram for the leaf nodes
        '''
        
        children = node.children
        if len(children) == 0:
            for img, count in node.occurrences_in_img.items():
                # print(img)
                if img not in self.img_to_histgram:
                    self.img_to_histgram[img] = np.zeros(self.word_idx_count)
                    self.img_to_histgram[img][node.index] += count
                else:
                    self.img_to_histgram[img][node.index] += count
        else:
            for c in children:
                self.build_histgram(c)
        
    def build_BoW(self):
        num_imgs = 0
        start_time = time.time()
        
        N = self.num_imgs
        print("word_idx_count: ", self.word_idx_count)
        idf = np.log(N / (self.word_count[:self.word_idx_count] + 1))
        for img in self.all_image:
            histogram = np.array(self.img_to_histgram[img])
            n_j = np.sum(histogram)  # Total count of words in this image

            # Term Frequency for each word in the image
            tf = histogram / n_j
            
            # TF-IDF calculation using broadcasting
            tf_idf = tf * idf
            
            # Store the result in a dictionary
            self.BoW[img] = tf_idf
            
            num_imgs += 1
            if num_imgs % 100 == 0:
                print('Building BoW: {} out of {}'.format(num_imgs, len(self.all_image)))
                print('Estimated time left: {}'.format((time.time() - start_time) * (len(self.all_image) - num_imgs) / num_imgs))


    def spatial_verification(self, query, img_path_lst, method):
        fd = FeatureDetector()
        best_inliers = np.NINF
        best_img_path = None
        best_img = None
        best_H = None
        for img_path in img_path_lst:
            img = cv2.imread(img_path)
            correspondences = fd.detect_and_match(img, query, method=method)
            inliers, optimal_H = RANSAC_find_optimal_Homography(correspondences, num_rounds=2000)
            print('Running RANSAC... Image: {} Inliers: {}'.format(img_path, inliers))
            if inliers is None:
                continue

            if best_inliers < inliers:
                best_inliers = inliers
                best_img_path = img_path
                best_img = img
                best_H = optimal_H
        return best_img, best_img_path, best_H, best_inliers

    def get_leaf_nodes(self, root, des):
        children = root.children
        if len(children) == 0:
            # import pdb;pdb.set_trace()
            return root
        
        norm = np.linalg.norm(root.kmeans.cluster_centers_ - des, axis=1)
        child_idx = np.argmin(norm)
        return self.get_leaf_nodes(children[child_idx], des)


    def add_extra_node(self, img_path, input_img, method):
        # compute the features
        fd = FeatureDetector(n_features=5)
        kpts, des = fd.detect(input_img, method=method)

        if des is None:
            print('Cannot compute the features for the image: {}'.format(img_path))
            return
        q = np.zeros(self.word_idx_count)
        node_lst = []
        for d in des:
            node = self.get_leaf_nodes(self.vocabulary_tree, d)
            node_lst.append(node)
            q[node.index] += 1

        for w in range(self.word_idx_count):
            n_w = self.word_count[w]
            N = self.num_imgs
            n_wq = q[w]
            n_q = np.sum(q)
            q[w] = (n_wq / n_q) * np.log(N/n_w)

        for n in node_lst:
            if img_path not in n.occurrences_in_img:
                n.occurrences_in_img[img_path] = 1
            else:
                n.occurrences_in_img[img_path] += 1

        self.BoW[img_path] = q

        return


    def add_extra_image(self):
        count = 0
        for img_path in self.extra_image:
            img = cv2.imread(img_path)
            if img is None:
                print('Cannot read the image: {}'.format(img_path))
                continue
            self.add_extra_node(img_path, img, method='SIFT')
            count += 1
            if count % 500 == 0:
                print('Adding extra images: {} out of {}'.format(count, len(self.extra_image)))


    def query(self, input_img, top_K, method, n_features=70, affine=False):
        # compute the features
        fd = FeatureDetector(n_features=n_features)

        if not affine:

            kpts, des = fd.detect(input_img, method=method)

            print('number of features: {}'.format(len(des)))

            q = np.zeros(self.word_idx_count)
            node_lst = []
            for d in des:
                node = self.get_leaf_nodes(self.vocabulary_tree, d)
                node_lst.append(node)
                q[node.index] += 1

            N = self.num_imgs
            for w in range(self.word_idx_count):
                n_w = self.word_count[w]
                n_wq = q[w]
                n_q = np.sum(q)
                q[w] = (n_wq / n_q) * np.log(N/n_w)

            # get a list of img from database that have the same visual words
            target_img_dct = {}
            target_img_lst = []
            node_count = 0
            for n in node_lst:
                # node_count += 1
                # print('number of occurrences_in_img: {}'.format(len(n.occurrences_in_img)))
                for img, count in n.occurrences_in_img.items():
                    if img in target_img_lst:
                        continue
                    if target_img_dct.get(img) is None:
                        target_img_dct[img] = count
                    else:
                        target_img_dct[img] += count
                    if target_img_dct[img] > 2 and img not in target_img_lst:
                        target_img_lst.append(img)

            # size of the target image
            print('numbers of all target images: {}'.format(len(target_img_dct)))
            # size of the target image
            print('numbers of accepted target images: {}'.format(len(target_img_lst)))

            # compute similarity between query BoW and the all targets 
            score_lst = np.zeros(len(target_img_lst))
            for j in range(len(target_img_lst)):
                img = target_img_lst[j]
                t = self.BoW[img]

                # l1 norm similarity or L2 norm similarity
                # score_lst[j] = 2 + np.sum(np.abs(q - t) - np.abs(q) - np.abs(t)) 
                score_lst[j] = np.sum(np.abs(q / np.sum(np.abs(q)) - t / np.sum(np.abs(t))))
                # score_lst[j] = np.linalg.norm(q / np.linalg.norm(q) - t / np.linalg.norm(t))

            # sort the similarity and take the top_K most similar image
            # best_K_match_imgs_idx = np.argsort(score_lst)[-top_K:][::-1]
            best_K_match_imgs_idx = np.argsort(score_lst)[:top_K]
            best_K_match_imgs = [target_img_lst[i] for i in best_K_match_imgs_idx]

            for i in range(top_K):
                print("{}: scores: {}, image: {}".format(i, score_lst[best_K_match_imgs_idx[i]], best_K_match_imgs[i]))

            best_img, best_img_path, best_H, best_inliners= self.spatial_verification(input_img, best_K_match_imgs, method)
            print('The best match image: {}'.format(best_img_path))
            print('Homography: {}'.format(best_H))
            visualize_homograpy(best_img, input_img, best_H)


            return best_img, best_img_path, best_H, best_K_match_imgs, best_inliners

        else:

            affined_imgs = tilt(input_img)
            best_img = None
            best_img_path = None
            best_H = None
            best_inliners = 0
            for affined_img in affined_imgs:

                print('processing the affined image {} out of {}'.format(affined_imgs.index(affined_img), len(affined_imgs)))
                kpts, des = fd.detect(affined_img, method=method)

                q = np.zeros(self.word_idx_count)
                node_lst = []
                for d in des:
                    node = self.get_leaf_nodes(self.vocabulary_tree, d)
                    node_lst.append(node)
                    q[node.index] += 1

                N = self.num_imgs
                for w in range(self.word_idx_count):
                    n_w = self.word_count[w]
                    n_wq = q[w]
                    n_q = np.sum(q)
                    q[w] = (n_wq / n_q) * np.log(N/n_w)

                # get a list of img from database that have the same visual words
                target_img_lst = []
                node_count = 0
                for n in node_lst:

                    for img, count in n.occurrences_in_img.items():
                        if img not in target_img_lst:
                            target_img_lst.append(img)

                # compute similarity between query BoW and the all targets 
                score_lst = np.zeros(len(target_img_lst))
                for j in range(len(target_img_lst)):
                    img = target_img_lst[j]
                    t = self.BoW[img]
                    # score_lst[j] = 2 + np.sum(np.abs(q - t) - np.abs(q) - np.abs(t)) 
                    score_lst[j] = np.sum(np.abs(q / np.sum(np.abs(q)) - t / np.sum(np.abs(t))))
                    # score_lst[j] = np.linalg.norm(q / np.linalg.norm(q) - t / np.linalg.norm(t))

                # sort the similarity and take the top_K most similar image
                best_K_match_imgs_idx = np.argsort(score_lst)[:top_K]
                best_K_match_imgs = [target_img_lst[i] for i in best_K_match_imgs_idx]

                ibest_img, ibest_img_path, ibest_H, ibest_inliners= self.spatial_verification(affined_img, best_K_match_imgs, method)
                
                if ibest_inliners > best_inliners:
                    best_img = ibest_img
                    best_img_path = ibest_img_path
                    best_H = ibest_H
                    best_inliners = ibest_inliners

            print('The best match image: {}'.format(best_img_path))
            print('with inliners: {}'.format(best_inliners))
            # visualize_homograpy(best_img, input_img, best_H)

            return best_img, best_img_path, best_H, best_K_match_imgs, best_inliners




    def save(self, db_name):
        file = open(db_name,'wb')
        pickle.dump(self.__dict__, file)
        file.close()


    def load(self, db_name):
        file = open(db_name,'rb')
        self.__dict__.update(pickle.load(file))
        file.close()


    def score(self, img, img_path):
        t = self.BoW[img_path]
        
        fd = FeatureDetector(n_features = 70)
        kpts, des = fd.detect(img, method='SIFT')

        print('number of features: {}'.format(len(des)))
        q = np.zeros(self.word_idx_count)
        node_lst = []
        for d in des:
            node = self.get_leaf_nodes(self.vocabulary_tree, d)
            node_lst.append(node)
            q[node.index] += 1

        for w in range(self.word_idx_count):
            n_w = self.word_count[w]
            N = self.num_imgs
            n_wq = q[w]
            n_q = np.sum(q)
            q[w] = (n_wq / n_q) * np.log(N/n_w)

        # score = 2 + np.sum(np.abs(q - t) - np.abs(q) - np.abs(t)) 
        score = np.sum(np.abs(q / np.sum(np.abs(q)) - t / np.sum(np.abs(t))))
        # score = np.linalg.norm(q / np.linalg.norm(q) - t / np.linalg.norm(t))
        print('score with 286: {}'.format(score))
        


        
def build_database(load_path, k, L, method, save_path):
    print('Initial the Database')
    db = Database()

    print('Loading the images from {}, use {} for features'.format(load_path, method))
    db.loadImgs(load_path, method=method)

    print('Building Vocabulary Tree, with {} clusters, {} levels'.format(k, L))
    db.run_KMeans(k=k, L=L, max_iter=1000)
    print('Vocabulary Tree has been built with total indexes: {}'.format(db.word_idx_count))

    print('Release the memory for all_des')
    del db.all_des
    gc.collect()  # Force garbage collection

    print('Building Histgram for each images')
    db.build_histgram(db.vocabulary_tree)

    print('Building BoW for each images')
    db.build_BoW()

    print('Release the memory for img_to_histgram')
    del db.img_to_histgram
    gc.collect()  # Force garbage collection

    ## Add images after the vocab tree database has been built
    # print('Adding extra images to the database')
    # db.add_extra_image()

    print('Saving the database to {}'.format(save_path))
    db.save(save_path)


def resize_image(img, target_width):
    # inport a image
    ratio = target_width / img.shape[1]
    target_height = int(img.shape[0] * ratio)
    resized_image = cv2.resize(img, (target_width, target_height))
    return resized_image


def query_list(db, img_lst, top_K, method, n_features, inliner_threshold=20):
    num = len(img_lst)
    count = 0
    for image in img_lst:
        # resize image into 256
        img = resize_image(image, 256)
        print('Querying the image')
        affine = False
        if method == 'ASIFT':
            affine = True
        # db.score(img, '..\data\\train\\t\\ans (1).jpg')
        best_img, best_img_path, best_H, best_K, best_inliners= db.query(img, top_K = top_K, method=method, n_features=n_features, affine=affine)
        # if best_inliners < inliner_threshold:
        #     continue
        plt.subplot(num, 2, 2 * count + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if count == 0:
            plt.title('Query Image')
        plt.subplot(num, 2, 2 * count +2)
        plt.imshow(cv2.cvtColor(best_img, cv2.COLOR_BGR2RGB))
        if count == 0:
            plt.title('Best Match Image')
        count += 1
        
    plt.show()

if __name__ == '__main__':

    # Define the test path and DVD cover path
    test_path = '..\\data\\test'
    cover_path = '..\\data\\train'

    # Initial and build the database
    db = Database()
    build_database(cover_path, k=5, L=7, method='SIFT', save_path='..\\model\\data_sift')

    # Test Load
    # print('Loading the database')
    # db.load('..\\model\\data_sift')

    # query a folder
    # query_list(db, test_path, top_K = 10, method='SIFT')

    # query a image
    # test = test_path + '/image_04.jpg'
    # test = cv2.imread(test)
