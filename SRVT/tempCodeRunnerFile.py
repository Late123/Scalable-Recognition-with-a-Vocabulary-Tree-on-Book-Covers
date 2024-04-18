
test = test_path + '/image_01.jpeg'
test = cv2.imread(test)
best_img, best_img_path, best_H, best_K= db.query(test, top_K = 10, method='SIFT')