from preprocess import *
from database import *


if __name__ == '__main__':
    model_dir = '..\\model\\efficientdet_d6_coco17_tpu-32\\saved_model'
    # model_dir = '..\\model\\centernet_hg104_512x512_coco17_tpu-8\\saved_model'
    image_folder = '..\\data\\10testers\\5book'
    database_dir = '..\\model\\data_sift'

    query_top_K = 10
    preprocess_threshold = 0.7
    n_features = 130
    inliner_threshold = 20
    method = 'SIFT' # SIFT or ASIFT or ORB

    # Load the model
    print('loading model...')
    model = import_model(model_dir)
    # Load the database
    db = Database()
    db.load(database_dir)
    print('model loaded!')

    # Preprocess the images
    
    while True:
        for subdir, dirs, files in os.walk(image_folder):
            for file in files:
                print(f"Start processing {file}")
                image_path = os.path.join(subdir, file)
                image_books = image_preprocess(model, image_path, threshold =  preprocess_threshold)
                query_list(db, image_books, top_K = query_top_K, method = 'SIFT', n_features = n_features, inliner_threshold = inliner_threshold)