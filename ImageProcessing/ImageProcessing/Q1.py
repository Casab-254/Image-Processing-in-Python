import numpy as np
import cv2
import os

class Q1:
    
    def __init__(self, image_feed_1, image_feed_2):
        self.image_feed_1 = image_feed_1
        self.image_feed_2 = image_feed_2 

    # Double Exposure
    def double_exposure(self, images):
        # Ensure images have the same dimensions
        min_height = min(image.shape[0] for image in images)
        min_width = min(image.shape[1] for image in images)

        # Resize images to the minimum dimensions
        resized_images = [cv2.resize(image, (min_width, min_height)) for image in images]

        # Calculate the mean image
        combined_image = np.mean(np.array(resized_images), axis=0).astype(np.uint8)
        return combined_image

    # Change Detection
    def change_detection(self, frame1, frame2):
        # Subtract frames to detect changes
        diff = cv2.absdiff(frame1, frame2)
        return diff

    # Synthesize Novel Image
    def synthesize_novel_image(self, images, masks):
        # Apply masks and combine images
        masked_images = [image * mask for image, mask in zip(images, masks)]
        combined_image = np.sum(masked_images, axis=0) / len(images)
        combined_image = combined_image.astype(np.uint8)
        return combined_image 
    
    def read_image(self, image_path):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image
    
    def resize_image(self, image, height, width):
        image = cv2.resize(image, (height, width))
        return image
    
    def show_image(self, image_object):
        # For printing the images preprocessed on the screen, Uncomment the following cv2 function calls.
        # cv2.imshow('Double Exposure in Action', image_object[0])
        # cv2.imshow('Change Detection in Action', image_object[1])
        # cv2.imshow('Synthesized Image in Action', image_object[2])
        
        try:
            if not os.path.exists('./output/Q1'):
                os.makedirs('./output/Q1')
        except Exception as e:
            pass
            
        self.save_image(image_object[0], './output/Q1/combined_image.jpg')
        self.save_image(image_object[1], './output/Q1/change_diff.jpg')
        self.save_image(image_object[2], './output/Q1/novel_image.jpg')
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def save_image(self, image, image_path):
        cv2.imwrite(image_path, image)
    
    def run(self):
        # Load images and masks (assuming grayscale images)
        image_feed_1 = self.read_image(self.image_feed_1)
        image_feed_2 = self.read_image(self.image_feed_2)

        # Ensure images have the same dimensions
        image_feed_1 = self.resize_image(image_feed_1, np.array(image_feed_2).shape[1], np.array(image_feed_2).shape[0])
        
        # Assuming you have masks for synthesis
        mask1 = np.random.random(image_feed_1.shape)
        mask2 = np.random.random(image_feed_2.shape)

        # Call functions for each task
        combined_image = self.double_exposure([image_feed_1, image_feed_2])
        change_diff = self.change_detection(image_feed_1, image_feed_2)
        novel_image = self.synthesize_novel_image([image_feed_1, image_feed_2], [mask1, mask2])
        
        image_object = [combined_image,  change_diff, novel_image]
        
        self.show_image(image_object)

if __name__ == '__main__':
    image_feed_1 = './1.jpg'
    image_feed_2 = './2.jpg' 
    Q1(image_feed_1, image_feed_2).run()
