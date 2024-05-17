import cv2
import numpy as np
import os

class Q2:
    
    def __init__(self, images):
        self.images_array = images 

    # Image Resolution Simulation
    def simulate_image_resolution(self, image):
        resolutions = [(1024, 1024), (512, 512), (256, 256), (128, 128), (64, 64), (32, 32)]
        resized_images = [cv2.resize(np.array(image), (width, height)) for width, height in resolutions]
        return resized_images

    # Quantization Simulation 
    def simulate_quantization(self, image):
        quantized_images = []
        for bits in range(1, 9):
            # Calculate the quantization levels
            levels = 2 ** bits

            # Perform quantization
            quantized_image = np.floor(image / (256 / levels)) * (256 / levels)

            quantized_images.append(quantized_image.astype(np.uint8))

        return quantized_images

    # Multi-resolution Simulation
    def simulate_multi_resolution(self, image):
        scales = [1.72**i for i in range(6)]
        resized_images = [cv2.resize(image, (int(image.shape[1] / scale), int(image.shape[0] / scale))) for scale in scales]
        return resized_images

    def read_image(self, image_path):
        image = cv2.imread(image_path) 
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image

    def save_image(self, image, image_path):
        cv2.imwrite(image_path, image)
    
    def show_image(self, image_object, count):
        for i, resized_image in enumerate(image_object[0]):
            self.save_image(resized_image, './output/Q2/' + str(count) + '/image_resolution_' + str(i) + '.jpg') 
            
            # For printing the images preprocessed on the screen, Uncomment the following function calls.
            # cv2.imshow(f'Image Resolution {i}', resized_image)

        for i, quantized_image in enumerate(image_object[1]):
            self.save_image(quantized_image, './output/Q2/' + str(count) + '/quantization_' + str(i) + '.jpg')
            
            # For printing the images preprocessed on the screen, Uncomment the following function calls.
            # cv2.imshow(f'Quantization {i}', quantized_image)

        for i, multi_resolution_image in enumerate(image_object[2]):
            self.save_image(multi_resolution_image, './output/Q2/' + str(count) + '/multi_resolution_' + str(i) + '.jpg')
            
            # For printing the images preprocessed on the screen, Uncomment the following function calls.
            # cv2.imshow(f'Multi-Resolution {i}', multi_resolution_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def run(self):
        count = 0
        for image in self.images_array:
            count += 1
            image = self.read_image(image)
            
            # Perform image resolution simulation
            resized_images = self.simulate_image_resolution(image)

            # Perform quantization simulation
            quantized_images = self.simulate_quantization(image)

            # Perform multi-resolution simulation
            multi_resolution_images = self.simulate_multi_resolution(image)
            
            try:
                if not os.path.exists('./output/Q2/' + str(count)):
                    os.makedirs('./output/Q2/' + str(count))
            except Exception as e:
                pass
            
            self.show_image([resized_images, quantized_images, multi_resolution_images], count)
            

if __name__ == '__main__':
    images = ['./3.jpg', './4.jpg']
    Q2(images).run()