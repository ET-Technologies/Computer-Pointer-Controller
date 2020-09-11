def preprocess_output(self, image, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        eye_coords = []
        box = []
        area = 20
        left_x = outputs[0][0] * image.shape[1]
        left_y = outputs[0][1] * image.shape[0]
        right_x = outputs[0][2] * image.shape[1]
        right_y = outputs[0][3] * image.shape[0]
        left_xmin = outputs[0][0] * image.shape[1] - area
        left_xmax = outputs[0][0] * image.shape[1] + area
        left_ymin = outputs[0][1] * image.shape[0] - area
        left_ymax = outputs[0][1] * image.shape[0] + area
        right_xmin = outputs[0][2] * image.shape[1] - area
        right_xmax = outputs[0][2] * image.shape[1] + area
        right_ymin = outputs[0][3] * image.shape[0] - area
        right_ymax = outputs[0][3] * image.shape[0] + area
        cv2.rectangle(image, (left_xmin, left_ymin),
                             (left_xmax, left_ymax), (0, 255, 0), 2)
        cv2.rectangle(image, (right_xmin, right_ymin),
                             (right_xmax, right_ymax), (0, 255, 0), 2)
        box = [[int(left_xmin), int(left_ymin), int(left_xmax), int(left_ymax)],
                 [int(right_xmin), int(right_ymin), int(right_xmax), int(right_ymax)]]
        eye_coords = [int(left_x.squeeze()), int(left_y.squeeze()),
                      int(right_x.squeeze()), int(right_y.squeeze())]
        return image, box, eye_coords