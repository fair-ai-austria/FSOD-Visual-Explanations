import shutil

import requests
import json
import os
import time
from PIL import Image
from fiftyone import ViewField as F
import fiftyone as fo
import numpy as np
import threading

from service_settings import Settings, sanitize_directory_name


def insert_results_in_dataset(dataset, results, images_dir, threshold, object_list, field_name, view_index=-1):
    if os.getcwd() in images_dir:
        images_dir_abs = images_dir
    else:
        images_dir_abs = os.path.join(os.getcwd(), images_dir)

    for i, img_res in enumerate(results):
        img_file = img_res['image_file']
        result_list = img_res['features']
        search_view = dataset.match(F("filepath") == os.path.join(images_dir_abs, img_file))
        if len(search_view) > 0:
            sample = search_view.first()

            if sample.metadata:
                img_width = sample.metadata.width
                img_height = sample.metadata.height
            else:
                img = Image.open(sample['filepath'])
                img_width, img_height = img.size

            detection_list = []
            for result in result_list:
                if not result['filtered'] and result['confidence'] >= threshold and result['class'] in object_list:
                    if view_index >= 0 and result['view_index'] != view_index:
                        continue
                    scores = result['confidence']

                    x = result['region'][0] / img_width
                    y = result['region'][1] / img_height
                    width = (result['region'][2] - result['region'][0]) / img_width
                    height = (result['region'][3] - result['region'][1]) / img_height
                    bounding_box = [x, y, width, height]

                    # add detection to sample
                    new_detection = fo.Detection(
                        label=result['class'],
                        bounding_box=bounding_box,
                        confidence=scores,
                        tags=["inference"]
                    )

                    detection_list.append(new_detection)

            sample[field_name] = fo.Detections(detections=detection_list)
            sample.save()
    dataset.save()

#not used currently
def filter_overlapped_objects(image_result_list, det_threshold):
    for i, img_res in enumerate(image_result_list):
        # check if the image result is empty
        if img_res['features'] is None or len(img_res['features']) == 0:
            continue
        # reset filter status
        features = img_res['features']
        class_list = [res['class'] for res in features]
        # create unique class list
        unique_class_list = list(set(class_list))

        for res in features:
            res['filtered'] = False

        if det_threshold > 0.0:
            for c in unique_class_list:
                # get the indexes of the boxes with the same class
                box_id_indexes = [i for i, x in enumerate(class_list) if x == c]
                for i in range(len(box_id_indexes)):
                    if features[box_id_indexes[i]]['confidence'] >= det_threshold:
                        for j in range(i + 1, len(box_id_indexes)):
                            if features[box_id_indexes[j]]['confidence'] >= det_threshold:
                                bid1 = box_id_indexes[i]
                                bid2 = box_id_indexes[j]
                                arr1 = np.array(features[bid1]['region'])
                                arr2 = np.array(features[bid2]['region'])
                                a1 = np.prod(arr1[2:] - arr1[:2])
                                a2 = np.prod(arr2[2:] - arr2[:2])
                                top_left = np.maximum(arr1[:2], arr2[:2])  # [[x, y]]
                                bottom_right = np.minimum(arr1[2:], arr2[2:])  # [[x, y]]
                                wh = bottom_right - top_left
                                ia = wh[0].clip(0) * wh[1].clip(0)
                                if ia >= 0.9 * min(a1,
                                                   a2):  # same class overlapping case, and larger one is much larger than small
                                    score1 = features[bid1]['confidence']
                                    score2 = features[bid2]['confidence']
                                    if a1 >= a2:
                                        if score2 < score1:
                                            features[bid2]['filtered'] = True
                                    else:
                                        if score1 < score2:
                                            features[bid1]['filtered'] = True
                    else:
                        features[box_id_indexes[i]]['filtered'] = True


class session_logger:
    def __init__(self):
        self.log_messages = []
        self.status_message = ""
        self.last_queried_log_messages = -1
        self._lock = threading.RLock()

    def clear_log_messages(self):
        with self._lock:
            self.log_messages = []
            self.last_queried_log_messages = -1

    def clear_log_messages_to_line_with_text(self, text):
        log_message = ""
        with self._lock:
            index = -1
            for i in range(len(self.log_messages)):
                if text in self.log_messages[i]:
                    index = i
                    break
            if index > 0:
                self.log_messages = self.log_messages[index:]
                self.last_queried_log_messages = len(self.log_messages)

                log_message = "\n".join(self.log_messages)
                # remove space characters at the beginning
                log_message = log_message.lstrip()
        return log_message

    def get_log_messages_to_line_with_text(self, text):
        log_message = ""
        status_message = ""
        with self._lock:
            index = -1
            for i in range(len(self.log_messages)):
                if text in self.log_messages[i]:
                    index = i
                    break
            if index > 0:
                log_message = "\n".join(self.log_messages[:index])
                # remove space characters at the beginning
                log_message = log_message.lstrip()
                status_message = self.log_messages[-1]
            else:
                log_message = "\n".join(self.log_messages)
            self.last_queried_log_messages = len(self.log_messages)
        return log_message, status_message


    def is_first_message(self):
        with self._lock:
            return len(self.log_messages) > 0 and self.last_queried_log_messages < 0

    def is_new_message(self):
        with self._lock:
            return len(self.log_messages) > 0 and len(self.log_messages) > self.last_queried_log_messages

    def new_log_message(self, message):
        if message is not None and len(message) > 0:
            # devide the message into multiple lines if it contains newline characters and add each line as a separate message
            message_lines = message.split('\n')
            with self._lock:
                for message_line in message_lines:
                    self.log_messages.append(message_line)

    def new_status_message(self, message):
        if message is not None and len(message) > 0:
            with self._lock:
                self.status_message = message

    def get_log_messages(self):
        with self._lock:
            self.last_queried_log_messages = len(self.log_messages)
            return self.log_messages

    def get_status_message(self):
        with self._lock:
            return self.status_message

    def get_full_log_message(self):
        with self._lock:
            if len(self.log_messages) == 0:
                self.last_queried_log_messages = len(self.log_messages)
                return ""
            self.last_queried_log_messages = len(self.log_messages)
            return "\n".join(self.log_messages)

    def get_messages_since_last_query(self):
        with self._lock:
            if self.last_queried_log_messages < 0:
                self.last_queried_log_messages = len(self.log_messages)
                return list(self.log_messages)
            if len(self.log_messages) <= self.last_queried_log_messages:
                return []
            new_messages = list(self.log_messages[self.last_queried_log_messages:])
            self.last_queried_log_messages = len(self.log_messages)
            return new_messages

    def get_full_log_message_since_last_query(self):
        new_messages = self.get_messages_since_last_query()
        return "\n".join(new_messages)


class object_detection_client:
    def __init__(self, service_url, service_port, training_dir=None):
        self.service_url = service_url
        self.service_port = service_port
        self.label_list_base = []
        self.label_list_base_sorted = []
        self.super_category_list_sorted = []
        if training_dir is None:
            self.training_dir = os.path.join('datasets', 'inc_images')
        else:
            self.training_dir = training_dir

    def status(self):
        error_msg = ""
        try:
            url = f"{self.service_url}:{self.service_port}/status"
            response = requests.get(url)
            if response.status_code == 200:
                status = response.json()
                return status, ""
            else:
                try:
                    message_json = response.json()
                    if message_json is not None and 'description' in message_json:
                        error_msg = message_json['description']
                    else:
                        error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                except Exception as e:
                    error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Connection to object detection service failed!"
        return None, error_msg

    def get_models(self):
        error_msg = ""
        try:
            url = f"{self.service_url}:{self.service_port}/get_models"
            response = requests.get(url)
            if response.status_code == 200:
                datasets = response.json()
                return datasets, ""
            else:
                try:
                    message_json = response.json()
                    if message_json is not None and 'description' in message_json:
                        error_msg = message_json['description']
                    else:
                        error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                except Exception as e:
                    error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Connection to object detection service failed!"
        return {}, error_msg


    def set_model(self, dataset_id, dataset_name, pretrained_dataset, weighted_patch_calculation,
                  check_mask_overlap, model_type, model_size, model_path, finetuned_labels, pretrained_model_files,
                  shot_input):
        error_msg = ""
        # check if finetuned_labels is a str
        if isinstance(finetuned_labels, str):
            finetuned_labels = [label.strip() for label in finetuned_labels.split(',')]
        if len(pretrained_model_files) == 0:
            pretrained_model_files = None
        try:
            url = f"{self.service_url}:{self.service_port}/set_model"
            request_body = {
                "model_id": dataset_id,
                "name": dataset_name,
                "model_path": model_path,
                "finetuned_labels": finetuned_labels,
                "pretrained_prototype_files": pretrained_model_files,
                "settings": {
                    "pretrained_dataset": pretrained_dataset,
                    "weighted_patch_calculation": weighted_patch_calculation,
                    "check_mask_overlap": check_mask_overlap,
                    "model_size": model_size,
                    "model_type": model_type,
                    "shot_number": shot_input
                }
            }
            request_body_json = json.dumps(request_body)
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, data=request_body_json, headers=headers)
            if response.status_code == 200:
                return ""
            else:
                message_json = response.json()
                if message_json is not None and 'description' in message_json:
                    error_msg = message_json['description']
                else:
                    error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
        except Exception as e:
            error_msg = f"Error: Connection to object detection service failed!"
        return error_msg

    def delete_model(self, model_id):
        error_msg = ""
        try:
            url = f"{self.service_url}:{self.service_port}/delete_model?id={model_id}"
            response = requests.post(url)
            if response.status_code == 200:
                return ""
            else:
                message_json = response.json()
                if message_json is not None and 'description' in message_json:
                    error_msg = message_json['description']
                else:
                    error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
        except Exception as e:
            error_msg = f"Error: Connection to object detection service failed!"
        return error_msg

    def reload_models(self):
        error_msg = ""
        try:
            url = f"{self.service_url}:{self.service_port}/reload_models"
            response = requests.post(url)
            if response.status_code == 200:
                return ""
            else:
                message_json = response.json()
                if message_json is not None and 'description' in message_json:
                    error_msg = message_json['description']
                else:
                    error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
        except Exception as e:
            error_msg = f"Error: Connection to object detection service failed!"
        return error_msg

    def get_label_list_with_multi_view(self, model_id, class_selection_mode):
        try:
            url = f"{self.service_url}:{self.service_port}/label_list?class_selection_mode={class_selection_mode}&model_id={model_id}&with_multi_view=true"
            response = requests.get(url)
            if response.status_code == 200:
                label_list = response.json()
                return label_list, ""
            else:
                try:
                    message_json = response.json()
                    if message_json is not None and 'description' in message_json:
                        error_msg = message_json['description']
                    else:
                        error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                except Exception as e:
                    error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Connection to object detection service failed!"
        return self.label_list_base, error_msg


    def get_label_list(self, model_id, class_selection_mode, detect_supercategories=False):
        try:
            if model_id is None:
                url = f"{self.service_url}:{self.service_port}/label_list_with_super_category?class_selection_mode={class_selection_mode}&with_multi_view=false"
            else:
                url = f"{self.service_url}:{self.service_port}/label_list_with_super_category?class_selection_mode={class_selection_mode}&model_id={model_id}&with_multi_view=false"
            response = requests.get(url)
            if response.status_code == 200:
                label_list = response.json()['label_list']
                super_category_list = response.json()['super_category_list']
                if 'super_category_use_area_list' in response.json():
                    super_category_use_area_list = response.json()['super_category_use_area_list']
                else:
                    super_category_use_area_list = [False] * len(super_category_list)


                self.label_list_base = label_list

                # Sort label_list_base and rearrange super_category_list accordingly
                sorted_values = sorted(zip(self.label_list_base, super_category_list, super_category_use_area_list), key=lambda x: x[0])
                self.label_list_base_sorted = [pair[0] for pair in sorted_values]
                self.super_category_list_sorted = [pair[1] for pair in sorted_values]
                self.super_category_use_area_list_sorted = [pair[2] for pair in sorted_values]

                return (self.label_list_base, self.label_list_base_sorted, self.super_category_list_sorted,
                        self.super_category_use_area_list_sorted, "")
            else:
                try:
                    message_json = response.json()
                    if message_json is not None and 'description' in message_json:
                        error_msg = message_json['description']
                    else:
                        error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                except Exception as e:
                    error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Connection to object detection service failed!"
        return (self.label_list_base, self.label_list_base_sorted, self.super_category_list_sorted,
                self.super_category_use_area_list_sorted, error_msg)

    def get_all_prototypes(self, model_id, class_selection_mode):
        try:
            url = f"{self.service_url}:{self.service_port}/get_all_prototypes?class_selection_mode={class_selection_mode}&model_id={model_id}"
            response = requests.get(url)
            if response.status_code == 200:
                prototypes = response.json()
                return prototypes, ""
            else:
                try:
                    message_json = response.json()
                    if message_json is not None and 'description' in message_json:
                        error_msg = message_json['description']
                    else:
                        error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                except Exception as e:
                    error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Connection to object detection service failed!"
        return [], error_msg

    def poll_job_status(self, job_id, session_logger=None):
        # poll status of the job
        error_msg = ""
        url = self.service_url + ':' + str(self.service_port) + '/job_status/' + str(job_id)
        data = {
            "jobid": job_id
        }

        while True:
            try:
                response = requests.get(url, data=data)
                if response.status_code == 200:
                    return_val = response.json()
                    if return_val is not None:

                        if session_logger is not None:
                            if 'log_message' in return_val:
                                session_logger.new_log_message(return_val['log_message'])
                            if 'status_message' in return_val:
                                session_logger.new_status_message(return_val['status_message'])

                        if return_val['status'] == 'completed':

                            return "completed", return_val, ""

                        elif return_val['status'] == 'cancelled':
                            error_msg = 'Error: ' + return_val['description']
                            return "cancelled", return_val, error_msg

                    # sleep for 0.5 sec
                    time.sleep(0.5)
                else:
                    try:
                        message_json = response.json()
                        if message_json is not None and 'description' in message_json:
                            error_msg = message_json['description']
                        else:
                            error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                        return "cancelled", None, error_msg
                    except Exception as e:
                        error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                        return "cancelled", None, error_msg
            except requests.exceptions.ConnectionError:
                error_msg = "Error: Connection to object detection service failed!"
                return "cancelled", None, error_msg

    def upload_images_to_server(self, frame_batch, image_info):
        url = self.service_url + ':' + str(self.service_port) + '/send_image'
        first_image = True
        error_msg = ""
        batch_id = None
        for image_file in frame_batch:
            # send image to the server
            print('Send image: ', image_file)
            # Open the image file in binary mode
            try:
                with open(image_file, 'rb') as image_file_binary:
                    # Prepare the request payload

                    image_info_json = json.dumps(image_info)
                    # Send the request
                    try:
                        response = requests.post(url, files={'image': image_file_binary},
                                                 data={'image_info': image_info_json})
                        # Check if the request was successful
                        if response.status_code == 200:
                            # Parse the JSON content
                            data = response.json()
                            # Access data using the keys, e.g., data['batch_id'], data['numb_files']
                            batch_id = data.get("batch_id")

                            if first_image:
                                image_info['batch_id'] = batch_id
                        else:
                            # Handle errors
                            ret_error_msg = response.text
                            error_msg = f"Error: Image could not be uploaded to the server: (HTTP error {response.status_code}, {ret_error_msg}!"
                            return None, error_msg
                    except requests.exceptions.ConnectionError:
                        error_msg = "Error: Connection to object detection service failed!"
                        return None, error_msg
            except IOError as e:
                # If an error occurs, print an error message or handle it as needed
                error_msg = f"Error: Failed to open file {image_file}. Error: {e}"
                return None, error_msg
        return batch_id, ""

    def delete_objects(selfs, model_id, objects):
        try:
            url = selfs.service_url + ':' + str(selfs.service_port) + '/delete_classes'
            request_body = {
                "class_names": objects,
                "model_id": model_id
            }
            request_body_json = json.dumps(request_body)
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, data=request_body_json, headers=headers)
            if response.status_code == 200:
                return ""
            else:
                try:
                    message_json = response.json()

                    if message_json is not None and 'description' in message_json:
                        error_msg = message_json['description']
                    else:
                        error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                except Exception as e:
                    error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Connection to object detection service failed!"
        return error_msg

    def reset_model(self, model_id, reset_finetuning=False):
        global args

        try:
            url = f"{self.service_url}:{self.service_port}/reset_model?model_id={model_id}&reset_finetuning={'true' if reset_finetuning else 'false'}"
            response = requests.post(url)
            if response.status_code == 200:
                # poll status of the job
                return_val = response.json()
                job_id = return_val['jobid']
                status, return_val, error_msg = self.poll_job_status(job_id)
                if status == 'completed':
                    self.label_list_base, self.label_list_base_sorted, \
                    self.super_category_list_sorted, self.super_category_use_area_list_sorted, \
                                        error_msg = self.get_label_list(model_id, 'all')
                    return self.label_list_base_sorted, error_msg
            else:
                try:
                    message_json = response.json()
                    if message_json is not None and 'description' in message_json:
                        error_msg = message_json['description']
                    else:
                        error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                except Exception as e:
                    error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"

        except requests.exceptions.ConnectionError:
            error_msg = "Error: Connection to object detection service failed!"
        return self.label_list_base_sorted, error_msg


    # get embeddings for the images in the image_annotation_data dictionary
    # image_annotation_data: dictionary with image file names as keys and a list of bounding boxes as values
    # if no bounding boxes are specified, the whole image is used for embedding calculation
    def get_embeddings(self, model_id, image_annotation_data, image_dir, upload_images=False):
        embeddings = {}
        error_msg = ""

        batch_id = None
        if upload_images:
            image_info = {"process_type": "calc_embeddings"}
            # get all key values of the dictionary image_annotation_data
            image_file_list = [os.path.join(image_dir, key) for key in image_annotation_data]
            batch_id, error_msg = self.upload_images_to_server(image_file_list, image_info)

        if batch_id is not None or not upload_images:
            url = self.service_url + ':' + str(self.service_port) + '/calc_embeddings'

            request_body = {
                "model_id": model_id,
                "annotation_data": image_annotation_data,
            }
            if not upload_images:
                request_body["image_dir"] = image_dir
            else:
                request_body['batch_id'] = None

            request_body_json = json.dumps(request_body)
            headers = {'Content-Type': 'application/json'}
            try:
                response = requests.post(url, data=request_body_json, headers=headers)
                if response.status_code == 200:
                    # poll status of the job
                    return_val = response.json()
                    job_id = return_val['jobid']
                    status, return_val, error_msg = self.poll_job_status(job_id)
                    if status == 'completed':
                        # get dictionary with image file names as keys and embeddings as values
                        # (number is equal to the number of bounding boxes)
                        embeddings = return_val['embeddings']
                    else:
                        error_msg = 'Error: Service job failed (status: ' + status + ', error message: ' + error_msg + ')'
                else:
                    try:
                        message_json = response.json()
                        if message_json is not None and 'description' in message_json:
                            error_msg = message_json['description']
                        else:
                            error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                    except Exception as e:
                        error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
            except requests.exceptions.ConnectionError:
                error_msg = "Error: Connection to object detection service failed!"

        return error_msg, embeddings

    def send_training_request(self, model_id, new_class_info: dict, batch_id=None, training_mode="new_object",
                              mask_background=False, re_calc_embeddings=True, zip_file_path: str = None):
        """
        Sends a POST request to the /inc_training endpoint with optional zip file and class info.

        Parameters:
            new_class_info (dict): Dictionary containing the new class information.
            zip_file_path (str, optional): Path to the zip file containing training images.

        Returns:
            Response object from the POST request.
        """
        url = self.service_url + ':' + str(self.service_port) + '/inc_training'
        headers = {'Content-Type': 'application/json'}
        #headers = {
        #    'accept': 'application/json'
        #}

        # Prepare form data
        files = {}
        data = {
            'batch_id': batch_id,
            'new_class_info': new_class_info, #json.dumps(new_class_info),
            'training_mode': training_mode,
            'mask_background': mask_background,
            'model_id': model_id,
            're_calc_embeddings': re_calc_embeddings  # always recalculate embeddings at training
        }

        if zip_file_path:
            try:
                files['zip_training_images'] = open(zip_file_path, 'rb')

            except FileNotFoundError:
                raise FileNotFoundError(f"Zip file not found: {zip_file_path}")

        try:
            request_body_json = json.dumps(data)
            response = requests.post(url, data=request_body_json, headers=headers) #, files=files, data=data)
        finally:
            # Ensure file is closed after the request
            if 'zip_training_images' in files:
                files['zip_training_images'].close()

        return response

    def training(self, model_id, training_images, image_dir, object_name, super_category_name, super_category_use_area,
                 train_index=0, upload_images=True, mask_background=False, max_train_index=5):
        prototype = None
        image_files = None
        embeddings = []
        error_msg = ""

        batch_id = None
        if upload_images:
            image_info = {
                "process_type": "inc_training",
            }
            batch_id, error_msg = self.upload_images_to_server(training_images, image_info)
            print('Upload images for object name: ', object_name, '(batch_id: ', batch_id, ') to server: ', training_images)

            object_dir = sanitize_directory_name(object_name)
        else:
            # check if training_dir is a parent directory of image_dir even if the training directory is set relative
            if not os.path.isabs(self.training_dir):
                training_dir_abs = os.path.join(os.getcwd(), self.training_dir)
            else:
                training_dir_abs = self.training_dir

            if not os.path.isabs(image_dir):
                image_dir_abs = os.path.join(os.getcwd(), image_dir)
            else:
                image_dir_abs = image_dir

            if not os.path.normpath(image_dir_abs).startswith(os.path.normpath(training_dir_abs)):
                # copy images to the dataset directory of the object detection service
                object_dir = sanitize_directory_name(object_name)
                object_dir_full_path = os.path.join(self.training_dir, object_dir)

                if not os.path.exists(object_dir_full_path):
                    # create the directory if it does not exist
                    os.makedirs(object_dir_full_path)
                #copy 'image_info.json' file to the object directory
                if os.path.exists(os.path.join(image_dir, 'image_info.json')):
                    # copy the image_info.json file to the object directory
                    shutil.copy(os.path.join(image_dir, 'image_info.json'), object_dir_full_path)
                file_list_source = os.listdir(image_dir)
                file_list_dest = os.listdir(object_dir_full_path)
                # check if the file lists are the same
                copy_all_files = False
                if len(file_list_source) != len(file_list_dest):
                    copy_all_files = True
                else:
                    # check if all files in the image directory are in the object directory
                    for file in file_list_source:
                        if file not in file_list_dest:
                            copy_all_files = True
                            break
                if copy_all_files:
                    # remove all files in the object directory
                    for file in file_list_dest:
                        file_full_path = os.path.join(object_dir_full_path, file)
                        if os.path.isfile(file_full_path):
                            os.remove(file_full_path)
                    # copy all files from the image directory to the object directory
                    for file in file_list_source:
                        file_full_path = os.path.join(image_dir, file)
                        if os.path.isfile(file_full_path):
                            shutil.copy(file_full_path, object_dir_full_path)

            else:
                # get the relative path of the object directory
                object_dir = os.path.basename(image_dir)

            '''            
            if train_index > 0:
                object_dir = object_dir + '_multi_view_' + str(train_index)
            else:
                if os.path.exists(object_dir_full_path):
                    # move all images to this directory
                    for mv_index in range(1, max_train_index):
                        mv_object_dir = object_dir + '_multi_view_' + str(mv_index)
                        mv_object_dir_full_path = os.path.join(self.training_dir, mv_object_dir)
                        if os.path.exists(mv_object_dir_full_path):
                            # move all images from this directory to the new directory
                            mv_file_list = os.listdir(mv_object_dir_full_path)
                            for mv_file in mv_file_list:
                                mv_file_full_path = os.path.join(mv_object_dir_full_path, mv_file)
                                if os.path.isfile(mv_file_full_path):
                                    shutil.move(mv_file_full_path, object_dir_full_path)

            if not os.path.exists(object_dir_full_path):
                os.makedirs(object_dir_full_path)
                for image_file in training_images:
                    shutil.copy(image_file, object_dir_full_path)
            else:
                # check if all training images are already in the directory
                all_images_in_dir = True
                image_files = os.listdir(object_dir_full_path)
                if len(image_files) != len(training_images):
                    all_images_in_dir = False
                else:
                    for image_file in training_images:
                        if not os.path.exists(os.path.join(object_dir_full_path, os.path.basename(image_file))):
                            all_images_in_dir = False
                            break
                if not all_images_in_dir:
                    # remove dir tree
                    shutil.rmtree(object_dir_full_path)
                    os.makedirs(object_dir_full_path)
                    for image_file in training_images:
                        shutil.copy(image_file, object_dir_full_path)
            '''

        if batch_id is not None or not upload_images:
            object_name_index = object_name
            try:
                if train_index == 0:
                    training_mode = "new_object"
                    response = self.send_training_request(model_id, {object_name: [super_category_name, super_category_use_area]},
                                                          batch_id, training_mode, mask_background, upload_images)
                else:
                    training_mode = "multi_view"
                    object_name_index = object_name + '_multi_view_' + str(train_index)
                    #object_dir = object_dir + '_multi_view_' + str(train_index)
                    response = self.send_training_request(model_id, {object_name_index: [super_category_name, super_category_use_area]},
                                                          batch_id, training_mode, mask_background, upload_images)

                if response.status_code == 200:
                    # poll status of the job
                    return_val = response.json()
                    job_id = return_val['jobid']
                    status, return_val, error_msg = self.poll_job_status(job_id)
                    if status == 'completed':
                        embeddings = return_val['example_embeddings'][object_name_index]

                        if len(embeddings) == 0:
                            error_msg = "object_detection_client::training error: No embeddings found for the object!"
                            print(error_msg)
                        #print('object_detection_client::training: embeddings: found')

                        image_files = return_val['example_files'][object_name_index]
                        prototype = return_val['prototype']
                    else:
                        error_msg = 'Error: Service job failed (status: ' + status + ', error message: ' + error_msg + ')'
                else:
                    try:
                        message_json = response.json()
                        if message_json is not None and 'description' in message_json:
                            error_msg = message_json['description']
                        else:
                            error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                    except Exception as e:
                        error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
            except requests.exceptions.ConnectionError:
                error_msg = "Error: Connection to object detection service failed!"

        else:
            print('else branch of: batch_id is not None or not upload_images')

        return error_msg, image_files, embeddings, prototype

    def finetune_model(self, model_id, numb_iter, top_k, eval_period, numb_gpus, batch_size, base_lr, is_eval,
                       training_annotation_dir, session_logger):
        error_msg = ""
        try:
            url = self.service_url + ':' + str(self.service_port) + '/finetune_model'
            request_body = {
                "batch_id": None,
                "model_id": model_id,
                "training_annotation_dir": training_annotation_dir,
                "trainings_parameters": {"numb_gpus": numb_gpus, "max_iter": numb_iter, "top_k": top_k, "eval_period": eval_period,
                                         'ims_per_batch': batch_size, 'base_lr': base_lr, 'eval': is_eval}
            }

            request_body_json = json.dumps(request_body)
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, data=request_body_json, headers=headers)

            if response.status_code == 200:
                # poll status of the job
                return_val = response.json()
                job_id = return_val['jobid']
                session_logger.clear_log_messages()
                status, return_val, error_msg = self.poll_job_status(job_id, session_logger)
                if status == 'completed':
                    finetuned_model_path = return_val['finetuned_model_path']
                    finetuned_labels = return_val['finetuned_labels']

                    if 'val_results' in return_val:
                        val_result = return_val['val_results']
                    else:
                        val_result = None

                    #if 'eval_report_path' in return_val:
                    #    eval_report_path = return_val['eval_report_path']
                    #else:
                    #    eval_report_path = None

                    #if "evaluation_results" in return_val:
                    #    evaluation_results = return_val["evaluation_results"]
                    #else:
                    #    evaluation_results = None


                    return "", finetuned_model_path, finetuned_labels, val_result #, eval_report_path
                else:
                    error_msg = 'Error: Service job failed (status: ' + status + ', error message: ' + error_msg + ')'
            else:
                try:
                    message_json = response.json()
                    if message_json is not None and 'description' in message_json:
                        error_msg = message_json['description']
                    else:
                        error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                except Exception as e:
                    error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Connection to object detection service failed!"
        return error_msg, None, None, None #, None

    # set image_dir to None if the images are already uploaded
    # set batch_id to None if the images are already on in the specified directory on the server
    def det_images_on_server(self, model_id, images_dir, batch_id, class_selection_mode,
                             detect_supercategories, del_uploaded_images,
                             with_embeddings=False, session_logger=None, top_k=1):
        url = self.service_url + ':' + str(self.service_port) + '/detect_objects'
        if batch_id is None:
            request_body = {
                "batch_id": None,
                "model_id": model_id,
                "top_k": top_k,
                "class_selection_mode": class_selection_mode,
                "detect_supercategories": detect_supercategories,
                "del_data": del_uploaded_images,
                "server_data_path": images_dir,
                "with_embeddings": with_embeddings
            }
        else:
            request_body = {
                "batch_id": batch_id,
                "model_id": model_id,
                "class_selection_mode": class_selection_mode,
                "detect_supercategories": detect_supercategories,
                "del_data": del_uploaded_images,
                "validation_data": None,
                "with_embeddings": with_embeddings
            }
        request_body_json = json.dumps(request_body)
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(url, data=request_body_json, headers=headers)
            if response.status_code == 200:
                # poll status of the job
                return_val = response.json()
                job_id = return_val['jobid']
                status, return_val, error_msg = self.poll_job_status(job_id, session_logger)
                if status == 'completed':
                    #print('det_images_on_server return_val: ', return_val)

                    if 'result_list' in return_val:
                        result_list = return_val['result_list']

                        for img_res in result_list:
                            features = img_res['features']
                            for res in features:
                                res['filtered'] = False

                        return result_list, error_msg
                    else:
                        if 'val_results' in return_val:
                            val_result = return_val['val_results']
                            return val_result, error_msg
                        else:
                            error_msg = "Error: No result_list or val_results found in the response from the object detection service!"
                            return [], error_msg
            else:
                try:
                    message_json = response.json()
                    if message_json is not None and 'description' in message_json:
                        error_msg = message_json['description']
                    else:
                        error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                except Exception as e:
                    error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Connection to object detection service failed!"
        return [], error_msg


    def det_preloaded_image_batches(self, model_id, batch_id_list, class_selection_mode, detect_supercategories,
                                    del_uploaded_images=True, top_k=1):
        image_result_list = []
        error_msg = ""
        for batch_id in batch_id_list:
            image_result, error_msg = self.det_images_on_server(model_id, "", batch_id,
                                                                class_selection_mode, detect_supercategories,
                                                                del_uploaded_images, False, None, top_k)
            if error_msg != "":
                break
            image_result_list.extend(image_result)
        return image_result_list, error_msg


    def det_image_batch(self, model_id, frame_batch, frame_pos_batch, batch_offset, batch_size, class_selection_mode,
                        detect_supercategories, del_uploaded_images=True, with_embeddings=False, top_k=1):

        image_info = {
            "process_type": "extract",
        }
        batch_id, error_msg = self.upload_images_to_server(frame_batch, image_info)
        if batch_id is None or error_msg != "":
            return [], frame_batch, frame_pos_batch, batch_offset, error_msg, None

        image_result, error_msg = self.det_images_on_server(model_id, "", batch_id, class_selection_mode,
                                                            detect_supercategories, del_uploaded_images,
                                                            with_embeddings, None, top_k)
        if error_msg != "":
            return [], frame_batch, frame_pos_batch, batch_offset, error_msg, None
        else:
            # delete frame_batch
            frame_batch = []
            frame_pos_batch = []
            batch_offset += batch_size

            if del_uploaded_images:
                batch_id = None

            return image_result, frame_batch, frame_pos_batch, batch_offset, error_msg, batch_id

    def det_image_sequence(self, model_id, image_sequence_files, class_selection_mode, detect_supercategories,
                           batch_size, del_uploaded_images=True, with_embeddings=False):

        results = []
        frame_batch = []
        frame_pos_batch = []

        error_msg = ""
        batch_id_list = []

        batch_offset = 0
        for i in range(len(image_sequence_files)):
            frame_batch.append(image_sequence_files[i])
            frame_pos_batch.append(0)
            if len(frame_batch) == batch_size:
                # recognize the landmarks in the frame_batch
                result_list, frame_batch, frame_pos_batch, batch_offset, error_msg, batch_id = \
                    self.det_image_batch(model_id, frame_batch, frame_pos_batch, 0, batch_size,
                                         class_selection_mode, detect_supercategories, del_uploaded_images,
                                         with_embeddings)
                if not del_uploaded_images and batch_id_list is not None:
                    batch_id_list.append(batch_id)
                results.extend(result_list)
                if len(error_msg) > 0:
                    return [], [], error_msg
        if len(frame_batch) > 0:
            # recognize the landmarks in the frame_batch
            result_list, frame_batch, frame_pos_batch, batch_offset, error_msg, batch_id = \
                self.det_image_batch(model_id, frame_batch, frame_pos_batch, batch_offset, batch_size,
                                     class_selection_mode, detect_supercategories, del_uploaded_images,
                                     with_embeddings)
            if not del_uploaded_images and batch_id_list is not None:
                batch_id_list.append(batch_id)
            results.extend(result_list)

        return results, batch_id_list, error_msg

    """
    Calculate class scores for each patch in a specified region of the image.

    Parameters:
    - image_file: (Optional) The file path of the image to analyze. Must be accessible from the server.
    - bbox: The bounding box specifying the region of interest in the image.
    - batch_id: (Optional) The batch ID if the image has been uploaded to the server.
    - class_selection: (Optional) The class selection mode for filtering results.

    Returns:
    - A dictionary with the following keys:
      - 'box_patch': The recalculated bounding box assigned to patches.
      - 'image_patch_class_scores': The class scores for each patch.
      - 'image_patch_topk_class_indices': The top class indices for each patch (indices of classes with the highest scores).
      - 'class_indices_sorted': The sorted (based on area) class indices which appear in the 'image_patch_topk_class_indices'.
    """
    def analyse_region(self, model_id, image_file, bbox, background_mask_image = None, batch_id=None,
                       class_selection_mode='all', top_k=1, class_selection=None, analyse_label=None,
                        use_background_mask=False):
        url = self.service_url + ':' + str(self.service_port) + '/analyse_region'

        # convert int64 values in bbox to int
        if bbox is not None:
            bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]


        # Prepare the request payload
        if batch_id is None:
            request_body = {
                "image_file": image_file,
                "background_mask_image": background_mask_image,
                "region": bbox,
                "class_selection_mode": class_selection_mode,
                "top_k": top_k,
                "class_selection": class_selection,
                "analyse_label": analyse_label,
                "use_background_mask": use_background_mask,
                "model_id": model_id
            }
        else:
            request_body = {
                "batch_id": batch_id,
                "region": bbox,
                "class_selection_mode": class_selection_mode,
                "top_k": top_k,
                "class_selection": class_selection,
                "analyse_label": analyse_label,
                "use_background_mask": use_background_mask,
                "model_id": model_id
            }

        request_body_json = json.dumps(request_body)
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(url, data=request_body_json, headers=headers)
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON content
                data = response.json()
                data['image_patch_class_scores'] = np.array(data['image_patch_class_scores'])
                data['image_patch_top_class_indices'] = np.array(data['image_patch_top_class_indices'])

                return data, ""
            else:
                # Handle errors
                ret_error_msg = response.text
                error_msg = f"Error: Image region could not be analysed at the server: (HTTP error {response.status_code}, {ret_error_msg}!"
                return None, error_msg
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Connection to object detection service failed!"
            return None, error_msg


    def inference_of_dataset(self, model_id, dataset, image_files, images_dir, threshold, object_list,
                             class_selection_mode='all', detect_supercategories=False, field_name='predictions',
                             batch_size=10, del_uploaded_images=True, del_prev_detections=True, top_k=1):
        error_msg = ""
        field_schema = dataset.get_field_schema()
        if field_name in field_schema:
            if del_prev_detections:
                dataset.delete_sample_field(field_name)
                dataset.add_sample_field(field_name, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)
        else:
            dataset.add_sample_field(field_name, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)

        if len(images_dir) == 0 or (image_files is not None and len(image_files) > 0):
            results, batch_id_list, error_msg = self.det_image_sequence(model_id, image_files, class_selection_mode,
                                                                        detect_supercategories, batch_size, del_uploaded_images)
        else:
            results, error_msg = self.det_images_on_server(model_id, images_dir, None, class_selection_mode,
                                                           detect_supercategories, del_uploaded_images, False, None, top_k)

        #filter_overlapped_objects(results, threshold)

        if len(error_msg) == 0:

            insert_results_in_dataset(dataset, results, images_dir, threshold, object_list, field_name)

        return results, error_msg

    def evaluate(self, model_id, validation_dataset, val_images_dir, validation_image_list, label, batch_size=10):

        batch_id_list = None
        results, error_msg = self.inference_of_dataset(model_id, validation_dataset, validation_image_list,
                                              val_images_dir, 0.0,[label], 'all',
                                             'predictions', batch_size, False, True)
        if len(error_msg) == 0:
            evaluation_result = validation_dataset.evaluate_detections(
                "predictions",
                gt_field="detections",
                eval_key="eval",
                compute_mAP=True
            )

            return evaluation_result, error_msg
        else:
            return None, error_msg

    def evaluate_clusters(self, model_id, validation_dataset, val_images_dir, validation_image_list, label,
                          numb_clusters, batch_size=10, top_k=1):
        error_msg = ""
        field_name = 'predictions'
        evaluation_result = None
        evaluation_result_clusters= [None] * numb_clusters

        if len(val_images_dir) == 0:
            results, batch_id_list, error_msg = self.det_image_sequence(validation_image_list, 'all',
                                                                        False, batch_size, False)
        else:
            results, error_msg = self.det_images_on_server(model_id, val_images_dir, None, 'all',
                                                           False, False, False, None, top_k)

        if len(error_msg) == 0:
            #filter_overlapped_objects(results, 0.0)

            for cluster_numb in range(numb_clusters):
                field_schema = validation_dataset.get_field_schema()
                if field_name in field_schema:
                    validation_dataset.delete_sample_field(field_name)
                    validation_dataset.add_sample_field(field_name, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)
                else:
                    validation_dataset.add_sample_field(field_name, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)

                insert_results_in_dataset(validation_dataset, results, val_images_dir, 0.0,
                                          [label], field_name, cluster_numb)

                evaluation_result_clusters[cluster_numb] = validation_dataset.evaluate_detections(
                    "predictions",
                    gt_field="detections",
                    eval_key="eval",
                    compute_mAP=True
                )

            if numb_clusters == 1:
                evaluation_result = evaluation_result_clusters[0]
            else:
                validation_dataset.delete_sample_field(field_name)
                validation_dataset.add_sample_field(field_name, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)

                insert_results_in_dataset(validation_dataset, results, val_images_dir, 0.0, [label], field_name, -1)

                evaluation_result = validation_dataset.evaluate_detections(
                    "predictions",
                    gt_field="detections",
                    eval_key="eval",
                    compute_mAP=True
                )

        return evaluation_result, evaluation_result_clusters, error_msg
