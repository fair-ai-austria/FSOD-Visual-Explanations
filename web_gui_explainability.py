"""Object detection explainability demo (web user interface).

The application visualizes how the object detection model classifies the image patches of a query image
(patch scores and patch classes), optionally also for the foreground of the image defined by a mask image.

Data layout (below --data_dir), one folder per dataset / model type:

    <data_dir>/coco/images/    source (query) images and their mask images (*mask.png)
    <data_dir>/coco/results/   saved analysis results (<image file name>.npz), label list (labels.json)
    <data_dir>/lvis/images/    ... same for the lvis model
    <data_dir>/lvis/results/

Usage:
  * The gallery of pre-analysed images shows thumbnails of all images of the selected model that have a saved
    analysis result. A click on a thumbnail creates the visualizations from the saved result (image and, if there was
    one, its mask). The object detection service is not needed.
  * A new image (and optional its mask) that is uploaded with the upload button is analysed by the object detection
    service (which must be available). The image, the mask and the result are saved in the folders above, so
    that the image appears in the gallery of pre-analysed images afterwards.
"""
import argparse
import json
import os
import shutil

import cv2
import gradio as gr
import numpy as np

import explainability_util
from object_detection_client import object_detection_client

"""Parse arguments."""
parser = argparse.ArgumentParser(description='Object Detection Explainability Demo')

# setting for connection to the object detection service
parser.add_argument('--service_url', default='localhost', type=str,
                    help='URL to the object detection service')
parser.add_argument('--service_port', default=8001, type=int,
                    help='Port of the object detection service')
parser.add_argument('--port', default=8003, type=int,
                    help='Port of web user interface')

# settings for the demo data
parser.add_argument('--data_dir', default='.', type=str,
                    help='Directory containing the folders "coco" and "lvis". Each of them has the sub folders '
                         '"images" (source images and masks) and "results" (saved analysis results)')
parser.add_argument('--coco_model_id', default='default_coco_dinov3', type=str,
                    help='Model id used for the "coco" model')
parser.add_argument('--lvis_model_id', default='default_lvis_dinov3', type=str,
                    help='Model id used for the "lvis" model')

args = parser.parse_args()
detection_client = None

MODEL_CHOICES = ['coco', 'lvis']
IMAGES_SUBDIR = 'images'
RESULTS_SUBDIR = 'results'
LABELS_FILE = 'labels.json'
MASK_SUFFIX = 'mask.png'  # mask images must end with this suffix
TOP_K_INDEX = 0  # the visualizations show the best class of each patch
HOVER_TOP_K = 3  # number of best classes per patch shown in the hover text
THRESHOLD = 0.0  # patches with a lower score are not shown in the class figure


# ----------------------------------------------------------------------------------------------------------------------
# Data folders
# ----------------------------------------------------------------------------------------------------------------------

def get_model_id(model_choice):
    """Map the selection of the user interface ('coco' or 'lvis') to the model id of the service."""
    return args.lvis_model_id if model_choice == 'lvis' else args.coco_model_id


def get_dataset_name(model_id):
    """The data folder ('coco' or 'lvis') depends on the used model_id."""
    return 'lvis' if 'lvis' in model_id.lower() else 'coco'


def get_data_dirs(model_id, create=False):
    """Return the absolute paths of the folder with the source images and of the folder with the saved results."""
    base_dir = os.path.abspath(os.path.join(args.data_dir, get_dataset_name(model_id)))
    images_dir = os.path.join(base_dir, IMAGES_SUBDIR)
    results_dir = os.path.join(base_dir, RESULTS_SUBDIR)
    if create:
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
    return images_dir, results_dir


def get_result_path(model_id, image_name):
    """Path of the file with the saved analysis result of an image (identified by its file name)."""
    _, results_dir = get_data_dirs(model_id)
    return os.path.join(results_dir, image_name + '.npz')


def list_samples(model_id):
    """List the file names of the pre-analysed images of the model (saved result and source image exist)."""
    images_dir, results_dir = get_data_dirs(model_id)
    if not os.path.isdir(results_dir):
        return []
    names = [f[:-len('.npz')] for f in os.listdir(results_dir) if f.endswith('.npz')]
    return sorted(name for name in names if os.path.isfile(os.path.join(images_dir, name)))


# ----------------------------------------------------------------------------------------------------------------------
# Label list
# ----------------------------------------------------------------------------------------------------------------------

def get_labels(model_id):
    """Get the label list from the service and save it. If the service is not available the saved list is used.

    Returns the label list (None if not available at all) and a flag if the service is available."""
    _, results_dir = get_data_dirs(model_id)
    labels_path = os.path.join(results_dir, LABELS_FILE)
    try:
        label_list, _, _, _, error_msg = detection_client.get_label_list(model_id, 'all')
    except Exception as e:
        label_list, error_msg = None, str(e)

    # the client returns an error message if the service could not be reached
    if len(error_msg) == 0 and label_list:
        get_data_dirs(model_id, create=True)
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(list(label_list), f)
        return list(label_list), True

    if os.path.isfile(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            return json.load(f), False
    return None, False


def get_saved_labels(model_id):
    """Get the saved label list. Only if there is none, the service is asked (the service is not needed
    to show saved results)."""
    _, results_dir = get_data_dirs(model_id)
    labels_path = os.path.join(results_dir, LABELS_FILE)
    if os.path.isfile(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    label_list, _ = get_labels(model_id)
    return label_list


# ----------------------------------------------------------------------------------------------------------------------
# Saving and loading of the analysis results
# ----------------------------------------------------------------------------------------------------------------------

def save_result(model_id, image_file, mask_file, result, result_foreground):
    """Save the source image, the mask and the analysis results in the folders of the model.

    The result is stored in one .npz file (named like the image) with the arrays 'scores', 'class_indices', 'bbox',
    'mask_name' and, if a mask was used, 'scores_foreground' and 'class_indices_foreground'.
    Returns the paths of the saved image and mask (mask is None if there is none)."""
    images_dir, results_dir = get_data_dirs(model_id, create=True)

    # copy the source files into the image folder
    saved_files = []
    for source_file in [image_file, mask_file]:
        if source_file is None:
            saved_files.append(None)
            continue
        target_file = os.path.join(images_dir, os.path.basename(source_file))
        try:
            shutil.copyfile(source_file, target_file)
        except shutil.SameFileError:
            pass  # the file was already selected from the image folder
        saved_files.append(target_file)

    data = {
        'scores': result['scores'],
        'class_indices': result['class_indices'],
        'bbox': np.array(result['bbox']),
        'mask_name': np.array(os.path.basename(mask_file) if mask_file is not None else ''),
    }
    if result_foreground is not None:
        data['scores_foreground'] = result_foreground['scores']
        data['class_indices_foreground'] = result_foreground['class_indices']
    np.savez_compressed(get_result_path(model_id, os.path.basename(image_file)), **data)
    return saved_files[0], saved_files[1]


def load_result(model_id, image_name):
    """Load the saved analysis result of an image of the image folder.

    Returns the paths of the image and the mask (None if there is none), the result and the result of the
    foreground analysis (None if there is none). A result is a dict with 'scores', 'class_indices' and 'bbox'."""
    images_dir, _ = get_data_dirs(model_id)
    with np.load(get_result_path(model_id, image_name)) as data:
        result = {
            'scores': data['scores'],
            'class_indices': data['class_indices'],
            'bbox': [int(v) for v in data['bbox']],
        }
        mask_name = str(data['mask_name'])
        result_foreground = None
        if 'scores_foreground' in data:
            result_foreground = {
                'scores': data['scores_foreground'],
                'class_indices': data['class_indices_foreground'],
                'bbox': result['bbox'],  # the foreground is analysed with the same region as the whole image
            }

    image_file = os.path.join(images_dir, image_name)
    mask_file = os.path.join(images_dir, mask_name) if len(mask_name) > 0 else None
    if mask_file is not None and not os.path.isfile(mask_file):
        # the mask is needed to create the foreground image
        mask_file = None
        result_foreground = None
    return image_file, mask_file, result, result_foreground


# ----------------------------------------------------------------------------------------------------------------------
# Analysis (object detection service) and visualization
# ----------------------------------------------------------------------------------------------------------------------

def create_foreground_image(image_file, mask_file, out_file):
    """Mask out the background of the image and save the result."""
    image = cv2.imread(image_file)
    mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    if mask_image.shape[:2] != image.shape[:2]:
        mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    image = cv2.bitwise_and(image, mask_image)
    cv2.imwrite(out_file, image)


def get_foreground_path(model_id, image_file):
    """Path of the generated foreground image (it is stored in the result folder)."""
    _, results_dir = get_data_dirs(model_id, create=True)
    return os.path.join(results_dir, os.path.basename(image_file) + '_foreground.jpg')


def analyse_image(model_id, image_file, label_list_base):
    """Analyse the whole image with the object detection service.

    The best HOVER_TOP_K classes (or less if there are fewer classes) are requested for each patch.
    Returns the result (dict with 'scores', 'class_indices' and 'bbox') and an error message."""
    top_k = max(1, min(HOVER_TOP_K, len(label_list_base)))
    image_info = {
        "process_type": "extract",
    }
    batch_id, error = detection_client.upload_images_to_server([image_file], image_info)
    if len(error) > 0:
        return None, f"Error uploading image: {error}"

    results, error = detection_client.analyse_region(model_id, None, None, None, batch_id, 'all', top_k)
    if results is None or len(error) > 0:
        return None, f"Error analysing region: {error}"

    return {
        'scores': results['image_patch_class_scores'],
        'class_indices': results['image_patch_top_class_indices'],
        'bbox': [int(v) for v in results['bbox']],
    }, ""


def create_figures(image_file, result, label_list_base, max_score=None):
    """Create the patch score heatmap and the patch class figure of a result.

    max_score is the upper limit of the heatmap color scale (default: the maximum score of the result)."""
    scores = result['scores']
    class_indices = result['class_indices']
    hover_text = explainability_util.get_hover_text(class_indices, scores, label_list_base)
    if max_score is None:
        max_score = np.max(scores)

    score_heatmap = explainability_util.create_patch_score_heatmap(scores[:, :, TOP_K_INDEX], 0, max_score,
                                                                   hover_text)
    class_fig = explainability_util.create_patch_class_figure(image_file, None, scores, class_indices,
                                                              result['bbox'], TOP_K_INDEX, label_list_base,
                                                              hover_text, THRESHOLD)
    return score_heatmap, class_fig


def visualize(model_id, image_file, mask_file, result, result_foreground, label_list_base):
    """Create the visualization images from the (saved) analysis results.

    Returns the images for the gallery and the four figures (score and class figure of the whole image,
    score and class figure of the foreground; the foreground figures are None without mask)."""
    score_heatmap, class_fig = create_figures(image_file, result, label_list_base)

    score_heatmap_foreground = None
    class_fig_foreground = None
    gallery_images = [image_file]
    if mask_file is not None:
        gallery_images.append(mask_file)
        if result_foreground is not None:
            # the foreground figures are drawn over the masked image; the same color scale as for the whole image
            # is used so that the scores are comparable
            image_file_foreground = get_foreground_path(model_id, image_file)
            create_foreground_image(image_file, mask_file, image_file_foreground)
            score_heatmap_foreground, class_fig_foreground = create_figures(
                image_file_foreground, result_foreground, label_list_base, np.max(result['scores']))

    return gallery_images, score_heatmap, class_fig, score_heatmap_foreground, class_fig_foreground


# ----------------------------------------------------------------------------------------------------------------------
# Event handlers of the user interface
# ----------------------------------------------------------------------------------------------------------------------

def error_outputs(error_msg):
    """Outputs of the handlers in case of an error: empty figures and the message in the status line."""
    return None, None, None, None, None, error_msg


def split_image_and_mask(files):
    """Split the selected files into the image and the optional mask (a file ending with 'mask.png').

    Returns image file, mask file and an error message."""
    if files is None or isinstance(files, str):
        files = [files] if files else []
    files = [f for f in files if os.path.isfile(f)]  # ignore folders
    mask_files = [f for f in files if f.endswith(MASK_SUFFIX)]
    image_files = [f for f in files if not f.endswith(MASK_SUFFIX)]

    if len(image_files) != 1 or len(mask_files) > 1:
        return None, None, "Please select an image and optional its mask image (file name ends with 'mask.png')."
    return image_files[0], (mask_files[0] if mask_files else None), ""


def show_saved_result(model_id, image_name, status=""):
    """Create the visualizations of an image of the image folder from its saved analysis result."""
    label_list_base = get_saved_labels(model_id)
    if label_list_base is None:
        return error_outputs("No label list available (no service and no saved label list).")

    try:
        image_file, mask_file, result, result_foreground = load_result(model_id, image_name)
    except Exception as e:
        return error_outputs(f"Saved analysis result could not be loaded: {e}")

    outputs = visualize(model_id, image_file, mask_file, result, result_foreground, label_list_base)
    return (*outputs, status)


def analyse_and_save(model_id, image_file, mask_file, label_list_base):
    """Analyse the image (and its foreground if there is a mask) with the service, save the source files and the
    results and create the visualizations."""
    result, error = analyse_image(model_id, image_file, label_list_base)
    if result is None:
        print(error)
        return error_outputs(error)

    result_foreground = None
    if mask_file is not None:
        # mask out the background and analyse the foreground separately
        image_file_foreground = get_foreground_path(model_id, image_file)
        create_foreground_image(image_file, mask_file, image_file_foreground)
        result_foreground, error = analyse_image(model_id, image_file_foreground, label_list_base)
        if result_foreground is None:
            error_msg = "Error analysing region with mask: " + error
            print(error_msg)
            return error_outputs(error_msg)
        result_foreground['bbox'] = result['bbox']

    image_file, mask_file = save_result(model_id, image_file, mask_file, result, result_foreground)

    outputs = visualize(model_id, image_file, mask_file, result, result_foreground, label_list_base)
    return (*outputs, "")


def analyse_new_image(files, model_choice):
    """Handler of the upload button: analyse a new image and optional its mask with the object detection service.

    The image, the mask and the result are saved, so that the image is a pre-analysed image afterwards."""
    if not files:
        return error_outputs("")

    image_file, mask_file, error = split_image_and_mask(files)
    if len(error) > 0:
        return error_outputs(error)

    model_id = get_model_id(model_choice)
    label_list_base, service_available = get_labels(model_id)
    if not service_available:
        return error_outputs("The object detection service is not available. Please select one of the "
                             "pre-analysed images.")
    return analyse_and_save(model_id, image_file, mask_file, label_list_base)


def select_sample(model_choice, evt: gr.SelectData):
    """Handler of the gallery of pre-analysed images: show the saved result of the clicked image."""
    samples = list_samples(get_model_id(model_choice))
    if evt.index is None or evt.index >= len(samples):
        return error_outputs("Please select a pre-analysed image.")
    return show_saved_result(get_model_id(model_choice), samples[evt.index])


def get_sample_thumbnails(model_choice):
    """Content of the gallery of pre-analysed images: (image file, file name) of the images of the model."""
    model_id = get_model_id(model_choice)
    images_dir, _ = get_data_dirs(model_id)
    return [(os.path.join(images_dir, name), name) for name in list_samples(model_id)]


def refresh_samples(model_choice):
    """Update of the gallery of pre-analysed images."""
    return gr.update(value=get_sample_thumbnails(model_choice))


def change_model(model_choice):
    """Handler for the model selection: show the pre-analysed images of the selected model and the service status."""
    model_id = get_model_id(model_choice)
    label_list_base, service_available = get_labels(model_id)
    samples = list_samples(model_id)
    if service_available:
        status = ""
    elif label_list_base is not None and len(samples) > 0:
        status = "Object detection service not available: only the pre-analysed images can be used."
    else:
        status = "Object detection service not available and no pre-analysed images found."

    return refresh_samples(model_choice), status


def build_gradio_app():
    global detection_client
    global args

    if gr.__version__ > '4.40.0':
        css = """
         #status_id {color: red; padding: 10px}
         .bordered-group {
             border: 1px solid #ddd;
             border-radius: 5px;
             padding: 10px;
             margin: 10px 0;
         }
         .custom-gallery { 
             overflow-y: auto !important;
         }
         .grid-wrap {
             overflow: auto;
         }
         .gr  

         () => {
             const params = new URLSearchParams(window.location.search);
             if (!params.has('__theme')) {
                 params.set('__theme', 'dark');
                 window.location.search = params.toString();
             }
         }

         """
    else:
        css = """
         #status_id {color: red; padding: 10px}
         .bordered-group {
             border: 1px solid #ddd;
             border-radius: 5px;
             padding: 10px;
             margin: 10px 0;
         }
         .custom-gallery { 
         } 
         () => {
             const params = new URLSearchParams(window.location.search);
             if (!params.has('__theme')) {
                 params.set('__theme', 'dark');
                 window.location.search = params.toString();
             }
         }   
         """

    detection_client = object_detection_client('http://' + args.service_url, args.service_port)

    # the folders must exist so that the pre-analysed images can be listed and new ones saved
    for model_choice in MODEL_CHOICES:
        get_data_dirs(get_model_id(model_choice), create=True)

    with (gr.Blocks(css=css, delete_cache=(86400, 86400)) as app):
        with gr.Column():
            model_selection = gr.Radio(choices=MODEL_CHOICES, value=MODEL_CHOICES[0], label="Model")

            # thumbnails of the pre-analysed images of the selected model; a click shows the saved result
            sample_gallery = gr.Gallery(label="Pre-analysed images (click an image to show its analysis)",
                                        columns=8, height=200, object_fit='cover', allow_preview=False,
                                        show_download_button=False)

            # analysis of a new image (and optional its mask): requires the object detection service
            image_sequ_upload = gr.UploadButton(label='Analyse a new image and optional its mask (requires service)',
                                                file_count="multiple", file_types=["image"])

            input_images_selected = gr.Gallery(label="Selected Images", object_fit='contain')
            with gr.Row():
                region_score_display = gr.Plot(label="Region Patch Score Result")
                region_class_display = gr.Plot(label="Region Patch Class Result")
            with gr.Row():
                region_score_display_foreground = gr.Plot(label="Region Patch Score Result (foreground)")
                region_class_display_foreground = gr.Plot(label="Region Patch Class Result (foreground)")

            # insert status line
            status_line = gr.Textbox(interactive=False, show_label=False, container=False, elem_id="status_id",
                                     max_lines=1)

        outputs = [input_images_selected,
                   region_score_display,
                   region_class_display,
                   region_score_display_foreground,
                   region_class_display_foreground,
                   status_line]

        sample_gallery.select(fn=select_sample, inputs=model_selection, outputs=outputs)
        # after the analysis the new image is added to the gallery of pre-analysed images
        image_sequ_upload.upload(fn=analyse_new_image, inputs=[image_sequ_upload, model_selection],
                                 outputs=outputs).then(fn=refresh_samples, inputs=model_selection,
                                                       outputs=sample_gallery)

        model_selection_outputs = [sample_gallery, status_line]
        model_selection.change(fn=change_model, inputs=model_selection, outputs=model_selection_outputs)
        app.load(fn=change_model, inputs=model_selection, outputs=model_selection_outputs)

        # the data folder may be located outside of the working directory
        app.launch(server_name='0.0.0.0', server_port=args.port, share=False,
                   allowed_paths=[os.path.abspath(args.data_dir)])


if __name__ == "__main__":
    build_gradio_app()
