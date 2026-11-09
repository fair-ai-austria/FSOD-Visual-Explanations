import plotly.express as px
import numpy as np
import base64
import cv2

def get_hover_text(class_indices, score_values, label_list_base, from_index=0):

    '''
    hover_text = np.array(
        [[f"{label_list_base[class_index]}: {results['image_patch_class_scores'][row_index][col_index][top_k]:.2f}"
          if class_index >= 0 else f"pre-filtered: {results['image_patch_class_scores'][row_index][col_index][top_k]:.2f}" for
          col_index, class_index in enumerate(row)] for row_index, row in
         enumerate(results['image_patch_top_class_indices'])])
    '''

    hover_text = np.full((class_indices.shape[0], class_indices.shape[1]), "", dtype=object)

    valid_indices_0 = class_indices[:,:,from_index] >= 0
    if score_values.shape[2] > from_index+2:
        valid_indices_2 = class_indices[:,:,from_index+2] >= 0

        hover_text[valid_indices_2] = [
            f"{label_list_base[class_indices[row_index, col_index, from_index+0]]}: {score_values[row_index, col_index, from_index+0]:.2f}" +
            f" ({label_list_base[class_indices[row_index, col_index, from_index+1]]}: {score_values[row_index, col_index, from_index+1]:.2f}" +
            f" {label_list_base[class_indices[row_index, col_index, from_index+2]]}: {score_values[row_index, col_index, from_index+2]:.2f})"

            for row_index, col_index in zip(*np.where(valid_indices_2))
        ]

        valid_indices_2_1 = np.logical_and(class_indices[:, :, from_index+1] >= 0, class_indices[:, :, from_index+2] < 0, class_indices[:, :, from_index+2] < 0)
        hover_text[valid_indices_2_1] = [
            f"{label_list_base[class_indices[row_index, col_index, from_index+0]]}: {score_values[row_index, col_index, from_index+0]:.2f}" +
            f" ({label_list_base[class_indices[row_index, col_index, from_index+1]]}: {score_values[row_index, col_index, from_index+1]:.2f})"
            for row_index, col_index in zip(*np.where(valid_indices_2_1))
        ]

        valid_indices_2_0 = np.logical_and(class_indices[:, :, from_index+0] >= 0, class_indices[:, :, from_index+1] < 0, class_indices[:, :, from_index+2] < 0)
        hover_text[valid_indices_2_0] = [
            f"{label_list_base[class_indices[row_index, col_index, from_index+0]]}: {score_values[row_index, col_index, from_index+0]:.2f}"
            for row_index, col_index in zip(*np.where(valid_indices_2_0))
        ]
    elif score_values.shape[2] == from_index+2:
        valid_indices_1 = class_indices[:, :, from_index+1] >= 0
        hover_text[valid_indices_1] = [
            f"{label_list_base[class_indices[row_index, col_index, from_index+0]]}: {score_values[row_index, col_index, from_index+0]:.2f}" +
            f" ({label_list_base[class_indices[row_index, col_index, from_index+1]]}: {score_values[row_index, col_index, from_index+1]:.2f})"
            for row_index, col_index in zip(*np.where(valid_indices_1))
        ]

        valid_indices_1_0 = np.logical_and(class_indices[:, :, from_index+0] >= 0, class_indices[:, :, from_index+1] < 0)
        hover_text[valid_indices_1_0] = [
            f"{label_list_base[class_indices[row_index, col_index, from_index+0]]}: {score_values[row_index, col_index, from_index+0]:.2f}"
            for row_index, col_index in zip(*np.where(valid_indices_1_0))
        ]
    else:
        hover_text[valid_indices_0] = [
            f"{label_list_base[class_indices[row_index, col_index, from_index+0]]}: {score_values[row_index, col_index, from_index+0]:.2f}"
            for row_index, col_index in zip(*np.where(valid_indices_0))
        ]
    hover_text[~valid_indices_0] = [
        f"pre-filtered: {score_values[row_index, col_index, from_index+0]:.2f}"
        for row_index, col_index in zip(*np.where(~valid_indices_0))
    ]

    return hover_text


def create_patch_score_heatmap(scores, min_value, max_value, hover_text):
    fig = px.imshow(scores, zmin=min_value, zmax=max_value, height=600)  # Set the color scale range# , text_auto=True)
    fig.update_layout(
        margin=dict(l=40, r=80, t=0, b=0)  # Add the same right margin as the first plot
    )
    fig.data[0].customdata = hover_text
    # Set hovertemplate to show the label
    fig.data[0].hovertemplate = "Label: %{customdata}<extra></extra>"

    return fig

def convert_to_rgba(colors, alpha=0.5):
    # convert colors from rgb ('rgb(228,10,5)') to rgba ('rgba(228,10,5,0.5)'); the values can have up one, two, or three digits
    for i in range(len(colors)):
        color = colors[i]
        if color.startswith('rgb'):
            # Extract the RGB values
            rgb_values = color[color.index('(') + 1:color.index(')')].split(',')
            # Convert to RGBA with 0.5 alpha
            colors[i] = f'rgba({rgb_values[0]},{rgb_values[1]},{rgb_values[2]},{alpha})'
        else:
            # Extract the RGB values from hex
            rgb_values = [0, 0, 0]
            if color.startswith('#'):
                # Convert hex to RGB
                hex_color = color.lstrip('#')
                rgb_values = [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]
            # Convert to RGBA with 0.5 alpha
            colors[i] = f'rgba({rgb_values[0]},{rgb_values[1]},{rgb_values[2]},{alpha})'
    return colors


def create_patch_class_figure(image_file, background_mask_image, scores, class_indices, bbox, top_k, label_list_base, hover_text, threshold,
                              mask_background=False):
    # create a 2D array of the class labels by using the
    #class_label_array = [[label_list_base[index] for index in row] for row in results['image_patch_top_class_indices']]

    n_colors = len(label_list_base)
    #colors = px.colors.sample_colorscale("Viridis", [i / (n_colors - 1) for i in range(n_colors)])
    colors = px.colors.qualitative.Light24
    colors = convert_to_rgba(colors)

    if len(colors) < n_colors:
        while len(colors) < n_colors:
            colors.extend(colors)

    # add a transparent color at the front of the list
    colors = ['rgba(0,0,0,0)'] + colors
    len_colors = len(colors)
    # Set the class label to None if the score is below the threshold
    '''
    class_image = [[None if scores[row_index][col_index][top_k] < threshold else \
                    class_indices[row_index][col_index][top_k] \
                    for col_index, index in enumerate(row)] \
                    for row_index, row in enumerate(class_indices[:, :, top_k])]
    '''
    class_image = [
        [
            class_indices[row_index][col_index][top_k] if scores[row_index][col_index][top_k] >= threshold else None
            for col_index, _ in enumerate(row)
        ]
        for row_index, row in enumerate(class_indices[:, :, top_k])
    ]

    # get list of unique values in the class_image
    unique_values = sorted({value for row in class_image for value in row if value is not None})
    # map the unique values to the colors so that the gab between the colors is as big as possible
    color_map = {}
    prev_color_index = -10
    for i, value in enumerate(unique_values):
        color_index = i % len_colors
        # if the color index is the same as the previous one, add 1 to it
        if abs(color_index-prev_color_index) < 10:
            color_index = (color_index + int(len_colors/2)) % len_colors
        color_map[value] =color_index
        prev_color_index = color_index

    # create a new image with the colors
    color_image = [[color_map[value] if value is not None else None for value in row] for row in class_image]

    fig = px.imshow(class_image, text_auto=True, color_continuous_scale=colors, height=600)
    #                color_continuous_scale=None,  # Disable continuous scale
    #                color_discrete_sequence=colors)  # Use discrete colors

    # has no effect!!!!!!!
    fig.update_layout(
        margin=dict(l=40, t=0, b=0)  # Add the same right margin as the first plot
    )
    # does not work because only rgb colors are supported
    '''
    color_scale = []
    for i in range(len(colors)):
        color_scale.append((i / len(colors), colors[i]))

    fig.update_coloraxes(
        showscale=False,
        colorscale=color_scale
    )
    '''

    fig.update_xaxes(showgrid=False, showline=False, visible=True)
    fig.update_yaxes(showgrid=False, showline=False, visible=True)
    fig.update_layout(coloraxis_showscale=False)


    fig.data[0].customdata = hover_text
    # Set hovertemplate to show the label
    fig.data[0].hovertemplate = "Label: %{customdata}<extra></extra>"

    img = cv2.imread(image_file)
    if background_mask_image is not None and mask_background:
        mask_image = cv2.imread(background_mask_image, cv2.IMREAD_GRAYSCALE)
        # mask out the background
        img = cv2.bitwise_and(img, img, mask=mask_image)

    if (bbox[0] + bbox[2] > img.shape[1]) or (bbox[1] + bbox[3] > img.shape[0]):
        # create a new image with the required size
        bg_img = np.zeros((bbox[3], bbox[2], 3), dtype=np.uint8)
        # copy the original image to the new image in the box area
        copy_img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        bg_img[0:copy_img.shape[0], 0: copy_img.shape[1]] = copy_img
    else:
        bg_img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]  # Crop the image to the bounding box

    # Set image dimensions (in data coordinates)
    img_width, img_height = img.shape[1], img.shape[0]

    # Convert BGR to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert the image to a base64 string
    _, buffer = cv2.imencode('.png', bg_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    bg_img = f"data:image/png;base64,{img_base64}"

    # Add the background image
    fig.add_layout_image(
        source=bg_img,
        xref="x",
        yref="y",
        x=-0.5,
        y=-0.5,
        xanchor="left",
        yanchor="top",
        layer="below",
        sizing="stretch",
        sizex=len(class_indices[0]),
        sizey=len(class_indices),
        # opacity=0.5
    )

    # Set axes to match image dimensions for correct alignment
    # fig.update_xaxes(showgrid=False, range=[0, img_width])
    # fig.update_yaxes(showgrid=False, range=[0, img_height], scaleanchor="x")

    return fig